#if canImport(Metal)
import Foundation
import Metal

/// The render from `rgb_pre` onward on the GPU.
///
/// The frame is uploaded once, stays float32 on the GPU through filming, printing and scanning,
/// and is downloaded once at the collected tap. Each stage mirrors its CPU counterpart operator for
/// operator, with the same constants, which the CPU stages compute and this reads.
final class MetalPipeline {
    private let c: MetalContext
    private let params: RuntimePhotoParams
    private let resizing: ResizingService
    private let filming: FilmingStage
    private let printing: PrintingStage
    private let scanning: ScanningStage

    private let converter: Hanatos2025RawConverter
    private let lut: MTLBuffer
    private let lutSize: Int
    private let filmCurves: [Double]
    private let filmTable: MetalFilm.CurveTable
    private let compressor: OutputGamutCompressor

    /// Whether every operator these parameters reach has a Metal path. The FFT diffusion filters
    /// and the non-default upsampling do not, and those renders stay on the CPU.
    static func supports(_ params: RuntimePhotoParams) -> Bool {
        params.settings.rgbToRawMethod == .hanatos2025
            && !params.camera.diffusionFilter.active
            && !params.enlarger.diffusionFilter.active
    }

    init(
        context: MetalContext, params: RuntimePhotoParams, resizing: ResizingService,
        filming: FilmingStage, printing: PrintingStage, scanning: ScanningStage
    ) throws {
        guard let tcLUT = filming.tcLUT else {
            throw SpektraError.unsupportedSetting("Metal pipeline", value: "no tc_lut")
        }
        c = context
        self.params = params
        self.resizing = resizing
        self.filming = filming
        self.printing = printing
        self.scanning = scanning

        converter = Hanatos2025RawConverter(
            colourSpace: filming.inputColourSpace, applyCCTFDecoding: params.io.inputCCTFDecoding,
            referenceIlluminant: filming.referenceIlluminant, tcLUT: tcLUT)
        lut = try context.buffer(from: tcLUT.values)
        lutSize = tcLUT.height

        let data = params.film.data
        filmCurves = DensityCurves.normalized(
            curves: data.densityCurves, minima: data.densityCurveMinima)
        let gamma = params.filmRender.densityCurveGamma
        filmTable = try MetalFilm.CurveTable(
            context, curves: filmCurves, logExposure: data.logExposure,
            gamma: (gamma, gamma, gamma))
        compressor = try OutputGamutCompressor(
            spec: params.io.outputGamutCompress, colourSpace: scanning.outputColourSpace)
    }

    /// Runs from `rgb_pre` to `collect`, adding each stage's wall-clock time to `timings` under
    /// the CPU topology's labels.
    func run(
        _ rgbPre: ImageBuffer, collect: Tap, timings: inout [String: TimeInterval]
    ) throws -> ImageBuffer {
        func timed<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
            let start = DispatchTime.now().uptimeNanoseconds
            defer {
                timings[label, default: 0] +=
                    Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
            }
            return try body()
        }

        // Each stage takes the only reference to its input, so a consumed frame is freed before
        // the next one is allocated. Peak memory is what caps export size.
        var frame: GPUFrame? = try timed("filming.expose") { try expose(rgbPre) }
        if collect == .logExposureFilm { return frame!.download() }

        frame = try timed("filming.develop") { try developFilm(taking: &frame) }
        if collect == .cmyFilm { return frame!.download() }

        if params.io.scanFilm {
            return try timed("scanning.scan_film") { try scan(taking: &frame) }.download()
        }

        frame = try timed("printing.expose") { try exposePrint(taking: &frame) }
        if collect == .logExposurePrint { return frame!.download() }

        frame = try timed("printing.develop") { try developPrint(frame!) }
        if collect == .cmyPrint { return frame!.download() }

        return try timed("scanning.scan_print") { try scan(taking: &frame) }.download()
    }

    /// Moves the frame out of `slot`, leaving it empty.
    private func take(_ slot: inout GPUFrame?) -> GPUFrame {
        defer { slot = nil }
        return slot!
    }

    // MARK: - Filming

    /// ``FilmingStage/expose(_:)``.
    private func expose(_ rgbPre: ImageBuffer) throws -> GPUFrame {
        var raw: GPUFrame
        do {
            let rgb = try GPUFrame(c, uploading: rgbPre)
            raw = try MetalFilm.rgbToRaw(c, rgb, converter: converter, lut: lut, lutSize: lutSize)
        }
        try MetalElementwise.scale(c, raw, by: pow(2.0, params.camera.exposureCompensationEV))

        let halation = params.filmRender.halation
        try MetalStage.boostHighlights(
            c, &raw, boostEV: halation.boostEV, boostRange: halation.boostRange,
            protectEV: halation.protectEV)

        if let pixelSize = resizing.pixelSizeMicrons {
            let lensBlur = params.camera.lensBlurMicrons
            if lensBlur > 0, lensBlur / pixelSize > 0 {
                raw = try MetalBlur.gaussian(
                    c, raw, sigmaPerChannel: [Double](repeating: lensBlur / pixelSize, count: 3))
            }
            if halation.active {
                try MetalStage.halation(
                    c, raw, halation, pixelSizeMicrons: pixelSize,
                    planes: MetalBlur.Planes(c, like: raw))
            }
        }

        let correction = try filming.colourReference.filmingExposureCorrection()
        try MetalElementwise.scaleLog10Guard(c, raw, scale: correction)
        return raw
    }

    /// ``Develop/film``: the density curves, the coupler correction and grain.
    private func developFilm(taking slot: inout GPUFrame?) throws -> GPUFrame {
        let data = params.film.data
        let density: GPUFrame
        do {
            // The log exposure is last read by the coupler correction, so it is freed before grain.
            let logRaw = take(&slot)
            density = try GPUFrame(c, height: logRaw.height, width: logRaw.width, channels: 3)
            try MetalFilm.interpolate(c, logRaw, into: density, table: filmTable)

            let couplers = params.filmRender.dirCouplers
            if couplers.active {
                let setup = Couplers.CorrectionSetup(
                    pixelSizeMicrons: resizing.pixelSizeMicrons, logExposure: data.logExposure,
                    curves: filmCurves, params: couplers, positive: params.film.isPositive)
                let gamma = params.filmRender.densityCurveGamma
                let before = try MetalFilm.CurveTable(
                    c, curves: setup.curvesBefore, logExposure: data.logExposure,
                    gamma: (gamma, gamma, gamma))
                try MetalStage.couplerCorrection(
                    c, density: density, logRaw: logRaw, setup: setup,
                    tailWeight: couplers.diffusionTailWeight, positive: params.film.isPositive,
                    before: before,
                    planes: setup.diffusionSizePixels > 0
                        ? MetalBlur.Planes(c, like: density) : nil)
            }
        }

        guard let pixelSize = resizing.pixelSizeMicrons else { return density }
        return try grain(density, pixelSizeMicrons: pixelSize)
    }

    /// ``Grain/apply``, with the CPU's seed.
    private func grain(_ density: GPUFrame, pixelSizeMicrons: Double) throws -> GPUFrame {
        try MetalGrain.apply(
            c, density, pixelSizeMicrons: pixelSizeMicrons, params: params.filmRender.grain,
            densityCurves: filmCurves, densityCurvesLayers: params.film.data.densityCurvesLayers,
            positive: params.film.isPositive, seed: 0, bypass: false)
    }

    // MARK: - Printing

    /// ``PrintingStage/expose(_:)``.
    private func exposePrint(taking slot: inout GPUFrame?) throws -> GPUFrame {
        printing.prepareColourReferences()
        let illuminant = printing.enlarger.filteredIlluminant(printing.lampSpectrum)
        let raw = try MetalSpectralContraction.project(
            c, frame: take(&slot), channelDensity: params.film.data.channelDensity,
            baseDensity: params.film.data.baseDensity, illuminant: illuminant,
            response: printing.paperSensitivity)
        try MetalElementwise.affine3(
            c, raw, factor: printing.exposureFactorMidgray(printIlluminant: illuminant),
            offset: printing.rawPreflash(printIlluminant: illuminant))
        try MetalElementwise.scaleLog10Guard(c, raw)
        try MetalElementwise.exp10Scale(c, raw, scale: try printing.printExposureScale())
        try MetalElementwise.scaleLog10Guard(c, raw)
        return raw
    }

    /// ``Develop/print(logRaw:profile:morph:)``.
    private func developPrint(_ logRaw: GPUFrame) throws -> GPUFrame {
        let data = params.print.data
        let curves = try PrintCurvesMorph.apply(
            logExposure: data.logExposure, model: data.densityCurvesModel,
            params: params.printRender.densityCurvesMorph, positive: params.print.isPositive)
        let table = try MetalFilm.CurveTable(
            c, curves: curves, logExposure: data.logExposure, gamma: (1, 1, 1))
        try MetalFilm.interpolate(c, logRaw, into: logRaw, table: table)
        return logRaw
    }

    // MARK: - Scanning

    /// ``ScanningStage/scan(_:)``.
    private func scan(taking slot: inout GPUFrame?) throws -> GPUFrame {
        let s = scanning
        let xyz = try MetalSpectralContraction.project(
            c, frame: take(&slot), channelDensity: s.channelDensity, baseDensity: s.baseDensity,
            illuminant: s.scanIlluminant, response: Observer.cmfs, scale: 1.0 / s.normalisation)
        try MetalElementwise.scaleLog10Guard(c, xyz)
        try MetalElementwise.exp10Scale(c, xyz)

        if let line = try s.colourReference.xyzCorrection() {
            var k = SIMD2<Float>(Float(line.m), Float(line.q))
            var pixels = UInt32(xyz.pixelCount)
            try c.dispatch("xyz_correct", count: xyz.pixelCount) { e in
                e.setBuffer(xyz.buffer, offset: 0, index: 0)
                e.setBytes(&k, length: 8, index: 1)
                e.setBytes(&pixels, length: 4, index: 2)
            }
        }

        let illuminantXYZ = Observer.illuminantXYZ(s.scanIlluminant)
        try addGlare(xyz, illuminantXYZ: illuminantXYZ)

        let space = s.outputColourSpace
        let cat = Colour.chromaticAdaptationMatrix(
            from: Colour.XYZToxy(illuminantXYZ).XYZ, to: space.whitepoint.XYZ, transform: .cat02)
        try MetalElementwise.matrix3(c, xyz, cat)
        try MetalElementwise.matrix3(c, xyz, space.matrixXYZToRGB)

        var rgb = try compress(xyz)

        if s.scanner.lensBlur > 0 {
            rgb = try MetalBlur.gaussian(
                c, rgb, sigmaPerChannel: [Double](repeating: s.scanner.lensBlur, count: 3))
        }
        let (sigma, amount) = s.scanner.unsharpMask
        if sigma > 0 && amount > 0 {
            let blurred = try MetalBlur.gaussian(
                c, rgb, sigmaPerChannel: [Double](repeating: sigma, count: 3))
            try MetalElementwise.unsharp(c, rgb, blurred, amount: amount)
        }

        if params.io.outputCCTFEncoding {
            try MetalElementwise.matrix3(
                c, rgb, Colour.matrixRGBToRGB(from: space, to: space))
            try MetalElementwise.transfer(c, rgb, space.transfer, encode: true)
        }
        return rgb
    }

    /// ``ScanningStage/addGlare(_:illuminantXYZ:)``.
    private func addGlare(_ xyz: GPUFrame, illuminantXYZ: (Double, Double, Double)) throws {
        guard !params.io.scanFilm else { return }
        let glare = params.printRender.glare
        guard glare.active, glare.percent > 0 else { return }

        var field = try GPUFrame(c, height: xyz.height, width: xyz.width, channels: 1)
        let (mu, sigma) = Distributions.lognormalLogParameters(
            mean: glare.percent, std: glare.roughness * glare.percent)
        var k = SIMD2<Float>(Float(mu), Float(sigma))
        let key = Glare.key(seed: 0)
        var seed = key.seed
        var stream = SIMD2<UInt32>(key.channel, key.sublayer)
        var pixels = UInt32(xyz.pixelCount)
        try c.dispatch("glare_field", count: xyz.pixelCount) { e in
            e.setBuffer(field.buffer, offset: 0, index: 0)
            e.setBytes(&k, length: 8, index: 1)
            e.setBytes(&seed, length: 8, index: 2)
            e.setBytes(&stream, length: 8, index: 3)
            e.setBytes(&pixels, length: 4, index: 4)
        }
        if glare.blur > 0 {
            field = try MetalBlur.gaussian(c, field, sigmaPerChannel: [glare.blur])
        }
        var illuminant = SIMD3<Float>(
            Float(illuminantXYZ.0), Float(illuminantXYZ.1), Float(illuminantXYZ.2))
        try c.dispatch("glare_add", count: xyz.pixelCount) { e in
            e.setBuffer(xyz.buffer, offset: 0, index: 0)
            e.setBuffer(field.buffer, offset: 0, index: 1)
            e.setBytes(&illuminant, length: 16, index: 2)
            e.setBytes(&pixels, length: 4, index: 3)
        }
    }

    /// ``OutputGamutCompressor/apply(to:)``, in place.
    private func compress(_ rgb: GPUFrame) throws -> GPUFrame {
        try MetalGamut.apply(c, compressor, to: rgb)
        return rgb
    }
}
#endif

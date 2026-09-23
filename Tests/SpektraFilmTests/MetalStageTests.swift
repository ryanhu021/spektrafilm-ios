#if canImport(Metal)
import Foundation
import Testing

@testable import SpektraFilm

/// Highlight boost, halation and the coupler correction against the CPU operators.
@Suite("Metal stage operators", .enabled(if: MetalContext.shared != nil))
struct MetalStageTests {

    static func image(height: Int = 83, width: Int = 117, scale: Double) -> ImageBuffer {
        var v = [Double](repeating: 0, count: height * width * 3)
        for i in v.indices { v[i] = Double((i * 7919) % 1000) / 1000.0 * scale }
        return ImageBuffer(height: height, width: width, channels: 3, values: v)
    }

    static func digested() throws -> RuntimePhotoParams {
        try ParamsBuilder.digest(
            RuntimePhotoParams.make(film: "kodak_portra_400", print: "kodak_portra_endura"))
    }

    static func worstRelative(_ a: [Double], _ b: [Double]) -> Double {
        zip(a, b).map { abs($0 - $1) / Swift.max(1e-3, abs($1)) }.max()!
    }

    @Test("highlight boost matches")
    func boost() throws {
        let c = try #require(MetalContext.shared)
        let input = Self.image(scale: 3.0)
        var cpu = input
        Diffusion.boostHighlights(&cpu, boostEV: 1.5, boostRange: 0.4, protectEV: 1.0)
        var frame = try GPUFrame(c, uploading: input)
        try MetalStage.boostHighlights(c, &frame, boostEV: 1.5, boostRange: 0.4, protectEV: 1.0)
        let worst = Self.worstRelative(frame.download().values, cpu.values)
        // Measured at 4.4e-7.
        #expect(worst < 1e-6, "worst relative difference \(worst)")
    }

    /// 8.75 um puts the widths below the IIR switch, 1.5 um above it.
    @Test("halation matches", arguments: [8.75, 1.5])
    func halation(pixelSize: Double) throws {
        let c = try #require(MetalContext.shared)
        let params = try Self.digested().filmRender.halation
        let input = Self.image(scale: 2.0)
        let cpu = Diffusion.applyHalation(input, params, pixelSizeMicrons: pixelSize)
        let frame = try GPUFrame(c, uploading: input)
        try MetalStage.halation(
            c, frame, params, pixelSizeMicrons: pixelSize, planes: MetalBlur.Planes(c, like: frame))
        let worst = Self.worstRelative(frame.download().values, cpu.values)
        // Measured at 2.7e-7 to 2.8e-7.
        #expect(worst < 1e-6, "worst relative difference \(worst)")
    }

    @Test("the coupler correction matches", arguments: [nil, 8.75, 2.0] as [Double?])
    func couplers(pixelSize: Double?) throws {
        let c = try #require(MetalContext.shared)
        let params = try Self.digested()
        let data = params.film.data
        let curves = DensityCurves.normalized(
            curves: data.densityCurves, minima: data.densityCurveMinima)
        var logRaw = Self.image(scale: 4.5)
        logRaw.transformInPlace { $0 - 3.0 }
        let gamma = params.filmRender.densityCurveGamma

        let density = DensityCurves.densityFromLogExposure(
            logExposure: logRaw, curves: curves, axis: data.logExposure, gammaFactor: gamma)
        let cpu = Couplers.applyDensityCorrection(
            density: density, logRaw: logRaw, pixelSizeMicrons: pixelSize,
            logExposure: data.logExposure, curves: curves, params: params.filmRender.dirCouplers,
            positive: false, gammaFactor: gamma, spatial: FastSpatialFilter())

        let setup = Couplers.CorrectionSetup(
            pixelSizeMicrons: pixelSize, logExposure: data.logExposure, curves: curves,
            params: params.filmRender.dirCouplers, positive: false)
        let raw = try GPUFrame(c, uploading: logRaw)
        let table = try MetalFilm.CurveTable(
            c, curves: curves, logExposure: data.logExposure, gamma: (gamma, gamma, gamma))
        let before = try MetalFilm.CurveTable(
            c, curves: setup.curvesBefore, logExposure: data.logExposure,
            gamma: (gamma, gamma, gamma))
        let gpuDensity = try GPUFrame(c, height: raw.height, width: raw.width, channels: 3)
        try MetalFilm.interpolate(c, raw, into: gpuDensity, table: table)
        try MetalStage.couplerCorrection(
            c, density: gpuDensity, logRaw: raw, setup: setup,
            tailWeight: params.filmRender.dirCouplers.diffusionTailWeight, positive: false,
            before: before, planes: MetalBlur.Planes(c, like: gpuDensity))
        let gpu = gpuDensity.download()
        let worst = zip(gpu.values, cpu.values).map { abs($0 - $1) }.max()!
        // Measured at 2.3e-7 to 2.6e-7, with and without the spatial diffusion.
        #expect(worst < 1e-5, "worst density difference \(worst)")
    }
}
#endif

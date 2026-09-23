import Foundation

/// Scanning: dye density to output RGB.
///
/// Ports `runtime/stages/scanning.py`. The medium is lit by its viewing illuminant, the transmitted
/// or reflected spectrum is projected onto the CIE observer, and the resulting XYZ is converted to
/// the output colourspace, compressed into its gamut, and encoded.
///
/// Whether the negative or the print is scanned is decided by `io.scanFilm`, and it changes which
/// profile supplies the dyes, the base density and the illuminant.
public final class ScanningStage {
    private let film: Profile
    private let filmRender: FilmRenderingParams
    private let print: Profile
    private let printRender: PrintRenderingParams
    private let scanner: ScannerParams
    private let io: IOParams
    private let settings: SettingsParams
    private let colourReference: ColorReferenceService
    private let spatial: any SpatialFilter
    private let outputColourSpace: ColourSpace

    /// The medium being scanned, and the light it is viewed under.
    private let channelDensity: [Double]
    private let baseDensity: [Double]
    private let scanIlluminant: [Double]
    private let normalisation: Double

    public init(
        film: Profile,
        filmRender: FilmRenderingParams,
        print: Profile,
        printRender: PrintRenderingParams,
        scanner: ScannerParams,
        io: IOParams,
        settings: SettingsParams,
        colourReference: ColorReferenceService,
        spatial: any SpatialFilter
    ) throws {
        self.film = film
        self.filmRender = filmRender
        self.print = print
        self.printRender = printRender
        self.scanner = scanner
        self.io = io
        self.settings = settings
        self.colourReference = colourReference
        self.spatial = spatial
        outputColourSpace = try ColourSpace.named(io.outputColourSpace)

        let medium = io.scanFilm ? film : print
        channelDensity = medium.data.channelDensity
        baseDensity = medium.data.baseDensity
        scanIlluminant = try Illuminant(label: medium.info.viewingIlluminant).spectrum
        normalisation = Observer.luminanceNormalisation(illuminant: scanIlluminant)

        // The colour reference service needs this conversion to measure its black and white points,
        // and only this stage knows which medium and illuminant apply.
        let dyes = channelDensity
        let base = baseDensity
        let illuminant = scanIlluminant
        let norm = normalisation
        colourReference.cmyToLogXYZ = { cmy in
            Self.cmyToLogXYZ(
                cmy, channelDensity: dyes, baseDensity: base, illuminant: illuminant,
                normalisation: norm)
        }
    }

    /// `scan`.
    public func scan(_ density: ImageBuffer) throws -> ImageBuffer {
        var rgb = try densityToRGB(density)
        rgb = applyBlurAndUnsharp(rgb)
        return applyCCTFEncoding(rgb)
    }

    // MARK: - Density to RGB

    private func densityToRGB(_ density: ImageBuffer) throws -> ImageBuffer {
        let glare = io.scanFilm ? nil : printRender.glare

        let logXYZ = spectralCompute(density)
        var xyz = logXYZ
        xyz.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] = Foundation.pow(10.0, p[i]) }
        }
        xyz = try colourReference.correctXYZ(xyz)

        let illuminantXYZ = Observer.illuminantXYZ(scanIlluminant)
        let illuminantXY = Colour.XYZToxy(illuminantXYZ)
        xyz = Glare.add(
            xyz,
            illuminantXYZ: [illuminantXYZ.0, illuminantXYZ.1, illuminantXYZ.2],
            glare: glare,
            spatial: spatial)

        Colour.XYZToRGB(&xyz, colourspace: outputColourSpace, illuminant: illuminantXY)
        try OutputGamutCompression.compress(
            &xyz, spec: io.outputGamutCompress, colourSpace: outputColourSpace)
        return xyz
    }

    /// The per-pixel spectral map, in row bands.
    ///
    /// `use_scanner_lut` replaces this with a coarse 3D LUT in the reference. It defaults off and
    /// the reference itself calls the LUT an approximation, so the direct path is the only one here.
    private func spectralCompute(_ density: ImageBuffer) -> ImageBuffer {
        let bandRows = ImageBuffer.bandRows(
            width: density.width, channels: ColourTables.wavelengthCount)
        return density.mapPerPixel(bandRows: bandRows, channelsOut: 3) { band in
            Self.cmyToLogXYZ(
                band, channelDensity: channelDensity, baseDensity: baseDensity,
                illuminant: scanIlluminant, normalisation: normalisation)
        }
    }

    /// `cmy_to_log_xyz`. Density to spectrum, spectrum to transmitted light, light to XYZ.
    static func cmyToLogXYZ(
        _ cmy: ImageBuffer,
        channelDensity: [Double],
        baseDensity: [Double],
        illuminant: [Double],
        normalisation: Double
    ) -> ImageBuffer {
        let spectral = DensityCurves.spectralDensity(
            cmy: cmy, channelDensity: channelDensity, baseDensity: baseDensity)
        let light = DensityCurves.densityToLight(spectral, illuminant: illuminant)
        var xyz = DensityCurves.project(light, onto: Observer.cmfs, scale: 1.0 / normalisation)
        xyz.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] = log10Guard(p[i]) }
        }
        return xyz
    }

    // MARK: - Sharpening

    private func applyBlurAndUnsharp(_ rgb: ImageBuffer) -> ImageBuffer {
        var out = rgb
        if scanner.lensBlur > 0 {
            out = Diffusion.applyGaussianBlur(out, sigmaPixels: scanner.lensBlur)
        }
        let (sigma, amount) = scanner.unsharpMask
        if sigma > 0 && amount > 0 {
            out = Diffusion.applyUnsharpMask(out, sigma: sigma, amount: amount)
        }
        return out
    }

    private func applyCCTFEncoding(_ rgb: ImageBuffer) -> ImageBuffer {
        guard io.outputCCTFEncoding else { return rgb }
        var out = rgb
        // The reference routes this through RGB_to_RGB with the same space in and out, which applies
        // a near-identity matrix before the transfer function.
        Colour.RGBToRGB(
            &out, from: outputColourSpace, to: outputColourSpace, applyEncoding: true)
        return out
    }
}

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
        applyBlurAndUnsharp(&rgb)
        applyCCTFEncoding(&rgb)
        return rgb
    }

    // MARK: - Density to RGB

    private func densityToRGB(_ density: ImageBuffer) throws -> ImageBuffer {
        // One binding for the frame: a second one keeps the storage shared, and the in-place pass
        // below then copies it.
        var xyz = spectralCompute(density)
        xyz.transformInPlace { Foundation.pow(10.0, $0) }
        xyz = try colourReference.correctXYZ(xyz)

        let illuminantXYZ = Observer.illuminantXYZ(scanIlluminant)
        let illuminantXY = Colour.XYZToxy(illuminantXYZ)
        addGlare(&xyz, illuminantXYZ: illuminantXYZ)

        Colour.XYZToRGB(&xyz, colourspace: outputColourSpace, illuminant: illuminantXY)
        try OutputGamutCompression.compress(
            &xyz, spec: io.outputGamutCompress, colourSpace: outputColourSpace)
        return xyz
    }

    /// `add_glare`, applied in place.
    ///
    /// ``Glare/add(_:illuminantXYZ:glare:seed:spatial:)`` returns a new frame while this stage still
    /// holds the old one, so the two would overlap. The glare field is still a whole plane: it is
    /// one channel, and the blur needs all of it.
    private func addGlare(_ xyz: inout ImageBuffer, illuminantXYZ: (Double, Double, Double)) {
        // The reference passes no glare on the scan-film branch. `film_render.glare` is dead.
        guard !io.scanFilm else { return }
        let params = printRender.glare
        guard params.active, params.percent > 0 else { return }

        let field = Glare.randomAmount(
            amount: params.percent,
            roughness: params.roughness,
            blur: params.blur,
            height: xyz.height,
            width: xyz.width,
            spatial: spatial)
        let illuminant = [illuminantXYZ.0, illuminantXYZ.1, illuminantXYZ.2]
        xyz.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for pixel in 0..<field.values.count {
                let flare = field.values[pixel]
                for channel in 0..<3 { p[pixel * 3 + channel] += flare * illuminant[channel] }
            }
        }
    }

    /// The per-pixel spectral map.
    ///
    /// In the reference, `use_scanner_lut` replaces this with a coarse 3D LUT. It defaults off and
    /// the reference calls the LUT an approximation, so only the direct path is ported.
    private func spectralCompute(_ density: ImageBuffer) -> ImageBuffer {
        Self.cmyToLogXYZ(
            density, channelDensity: channelDensity, baseDensity: baseDensity,
            illuminant: scanIlluminant, normalisation: normalisation)
    }

    /// `cmy_to_log_xyz`. Density to spectrum, spectrum to transmitted light, light to XYZ.
    static func cmyToLogXYZ(
        _ cmy: ImageBuffer,
        channelDensity: [Double],
        baseDensity: [Double],
        illuminant: [Double],
        normalisation: Double
    ) -> ImageBuffer {
        var xyz = SpectralContraction.project(
            cmy: cmy, channelDensity: channelDensity, baseDensity: baseDensity,
            illuminant: illuminant, response: Observer.cmfs, scale: 1.0 / normalisation)
        xyz.transformInPlace { log10Guard($0) }
        return xyz
    }

    // MARK: - Sharpening

    /// The unsharp mask is expanded here instead of calling
    /// ``Diffusion/applyUnsharpMask(_:sigma:amount:)``, which returns a new frame while this stage
    /// still holds the old one and the blur, three frames at once.
    private func applyBlurAndUnsharp(_ rgb: inout ImageBuffer) {
        if scanner.lensBlur > 0 {
            rgb = Diffusion.applyGaussianBlur(rgb, sigmaPixels: scanner.lensBlur)
        }
        let (sigma, amount) = scanner.unsharpMask
        guard sigma > 0 && amount > 0 else { return }
        let blurred = Diffusion.applyGaussianBlur(rgb, sigmaPixels: sigma)
        rgb.combineInPlace(with: blurred) { value, blurredValue in
            value + amount * (value - blurredValue)
        }
    }

    private func applyCCTFEncoding(_ rgb: inout ImageBuffer) {
        guard io.outputCCTFEncoding else { return }
        // The reference routes this through RGB_to_RGB with the same space in and out, which applies
        // a near-identity matrix before the transfer function.
        Colour.RGBToRGB(
            &rgb, from: outputColourSpace, to: outputColourSpace, applyEncoding: true)
    }
}

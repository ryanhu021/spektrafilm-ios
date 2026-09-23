import Foundation

/// RGB to per-film camera raw.
///
/// `raw(rgb) = b(rgb) * tcLUT[tc(rgb)]`, where `b = X + Y + Z` is a scalar brightness and `tc` is the
/// reparameterised chromaticity of ``ChromaticityCoordinates``. The reconstructed spectrum depends
/// only on chromaticity and the sensitivity integral is linear in it, so the whole 81-sample
/// reconstruction collapses into a 2D table built once per film. Per pixel this is one 3x3 matrix
/// product, two divides, a 16-tap LUT fetch and a multiply.
public enum SpectralUpsampling {

    /// Dispatch on `settings.rgb_to_raw_method`.
    ///
    /// `mallett2019` is not ported. It is a single 3x3 matrix valid only inside sRGB, it is not the
    /// default, and it needs a colour table (`MSDS_BASIS_FUNCTIONS_sRGB_MALLETT2019`) that
    /// ``ColourTables`` does not have.
    public static func rgbToRaw(
        method: SettingsParams.RGBToRawMethod,
        rgb: ImageBuffer,
        sensitivity: SpectralMatrix,
        colourSpace: ColourSpace,
        applyCCTFDecoding: Bool,
        referenceIlluminant: Illuminant,
        tcLUT: ImageBuffer? = nil
    ) throws -> ImageBuffer {
        switch method {
        case .hanatos2025:
            return try rgbToRawHanatos2025(
                rgb: rgb,
                sensitivity: sensitivity,
                colourSpace: colourSpace,
                applyCCTFDecoding: applyCCTFDecoding,
                referenceIlluminant: referenceIlluminant,
                tcLUT: tcLUT)
        case .mallett2019:
            throw SpektraError.unsupportedSetting("settings.rgb_to_raw_method", value: method.rawValue)
        }
    }

    /// `rgb_to_raw_hanatos2025`.
    ///
    /// `sensitivity` is read **only** when `tcLUT` is nil, which is the reference's uncached fallback:
    /// it rebuilds the LUT with `HANATOS2025_NO_ADAPTATION` on every call. The render path passes the
    /// LUT that `FilmingStage` builds once at construction.
    public static func rgbToRawHanatos2025(
        rgb: ImageBuffer,
        sensitivity: SpectralMatrix,
        colourSpace: ColourSpace,
        applyCCTFDecoding: Bool,
        referenceIlluminant: Illuminant,
        tcLUT: ImageBuffer? = nil,
        sampler: any LUT2DSampler = MitchellLUT2DSampler()
    ) throws -> ImageBuffer {
        let lut =
            try tcLUT
            ?? TCLUTBuilder.computeHanatos2025TCLUT(
                sensitivity: sensitivity, adaptation: .noAdaptation)
        let converter = Hanatos2025RawConverter(
            colourSpace: colourSpace,
            applyCCTFDecoding: applyCCTFDecoding,
            referenceIlluminant: referenceIlluminant,
            tcLUT: lut,
            sampler: sampler)
        return converter.raw(rgb: rgb)
    }

    /// `_rgb_to_tc_b`. The tc buffer has 2 channels; brightness is one value per pixel.
    public static func rgbToTCB(
        rgb: ImageBuffer,
        colourSpace: ColourSpace,
        applyCCTFDecoding: Bool,
        referenceIlluminant: Illuminant
    ) -> (tc: ImageBuffer, brightness: [Double]) {
        Hanatos2025RawConverter(
            colourSpace: colourSpace,
            applyCCTFDecoding: applyCCTFDecoding,
            referenceIlluminant: referenceIlluminant,
            tcLUT: nil
        ).tcAndBrightness(rgb: rgb)
    }

    /// `colour.RGB_to_XYZ(..., illuminant:, chromatic_adaptation_transform: 'CAT16')` collapsed to
    /// one matrix: `M = M_cat16(colourspace whitepoint -> illuminant xy) * matrix_RGB_to_XYZ`.
    ///
    /// Verified against `colour.RGB_to_XYZ` at 4.3e-19 max abs, i.e. exact. CAT16 is named
    /// explicitly here. The rest of the engine's `RGB_to_RGB` calls take colour-science's CAT02
    /// default, so the pipeline uses two adaptation transforms.
    public static func composedRGBToXYZMatrix(
        colourSpace: ColourSpace, referenceIlluminant: Illuminant
    ) -> Matrix3 {
        let cat = Colour.chromaticAdaptationMatrix(
            from: colourSpace.whitepoint.XYZ,
            to: referenceIlluminant.chromaticity.XYZ,
            transform: .cat16)
        return cat * colourSpace.matrixRGBToXYZ
    }
}

/// A film's RGB-to-raw conversion with everything per-film hoisted out of the pixel loop.
///
/// The composed CAT16 matrix, the transfer function and the `tc_lut` depend only on the film and
/// the input colour space, so they are computed before the pixel loop.
public struct Hanatos2025RawConverter: Sendable {
    public let matrix: Matrix3
    public let transfer: TransferFunction
    public let applyCCTFDecoding: Bool
    public let referenceIlluminantXY: Chromaticity
    public let tcLUT: ImageBuffer?
    public let sampler: any LUT2DSampler

    public init(
        colourSpace: ColourSpace,
        applyCCTFDecoding: Bool,
        referenceIlluminant: Illuminant,
        tcLUT: ImageBuffer?,
        sampler: any LUT2DSampler = MitchellLUT2DSampler()
    ) {
        self.matrix = SpectralUpsampling.composedRGBToXYZMatrix(
            colourSpace: colourSpace, referenceIlluminant: referenceIlluminant)
        self.transfer = colourSpace.transfer
        self.applyCCTFDecoding = applyCCTFDecoding
        self.referenceIlluminantXY = referenceIlluminant.chromaticity
        self.tcLUT = tcLUT
        self.sampler = sampler
    }

    /// `_rgb_to_tc_b`.
    ///
    /// `b` is **not** clamped, only `nan_to_num`'d, so a wide-gamut input can leave with negative
    /// brightness and therefore negative raw. `filming.py` floors that later with
    /// `log10(fmax(raw, 0) + 1e-10)`. The chromaticity divide guards with `fmax(b, 1e-10)`, so a
    /// negative or NaN `b` sends `tc` off to a clamped corner while `b` keeps its sign.
    public func tcAndBrightness(rgb: ImageBuffer) -> (tc: ImageBuffer, brightness: [Double]) {
        precondition(rgb.channels == 3, "RGB buffer must have 3 channels")
        var tc = ImageBuffer(height: rgb.height, width: rgb.width, channels: 2)
        var brightness = [Double](repeating: 0, count: rgb.pixelCount)
        rgb.values.withUnsafeBufferPointer { source in
            for pixel in 0..<rgb.pixelCount {
                let value = tcPixel(source, pixel)
                tc.values[pixel * 2] = value.x
                tc.values[pixel * 2 + 1] = value.y
                brightness[pixel] = value.brightness
            }
        }
        return (tc, brightness)
    }

    /// `rgb_to_raw_hanatos2025` with a prebuilt LUT.
    ///
    /// One pass. Decoding, the matrix, the chromaticity divide and the brightness multiply all happen
    /// inside the LUT fetch, so the only full-frame buffer this allocates is the result: no decoded
    /// copy of the input, no `tc` frame and no brightness frame.
    public func raw(rgb: ImageBuffer) -> ImageBuffer {
        guard let tcLUT else {
            preconditionFailure("Hanatos2025RawConverter.raw needs a tc_lut")
        }
        precondition(rgb.channels == 3, "RGB buffer must have 3 channels")
        var out = ImageBuffer(height: rgb.height, width: rgb.width, channels: tcLUT.channels)
        rgb.values.withUnsafeBufferPointer { source in
            sampler.sample(lut: tcLUT, into: &out) { pixel in
                let value = tcPixel(source, pixel)
                return (value.x, value.y, value.brightness)
            }
        }
        return out
    }

    /// The `tc` coordinates and the brightness of one pixel of a 3-channel RGB buffer.
    ///
    /// Shared by ``tcAndBrightness(rgb:)`` and ``raw(rgb:)`` so the buffered and the fused path
    /// cannot drift apart.
    @inline(__always)
    private func tcPixel(
        _ source: UnsafeBufferPointer<Double>, _ pixel: Int
    ) -> (x: Double, y: Double, brightness: Double) {
        let base = pixel * 3
        var rgb = (source[base], source[base + 1], source[base + 2])
        if applyCCTFDecoding {
            rgb = (transfer.decode(rgb.0), transfer.decode(rgb.1), transfer.decode(rgb.2))
        }
        let (x, y, z) = matrix.apply(rgb)
        let b = x + y + z
        let scale = npFmax(b, 1e-10)
        let coordinate = ChromaticityCoordinates.triToQuad(x: x / scale, y: y / scale)
        return (coordinate.x, coordinate.y, nanToNum(b))
    }
}

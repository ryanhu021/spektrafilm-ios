import Foundation

/// Bakes input gamut compression into a finished `tc_lut`.
///
/// The interface between this subsystem and `gamut_compression`. `remap_tc_lut_for_compression`
/// rewrites the LUT so `new_lut[xy]` returns what the uncompressed LUT would have returned for
/// `compress(xy)`, which keeps the per-pixel fetch compression-agnostic. Its resample is **bilinear
/// with clamp-to-edge**, a different interpolator and a different boundary rule from the runtime
/// Mitchell fetch. Do not unify the two.
///
/// Nothing in this subsystem implements it. When a spec asks for compression and no bake is
/// supplied, ``TCLUTBuilder`` throws instead of rendering uncompressed.
public protocol InputGamutCompressionBake: Sendable {
    /// - Parameters:
    ///   - tcLUT: `[tc.x][tc.y][rgb]`, as ``TCLUTBuilder`` produced it.
    ///   - referenceIlluminantXY: the film's white, the same one `rgbToTCB` projects against, so the
    ///     compression's achromatic axis is the film's white.
    func remap(
        tcLUT: ImageBuffer, referenceIlluminantXY: Chromaticity, spec: InputGamutCompressSpec
    ) throws -> ImageBuffer
}

/// `compute_hanatos2025_tc_lut`: the per-film raw LUT.
///
/// Blur the irradiance spectra along wavelength, fold the normalised window into the sensitivities,
/// contract the 192 x 192 x 81 table down to 192 x 192 x 3, scale by `exp2(surface)`, then hand the
/// result to the compression bake. Roughly 5 ms of work, once per film.
public enum TCLUTBuilder {

    /// - Parameters:
    ///   - sensitivity: `nan_to_num(10 ** profile.log_sensitivity)`, `(81, 3)`. Built by the caller.
    ///   - gamutCompress: pass nil, or a spec with `active == false`, to skip the bake.
    ///   - compressionBake: required when `gamutCompress?.active` is true.
    public static func computeHanatos2025TCLUT(
        sensitivity: SpectralMatrix,
        adaptation: Hanatos2025SensitivityAdaptation,
        gamutCompress: InputGamutCompressSpec? = nil,
        compressionBake: (any InputGamutCompressionBake)? = nil
    ) throws -> ImageBuffer {
        let spectra = try IrradianceSpectraLUT.shared()
        let blur = HanatosSpectralBlur(sigma: adaptation.spectralGaussianBlur)

        let operand: SpectralMatrix
        if adaptation.applyWindow {
            operand = sensitivity.multiplied(
                by: try adaptation.normalisedWindow(sensitivity: sensitivity))
        } else {
            operand = sensitivity
        }
        var rawLUT = spectra.contracted(with: operand, blur: blur)

        if adaptation.applySurface {
            let surface = try LogExposureCorrectionSurface.poly4.evaluate(
                params: adaptation.surfaceParams,
                illuminantXY: adaptation.referenceIlluminant.chromaticity,
                gridSize: spectra.gridSize)
            // exp2, not pow(2, x): they differ in the last bits on some libms.
            for i in rawLUT.values.indices { rawLUT.values[i] *= exp2(surface.values[i]) }
        }

        if let gamutCompress, gamutCompress.active {
            guard let compressionBake else {
                throw SpektraError.unsupportedSetting(
                    "io.input_gamut_compress.active", value: "true")
            }
            rawLUT = try compressionBake.remap(
                tcLUT: rawLUT,
                referenceIlluminantXY: adaptation.referenceIlluminant.chromaticity,
                spec: gamutCompress)
        }
        return rawLUT
    }
}

/// `SpectralLUTService`'s filming `tc_lut` cache.
///
/// The reference compares its cache key by value across the sensitivity array, the adaptation record
/// and the gamut-compress spec, and rebuilds when any of them differs. `_same_hanatos2025_adaptation`
/// ignores the adaptation's dead `active` field. The Swift record has no such field, so plain
/// `Equatable` is the same comparison.
///
/// The reference's `FilmingStage` mutates the adaptation object it was handed, so the reference
/// deep-copies it on set to notice the change. Swift's value semantics give the same behaviour,
/// including the case `test_filming_tc_lut_recomputes_when_spectral_gaussian_blur_changes` covers.
///
/// The compression bake is not part of the key. It is a deterministic function of the spec, so two
/// bakes of the same spec agree. A caller that swaps in a different implementation has to discard
/// the cache itself.
public struct FilmingTCLUTCache: Sendable {
    private struct Key: Equatable {
        let sensitivity: [Double]
        let adaptation: Hanatos2025SensitivityAdaptation
        let gamutCompress: InputGamutCompressSpec?
    }

    private var key: Key?
    private var cached: ImageBuffer?

    public init() {}

    /// `get_filming_tc_lut`.
    public mutating func lut(
        sensitivity: SpectralMatrix,
        adaptation: Hanatos2025SensitivityAdaptation,
        gamutCompress: InputGamutCompressSpec? = nil,
        compressionBake: (any InputGamutCompressionBake)? = nil
    ) throws -> ImageBuffer {
        let wanted = Key(
            sensitivity: sensitivity.values, adaptation: adaptation, gamutCompress: gamutCompress)
        if let cached, key == wanted { return cached }
        let lut = try TCLUTBuilder.computeHanatos2025TCLUT(
            sensitivity: sensitivity,
            adaptation: adaptation,
            gamutCompress: gamutCompress,
            compressionBake: compressionBake)
        key = wanted
        cached = lut
        return lut
    }

    /// True when the next ``lut(sensitivity:adaptation:gamutCompress:compressionBake:)`` with these
    /// arguments would return without rebuilding. For tests and for timing instrumentation.
    public func isCached(
        sensitivity: SpectralMatrix,
        adaptation: Hanatos2025SensitivityAdaptation,
        gamutCompress: InputGamutCompressSpec? = nil
    ) -> Bool {
        cached != nil
            && key
                == Key(
                    sensitivity: sensitivity.values, adaptation: adaptation,
                    gamutCompress: gamutCompress)
    }
}

import Foundation

/// The error function and everything the engine builds directly on it.
///
/// Three subsystems share this: the dichroic and UV/IR filters, the Hanatos `erf4` sensitivity
/// window, and the print-curve morph's layer CDFs.
///
/// Darwin libm supplies `erf` and `erfc`. Measured against `scipy.special` over a dense sweep of
/// 39,817 arguments spanning ±51: `erf` agrees to 3 ulps worst case (73% bit-exact, 26% one ulp),
/// max absolute difference 3.331e-16, and `erfc` to 4.441e-16 absolute. `erfc`'s *relative* error
/// reaches 5.6e-14 past x = 20, where it returns values near 1e-190, and Darwin keeps returning
/// subnormals past x = 26.55 where scipy has already flushed to zero. No call site reads the tail
/// that deep, and the Hanatos window normalisation divides one erf-weighted sum by another, so a
/// 1e-16 relative error passes straight through instead of accumulating.
public enum Erf {

    /// C99 `erf`.
    @inlinable
    public static func erf(_ x: Double) -> Double { Foundation.erf(x) }

    /// C99 `erfc`. Distinct from `1 - erf(x)`, which loses every significant digit past x = 6.
    @inlinable
    public static func erfc(_ x: Double) -> Double { Foundation.erfc(x) }

    /// The standard normal CDF, `scipy.special.ndtr`.
    ///
    /// `ndtr(z) == 0.5 * erfc(-z / sqrt(2))` to 2.220e-16 absolute over z in ±51, so the reference's
    /// `scipy.stats.norm.cdf` needs no special function of its own.
    @inlinable
    public static func normalCDF(_ z: Double) -> Double {
        0.5 * Foundation.erfc(-z / 2.0.squareRoot())
    }

    /// `-log(log(2))`, from `morph_curves._GUMBEL_LOCATION`.
    public static let gumbelLocation = 0.36651292058166435

    /// `0.5 * log(2) * sqrt(2 * pi)`, from `morph_curves._GUMBEL_WIDTH`.
    public static let gumbelWidth = 0.8687311606361591

    /// The Gumbel CDF width- and median-matched to the standard normal.
    ///
    /// `morph_curves._gumbel_matched_cdf`. Developer exhaustion blends this into the layer CDF, which
    /// skews the shoulder without moving the midpoint, because the two constants pin the Gumbel's
    /// median and slope to the normal's.
    @inlinable
    public static func gumbelMatchedCDF(_ z: Double) -> Double {
        Foundation.exp(-Foundation.exp(-(z / gumbelWidth + gumbelLocation)))
    }

    // MARK: - Band-pass filter

    /// `color_filters.sigmoid_erf`. A negative `width` flips the edge from rising to falling.
    @inlinable
    public static func sigmoid(_ x: Double, centre: Double, width: Double) -> Double {
        Foundation.erf((x - centre) / width) * 0.5 + 0.5
    }

    /// `color_filters.compute_band_pass_filter`, the camera's UV and IR cut, over the 81-sample grid.
    ///
    /// Two erf sigmoids multiplied, which is why it lives with them. The IR arm negates its width
    /// internally, so the tuple's `width` stays positive at every call site.
    ///
    /// Each arm is `1 - amplitude` at full block and 1 at full pass, so amplitude 0 yields an
    /// all-ones filter and `filming._rgb_to_film_raw` skips the whole block. That caller also
    /// rescales each channel by its own response to the reference illuminant, so this shape must not
    /// be normalised here.
    public static func bandPassFilter(
        uv: (amplitude: Double, centre: Double, width: Double),
        ir: (amplitude: Double, centre: Double, width: Double)
    ) -> [Double] {
        let ampUV = min(max(uv.amplitude, 0.0), 1.0)
        let ampIR = min(max(ir.amplitude, 0.0), 1.0)
        return Observer.wavelengths.map { lambda in
            let edgeUV = 1.0 - ampUV + ampUV * sigmoid(lambda, centre: uv.centre, width: uv.width)
            let edgeIR = 1.0 - ampIR + ampIR * sigmoid(lambda, centre: ir.centre, width: -ir.width)
            return edgeUV * edgeIR
        }
    }
}

import Foundation

/// Stray light in the viewing setup, added in XYZ.
///
/// Ports `model/glare.py`. A lognormal field with mean `percent` and standard deviation
/// `roughness * percent`, blurred, divided by 100, then added as a multiple of the viewing
/// illuminant. At the shipped defaults the additive term averages `3e-4 * illuminantXYZ`, which is
/// small but on the default render path.
///
/// The reference is not reproducible: `fast_lognormal_from_mean_std` is a Numba kernel whose RNG
/// state is thread-local and unreachable from `np.random.seed`, so run-to-run `max_abs` on a 16x16
/// default render is 1.71e-3, seventeen times the parity gate. This draws from the counter-based
/// Philox instead, keyed by the glare seed and countered by the linear pixel index, so the field is
/// fixed by the seed and independent of how the frame is split up.
public enum Glare {

    /// Sublayer slot the glare field reserves, chosen at the top of ``PhiloxKey``'s 16-bit range so
    /// no grain stream can reach it.
    ///
    /// Grain's particle streams take sublayer `0 ..< max(3, subLayerCount)` and its clumping field
    /// takes 3, and `subLayerCount` has no upper bound, so no small index is safe. Sharing a stream
    /// is not harmless: with `PhiloxKey(seed:)`, which is channel 0 and sublayer 0, the glare field
    /// drew from the same Philox words as the red channel's first particle sublayer at the same
    /// pixel index, and the two came out correlated at r = -0.139 over 65536 pixels against a
    /// sampling scale of 0.004. The reference draws them from unrelated generators.
    static let stream = 0xFFFF

    /// Stream the glare field draws from. Channel is zero: the field is 2D and shared across the
    /// three XYZ channels.
    static func key(seed: UInt64) -> PhiloxKey {
        PhiloxKey(seed: seed, channel: 0, sublayer: stream)
    }

    /// `compute_random_glare_amount`.
    ///
    /// - Parameters:
    ///   - amount: `glare.percent`. The field's mean before the division by 100.
    ///   - roughness: multiplies `amount` to give the field's standard deviation.
    ///   - blur: sigma in pixels, not micrometres.
    /// - Returns: a single-channel `[H, W]` field.
    public static func randomAmount(
        amount: Double,
        roughness: Double,
        blur: Double,
        height: Int,
        width: Int,
        seed: UInt64 = 0,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        var field = ImageBuffer(height: height, width: width, channels: 1)
        var source = Philox4x32(key: key(seed: seed))
        field.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for i in 0..<buffer.count {
                source.reset(counter: UInt64(i))
                p[i] = Distributions.lognormalFromMeanStd(
                    mean: amount, std: roughness * amount, &source)
            }
        }

        // The reference blurs unconditionally and relies on `fast_gaussian_filter` returning a copy
        // at a non-positive sigma. Same result, without depending on that guard.
        if blur > 0 {
            field = spatial.gaussian(field, sigma: blur)
        }

        field.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for i in 0..<buffer.count { p[i] /= 100.0 }
        }
        return field
    }

    /// `add_glare`.
    ///
    /// Called from the scanning stage on `[H, W, 3]` XYZ, after the black-and-white XYZ correction
    /// and before `XYZ_to_RGB`. `illuminantXYZ` is the viewing illuminant normalised to `Y = 1`.
    ///
    /// - Parameter glare: `nil` on the scan-film branch, where the reference passes no glare at
    ///   all. `film_render.glare` is dead; the print branch passes `print_render.glare`.
    public static func add(
        _ xyz: ImageBuffer,
        illuminantXYZ: [Double],
        glare: GlareParams?,
        seed: UInt64 = 0,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        guard let params = glare, params.active, params.percent > 0 else { return xyz }
        precondition(xyz.channels == 3, "glare is added in XYZ")
        precondition(illuminantXYZ.count == 3, "illuminant must be an XYZ triplet")

        let field = randomAmount(
            amount: params.percent,
            roughness: params.roughness,
            blur: params.blur,
            height: xyz.height,
            width: xyz.width,
            seed: seed,
            spatial: spatial)

        var out = xyz
        out.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for pixel in 0..<field.values.count {
                let flare = field.values[pixel]
                for channel in 0..<3 { p[pixel * 3 + channel] += flare * illuminantXYZ[channel] }
            }
        }
        return out
    }
}

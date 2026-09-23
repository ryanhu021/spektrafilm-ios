import Foundation

/// `(threshold, limit, power)` of a Reinhard knee, the shape both gamut compressors roll off with.
public typealias Knee = (threshold: Double, limit: Double, power: Double)

/// Smooth knee on a normalised distance: identity at and below `threshold`, asymptotic to `limit`.
///
///     n = (d - threshold) / (limit - threshold)
///     d' = threshold + (limit - threshold) * n / (1 + n^power)^(1/power)
///
/// Three properties the reference relies on, all pinned by the goldens:
///
/// - The mask is `d > threshold`, so `d == threshold` is returned untouched. The formula is
///   continuous there anyway, since `n = 0` gives `0`.
/// - Negative `d` is untouched. `compressLightness` leans on that to keep black at black.
/// - Nothing clamps. `d = 1e9` with the default knee returns `1.000000000000001`, just above the
///   limit, and the reference keeps it.
///
/// Upstream validation never checks `limit > threshold`. A spec with `limit < threshold` gives a
/// negative scale and the knee expands instead of compressing. The port reproduces that
/// arithmetic: nothing in the parameter surface can reach it, and rejecting it would change
/// behaviour.
@inlinable
public func reinhardKnee(_ d: Double, threshold: Double, limit: Double, power: Double) -> Double {
    // NaN fails this comparison and passes through unchanged, as it does under NumPy's boolean mask.
    guard d > threshold else { return d }
    let scale = limit - threshold
    let x = (d - threshold) / scale
    // Spelled as the reference spells it. A precomputed `1/power` rounds differently for powers
    // other than the default 6.
    let y = x / pow(1.0 + pow(x, power), 1.0 / power)
    return threshold + scale * y
}

@inlinable
public func reinhardKnee(_ d: Double, _ knee: Knee) -> Double {
    reinhardKnee(d, threshold: knee.threshold, limit: knee.limit, power: knee.power)
}

public func reinhardKnee(_ values: [Double], _ knee: Knee) -> [Double] {
    values.map { reinhardKnee($0, knee) }
}

/// One-sided roll-off on a perceptual lightness axis, normalised so `1.0` is perceptual white.
///
/// `lightnessWhite` is the lightness of the output colour space's whitepoint in whatever space the
/// caller works in: 1.0 for Oklab, `JzAzBz.whiteJz` for JzAzBz, 100 for CAM16-UCS. With the default
/// parameters the knee's threshold is `0.7 * lightnessWhite`, about a third of white in linear
/// grey, so this darkens midtones as well as highlights.
///
/// Black is anchored: 0 normalises to 0, which is at or below every valid threshold. Below-black
/// input passes through, because the knee ignores negatives.
@inlinable
public func compressLightness(_ L: Double, _ params: Knee, lightnessWhite: Double) -> Double {
    reinhardKnee(L / lightnessWhite, params) * lightnessWhite
}

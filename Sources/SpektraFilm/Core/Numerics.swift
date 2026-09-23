import Foundation

// MARK: - Signed power

/// colour-science's `colour.algebra.spow`: `sign(a) * |a|^p`.
///
/// Used wherever the reference raises a possibly-negative value to a fractional power: the Oklab
/// and CAM16 non-linearities, the PQ curve, the Reinhard knee. `pow(-0.5, 2.2)` is NaN in C, so
/// the sign has to come out first.
///
/// This matches colour-science's **array** path. Its scalar path has an extra clause,
/// `0 if a_p.ndim == 0 and np.isnan(a_p) else a_p`, which turns a NaN result into 0 for a 0-d
/// input only. Every engine call site passes an array, so NaN propagates: `spow(nan, 2) = nan`
/// and `spow(0, -1) = 0 * inf = nan`. The oracle confirms both, scalar and array.
@inlinable
public func spow(_ a: Double, _ p: Double) -> Double {
    // np.sign(-0.0) is +0.0, and NaN * anything is NaN, so the NaN branch only needs to propagate
    // the NaN.
    let sign: Double
    if a < 0 {
        sign = -1
    } else if a > 0 {
        sign = 1
    } else {
        sign = a.isNaN ? Double.nan : 0
    }
    return sign * pow(abs(a), p)
}

// MARK: - NaN-dropping maximum

/// NumPy's `np.fmax`: the larger operand, or the other operand when one of them is NaN.
///
/// Distinct from Swift's `max`, which is `y >= x ? y : x` and so returns NaN whenever NaN is the
/// first argument. The pipeline depends on NaN being dropped: the four `log10Guard` call sites feed
/// `fmax(raw, 0)` into `log10`, and the gamut code guards nine divisions with `fmax(d, 1e-12)`.
/// A NaN surviving either would turn a wrong colour into a trap at the next `Int(_:)` conversion.
///
/// Forwards to libm `fmax`, which agrees with `np.fmax` bit for bit on all 19 pairs measured in the
/// oracle, two NaN operands and both signed-zero orders included. `Double.maximum` breaks the
/// signed-zero tie the other way and must not be substituted here: `Double.maximum(0.0, -0.0)` is
/// `-0.0`, `np.fmax(0.0, -0.0)` is `+0.0`, and the goldens cannot see the difference because
/// `abs(-0.0 - 0.0)` is 0.
@inlinable
public func npFmax(_ a: Double, _ b: Double) -> Double { fmax(a, b) }

// MARK: - Non-finite substitution

/// NumPy's `np.nan_to_num` with its default substitutions, measured in the oracle:
/// NaN becomes 0, `+inf` becomes `Double.greatestFiniteMagnitude` (`np.finfo(float64).max`),
/// `-inf` becomes its negation.
@inlinable
public func nanToNum(_ x: Double) -> Double {
    if x.isNaN { return 0 }
    if x == .infinity { return .greatestFiniteMagnitude }
    if x == -.infinity { return -.greatestFiniteMagnitude }
    return x
}

public func nanToNum(_ values: [Double]) -> [Double] { values.map(nanToNum) }

public func nanToNum(_ image: ImageBuffer) -> ImageBuffer {
    var out = image
    for i in out.values.indices { out.values[i] = nanToNum(out.values[i]) }
    return out
}

// MARK: - Guarded log10

/// `log10(fmax(x, 0) + 1e-10)`, which the reference writes out at four places: the film and print
/// exposure conversions (`filming.py:70`, `printing.py:61`, `printing.py:91`) and the scanner's
/// XYZ-to-log step (`scanning.py:120`).
///
/// The `1e-10` floors the result at -10 for a zero or negative input. `fmax` maps a NaN input to
/// -10 as well, so NaN never reaches the downstream density lookup.
@inlinable
public func log10Guard(_ x: Double) -> Double { log10(npFmax(x, 0) + 1e-10) }

public func log10Guard(_ values: [Double]) -> [Double] { values.map(log10Guard) }

public func log10Guard(_ image: ImageBuffer) -> ImageBuffer {
    var out = image
    for i in out.values.indices { out.values[i] = log10Guard(out.values[i]) }
    return out
}

// MARK: - linspace

/// `numpy.linspace(start, stop, count, endpoint:)`.
///
/// Ported term by term because the result is an interpolation domain: a one-ulp difference in a
/// breakpoint moves every query near it. Two details matter:
///
/// - The step is formed **once** as `(stop - start) / div` and then multiplied by `i`. Computing
///   `Double(i) * (stop - start) / div` instead rounds twice and misses `linspace(-3, 4, 256)` by
///   8.9e-16, measured against the oracle.
/// - With `endpoint` the last sample is overwritten with `stop` exactly. The accumulated value is
///   often a different double: `linspace(0, 1, 50)` accumulates 0.9999999999999999, and 505 of the
///   4095 counts in `2...4096` do the same, measured in the oracle.
///
/// `LOG_EXPOSURE = linspace(-3, 4, 256)` is bit-identical to this, and to the `log_exposure` array
/// in all 28 bundled profiles.
public func linspace(
    _ start: Double, _ stop: Double, count: Int, endpoint: Bool = true
) -> [Double] {
    precondition(count >= 0, "linspace needs a non-negative count, got \(count)")
    if count == 0 { return [] }

    let delta = stop - start
    let div = endpoint ? count - 1 : count
    var out = [Double](repeating: 0, count: count)

    if div > 0 {
        let step = delta / Double(div)
        if step == 0 {
            // NumPy's denormal guard (gh-5437): dividing first keeps a subnormal delta alive.
            for i in 0..<count { out[i] = (Double(i) / Double(div)) * delta + start }
        } else {
            for i in 0..<count { out[i] = Double(i) * step + start }
        }
    } else {
        // count == 1 with endpoint: NumPy calls the step undefined and multiplies by delta.
        for i in 0..<count { out[i] = Double(i) * delta + start }
    }

    if endpoint && count > 1 { out[count - 1] = stop }
    return out
}

// MARK: - NaN-skipping reductions

/// Per-channel `np.nanmin(a, axis: 0)` over a flattened `[n][channels]` array.
///
/// An all-NaN slice returns NaN, as NumPy does after its `All-NaN slice encountered` warning.
/// Verified in the oracle, including on `fujifilm_c200`'s `channel_density`, the one bundled
/// profile array with missing samples.
public func nanMin(_ values: [Double], channels: Int) -> [Double] {
    reduceOverSamples(values, channels: channels) { Swift.min($0, $1) }
}

/// Per-channel `np.nanmax(a, axis: 0)`. All-NaN slices return NaN.
public func nanMax(_ values: [Double], channels: Int) -> [Double] {
    reduceOverSamples(values, channels: channels) { Swift.max($0, $1) }
}

/// Per-channel `np.nanmean(a, axis: 0)`. An all-NaN slice returns NaN, which is NumPy's
/// `Mean of empty slice` path.
public func nanMean(_ values: [Double], channels: Int) -> [Double] {
    precondition(channels > 0, "channels must be positive")
    precondition(values.count % channels == 0, "\(values.count) values is not a multiple of \(channels)")
    let samples = values.count / channels
    var sums = [Double](repeating: 0, count: channels)
    var counts = [Int](repeating: 0, count: channels)
    for i in 0..<samples {
        for c in 0..<channels {
            let v = values[i * channels + c]
            if v.isNaN { continue }
            sums[c] += v
            counts[c] += 1
        }
    }
    return (0..<channels).map { counts[$0] == 0 ? .nan : sums[$0] / Double(counts[$0]) }
}

/// `np.nanmean(a, axis: 1)` over a flattened `[n][channels]` array: one mean per sample, across
/// the channels. `color_reference.py` averages the three density curves this way.
public func nanMeanPerSample(_ values: [Double], channels: Int) -> [Double] {
    precondition(channels > 0, "channels must be positive")
    precondition(values.count % channels == 0, "\(values.count) values is not a multiple of \(channels)")
    let samples = values.count / channels
    return (0..<samples).map { i in
        var sum = 0.0
        var seen = 0
        for c in 0..<channels {
            let v = values[i * channels + c]
            if v.isNaN { continue }
            sum += v
            seen += 1
        }
        return seen == 0 ? .nan : sum / Double(seen)
    }
}

/// `np.nanmean(a)` over the whole array. NaN when every element is NaN.
public func nanMean(_ values: [Double]) -> Double {
    var sum = 0.0
    var seen = 0
    for v in values where !v.isNaN {
        sum += v
        seen += 1
    }
    return seen == 0 ? .nan : sum / Double(seen)
}

private func reduceOverSamples(
    _ values: [Double], channels: Int, _ combine: (Double, Double) -> Double
) -> [Double] {
    precondition(channels > 0, "channels must be positive")
    precondition(values.count % channels == 0, "\(values.count) values is not a multiple of \(channels)")
    let samples = values.count / channels
    var out = [Double](repeating: .nan, count: channels)
    for i in 0..<samples {
        for c in 0..<channels {
            let v = values[i * channels + c]
            if v.isNaN { continue }
            out[c] = out[c].isNaN ? v : combine(out[c], v)
        }
    }
    return out
}

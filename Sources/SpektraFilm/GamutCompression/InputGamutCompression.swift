import Foundation

// MARK: - The locus polygon

/// The closed CIE 1931 2° visible spectral locus, and the two geometric predicates the input
/// compressor asks of it.
///
/// 65 vertices at 5 nm from 380 to 700 nm with the first repeated at the end, straight out of
/// ``ColourTables/spectralLocusXY``. Tabulated rather than derived from `cie1931_2deg` because
/// `align()` moves the CMF values by ~9e-18, which is enough to flip ``contains(x:y:)`` and so change
/// a bisection step of the Oklch envelope.
public enum SpectralLocus {
    /// `(x, y)` per vertex, closed.
    public static let vertices: [(x: Double, y: Double)] = {
        let flat = ColourTables.spectralLocusXY
        return (0..<ColourTables.spectralLocusVertexCount).map { (flat[$0 * 2], flat[$0 * 2 + 1]) }
    }()

    /// Even-odd crossing test, standing in for `matplotlib.path.Path.contains_points`.
    ///
    /// Agrees with matplotlib everywhere off the polygon: zero disagreements over a 81 x 81 grid of
    /// `[-0.1, 1.0]²`, and the spec reports the same over 200 000 random points. Points exactly on the
    /// polygon are a coin toss in both implementations and the two disagree on 49 of the 131 vertices
    /// and edge midpoints. It makes no difference to the only consumer: the bisection that builds
    /// ``InputGamutCompression/locusEnvelope`` never lands on an edge, and that table comes out
    /// byte-identical to the oracle's.
    public static func contains(x: Double, y: Double) -> Bool {
        var inside = false
        for k in 0..<(vertices.count - 1) {
            let a = vertices[k]
            let b = vertices[k + 1]
            if (a.y > y) != (b.y > y) {
                let crossing = (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x
                if x < crossing { inside = !inside }
            }
        }
        return inside
    }

    /// Smallest positive parametric distance from `origin` along a unit `direction` to the polygon.
    ///
    /// Returns `+infinity` when no edge is hit, which happens only if the origin is outside the
    /// locus. `compressXYRadial` then multiplies `0 * infinity` and produces NaN. That is the
    /// reference's behaviour and the precondition it relies on: every film reference illuminant
    /// (D55, TH-KG3, T) sits well inside the locus.
    public static func rayDistance(
        originX: Double, originY: Double, directionX: Double, directionY: Double
    ) -> Double {
        var tMin = Double.infinity
        for k in 0..<(vertices.count - 1) {
            let a = vertices[k]
            let b = vertices[k + 1]
            let edgeX = b.x - a.x
            let edgeY = b.y - a.y
            let denominator = directionX * edgeY - directionY * edgeX
            guard abs(denominator) > 1e-12 else { continue }
            let offsetX = originX - a.x
            let offsetY = originY - a.y
            let t = (-offsetX * edgeY + offsetY * edgeX) / denominator
            let s = (-offsetX * directionY + offsetY * directionX) / denominator
            if t > 1e-9, s >= 0.0, s <= 1.0, t < tMin { tMin = t }
        }
        return tMin
    }
}

// MARK: - Input compression

/// Input gamut compression: pulls CIE 1931 chromaticities back inside the visible spectral locus,
/// where the Hanatos 2025 spectral upsampling is defined.
///
/// Runs at LUT bake time, not per pixel. The caller that bakes it into a film's `tc_lut` lives in the
/// spectral-upsampling subsystem; this type only maps xy to xy.
///
/// With the default knee `(0.0, 1.0, 6.0)` there is no identity region: `(0.35, 0.36)` moves at the
/// eighth decimal. A port that leaves anything untouched has the wrong knee.
public enum InputGamutCompression {

    /// One chromaticity. `white` is the film's reference illuminant, the achromatic axis the
    /// compression works around.
    public static func compress(
        _ xy: (x: Double, y: Double), white: Chromaticity, spec: InputGamutCompressSpec
    ) -> (x: Double, y: Double) {
        guard spec.active else { return xy }
        switch spec.algorithm {
        case .xy:
            return radial(xy, white: white, knee: spec.knee)
        case .oklch:
            return oklchChroma(xy, knee: spec.knee)
        }
    }

    /// A flat array of xy pairs, which is the shape the LUT bake works in.
    public static func compress(
        _ pairs: [Double], white: Chromaticity, spec: InputGamutCompressSpec
    ) -> [Double] {
        precondition(pairs.count % 2 == 0, "expected flat xy pairs, got \(pairs.count) values")
        guard spec.active else { return pairs }
        var out = pairs
        for i in stride(from: 0, to: out.count, by: 2) {
            let compressed = compress((out[i], out[i + 1]), white: white, spec: spec)
            out[i] = compressed.x
            out[i + 1] = compressed.y
        }
        return out
    }

    /// Radial compression from `white` toward the locus, the production default.
    ///
    /// Dominant wavelength is preserved by construction: the result stays on the ray.
    static func radial(
        _ xy: (x: Double, y: Double), white: Chromaticity, knee: Knee
    ) -> (x: Double, y: Double) {
        let deltaX = xy.x - white.x
        let deltaY = xy.y - white.y
        // `np.linalg.norm`, which is sqrt of the sum of squares rather than `hypot`.
        let distance = (deltaX * deltaX + deltaY * deltaY).squareRoot()
        // The reference passes through on `dist < 1e-9` and computes everywhere else, so a NaN
        // distance takes the compute path and comes back NaN in both components. Spelled as a
        // negated `<` for that reason: `distance >= 1e-9` would send NaN to the passthrough and keep
        // the finite component.
        guard !(distance < 1e-9) else { return xy }
        let safeDistance = npFmax(distance, 1e-12)
        let directionX = deltaX / safeDistance
        let directionY = deltaY / safeDistance
        let boundary = SpectralLocus.rayDistance(
            originX: white.x, originY: white.y, directionX: directionX, directionY: directionY)
        let normalised = distance / npFmax(boundary, 1e-12)
        let compressed = reinhardKnee(normalised, knee)
        return (
            white.x + directionX * (compressed * boundary),
            white.y + directionY * (compressed * boundary)
        )
    }

    /// Chroma reduction at constant Oklch lightness and hue, against the locus envelope.
    ///
    /// The reference takes `white_xy` here too and ignores it, for API symmetry; this signature drops
    /// it. White does not round-trip exactly on this path: the Oklab forward and inverse are not
    /// exact at `C = 0`, so `(1/3, 1/3)` comes back as `(0.33333333304, 0.33333333328)`.
    static func oklchChroma(_ xy: (x: Double, y: Double), knee: Knee) -> (x: Double, y: Double) {
        let (L, a, b) = Oklab.fromXYZ(xyToXYZUnitY(x: xy.x, y: xy.y))
        let chroma = hypot(a, b)
        let hue = atan2(b, a)
        let maximum = npFmax(locusEnvelope.lookup(L, hue), 1e-9)
        let compressed = reinhardKnee(chroma / maximum, knee) * maximum
        let xyz = Oklab.toXYZ((L, compressed * cos(hue), compressed * sin(hue)))
        let total = npFmax(xyz.0 + xyz.1 + xyz.2, 1e-12)
        return (xyz.0 / total, xyz.1 / total)
    }

    /// The Oklch chroma envelope of the locus, built once.
    ///
    /// Its lightness grid starts at 0.05, where the output envelopes start at 0.02 (Oklab), 0.002
    /// (JzAzBz) or 1.0 (CAM16-UCS).
    ///
    /// Known reference artefact, reproduced deliberately: 12 772 of the 46 080 cells (27.7%) run into
    /// the `hi = 0.5` bisection ceiling. At Y = 1 a near-monochromatic chromaticity has Oklch chroma
    /// well above 0.5, so over those cells the algorithm compresses against a flat 0.5 envelope
    /// instead of the locus, and the table records 0.49999809265136719 as its maximum. The band starts
    /// at lightness row 23 (`L = 0.3968253968253968`) with 3 of 720 hues and widens to 476 at the top
    /// row; no row saturates completely. Do not raise the bound; it would break parity.
    static let locusEnvelope = ChromaEnvelope(
        lightnessGrid: linspace(0.05, 1.0, count: ChromaEnvelope.lightnessCount),
        chromaUpper: 0.5
    ) { L, a, b in
        let xyz = Oklab.toXYZ((L, a, b))
        let total = npFmax(xyz.0 + xyz.1 + xyz.2, 1e-12)
        return SpectralLocus.contains(x: xyz.0 / total, y: xyz.1 / total)
    }
}

import Foundation

// MARK: - Chroma envelope

/// `C_max(L, h)`: the largest chroma at each lightness and hue whose reconstruction is still inside
/// the target gamut, on a 64 x 720 grid.
///
/// Shared by both sides of the subsystem. The output compressors build one per
/// `(perceptual space, output colour space)` against the RGB primaries cube; the input compressor's
/// `oklch` algorithm builds one against the visible spectral locus.
///
/// The builder is a fixed 18-step bisection with no convergence test, so the table is as coarse as
/// the reference's. The last step's resolution is `chromaUpper / 2^18`, 5.7e-4 for CAM16-UCS. That
/// quantisation alone costs about 1.8e-5 of output RGB, a fifth of the parity budget, so the
/// bisection is reproduced step for step. Replacing it with a solver would break parity.
struct ChromaEnvelope: Sendable {
    static let hueCount = 720
    static let lightnessCount = 64
    static let bisections = 18

    let lightnessGrid: [Double]
    let hueGrid: [Double]
    /// Row-major `[lightness][hue]`.
    let values: [Double]

    /// Bisects `inGamut` at every grid cell. `inGamut` takes `(L, a, b)` in the perceptual space.
    ///
    /// The reference bisects the whole mesh in lockstep, but each cell's interval depends only on
    /// that cell, so the per-cell loop gives the same table.
    init(
        lightnessGrid: [Double],
        chromaUpper: Double,
        inGamut: @Sendable (Double, Double, Double) -> Bool
    ) {
        let hueGrid = linspace(-Double.pi, Double.pi, count: Self.hueCount, endpoint: false)
        let hues = hueGrid.count
        var values = [Double](repeating: 0, count: lightnessGrid.count * hues)
        values.withUnsafeMutableBufferPointer { buffer in
            let out = buffer.baseAddress!
            Parallel.forEachChunk(of: buffer.count, cost: Self.bisections * 64) { cells in
                for cell in cells {
                    let L = lightnessGrid[cell / hues]
                    let h = hueGrid[cell % hues]
                    let cosH = cos(h)
                    let sinH = sin(h)
                    var lo = 0.0
                    var hi = chromaUpper
                    for _ in 0..<Self.bisections {
                        let mid = (lo + hi) * 0.5
                        if inGamut(L, mid * cosH, mid * sinH) {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    out[cell] = lo
                }
            }
        }
        self.lightnessGrid = lightnessGrid
        self.hueGrid = hueGrid
        self.values = values
    }

    /// Bilinear lookup with `L` clamped to the grid and `h` wrapped.
    ///
    /// `h = +π`, the top of `atan2`'s range, gives a hue index of 720.00000000000819 and so wraps onto
    /// the `-π` column. This is correct. Do not clamp it.
    ///
    /// A non-finite `L` or `h` returns 0. CAM16 produces NaN lightness for negative-luminance pixels.
    /// NumPy's `floor(nan).astype(int)` gives `INT_MIN` and then clips to 0, while a Swift `Int`
    /// conversion would trap. The output is garbage either way, but the port must not crash.
    func lookup(_ L: Double, _ h: Double) -> Double {
        guard L.isFinite, h.isFinite else { return 0 }
        let nL = lightnessGrid.count
        let nH = hueGrid.count
        let clampedL = min(max(L, lightnessGrid[0]), lightnessGrid[nL - 1])

        let hStep = hueGrid[1] - hueGrid[0]
        let hIndex = (h - hueGrid[0]) / hStep
        let hFloor = hIndex.rounded(.down)
        // NumPy's `%` is a floor modulo, so a negative index wraps to the top of the table.
        let hLo = ((Int(hFloor) % nH) + nH) % nH
        let hHi = (hLo + 1) % nH
        let hFraction = hIndex - hFloor

        let lIndex =
            (clampedL - lightnessGrid[0]) / (lightnessGrid[nL - 1] - lightnessGrid[0])
            * Double(nL - 1)
        let lLo = min(max(Int(lIndex.rounded(.down)), 0), nL - 2)
        let lHi = lLo + 1
        // Against the clipped index, not `floor(lIndex)`, so `L == L_grid.last` gives a fraction of 1.
        let lFraction = lIndex - Double(lLo)

        let v00 = values[lLo * nH + hLo]
        let v01 = values[lLo * nH + hHi]
        let v10 = values[lHi * nH + hLo]
        let v11 = values[lHi * nH + hHi]
        return v00 * (1 - lFraction) * (1 - hFraction) + v01 * (1 - lFraction) * hFraction
            + v10 * lFraction * (1 - hFraction) + v11 * lFraction * hFraction
    }
}

// MARK: - Spec surface

extension OutputGamutCompressSpec {
    /// `algorithm != .off`. A computed property on the output spec, where the input spec has a
    /// stored `active` flag.
    public var active: Bool { algorithm != .off }

    /// The three knee checks, applied to `knee` and to `lightnessCompression` when it is set.
    ///
    /// The reference never checks `limit > threshold`, and neither does this. See ``reinhardKnee``.
    public func validate() throws {
        try Self.validate(knee, label: "knee")
        if let lightnessCompression { try Self.validate(lightnessCompression, label: "lightness") }
    }

    private static func validate(_ knee: Knee, label: String) throws {
        guard knee.threshold >= 0, knee.threshold < 1 else {
            throw SpektraError.unsupportedSetting(
                "output_gamut_compress.\(label).threshold", value: "\(knee.threshold)")
        }
        guard knee.limit > 0 else {
            throw SpektraError.unsupportedSetting(
                "output_gamut_compress.\(label).limit", value: "\(knee.limit)")
        }
        guard knee.power > 0 else {
            throw SpektraError.unsupportedSetting(
                "output_gamut_compress.\(label).power", value: "\(knee.power)")
        }
    }
}

/// The four polar perceptual spaces an output compressor can reduce chroma in.
enum PerceptualSpace: String, Sendable, CaseIterable {
    case oklch
    case oklrab
    case jzazbz
    case cam16ucs

    init?(_ algorithm: OutputGamutCompressSpec.Algorithm) {
        switch algorithm {
        case .oklch: self = .oklch
        case .oklrab: self = .oklrab
        case .jzazbz: self = .jzazbz
        case .cam16ucs: self = .cam16ucs
        case .off, .acesRGC: return nil
        }
    }

    /// Lightness grid and initial chroma bound of the envelope. The grids differ per space because
    /// the lightness scales do: Oklab white is 1, JzAzBz white is 0.167 at 100 cd/m², CAM16-UCS
    /// white is 100.
    var envelopeConfiguration: (lightnessGrid: [Double], chromaUpper: Double) {
        switch self {
        case .oklch, .oklrab:
            return (linspace(0.02, 1.0, count: ChromaEnvelope.lightnessCount), 0.5)
        case .jzazbz:
            return (linspace(0.002, 0.18, count: ChromaEnvelope.lightnessCount), 0.3)
        case .cam16ucs:
            return (linspace(1.0, 110.0, count: ChromaEnvelope.lightnessCount), 150.0)
        }
    }
}

// MARK: - Compressor

/// Output gamut compression: reduces perceptual chroma against the output primaries cube, and rolls
/// above-white lightness back down into `[0, white]`.
///
/// Built once per render. Everything that depends only on the spec and the output colour space is
/// hoisted into it: the `C_max` envelope, the CAM16 viewing conditions, the perceptual lightness of
/// white, and both conversion matrices.
///
/// Containment is not exact. Measured over 2048 physically realizable pixels spanning
/// `[-14.4, 96.9]` in linear sRGB, `cam16ucs` (the default) brings everything into
/// `[1.8e-5, 0.99998]`, while `oklch` reaches 1.00148, `oklrab` 1.00110, `jzazbz` 1.00675, and
/// `aces_rgc` leaves the amplitude alone at 96.9. The reference has no clamp, and neither does this.
/// A clamp would fail parity and would hide the overshoot from whoever writes the file.
public struct OutputGamutCompressor: Sendable {
    /// JzAzBz needs absolute luminance. Linear RGB 1.0 maps to SDR diffuse white.
    public static let jzazbzWhiteLuminance = 100.0

    enum Kind: Sendable {
        case off
        case acesRGC
        case perceptual(PerceptualSpace)
    }

    let kind: Kind
    let knee: Knee
    let lightnessCompression: Knee?
    let rgbToXYZ: Matrix3
    let xyzToRGB: Matrix3
    let envelope: ChromaEnvelope?
    /// Perceptual lightness of the output whitepoint, the normaliser for the lightness knee.
    let lightnessWhite: Double
    let viewing: CAM16ViewingConditions?

    /// `colourSpace` is required by the four perceptual algorithms, which index a per-colour-space
    /// envelope. `off` and `aces_rgc` ignore it.
    public init(spec: OutputGamutCompressSpec, colourSpace: ColourSpace? = nil) throws {
        try spec.validate()
        knee = spec.knee
        rgbToXYZ = colourSpace?.matrixRGBToXYZ ?? .identity
        xyzToRGB = colourSpace?.matrixXYZToRGB ?? .identity

        switch spec.algorithm {
        case .off, .acesRGC:
            kind = spec.algorithm == .off ? .off : .acesRGC
            // The reference skips lightness compression on these paths.
            lightnessCompression = nil
            envelope = nil
            lightnessWhite = 1.0
            viewing = nil
        case .oklch, .oklrab, .jzazbz, .cam16ucs:
            guard let space = PerceptualSpace(spec.algorithm), let colourSpace else {
                throw SpektraError.unsupportedSetting(
                    "output_gamut_compress.algorithm",
                    value: "\(spec.algorithm.rawValue) without an output colour space")
            }
            kind = .perceptual(space)
            lightnessCompression = spec.lightnessCompression
            envelope = Self.envelope(space, colourSpace)
            let whiteXYZ = xyToXYZUnitY(x: colourSpace.whitepoint.x, y: colourSpace.whitepoint.y)
            switch space {
            case .oklch, .oklrab:
                // Oklab's perceptual white is 1.0 by construction, and the knee runs on L even for
                // oklrab, before the Lr remap.
                lightnessWhite = 1.0
                viewing = nil
            case .jzazbz:
                lightnessWhite =
                    JzAzBz.fromXYZ(
                        (
                            whiteXYZ.0 * Self.jzazbzWhiteLuminance,
                            whiteXYZ.1 * Self.jzazbzWhiteLuminance,
                            whiteXYZ.2 * Self.jzazbzWhiteLuminance
                        )
                    ).0
                viewing = nil
            case .cam16ucs:
                let vc = CAM16ViewingConditions(whitepointXYZ: whiteXYZ)
                viewing = vc
                // Exactly 100 for every colour space, since J = 100 * (A_w/A_w)^cz and
                // Jp = 170/1.7. Computed rather than hardcoded so a broken forward shows up here.
                lightnessWhite = CAM16UCS.forward(whiteXYZ, vc).0
            }
        }
    }

    /// One pixel of linear RGB in the output primaries.
    public func apply(_ rgb: (Double, Double, Double)) -> (Double, Double, Double) {
        switch kind {
        case .off:
            return rgb
        case .acesRGC:
            return acesRGC(rgb)
        case .perceptual(let space):
            return perceptual(rgb, space)
        }
    }

    /// A whole 3-channel buffer, in place.
    public func apply(to image: inout ImageBuffer) {
        precondition(image.channels == 3, "output gamut compression needs a 3-channel buffer")
        if case .off = kind { return }
        image.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            Parallel.forEachChunk(of: buffer.count / 3, cost: 16) { pixels in
                for pixel in pixels {
                    let k = pixel * 3
                    let out = apply((p[k], p[k + 1], p[k + 2]))
                    p[k] = out.0
                    p[k + 1] = out.1
                    p[k + 2] = out.2
                }
            }
        }
    }

    // MARK: ACES RGC

    /// ACES Reference Gamut Compression v1.3's per-channel form: knee the distance of each channel
    /// below the achromatic maximum, then rebuild against that maximum.
    ///
    /// The achromatic value itself is never touched, so a pixel above white stays above white. This
    /// path also skips the lightness knee, following the reference, so `aces_rgc` alone does not keep
    /// the output inside `[0, 1]`.
    ///
    /// Divergence from the published RGC: the standard has per-channel limits and thresholds
    /// (`limCyan`, `limMagenta`, `limYellow`). The reference collapses them to one triple applied to
    /// all three channels. The knee formula is the standard's.
    func acesRGC(_ rgb: (Double, Double, Double)) -> (Double, Double, Double) {
        let ach = Self.achromatic(rgb)
        // Matches the reference's `where(ach > 1e-12, compressed, rgb)`, including for a NaN channel,
        // which fails the comparison and so passes through.
        guard ach > 1e-12 else { return rgb }
        func channel(_ c: Double) -> Double {
            ach * (1.0 - reinhardKnee((ach - c) / ach, knee))
        }
        return (channel(rgb.0), channel(rgb.1), channel(rgb.2))
    }

    /// `np.max` over the three channels, which propagates NaN where Swift's `max` drops it.
    private static func achromatic(_ rgb: (Double, Double, Double)) -> Double {
        if rgb.0.isNaN || rgb.1.isNaN || rgb.2.isNaN { return .nan }
        return max(max(rgb.0, rgb.1), rgb.2)
    }

    // MARK: Polar perceptual

    /// The shared shape of `oklch`, `oklrab`, `jzazbz` and `cam16ucs`.
    ///
    /// The lightness knee runs before the chroma step so the `C_max` lookup happens at the corrected
    /// lightness, which makes the cube bound tight. Chroma and hue come from the *pre-knee*
    /// `a, b`: the lightness knee never rescales them.
    func perceptual(
        _ rgb: (Double, Double, Double), _ space: PerceptualSpace
    ) -> (Double, Double, Double) {
        guard let envelope else { return rgb }
        let xyz = rgbToXYZ.apply(rgb)

        var lightness: Double
        var a: Double
        var b: Double
        switch space {
        case .oklch, .oklrab:
            (lightness, a, b) = Oklab.fromXYZ(xyz)
        case .jzazbz:
            let scale = Self.jzazbzWhiteLuminance
            (lightness, a, b) = JzAzBz.fromXYZ((xyz.0 * scale, xyz.1 * scale, xyz.2 * scale))
        case .cam16ucs:
            // The initialiser builds the viewing conditions for, and only for, this space.
            (lightness, a, b) = CAM16UCS.forward(xyz, viewing!)
        }

        if let lightnessCompression {
            lightness = compressLightness(
                lightness, lightnessCompression, lightnessWhite: lightnessWhite)
        }

        let chroma = hypot(a, b)
        let hue = atan2(b, a)
        // oklrab indexes the envelope by the rebased lightness. The reconstructed triple still
        // uses L.
        let index = space == .oklrab ? Oklab.lightnessLr(lightness) : lightness
        let maximum = npFmax(envelope.lookup(index, hue), 1e-9)
        let compressed = reinhardKnee(chroma / maximum, knee) * maximum
        let newA = compressed * cos(hue)
        let newB = compressed * sin(hue)

        let newXYZ: (Double, Double, Double)
        switch space {
        case .oklch, .oklrab:
            newXYZ = Oklab.toXYZ((lightness, newA, newB))
        case .jzazbz:
            let scale = Self.jzazbzWhiteLuminance
            let back = JzAzBz.toXYZ((lightness, newA, newB))
            newXYZ = (back.0 / scale, back.1 / scale, back.2 / scale)
        case .cam16ucs:
            newXYZ = CAM16UCS.inverse((lightness, newA, newB), viewing!)
        }
        return xyzToRGB.apply(newXYZ)
    }

    // MARK: Envelope cache

    /// The envelopes cost 0.1 to 0.3 s each to build, so they are cached for the process rather than
    /// per compressor. Keyed the way the reference keys its cache, by
    /// `(perceptual space, output colour space)`.
    static func envelope(_ space: PerceptualSpace, _ colourSpace: ColourSpace) -> ChromaEnvelope {
        envelopeCache.value(for: "\(space.rawValue)|\(colourSpace.name)") {
            build(space, colourSpace)
        }
    }

    private static let envelopeCache = EnvelopeCache()

    private static func build(
        _ space: PerceptualSpace, _ colourSpace: ColourSpace
    )
        -> ChromaEnvelope
    {
        let configuration = space.envelopeConfiguration
        let toRGB = colourSpace.matrixXYZToRGB
        let whiteXYZ = xyToXYZUnitY(x: colourSpace.whitepoint.x, y: colourSpace.whitepoint.y)
        let viewing = CAM16ViewingConditions(whitepointXYZ: whiteXYZ)

        @Sendable func polarToXYZ(_ L: Double, _ a: Double, _ b: Double) -> (Double, Double, Double) {
            switch space {
            case .oklch:
                return Oklab.toXYZ((L, a, b))
            case .oklrab:
                return Oklab.toXYZ((Oklab.lightnessFromLr(L), a, b))
            case .jzazbz:
                let xyz = JzAzBz.toXYZ((L, a, b))
                return (
                    xyz.0 / jzazbzWhiteLuminance, xyz.1 / jzazbzWhiteLuminance,
                    xyz.2 / jzazbzWhiteLuminance
                )
            case .cam16ucs:
                return CAM16UCS.inverse((L, a, b), viewing)
            }
        }

        return ChromaEnvelope(
            lightnessGrid: configuration.lightnessGrid,
            chromaUpper: configuration.chromaUpper
        ) { L, a, b in
            let rgb = toRGB.apply(polarToXYZ(L, a, b))
            // The 1e-6 slack is the reference's, and it matters. Several colourspaces ship forward
            // and inverse matrices that are not exact inverses (4e-5 for sRGB). Without the slack
            // the largest in-gamut chroma comes out conservative.
            return rgb.0 >= -1e-6 && rgb.0 <= 1.0 + 1e-6 && rgb.1 >= -1e-6 && rgb.1 <= 1.0 + 1e-6
                && rgb.2 >= -1e-6 && rgb.2 <= 1.0 + 1e-6
        }
    }

    private final class EnvelopeCache: @unchecked Sendable {
        private let lock = NSLock()
        private var tables: [String: ChromaEnvelope] = [:]

        func value(for key: String, build: () -> ChromaEnvelope) -> ChromaEnvelope {
            lock.lock()
            if let cached = tables[key] {
                lock.unlock()
                return cached
            }
            lock.unlock()
            let built = build()
            lock.lock()
            tables[key] = built
            lock.unlock()
            return built
        }
    }
}

// MARK: - Convenience entry points

public enum OutputGamutCompression {
    /// One pixel. Builds a compressor per call, so use ``OutputGamutCompressor`` directly for more
    /// than a handful of pixels.
    public static func compress(
        _ rgb: (Double, Double, Double),
        spec: OutputGamutCompressSpec,
        colourSpace: ColourSpace? = nil
    ) throws -> (Double, Double, Double) {
        try OutputGamutCompressor(spec: spec, colourSpace: colourSpace).apply(rgb)
    }

    /// A whole buffer of linear RGB in the output primaries, in place.
    public static func compress(
        _ image: inout ImageBuffer,
        spec: OutputGamutCompressSpec,
        colourSpace: ColourSpace? = nil
    ) throws {
        try OutputGamutCompressor(spec: spec, colourSpace: colourSpace).apply(to: &image)
    }
}

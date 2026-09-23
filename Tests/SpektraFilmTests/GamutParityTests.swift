import CryptoKit
import Foundation
import Testing

@testable import SpektraFilm

/// Checks input and output gamut compression against colour-science.
///
/// The default render path runs `cam16ucs` output compression on every pixel, so the CAM16 forward
/// and inverse and the `C_max` envelope they index are gated here as hard as the compressors
/// themselves.
///
/// Measured against the oracle over these fixtures, max absolute difference:
///
/// | piece | max_abs |
/// |---|---|
/// | Reinhard knee, both compressors' input side, all five `C_max` envelopes | 0, bit-identical |
/// | Oklab and JzAzBz forward and inverse, CAM16-UCS inverse | 0, bit-identical |
/// | CAM16-UCS forward (`Jp` on 0 to 100) | 7.1e-15 |
/// | `oklch`, `oklrab`, `cam16ucs` output compression | 1.1e-14 |
/// | `jzazbz` output compression | 9.2e-13 |
///
/// The compressor tests keep the subsystem's 1e-4 contract rather than the measured figure, because
/// two error sources exist before the port adds any: the published forward/inverse matrix pairs are
/// not exact inverses (4e-5 of round-trip error for sRGB), and the envelope's 18-step bisection
/// quantises `C_max` (1.8e-5 of output RGB for CAM16-UCS). ``defaultPathTightAgreement()`` pins the
/// default path at the measured figure so a regression inside that budget still fails.
@Suite("Gamut compression parity")
struct GamutParityTests {

    static let defaultKnee: Knee = (0.0, 1.0, 6.0)
    static let softKnee: Knee = (0.815, 1.0, 1.2)
    static let lateKnee: Knee = (0.95, 1.0, 2.0)
    static let lightnessKnee: Knee = (0.7, 1.0, 2.2)

    static let whiteE = Chromaticity(x: 1.0 / 3.0, y: 1.0 / 3.0)
    /// kodak_portra_400's reference illuminant.
    static let whiteD55 = Chromaticity(x: 0.3324316389215494, y: 0.3474443989306461)

    // MARK: - Reinhard knee

    @Test(
        "the knee matches over a sweep that includes negatives and 1e9",
        arguments: [("default", defaultKnee), ("soft", softKnee), ("late", lateKnee)])
    func knee(label: String, knee: Knee) throws {
        let input = try Golden("gamut_knee_input").values
        try expectParity(input.map { reinhardKnee($0, knee) }, matches: "gamut_knee_\(label)")
    }

    /// The mask is `>`, so the threshold maps to itself and negatives never move. The lightness
    /// knee's black anchoring depends on the second property.
    @Test("the knee leaves the threshold and everything below it alone")
    func kneeIdentityRegion() {
        #expect(reinhardKnee(0.815, Self.softKnee) == 0.815)
        #expect(reinhardKnee(-0.3, Self.softKnee) == -0.3)
        #expect(reinhardKnee(-0.3, Self.defaultKnee) == -0.3)
        #expect(reinhardKnee(0.0, Self.defaultKnee) == 0.0)
        // Floating point puts 1e9 a hair above the limit. The reference does not clamp.
        #expect(reinhardKnee(1e9, Self.defaultKnee) > 1.0)
    }

    @Test(
        "the lightness knee matches at each algorithm's perceptual white",
        arguments: [("white1", 1.0), ("whitejz", 0.16717342769906365), ("white100", 100.0)])
    func lightnessCompression(label: String, white: Double) throws {
        let input = try Golden("gamut_lightness_input").values
        let actual = input.map {
            compressLightness($0 * white, Self.lightnessKnee, lightnessWhite: white)
        }
        try expectParity(actual, matches: "gamut_lightness_\(label)")
    }

    @Test("the lightness knee is one-sided and anchored at black")
    func lightnessAnchoring() {
        #expect(compressLightness(0.0, Self.lightnessKnee, lightnessWhite: 100.0) == 0.0)
        #expect(compressLightness(-20.0, Self.lightnessKnee, lightnessWhite: 100.0) == -20.0)
        #expect(compressLightness(50.0, Self.lightnessKnee, lightnessWhite: 100.0) == 50.0)
        #expect(compressLightness(120.0, Self.lightnessKnee, lightnessWhite: 100.0) < 100.0)
    }

    // MARK: - Spectral locus

    @Test("the locus polygon is the generated table, to the bit")
    func spectralLocus() throws {
        let golden = try Golden("gamut_spectral_locus")
        #expect(golden.shape == [66, 2])
        let flat = SpectralLocus.vertices.flatMap { [$0.x, $0.y] }
        #expect(flat == golden.values)
    }

    @Test(
        "ray-to-boundary distances match every 5 degrees",
        arguments: [("white_e", whiteE), ("d55", whiteD55)])
    func rayDistance(label: String, white: Chromaticity) throws {
        var actual: [Double] = []
        for step in 0..<72 {
            let radians = Double(step) * 5.0 * (Double.pi / 180.0)
            actual.append(
                SpectralLocus.rayDistance(
                    originX: white.x, originY: white.y,
                    directionX: cos(radians), directionY: sin(radians)))
        }
        try expectParity(actual, matches: "gamut_ray_distance_\(label)")
    }

    /// A ray from outside the locus can miss every edge. The reference returns `+inf` and
    /// `compress_xy_radial` then multiplies `0 * inf` into NaN. A documented precondition. The port
    /// reproduces the NaN and adds no guard.
    @Test("a missing ray returns infinity and compression produces NaN")
    func missingRay() {
        let outside = Chromaticity(x: 2.0, y: 2.0)
        let distance = SpectralLocus.rayDistance(
            originX: outside.x, originY: outside.y, directionX: 1, directionY: 0)
        #expect(distance == .infinity)
        var spec = InputGamutCompressSpec()
        spec.algorithm = .xy
        let compressed = InputGamutCompression.compress((3.0, 2.0), white: outside, spec: spec)
        #expect(compressed.x.isNaN)
        #expect(compressed.y.isNaN)
    }

    @Test("the inside/outside predicate matches matplotlib over a grid")
    func pointInPolygon() throws {
        let axis = linspace(-0.1, 1.0, count: 81)
        var actual: [Double] = []
        actual.reserveCapacity(axis.count * axis.count)
        for x in axis {
            for y in axis {
                actual.append(SpectralLocus.contains(x: x, y: y) ? 1 : 0)
            }
        }
        try expectParity(actual, matches: "gamut_point_in_polygon", maxAbsolute: 0, rootMeanSquare: 0)
    }

    /// Known divergence, left in place. Points exactly *on* the polygon have no well-defined
    /// answer, and the two implementations disagree on 49 of the 131 vertices and edge midpoints:
    /// matplotlib's Agg point-in-path routine calls 28 of them inside, the crossing rule calls 51.
    /// The spec says both call every boundary point outside; neither does.
    ///
    /// No output changes. The predicate's only consumer is the Oklch locus envelope, whose
    /// bisection never falls exactly on an edge, and that envelope hashes the same as the oracle's;
    /// see ``envelopeHashes()``. Matching Agg on the boundary would mean reimplementing it.
    @Test("on-boundary points follow the crossing rule, not matplotlib")
    func pointInPolygonBoundary() throws {
        let golden = try Golden("gamut_point_in_polygon_boundary").values
        let vertices = SpectralLocus.vertices
        var actual = vertices.map { SpectralLocus.contains(x: $0.x, y: $0.y) ? 1.0 : 0.0 }
        for k in 0..<(vertices.count - 1) {
            let midX = 0.5 * (vertices[k].x + vertices[k + 1].x)
            let midY = 0.5 * (vertices[k].y + vertices[k + 1].y)
            actual.append(SpectralLocus.contains(x: midX, y: midY) ? 1.0 : 0.0)
        }
        #expect(actual.count == golden.count)
        #expect(golden.reduce(0, +) == 28)
        #expect(actual.reduce(0, +) == 51)
        #expect(zip(actual, golden).count { $0 != $1 } == 49)
    }

    // MARK: - Input compression

    static let xyCases:
        [(name: String, algorithm: InputGamutCompressSpec.Algorithm, white: Chromaticity, knee: Knee)] = [
            ("xy_white_e", .xy, whiteE, defaultKnee),
            ("xy_d55", .xy, whiteD55, defaultKnee),
            ("oklch_white_e", .oklch, whiteE, defaultKnee),
            ("oklch_d55", .oklch, whiteD55, defaultKnee),
            ("xy_white_e_soft", .xy, whiteE, softKnee),
        ]

    @Test("input compression matches over 1608 chromaticities", arguments: xyCases)
    func compressXY(
        name: String, algorithm: InputGamutCompressSpec.Algorithm, white: Chromaticity, knee: Knee
    ) throws {
        let input = try Golden("gamut_xy_input").values
        var spec = InputGamutCompressSpec()
        spec.algorithm = algorithm
        spec.knee = knee
        let actual = InputGamutCompression.compress(input, white: white, spec: spec)
        try expectParity(actual, matches: "gamut_xy_\(name)")
    }

    @Test("an inactive input spec is exact identity")
    func compressXYInactive() throws {
        let input = try Golden("gamut_xy_input").values
        var spec = InputGamutCompressSpec()
        spec.active = false
        let actual = InputGamutCompression.compress(input, white: Self.whiteE, spec: spec)
        #expect(actual == input)
        try expectParity(actual, matches: "gamut_xy_inactive", maxAbsolute: 0, rootMeanSquare: 0)
    }

    /// The default knee has no identity region. A port that leaves a flat region near white has the
    /// wrong knee.
    @Test("the default input knee moves even a near-white chromaticity")
    func compressXYHasNoIdentityRegion() {
        var spec = InputGamutCompressSpec()
        let compressed = InputGamutCompression.compress(
            (0.35, 0.36), white: Self.whiteE, spec: spec)
        #expect(compressed.x != 0.35)
        #expect(compressed.y != 0.36)
        #expect(abs(compressed.x - 0.35) < 1e-7)
        // White itself is the one passthrough, by the 1e-9 distance test.
        spec.algorithm = .xy
        let atWhite = InputGamutCompression.compress(
            (Self.whiteE.x, Self.whiteE.y), white: Self.whiteE, spec: spec)
        #expect(atWhite.x == Self.whiteE.x)
        #expect(atWhite.y == Self.whiteE.y)
    }

    // MARK: - Perceptual transforms

    @Test("Oklab forward and inverse match colour-science")
    func oklab() throws {
        let xyz = try Golden("gamut_xyz_input").values
        try expectParity(
            mapTriples(xyz, Oklab.fromXYZ), matches: "gamut_oklab_forward",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
        let lab = try Golden("gamut_oklab_forward").values
        try expectParity(
            mapTriples(lab, Oklab.toXYZ), matches: "gamut_oklab_inverse",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
    }

    @Test("the Lr lightness remap and its inverse match")
    func oklrabLightness() throws {
        let L = try Golden("gamut_oklrab_l_input").values
        try expectParity(
            L.map(Oklab.lightnessLr), matches: "gamut_oklrab_lr",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
        let Lr = try Golden("gamut_oklrab_lr").values
        try expectParity(
            Lr.map(Oklab.lightnessFromLr), matches: "gamut_oklrab_l_from_lr",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
        #expect(Oklab.lightnessLr(0.0) == 0.0)
        #expect(abs(Oklab.lightnessLr(1.0) - 1.0) < 1e-15)
        for L in stride(from: 0.0, through: 1.0, by: 0.05) {
            #expect(abs(Oklab.lightnessFromLr(Oklab.lightnessLr(L)) - L) < 1e-12)
        }
    }

    @Test("JzAzBz forward and inverse match colour-science")
    func jzazbz() throws {
        let xyz = try Golden("gamut_xyz_input").values
        let absolute = xyz.map { $0 * 100.0 }
        try expectParity(
            mapTriples(absolute, JzAzBz.fromXYZ), matches: "gamut_jzazbz_forward",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
        let jab = try Golden("gamut_jzazbz_forward").values
        try expectParity(
            mapTriples(jab, JzAzBz.toXYZ), matches: "gamut_jzazbz_inverse",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
    }

    @Test("the CAM16 viewing-condition scalars match")
    func cam16ViewingConditions() throws {
        let vc = CAM16ViewingConditions(whitepointXYZ: Self.srgbWhiteXYZ)
        let actual = [
            vc.dRGB.0, vc.dRGB.1, vc.dRGB.2, vc.n, vc.F_L, vc.N_bb, vc.z, vc.A_w,
            vc.chromaExponentTerm,
        ]
        try expectParity(
            actual, matches: "gamut_cam16_viewing_conditions",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
        #expect(vc.N_cb == vc.N_bb)
    }

    @Test("CAM16-UCS forward and inverse match colour-science")
    func cam16ucs() throws {
        let vc = CAM16ViewingConditions(whitepointXYZ: Self.srgbWhiteXYZ)
        let xyz = try Golden("gamut_xyz_input").values
        try expectParity(
            mapTriples(xyz) { CAM16UCS.forward($0, vc) }, matches: "gamut_cam16ucs_forward",
            maxAbsolute: 1e-11, rootMeanSquare: 1e-12)
        let jab = try Golden("gamut_cam16ucs_forward").values
        try expectParity(
            mapTriples(jab) { CAM16UCS.inverse($0, vc) }, matches: "gamut_cam16ucs_inverse",
            maxAbsolute: 1e-11, rootMeanSquare: 1e-12)
    }

    /// Five divisions in the CAM16 inverse go through colour-science's `sdiv`, which maps a
    /// non-finite quotient to 0. Without it, the four hue axes come back NaN, which would show up
    /// as four radial artefacts in a hue sweep and nowhere else.
    @Test("the CAM16 inverse stays finite on the four hue axes")
    func cam16InverseHueAxes() {
        let vc = CAM16ViewingConditions(whitepointXYZ: Self.srgbWhiteXYZ)
        for (a, b) in [(20.0, 0.0), (0.0, 20.0), (-20.0, 0.0), (0.0, -20.0)] {
            let xyz = CAM16UCS.inverse((60.0, a, b), vc)
            #expect(xyz.0.isFinite && xyz.1.isFinite && xyz.2.isFinite)
        }
        // Zero chroma has t == 0, where the reference forces the opponent axes to zero.
        let grey = CAM16UCS.inverse((60.0, 0.0, 0.0), vc)
        #expect(grey.0.isFinite && grey.1.isFinite && grey.2.isFinite)
    }

    @Test("white lands at Jp = 100 exactly")
    func cam16White() {
        let vc = CAM16ViewingConditions(whitepointXYZ: Self.srgbWhiteXYZ)
        #expect(CAM16UCS.forward(Self.srgbWhiteXYZ, vc).0 == 100.0)
    }

    // MARK: - Output compression

    static let rgbCases:
        [(
            name: String, algorithm: OutputGamutCompressSpec.Algorithm, space: String?, knee: Knee,
            lightness: Knee?
        )] = [
            ("off", .off, nil, defaultKnee, lightnessKnee),
            ("aces_rgc", .acesRGC, nil, defaultKnee, lightnessKnee),
            ("aces_rgc_soft", .acesRGC, nil, softKnee, lightnessKnee),
            ("oklch_srgb", .oklch, "sRGB", defaultKnee, lightnessKnee),
            ("oklrab_srgb", .oklrab, "sRGB", defaultKnee, lightnessKnee),
            ("jzazbz_srgb", .jzazbz, "sRGB", defaultKnee, lightnessKnee),
            ("cam16ucs_srgb", .cam16ucs, "sRGB", defaultKnee, lightnessKnee),
            ("oklch_srgb_nolightness", .oklch, "sRGB", softKnee, nil),
            ("oklrab_srgb_nolightness", .oklrab, "sRGB", lateKnee, nil),
            ("jzazbz_srgb_nolightness", .jzazbz, "sRGB", softKnee, nil),
            ("cam16ucs_srgb_nolightness", .cam16ucs, "sRGB", lateKnee, nil),
            ("cam16ucs_display_p3", .cam16ucs, "Display P3", defaultKnee, lightnessKnee),
            ("cam16ucs_bt2020", .cam16ucs, "ITU-R BT.2020", defaultKnee, lightnessKnee),
        ]

    @Test("output compression matches over 1032 pixels", arguments: rgbCases)
    func compressRGB(
        name: String, algorithm: OutputGamutCompressSpec.Algorithm, space: String?, knee: Knee,
        lightness: Knee?
    ) throws {
        let input = try Golden("gamut_rgb_input").values
        var spec = OutputGamutCompressSpec()
        spec.algorithm = algorithm
        spec.knee = knee
        spec.lightnessCompression = lightness
        let compressor = try OutputGamutCompressor(
            spec: spec, colourSpace: space.map { try! ColourSpace.named($0) })
        try expectParity(mapTriples(input, compressor.apply), matches: "gamut_rgb_\(name)")
    }

    /// The default path measures 8.1e-15 over the 1032-pixel sweep and 1.5e-14 over the realizable
    /// sweep, twelve orders under the subsystem contract. The gate sits close to the measurement,
    /// so a regression that still fits inside the 1e-4 gate fails here.
    @Test("the default path agrees with the oracle to 1e-12")
    func defaultPathTightAgreement() throws {
        let compressor = try OutputGamutCompressor(
            spec: OutputGamutCompressSpec(), colourSpace: .sRGB)
        for golden in ["gamut_rgb", "gamut_realizable"] {
            let input = try Golden("\(golden)_input").values
            let expected = golden == "gamut_rgb" ? "gamut_rgb_cam16ucs_srgb" : "\(golden)_cam16ucs"
            try expectParity(
                mapTriples(input, compressor.apply), matches: expected, maxAbsolute: 1e-12,
                rootMeanSquare: 1e-13)
        }
    }

    @Test("the default compressor matches on physically realizable pixels")
    func compressRGBRealizable() throws {
        let input = try Golden("gamut_realizable_input").values
        let compressor = try OutputGamutCompressor(
            spec: OutputGamutCompressSpec(), colourSpace: .sRGB)
        let actual = mapTriples(input, compressor.apply)
        try expectParity(actual, matches: "gamut_realizable_cam16ucs")
        // The shipped path: 2048 chromaticities sampled inside the locus at Y in (0, 2], converted
        // to linear sRGB. 91% of the pixels are out of gamut and the input spans [-14.4, 96.9]. The
        // default algorithm maps all of it into [1.8e-5, 0.99998], so nothing downstream clips.
        #expect((input.min() ?? 0) < -14.0)
        #expect((input.max() ?? 0) > 96.0)
        #expect((actual.min() ?? 0) > 0.0)
        #expect((actual.max() ?? 0) < 1.0)
    }

    /// The reference docstrings promise output in [0, 1]. Only `cam16ucs` delivers it. Measured on
    /// the realizable sweep, the other algorithms leave the cube on 17 to 39% of pixels: `oklch`
    /// reaches 1.00148, `oklrab` 1.00110, `jzazbz` 1.00675, and `aces_rgc` 96.9, since it never
    /// touches the achromatic value. For the three perceptual ones, the cause is the 64-point
    /// lightness grid with bilinear interpolation, which underestimates `C_max`. Neither the
    /// reference nor the port clamps; the test pins the overshoot for downstream consumers.
    @Test("only cam16ucs keeps realizable pixels inside the cube")
    func cubeContainment() throws {
        let input = try Golden("gamut_realizable_input").values
        var maxima: [OutputGamutCompressSpec.Algorithm: Double] = [:]
        let algorithms: [OutputGamutCompressSpec.Algorithm] = [
            .oklch, .oklrab, .jzazbz, .cam16ucs, .acesRGC,
        ]
        for algorithm in algorithms {
            var spec = OutputGamutCompressSpec()
            spec.algorithm = algorithm
            let space: ColourSpace? = algorithm == .acesRGC ? nil : .sRGB
            let compressor = try OutputGamutCompressor(spec: spec, colourSpace: space)
            maxima[algorithm] = mapTriples(input, compressor.apply).max() ?? 0
        }
        #expect(maxima[.cam16ucs] ?? 0 < 1.0)
        #expect(abs((maxima[.oklch] ?? 0) - 1.001481) < 1e-5)
        #expect(abs((maxima[.oklrab] ?? 0) - 1.001103) < 1e-5)
        #expect(abs((maxima[.jzazbz] ?? 0) - 1.006746) < 1e-5)
        #expect((maxima[.acesRGC] ?? 0) > 90.0)
    }

    /// ACES RGC never touches the achromatic maximum, so it cannot bound amplitude. `lightness_
    /// compression` does not apply on this path either.
    @Test("aces_rgc preserves the achromatic maximum")
    func acesRGCAmplitude() throws {
        var spec = OutputGamutCompressSpec()
        spec.algorithm = .acesRGC
        let compressor = try OutputGamutCompressor(spec: spec)
        let out = compressor.apply((2.0, -0.1, 0.3))
        #expect(out.0 == 2.0)
        #expect(out.1 > 0.0)
        #expect(compressor.apply((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0))
        // Non-positive achromatic value is identity, including the negative channels.
        let shadow = compressor.apply((-0.2, -0.3, 0.0))
        #expect(shadow == (-0.2, -0.3, 0.0))
    }

    @Test("black stays exactly black in every algorithm")
    func blackIsPreserved() throws {
        for algorithm in OutputGamutCompressSpec.Algorithm.allCases {
            var spec = OutputGamutCompressSpec()
            spec.algorithm = algorithm
            let space: ColourSpace? = algorithm == .off || algorithm == .acesRGC ? nil : .sRGB
            let compressor = try OutputGamutCompressor(spec: spec, colourSpace: space)
            let out = compressor.apply((0.0, 0.0, 0.0))
            #expect(out.0 == 0.0, "\(algorithm.rawValue) moved black")
            #expect(out.1 == 0.0, "\(algorithm.rawValue) moved black")
            #expect(out.2 == 0.0, "\(algorithm.rawValue) moved black")
        }
    }

    /// Negative luminance makes CAM16's `J` negative, `spow(J/100, 0.5)` NaN, and the `C_max`
    /// lookup index non-finite. Unreachable from the pipeline, which guarantees `Y > 0`. The port
    /// must not trap on the integer conversion; the numbers themselves are not worth matching.
    @Test("negative-luminance pixels do not crash")
    func negativeLuminance() throws {
        let input = try Golden("gamut_negative_input").values
        let compressor = try OutputGamutCompressor(
            spec: OutputGamutCompressSpec(), colourSpace: .sRGB)
        let out = mapTriples(input, compressor.apply)
        #expect(out.count == input.count)
        // Measured agreement on this path is 6.3e-15, so the gate sits near the measurement.
        try expectParity(
            out, matches: "gamut_negative_cam16ucs", maxAbsolute: 1e-12, rootMeanSquare: 1e-13)
    }

    /// Non-finite pixels. The simulation does not produce them, and the reference does not simply
    /// propagate them: `sdiv` turns a NaN quotient into 0 inside the CAM16 lightness correlate, so
    /// the reference emits numbers where a plain division would leave NaN. ``expectParity`` fails
    /// if a NaN appears or disappears.
    @Test(
        "non-finite pixels match the reference on every algorithm",
        arguments: ["aces_rgc", "oklch", "oklrab", "jzazbz", "cam16ucs"])
    func nonFinitePixels(name: String) throws {
        let input = try Golden("gamut_nonfinite_input").values
        var spec = OutputGamutCompressSpec()
        spec.algorithm = OutputGamutCompressSpec.Algorithm(rawValue: name)!
        let compressor = try OutputGamutCompressor(
            spec: spec, colourSpace: name == "aces_rgc" ? nil : .sRGB)
        try expectParity(mapTriples(input, compressor.apply), matches: "gamut_nonfinite_\(name)")
    }

    /// The radial path's passthrough mask is `dist < 1e-9`, so a NaN distance takes the compute
    /// path and both components come back NaN. A `dist >= 1e-9` guard would keep the finite one.
    @Test(
        "non-finite chromaticities match on both input algorithms",
        arguments: [InputGamutCompressSpec.Algorithm.xy, .oklch])
    func nonFiniteChromaticities(algorithm: InputGamutCompressSpec.Algorithm) throws {
        let input = try Golden("gamut_nonfinite_xy_input").values
        var spec = InputGamutCompressSpec()
        spec.algorithm = algorithm
        let actual = InputGamutCompression.compress(input, white: Self.whiteE, spec: spec)
        try expectParity(actual, matches: "gamut_nonfinite_xy_\(algorithm.rawValue)")
    }

    @Test("the buffer path matches the per-pixel path")
    func bufferPath() throws {
        let input = try Golden("gamut_rgb_input")
        var image = ImageBuffer(
            height: input.shape[0], width: 1, channels: 3, values: input.values)
        try OutputGamutCompression.compress(
            &image, spec: OutputGamutCompressSpec(), colourSpace: .sRGB)
        try expectParity(image.values, matches: "gamut_rgb_cam16ucs_srgb")
    }

    // MARK: - Spec surface

    @Test("a perceptual algorithm without an output colour space throws")
    func missingColourSpace() {
        for algorithm in [.oklch, .oklrab, .jzazbz, .cam16ucs]
            as [OutputGamutCompressSpec.Algorithm]
        {
            var spec = OutputGamutCompressSpec()
            spec.algorithm = algorithm
            #expect(throws: SpektraError.self) {
                _ = try OutputGamutCompressor(spec: spec, colourSpace: nil)
            }
        }
        // off and aces_rgc ignore it.
        for algorithm in [.off, .acesRGC] as [OutputGamutCompressSpec.Algorithm] {
            var spec = OutputGamutCompressSpec()
            spec.algorithm = algorithm
            #expect(throws: Never.self) {
                _ = try OutputGamutCompressor(spec: spec, colourSpace: nil)
            }
        }
    }

    @Test("invalid knees are rejected on both specs")
    func kneeValidation() {
        let bad: [Knee] = [(-0.1, 1.0, 6.0), (1.0, 1.0, 6.0), (0.0, 0.0, 6.0), (0.0, 1.0, 0.0)]
        for knee in bad {
            var input = InputGamutCompressSpec()
            input.knee = knee
            #expect(throws: SpektraError.self) { try input.validate() }

            var output = OutputGamutCompressSpec()
            output.knee = knee
            #expect(throws: SpektraError.self) { try output.validate() }

            var lightness = OutputGamutCompressSpec()
            lightness.lightnessCompression = knee
            #expect(throws: SpektraError.self) { try lightness.validate() }
        }
        #expect(throws: Never.self) { try OutputGamutCompressSpec().validate() }
        #expect(throws: Never.self) { try InputGamutCompressSpec().validate() }
    }

    @Test("output activity is derived from the algorithm")
    func outputActive() {
        var spec = OutputGamutCompressSpec()
        #expect(spec.algorithm == .cam16ucs)
        #expect(spec.active)
        spec.algorithm = .off
        #expect(!spec.active)
    }

    // MARK: - Chroma envelopes

    /// The bisection is deterministic given the same in-gamut predicate, and the predicate flips
    /// only if a channel falls within about 1e-16 of the ±1e-6 slack, so the Swift tables should be
    /// byte-identical to the oracle's. This test hashes all 46 080 cells of each envelope and
    /// compares against the SHA-256 the oracle wrote. Committing the tables would add 2.5 MB of
    /// goldens and give a weaker check, since the parity gate tolerates 1e-4.
    @Test("every chroma envelope hashes to the oracle's table")
    func envelopeHashes() throws {
        let expected = try Self.envelopeHashes()
        for (key, entry) in expected {
            let table = try Self.envelope(named: key)
            #expect(table.values.count == 64 * 720, "\(key): wrong cell count")
            #expect(Self.sha256(table.values) == entry.sha256, "\(key): table differs from the oracle")
            #expect(
                table.values.filter { $0 == 0 }.count == entry.zeroCells,
                "\(key): wrong zero-cell count")
            // JSONSerialization loses the last bit of some decimals, so these four go through a
            // relative tolerance. The hash above is the bit-exact gate.
            expectSameDouble(table.values.max() ?? 0, entry.max, "\(key) maximum")
            expectSameDouble(table.values[0], entry.spot00, "\(key) [0, 0]")
            expectSameDouble(table.values[32 * 720 + 360], entry.spot32_360, "\(key) [32, 360]")
            expectSameDouble(table.values[63 * 720 + 719], entry.spot63_719, "\(key) [63, 719]")
        }
    }

    @Test("the default path's envelope matches elementwise on every fourth hue")
    func envelopeSlice() throws {
        let table = OutputGamutCompressor.envelope(.cam16ucs, .sRGB)
        var actual: [Double] = []
        for l in 0..<64 {
            for h in stride(from: 0, to: 720, by: 4) { actual.append(table.values[l * 720 + h]) }
        }
        try expectParity(
            actual, matches: "gamut_cmax_cam16ucs_srgb_h4", maxAbsolute: 0, rootMeanSquare: 0)
    }

    @Test(
        "the other envelopes match elementwise on five lightness rows",
        arguments: ["oklch", "oklrab", "jzazbz", "locus"])
    func envelopeRows(name: String) throws {
        let table = try Self.envelope(named: name == "locus" ? "locus|oklch" : "\(name)|sRGB")
        var actual: [Double] = []
        for l in [0, 16, 32, 48, 63] {
            actual.append(contentsOf: table.values[(l * 720)..<((l + 1) * 720)])
        }
        let golden = name == "locus" ? "gamut_cmax_locus_rows" : "gamut_cmax_\(name)_srgb_rows"
        try expectParity(actual, matches: golden, maxAbsolute: 0, rootMeanSquare: 0)
    }

    /// The input-side locus envelope hits its own `hi = 0.5` bisection ceiling over 27.7% of the
    /// table, all at L >= 0.3968253968253968. Over that region the algorithm compresses against a
    /// flat 0.5 instead of the locus. Reference behaviour, reproduced on purpose.
    @Test("the locus envelope saturates at the bisection ceiling as the reference does")
    func locusEnvelopeCeiling() {
        let table = InputGamutCompression.locusEnvelope
        func saturatedInRow(_ l: Int) -> Int {
            (0..<720).count { table.values[l * 720 + $0] > 0.4999 }
        }
        #expect((0..<64).reduce(0) { $0 + saturatedInRow($1) } == 12772)
        #expect(table.values.max() == 0.4999980926513672)
        let firstAffectedRow = (0..<64).first { saturatedInRow($0) > 0 }
        #expect(firstAffectedRow == 23)
        #expect(table.lightnessGrid[23] == 0.3968253968253968)
        #expect(saturatedInRow(22) == 0)
        // The spec says whole rows saturate above that lightness, but no row saturates completely:
        // the affected band grows from 3 hues at row 23 to 476 of 720 at the top.
        #expect(saturatedInRow(63) == 476)
    }

    @Test("the envelope lookup matches, including the wrap at +pi and the clamp above the grid")
    func envelopeLookup() throws {
        let L = try Golden("gamut_cmax_lookup_l").values
        let h = try Golden("gamut_cmax_lookup_h").values
        let table = OutputGamutCompressor.envelope(.oklch, .sRGB)
        let actual = zip(L, h).map { table.lookup($0, $1) }
        try expectParity(
            actual, matches: "gamut_cmax_lookup_oklch_srgb", maxAbsolute: 0, rootMeanSquare: 0)
        // +pi is the top of atan2's range. It wraps onto the -pi column rather than clamping onto
        // the last one. The hue index comes out as 720.00000000000819, not exactly 720, so the
        // wrapped lookup includes an 8e-12 fraction of the next column.
        #expect(abs(table.lookup(0.6, .pi) - table.lookup(0.6, -.pi)) < 1e-12)
        // A non-finite index returns 0 instead of trapping on the integer conversion.
        #expect(table.lookup(.nan, 0.5) == 0.0)
        #expect(table.lookup(0.5, .nan) == 0.0)
        #expect(table.lookup(.infinity, 0.5) == 0.0)
    }

    @Test("envelopes are cached per space and colour space")
    func envelopeCaching() {
        let first = OutputGamutCompressor.envelope(.cam16ucs, .sRGB)
        let second = OutputGamutCompressor.envelope(.cam16ucs, .sRGB)
        #expect(first.values == second.values)
        let other = OutputGamutCompressor.envelope(.cam16ucs, .displayP3)
        #expect(other.values != first.values)
    }

    // MARK: - Helpers

    static let srgbWhiteXYZ = xyToXYZUnitY(
        x: ColourSpace.sRGB.whitepoint.x, y: ColourSpace.sRGB.whitepoint.y)

    static func envelope(named key: String) throws -> ChromaEnvelope {
        let parts = key.split(separator: "|").map(String.init)
        if parts[0] == "locus" { return InputGamutCompression.locusEnvelope }
        guard let space = PerceptualSpace(rawValue: parts[0]) else {
            throw Golden.GoldenError.malformed(key, "unknown perceptual space")
        }
        return OutputGamutCompressor.envelope(space, try ColourSpace.named(parts[1]))
    }

    struct EnvelopeEntry {
        let sha256: String
        let max: Double
        let zeroCells: Int
        let spot00: Double
        let spot32_360: Double
        let spot63_719: Double
    }

    /// Reads the sidecar `gamut_cmax_tables.json` the fixture module writes.
    static func envelopeHashes() throws -> [String: EnvelopeEntry] {
        guard
            let url = Bundle.module.url(
                forResource: "gamut_cmax_tables", withExtension: "json", subdirectory: "Goldens")
        else {
            throw Golden.GoldenError.missing("gamut_cmax_tables.json")
        }
        guard
            let raw = try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
                as? [String: [String: Any]]
        else {
            throw Golden.GoldenError.malformed("gamut_cmax_tables.json", "not an object of objects")
        }
        var out: [String: EnvelopeEntry] = [:]
        for (key, entry) in raw {
            guard
                let sha = entry["sha256"] as? String,
                let maximum = entry["max"] as? Double,
                let zeros = entry["zero_cells"] as? Int,
                let spot = entry["spot"] as? [String: Double],
                let spot00 = spot["0,0"], let spot32 = spot["32,360"], let spot63 = spot["63,719"]
            else {
                throw Golden.GoldenError.malformed("gamut_cmax_tables.json", "bad entry \(key)")
            }
            out[key] = EnvelopeEntry(
                sha256: sha, max: maximum, zeroCells: zeros, spot00: spot00, spot32_360: spot32,
                spot63_719: spot63)
        }
        return out
    }

    /// Equality up to one last bit, for values that went through `JSONSerialization`.
    func expectSameDouble(
        _ actual: Double, _ expected: Double, _ label: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(
            abs(actual - expected) <= abs(expected) * 1e-15,
            "\(label): \(actual) vs \(expected)", sourceLocation: sourceLocation)
    }

    static func sha256(_ values: [Double]) -> String {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return CryptoKit.SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Applies a `(Double, Double, Double)` transform over a flat 3-channel array.
    func mapTriples(
        _ values: [Double], _ transform: ((Double, Double, Double)) -> (Double, Double, Double)
    ) -> [Double] {
        precondition(values.count % 3 == 0, "expected triples, got \(values.count) values")
        var out = [Double](repeating: 0, count: values.count)
        for i in stride(from: 0, to: values.count, by: 3) {
            let v = transform((values[i], values[i + 1], values[i + 2]))
            out[i] = v.0
            out[i + 1] = v.1
            out[i + 2] = v.2
        }
        return out
    }
}

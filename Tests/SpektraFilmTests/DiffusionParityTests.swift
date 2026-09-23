import Foundation
import Testing

@testable import SpektraFilm

/// Checks diffusion, halation and blur against the reference.
///
/// Three behaviours here can be wrong and still pass a loose check, so each has a test that does not
/// depend on a golden:
///
/// - the IIR blur is 3 to 11 percent wider than its nominal sigma, as upstream's is, and replicates
///   the edge where the FIR path reflects;
/// - the FIR and IIR paths disagree by about 1e-1 at the sigma 3 crossover, with no blend;
/// - the diffusion filter's boundary is the mirror fold, and reconstructing it with the other
///   "reflect" convention falls inside the parity gate on a smooth image.
@Suite("Diffusion, halation and blur parity")
struct DiffusionParityTests {

    // 35 mm at upscale 1: pixelSizeMicrons = 35_000 / longEdge.
    static let pixelSize512 = 35_000.0 / 512.0
    static let pixelSize1024 = 35_000.0 / 1024.0
    static let pixelSize4000 = 35_000.0 / 4000.0

    // MARK: - Inputs

    func plane(_ name: String) throws -> (values: [Double], height: Int, width: Int) {
        let golden = try Golden(name)
        #expect(golden.shape.count == 2, "\(name) should be a 2D plane, got \(golden.shape)")
        return (golden.values, golden.shape[0], golden.shape[1])
    }

    func image(_ name: String) throws -> ImageBuffer { try Golden(name).imageBuffer() }

    // MARK: - The Gaussian blur primitive

    @Test("the FIR kernel matches, including its truncate-3 radius")
    func firKernel() throws {
        for (sigma, tag) in [(0.7, "0p7"), (1.0, "1p0"), (2.0, "2p0")] {
            let (kernel, radius) = GaussianFilter.kernel1D(sigma: sigma, truncate: 3.0)
            let expected = try Golden("blur_kernel_1d_sigma\(tag)")
            #expect(kernel.count == expected.count, "sigma \(sigma): \(kernel.count) taps")
            #expect(radius == (kernel.count - 1) / 2)
            try expectParity(kernel, matches: "blur_kernel_1d_sigma\(tag)")
        }
        // Radius truncates toward zero, so a small enough sigma collapses to a single unit tap.
        let (identity, radius) = GaussianFilter.kernel1D(sigma: 0.032, truncate: 3.0)
        #expect(radius == 0)
        #expect(identity == [1.0])
    }

    @Test("the Young and van Vliet coefficients match")
    func yvvCoefficients() throws {
        var flat: [Double] = []
        for sigma in [3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0] {
            let c = GaussianFilter.yvvCoefficients(sigma: sigma)
            flat.append(contentsOf: [c.b, c.b1, c.b2, c.b3])
        }
        try expectParity(flat, matches: "blur_yvv_coeffs")
    }

    @Test(
        "the FIR path matches",
        arguments: [("0p032", 0.032), ("0p5", 0.5), ("1p0", 1.0), ("2p0", 2.0), ("2p99", 2.99)]
    )
    func firPath(tag: String, sigma: Double) throws {
        let input = try plane("blur_plane_64x80")
        let result = GaussianFilter.filterPlane(
            input.values, height: input.height, width: input.width, sigma: sigma)
        try expectParity(result, matches: "blur_fir_plane_sigma\(tag)")
    }

    @Test(
        "the IIR path matches, excess width and replicated edges included",
        arguments: [("3p0", 3.0), ("5p0", 5.0), ("20p0", 20.0), ("65p0", 65.0)]
    )
    func iirPath(tag: String, sigma: Double) throws {
        let input = try plane("blur_plane_64x80")
        let result = GaussianFilter.filterPlane(
            input.values, height: input.height, width: input.width, sigma: sigma)
        try expectParity(result, matches: "blur_iir_plane_sigma\(tag)")
    }

    @Test("a sigma of zero or less passes the plane through")
    func nonPositiveSigma() throws {
        let input = try plane("blur_plane_64x80")
        for sigma in [0.0, -1.0] {
            let result = GaussianFilter.filterPlane(
                input.values, height: input.height, width: input.width, sigma: sigma)
            #expect(result == input.values, "sigma \(sigma) should be an exact passthrough")
        }
    }

    @Test("the FIR boundary folds correctly on images smaller than the kernel")
    func firTinyBoundaries() throws {
        // 2 * radius = 18 >= 11, so the wrap-free interior split does not apply.
        let tiny = try plane("blur_tiny_3x11")
        try expectParity(
            GaussianFilter.filterPlane(
                tiny.values, height: tiny.height, width: tiny.width, sigma: 2.99),
            matches: "blur_fir_tiny_3x11_sigma2p99")

        // One row with a 19-tap kernel drives the fold into its modulo branch, where Swift's
        // truncating `%` needs the negative fixup that is dead code in Python.
        let row = try plane("blur_row_1x40")
        try expectParity(
            GaussianFilter.filterPlane(
                row.values, height: row.height, width: row.width, sigma: 2.99),
            matches: "blur_fir_row_1x40_sigma2p99")
    }

    @Test("dispatch is per channel: red and blue FIR, green IIR, inside one call")
    func perChannelDispatch() throws {
        let input = try image("diffusion_hdr_64x96x3")
        let sigmas = [2.9425, 3.0690, 2.8791]
        #expect(sigmas[0] < GaussianFilter.smallSigmaMax)
        #expect(sigmas[1] >= GaussianFilter.smallSigmaMax)
        #expect(sigmas[2] < GaussianFilter.smallSigmaMax)
        let result = GaussianFilter.apply(input, sigmaPerChannel: sigmas)
        try expectParity(result.values, matches: "blur_percth_hdr_mixed_dispatch")
    }

    /// The IIR path is wider than the sigma it is given. Upstream measures 1.0998 at sigma 3 and
    /// 1.1112 at sigma 5 from the impulse response. Correcting it would move every halation golden at
    /// sigma 3 and above by about 1e-1.
    @Test("the IIR blur stays wider than its nominal sigma")
    func iirExcessWidth() throws {
        let side = 601
        var impulse = [Double](repeating: 0, count: side * side)
        impulse[300 * side + 300] = 1.0

        var rows: [Double] = []
        for (index, sigma) in [3.0, 5.0].enumerated() {
            let response = GaussianFilter.filterPlane(
                impulse, height: side, width: side, sigma: sigma)
            let centre = Array(response[(300 * side)..<(301 * side)])
            rows.append(contentsOf: centre)

            var total = 0.0
            var firstMoment = 0.0
            for (i, v) in centre.enumerated() {
                total += v
                firstMoment += v * (Double(i) - 300.0)
            }
            let mean = firstMoment / total
            var variance = 0.0
            for (i, v) in centre.enumerated() {
                let d = Double(i) - 300.0 - mean
                variance += v * d * d
            }
            let ratio = (variance / total).squareRoot() / sigma
            let expected = [1.09983865794, 1.11119208009][index]
            #expect(
                abs(ratio - expected) < 1e-6,
                "sigma \(sigma) effective width ratio \(ratio), upstream measures \(expected)")
        }
        try expectParity(rows, matches: "blur_iir_impulse_rows")
    }

    /// There is no smoothing across the dispatch threshold: the branch flips at 3.0 and the two
    /// paths disagree by about 1e-1 on a random image, a thousand times the parity gate.
    @Test("the FIR to IIR crossover is discontinuous")
    func crossoverDiscontinuity() throws {
        let input = try plane("blur_plane_64x80")
        let fir = GaussianFilter.firPlane(
            input.values, height: input.height, width: input.width, sigma: 3.0, truncate: 3.0)
        let iir = GaussianFilter.iirPlane(
            input.values, height: input.height, width: input.width, sigma: 3.0)
        var worst = 0.0
        for i in fir.indices { worst = max(worst, abs(fir[i] - iir[i])) }
        #expect(worst > 0.05, "the two paths should not agree; max difference \(worst)")
    }

    /// Only the IIR path replicates the edge. Swapping in the FIR path's reflection would change the
    /// border rows and columns, and no other test would catch it.
    @Test("the IIR path replicates the edge rather than reflecting it")
    func iirEdgeReplication() throws {
        // A ramp: replication and reflection disagree strongly on a monotone signal.
        let width = 32
        let ramp = (0..<width).map { Double($0) }
        let blurred = GaussianFilter.filterPlane(ramp, height: 1, width: width, sigma: 5.0)
        // Edge replication pulls the left end up toward the interior and the right end down.
        // A reflecting fold would leave both ends nearer the untouched ramp values.
        #expect(blurred[0] > 2.0, "left edge \(blurred[0]) looks reflected, not replicated")
        #expect(
            blurred[width - 1] < Double(width - 1) - 2.0,
            "right edge \(blurred[width - 1]) looks reflected, not replicated")
    }

    // MARK: - The exponential surrogate

    @Test("the three-component fit still sums to 0.9999")
    func exponentialFitTotals() {
        let three = ExponentialFilter.fit(.three).map(\.amplitude).reduce(0, +)
        let two = ExponentialFilter.fit(.two).map(\.amplitude).reduce(0, +)
        #expect(abs(three - 0.9999) < 1e-12, "three-component amplitudes sum to \(three)")
        #expect(abs(two - 1.0) < 1e-12, "two-component amplitudes sum to \(two)")
    }

    @Test("the exponential filter matches, both fit tables")
    func exponentialFilter() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        try expectParity(
            ExponentialFilter.apply(hdr, decay: 5.0).values, matches: "expfilter_n3_hdr_decay5")
        try expectParity(
            ExponentialFilter.apply(hdr, decay: 5.0, mixture: .two).values,
            matches: "expfilter_n2_hdr_decay5")

        // At a 4000 px long edge the third component's sigma is (2.9425, 3.0690, 2.8791), so this
        // one call takes FIR, IIR, FIR across the three channels.
        let decay = [9.3, 9.7, 9.1].map { $0 / Self.pixelSize4000 }
        try expectParity(
            ExponentialFilter.apply(hdr, decayPerChannel: decay).values,
            matches: "expfilter_n3_hdr_decay_4000px")

        let input = try plane("blur_plane_64x80")
        try expectParity(
            ExponentialFilter.filterPlane(
                input.values, height: input.height, width: input.width, decay: 8.0),
            matches: "expfilter_n3_plane_decay8")
    }

    @Test("the real spatial filter drives the coupler model")
    func spatialFilterConformance() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        let filter = FastSpatialFilter()
        try expectParity(filter.exponential(hdr, decay: 5.0).values, matches: "expfilter_n3_hdr_decay5")
        try expectParity(filter.gaussian(hdr, sigma: 0.9).values, matches: "blur_px_hdr_sigma0p9")
        // NoSpatialFilter is only correct once the kernel sizes are zeroed.
        #expect(NoSpatialFilter().gaussian(hdr, sigma: 5.0).values == hdr.values)
    }

    // MARK: - Highlight boost

    @Test(
        "the highlight boost matches",
        arguments: [
            ("boost_hdr_ev3_r0p3_p4", 3.0, 0.3, 4.0),
            ("boost_hdr_ev6_r0p0_p0", 6.0, 0.0, 0.0),
            ("boost_hdr_ev1_r1p0_p2", 1.0, 1.0, 2.0),
        ]
    )
    func boost(golden: String, boostEV: Double, boostRange: Double, protectEV: Double) throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        let result = Diffusion.boostHighlights(
            hdr, boostEV: boostEV, boostRange: boostRange, protectEV: protectEV)
        try expectParity(result.values, matches: golden)
    }

    @Test("the boost curve matches over ten stops, and is anchored at both ends")
    func boostCurve() throws {
        let axis = geomspace(1e-6, pow(2.0, 10.0), count: 512)
        var input = ImageBuffer(height: 512, width: 1, channels: 3)
        for i in 0..<512 {
            for c in 0..<3 { input[i, 0, c] = axis[i] }
        }
        let result = Diffusion.boostHighlights(
            input, boostEV: 10.0, boostRange: 0.5, protectEV: 3.0)
        try expectParity(result.values, matches: "boost_curve_geomspace")

        // y(maxRaw) == maxRaw * 2 ** boostEV is what k solves for.
        let maxRaw = axis[511]
        #expect(abs(result[511, 0, 0] / (maxRaw * pow(2.0, 10.0)) - 1.0) < 1e-12)
        // Below the protected knee the curve is the identity.
        let knee = 0.184 * pow(2.0, 3.0)
        for i in 0..<512 where axis[i] <= knee {
            #expect(result[i, 0, 0] == axis[i])
        }
    }

    @Test("a zero boost is an exact passthrough")
    func boostZero() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        let result = Diffusion.boostHighlights(hdr, boostEV: 0.0, boostRange: 0.3, protectEV: 4.0)
        #expect(result.values == hdr.values)
    }

    /// A maximum of exactly zero fills the output with zeros, which discards negative values.
    /// Copying the input would match for an all-zero input and differ for any input with a negative.
    @Test("a zero maximum fills zeros rather than copying")
    func boostZeroMaximum() throws {
        let input = try image("boost_negatives_input")
        let result = Diffusion.boostHighlights(
            input, boostEV: 2.0, boostRange: 0.3, protectEV: 4.0)
        try expectParity(result.values, matches: "boost_negatives_maxzero")
        #expect(result.values.allSatisfy { $0 == 0.0 })
        #expect(input.values.contains { $0 < 0.0 }, "the fixture should carry negatives")
    }

    @Test("the boost writes through an aliased buffer")
    func boostInPlace() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        var aliased = hdr
        Diffusion.boostHighlights(&aliased, boostEV: 3.0, boostRange: 0.3, protectEV: 4.0)
        try expectParity(aliased.values, matches: "boost_hdr_ev3_r0p3_p4")
    }

    // MARK: - Halation

    @Test(
        "halation matches at the pixel sizes where the dispatch changes",
        arguments: [
            ("halation_hdr_default_512px", 35_000.0 / 512.0),
            ("halation_hdr_default_1024px", 35_000.0 / 1024.0),
            ("halation_hdr_default_4000px", 35_000.0 / 4000.0),
        ]
    )
    func halationDefaults(golden: String, pixelSize: Double) throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        let result = Diffusion.applyHalation(
            hdr, HalationParams(), pixelSizeMicrons: pixelSize)
        try expectParity(result.values, matches: golden)
    }

    @Test("halation matches on a hard edge")
    func halationStepEdge() throws {
        let edge = try image("diffusion_step_edge_64x64x3")
        let result = Diffusion.applyHalation(
            edge, HalationParams(), pixelSizeMicrons: Self.pixelSize4000)
        try expectParity(result.values, matches: "halation_step_edge_4000px")
    }

    /// With the scatter pass skipped, blue's strength of 0 makes pass 2 a bit-exact passthrough for
    /// that channel while red moves. The guards are `any`, not `all`, so the loop still runs on all
    /// three. Short-circuiting per channel would give the same result here and a different one
    /// elsewhere.
    @Test("halation's guards are per frame, not per channel")
    func halationChannelGuards() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        var noScatter = HalationParams()
        noScatter.scatterAmount = 0.0
        let result = Diffusion.applyHalation(
            hdr, noScatter, pixelSizeMicrons: Self.pixelSize4000)
        try expectParity(result.values, matches: "halation_hdr_no_scatter_4000px")

        var redMoved = 0.0
        for p in 0..<hdr.pixelCount {
            #expect(result.values[p * 3 + 2] == hdr.values[p * 3 + 2], "blue should be untouched")
            redMoved = max(redMoved, abs(result.values[p * 3] - hdr.values[p * 3]))
        }
        #expect(redMoved > 1e-3, "red should have moved; largest change \(redMoved)")
    }

    /// The scatter guard is "core or tail", so zeroing one of the two sizes still runs the pass with
    /// the zeroed term floored to 1e-6, an exact identity at that width. An "and" guard would skip
    /// the pass and return the input untouched.
    @Test("halation runs the scatter pass when only one of the two sizes is zero")
    func halationScatterSizeGuard() {
        let hdr = ImageBuffer(
            height: 2, width: 3, channels: 3,
            values: (0..<18).map { 0.1 + Double($0) * 0.37 })
        var params = HalationParams()
        // Pass 2 off, so only the scatter pass can move a value.
        params.halationStrength = (0.0, 0.0, 0.0)

        var coreOnly = params
        coreOnly.scatterTailMicrons = (0.0, 0.0, 0.0)
        var tailOnly = params
        tailOnly.scatterCoreMicrons = (0.0, 0.0, 0.0)

        for (label, one) in [("tail zeroed", coreOnly), ("core zeroed", tailOnly)] {
            let result = Diffusion.applyHalation(hdr, one, pixelSizeMicrons: Self.pixelSize4000)
            var moved = 0.0
            for i in result.values.indices {
                moved = max(moved, abs(result.values[i] - hdr.values[i]))
            }
            #expect(moved > 1e-6, "\(label): the scatter pass should still run; largest change \(moved)")
        }

        // Both sizes zero is the only case the guard rejects, and then the input is unchanged.
        var neither = params
        neither.scatterCoreMicrons = (0.0, 0.0, 0.0)
        neither.scatterTailMicrons = (0.0, 0.0, 0.0)
        #expect(
            Diffusion.applyHalation(hdr, neither, pixelSizeMicrons: Self.pixelSize4000).values
                == hdr.values)
    }

    @Test("halation skips pass 2 when every channel's strength is zero")
    func halationNoBounce() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        var params = HalationParams()
        params.halationStrength = (0.0, 0.0, 0.0)
        try expectParity(
            Diffusion.applyHalation(hdr, params, pixelSizeMicrons: Self.pixelSize4000).values,
            matches: "halation_hdr_no_bounce_4000px")
    }

    @Test("renormalising divides by 1 + strength, exactly")
    func halationRenormalise() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        var params = HalationParams()
        params.halationRenormalize = false
        let plain = Diffusion.applyHalation(
            hdr, params, pixelSizeMicrons: Self.pixelSize4000)
        try expectParity(plain.values, matches: "halation_hdr_no_renorm_4000px")

        let renormalised = Diffusion.applyHalation(
            hdr, HalationParams(), pixelSizeMicrons: Self.pixelSize4000)
        let strength = [0.05, 0.015, 0.0]
        for p in 0..<hdr.pixelCount {
            for c in 0..<3 {
                #expect(
                    renormalised.values[p * 3 + c] == plain.values[p * 3 + c] / (1.0 + strength[c]))
            }
        }
    }

    @Test("halation matches with one bounce, zero decay and off-default scales")
    func halationTuned() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        var params = HalationParams()
        params.scatterAmount = 0.6
        params.scatterSpatialScale = 2.0
        params.halationAmount = 1.5
        params.halationSpatialScale = 0.5
        params.halationStrength = (0.30, 0.10, 0.015)
        params.halationFirstSigmaMicrons = (50.0, 50.0, 50.0)
        params.halationBounceCount = 1
        params.halationBounceDecay = 0.0
        params.halationRenormalize = false
        try expectParity(
            Diffusion.applyHalation(hdr, params, pixelSizeMicrons: Self.pixelSize4000).values,
            matches: "halation_hdr_tuned_4000px")
    }

    @Test("an inactive halation returns the input")
    func halationInactive() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        var params = HalationParams()
        params.active = false
        #expect(
            Diffusion.applyHalation(hdr, params, pixelSizeMicrons: Self.pixelSize4000).values
                == hdr.values)
    }

    // MARK: - Lens blur and unsharp mask

    @Test("the lens blur matches in both unit systems")
    func lensBlur() throws {
        let hdr = try image("diffusion_hdr_64x96x3")
        try expectParity(
            Diffusion.applyGaussianBlur(
                hdr, sigmaMicrons: 30.0, pixelSizeMicrons: Self.pixelSize4000
            ).values,
            matches: "blur_um_hdr_sigma30_4000px")
        try expectParity(
            Diffusion.applyGaussianBlur(hdr, sigmaPixels: 0.9).values,
            matches: "blur_px_hdr_sigma0p9")

        // The micrometre form returns before dividing, so a LUT bake that never ran the resize
        // stage does not need a pixel size.
        #expect(
            Diffusion.applyGaussianBlur(hdr, sigmaMicrons: 0.0, pixelSizeMicrons: nil).values
                == hdr.values)
        #expect(Diffusion.applyGaussianBlur(hdr, sigmaPixels: 0.0).values == hdr.values)
    }

    @Test("the unsharp mask matches, planes and RGB")
    func unsharpMask() throws {
        let input = try plane("blur_plane_64x80")
        let planeBuffer = ImageBuffer(
            height: input.height, width: input.width, channels: 1, values: input.values)
        try expectParity(
            Diffusion.applyUnsharpMask(planeBuffer, sigma: 0.7, amount: 0.7).values,
            matches: "unsharp_plane_64x80_s0p7_a0p7")

        let rgb = try image("diffusion_rand_48x60x3")
        let sharpened = Diffusion.applyUnsharpMask(rgb, sigma: 1.5, amount: 1.2)
        try expectParity(sharpened.values, matches: "unsharp_rgb_48x60_s1p5_a1p2")
        // Nothing clips, so ringing at amount 1.2 pushes values below the input minimum.
        #expect(sharpened.values.min()! < rgb.values.min()!)
    }

    // MARK: - Diffusion-filter tables

    @Test("the group expansion tables match")
    func expandGroups() throws {
        var flat: [Double] = []
        for family in DiffusionFilterParams.Family.allCases {
            let shape = Diffusion.shape(for: family)
            for (group, isBloom) in [(shape.core, false), (shape.halo, false), (shape.bloom, true)] {
                let (lambdas, weights) = Diffusion.expandGroup(group, isBloom: isBloom)
                flat.append(contentsOf: lambdas)
                flat.append(contentsOf: weights)
            }
        }
        try expectParity(flat, matches: "diffusion_expand_tables")
    }

    @Test("the family enum keeps upstream's declaration order")
    func familyOrder() {
        #expect(
            DiffusionFilterParams.Family.allCases.map(\.rawValue) == [
                "glimmerglass", "black_pro_mist", "pro_mist", "cinebloom",
            ])
    }

    /// `numpy.interp` clamps outside its breakpoints. Extrapolating would run the fraction negative
    /// below strength 0.125 and past the 0.99 ceiling above 2.0.
    @Test("the strength to scatter-fraction table matches, and clamps at both ends")
    func strengthToScatter() throws {
        let strengths = [0.0, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 8.0]
        var flat: [Double] = []
        for family in DiffusionFilterParams.Family.allCases {
            flat.append(contentsOf: strengths.map { Diffusion.strengthToScatter($0, family: family) })
        }
        try expectParity(flat, matches: "diffusion_strength_scatter")

        // Below the first breakpoint and above the last, the fraction holds at the endpoint value.
        for family in DiffusionFilterParams.Family.allCases {
            #expect(
                Diffusion.strengthToScatter(0.0001, family: family)
                    == Diffusion.strengthToScatter(0.125, family: family))
            #expect(
                Diffusion.strengthToScatter(64.0, family: family)
                    == Diffusion.strengthToScatter(2.0, family: family))
        }
    }

    /// Cinebloom at its 0.85 family base drives red's innermost weight to -0.105, which clips to 0,
    /// so the renormalisation changes the row. Skipping the clip leaves a negative weight and a
    /// different halo colour, with no error.
    @Test("the halo warmth redistribution matches, clip and renormalise included")
    func haloChannelWeights() throws {
        let uniform = [Double](repeating: 1.0 / 3.0, count: 3)
        var flat: [Double] = []
        for family in DiffusionFilterParams.Family.allCases {
            let rows = Diffusion.haloChannelWeights(
                uniform, warmth: Diffusion.shape(for: family).haloWarmthBase)
            for row in rows {
                flat.append(contentsOf: row)
                #expect(abs(row.reduce(0, +) - 1.0) < 1e-15, "\(family) row does not sum to 1")
                #expect(row.allSatisfy { $0 >= 0 }, "\(family) row has a negative weight")
            }
        }
        try expectParity(flat, matches: "diffusion_halo_weights")

        // The clip fires for cinebloom's red channel.
        let cinebloom = Diffusion.haloChannelWeights(uniform, warmth: 0.85)
        #expect(cinebloom[0][0] == 0.0, "cinebloom red's innermost weight should clip to zero")
    }

    /// The effective warmth is the family base plus the user knob, and only then clamped to
    /// [-1.5, 1.5]. Cinebloom's base of 0.85 plus the GUI maximum of 1.5 reaches 2.35, so the clamp
    /// is reachable from the shipped parameter range and every golden sits inside it.
    @Test("the halo warmth is clamped to plus or minus 1.5 after the family base is added")
    func haloWarmthClamp() {
        let uniform = [Double](repeating: 1.0 / 3.0, count: 3)
        let atLimit = Diffusion.haloChannelWeights(uniform, warmth: 1.5)
        for warmth in [2.35, 4.0, 1e6] {
            #expect(
                Diffusion.haloChannelWeights(uniform, warmth: warmth) == atLimit,
                "warmth \(warmth) should behave exactly like 1.5")
        }
        let atNegativeLimit = Diffusion.haloChannelWeights(uniform, warmth: -1.5)
        #expect(Diffusion.haloChannelWeights(uniform, warmth: -2.65) == atNegativeLimit)
        // Without the clamp the row would keep moving, so the check above is not vacuous.
        #expect(atLimit != Diffusion.haloChannelWeights(uniform, warmth: 0.85))
        #expect(atLimit != atNegativeLimit)
    }

    @Test("the analytic radial profile matches")
    func radialProfile() throws {
        let radius = try Golden("diffusion_radial_radius_um").values
        let plain = Diffusion.diffusionFilterRadialProfile(
            radiusMicrons: radius, family: .blackProMist)
        try expectParity(plain.totalPerChannel.flatMap { $0 }, matches: "diffusion_radial_profile_bpm")

        let overridden = Diffusion.diffusionFilterRadialProfile(
            radiusMicrons: radius, family: .cinebloom, spatialScale: 1.5, haloWarmth: -0.4,
            overrides: Diffusion.PSFOverrides(
                coreIntensity: 0.5, haloIntensity: 2.0, bloomIntensity: 1.0, coreSize: 1.0,
                haloSize: 2.0, bloomSize: 0.5))
        try expectParity(
            overridden.totalPerChannel.flatMap { $0 },
            matches: "diffusion_radial_profile_cinebloom_overrides")
    }

    // MARK: - Diffusion-filter PSF

    @Test("the reference PSF matches, and each channel sums to 1")
    func referencePSF() throws {
        let psf = Diffusion.diffusionFilterPSF(
            kernelHeight: 29, kernelWidth: 29, family: .blackProMist, spatialScale: 1.0,
            pixelSizeMicrons: 100.0)
        try expectParity(psf.values, matches: "psf_bpm_29x29_px100")

        for c in 0..<3 {
            var sum = 0.0
            for p in 0..<psf.pixelCount { sum += psf.values[p * 3 + c] }
            #expect(abs(sum - 1.0) < 1e-12, "channel \(c) sums to \(sum)")
        }
        // The grid is odd-sided with the centre on a sample, so the PSF is exactly symmetric and
        // convolution equals correlation. FFTConvolve2D does not flip the kernel.
        for y in 0..<29 {
            for x in 0..<29 {
                for c in 0..<3 {
                    #expect(psf[y, x, c] == psf[28 - y, x, c])
                    #expect(psf[y, x, c] == psf[y, 28 - x, c])
                }
            }
        }
    }

    @Test("every family's PSF matches", arguments: DiffusionFilterParams.Family.allCases)
    func familyPSF(family: DiffusionFilterParams.Family) throws {
        let psf = Diffusion.diffusionFilterPSF(
            kernelHeight: 31, kernelWidth: 31, family: family, spatialScale: 1.0,
            pixelSizeMicrons: 150.0)
        try expectParity(psf.values, matches: "psf_\(family.rawValue)_31x31_px150")
    }

    @Test("the PSF matches with a negative warmth and with a doubled spatial scale")
    func psfKnobs() throws {
        try expectParity(
            Diffusion.diffusionFilterPSF(
                kernelHeight: 31, kernelWidth: 31, family: .blackProMist, spatialScale: 1.0,
                pixelSizeMicrons: 150.0, haloWarmth: -1.2
            ).values,
            matches: "psf_bpm_31x31_warmth_m1p2")
        try expectParity(
            Diffusion.diffusionFilterPSF(
                kernelHeight: 31, kernelWidth: 31, family: .blackProMist, spatialScale: 2.0,
                pixelSizeMicrons: 150.0
            ).values,
            matches: "psf_bpm_31x31_scale2")
    }

    /// Two override quirks that change the output: a negative intensity clamps to 0 and the other
    /// two renormalise around it, and all three intensities at 0 revert to the unmodified family,
    /// discarding the size overrides with them.
    @Test("the override quirks match")
    func psfOverrides() throws {
        try expectParity(
            Diffusion.diffusionFilterPSF(
                kernelHeight: 31, kernelWidth: 31, family: .cinebloom, spatialScale: 1.0,
                pixelSizeMicrons: 150.0, haloWarmth: 0.3,
                overrides: Diffusion.PSFOverrides(
                    coreIntensity: -5.0, haloIntensity: 1.0, bloomIntensity: 0.25, coreSize: 1.5,
                    haloSize: 1.0, bloomSize: 1.0)
            ).values,
            matches: "psf_cinebloom_31x31_overrides")

        let zeroed = Diffusion.PSFOverrides(
            coreIntensity: 0.0, haloIntensity: 0.0, bloomIntensity: 0.0, coreSize: 2.0,
            haloSize: 2.0, bloomSize: 2.0)
        try expectParity(
            Diffusion.diffusionFilterPSF(
                kernelHeight: 31, kernelWidth: 31, family: .blackProMist, spatialScale: 1.0,
                pixelSizeMicrons: 150.0, overrides: zeroed
            ).values,
            matches: "psf_bpm_31x31_zero_intensities")
        #expect(
            Diffusion.resolve(.blackProMist, overrides: zeroed)
                == Diffusion.shape(for: .blackProMist),
            "all-zero intensities should revert to the family, sizes and all")

        // A negative intensity clamps, and the other two renormalise.
        let negative = Diffusion.resolve(
            .blackProMist, overrides: Diffusion.PSFOverrides(coreIntensity: -5.0))
        #expect(negative.coreWeight == 0.0)
        #expect(abs(negative.haloWeight - 0.7833333333333333) < 1e-15)
        #expect(abs(negative.bloomWeight - 0.21666666666666667) < 1e-15)

        // A size of zero becomes 1e-6, not zero.
        let zeroSize = Diffusion.resolve(
            .blackProMist, overrides: Diffusion.PSFOverrides(coreSize: 0.0))
        #expect(zeroSize.core.lambdaMicrons == 16.0 * 1e-6)

        // The bloom truncation budget uses the overridden lambda and the untouched spread.
        #expect(
            Diffusion.bloomMaxLambdaMicrons(
                .cinebloom, overrides: Diffusion.PSFOverrides(bloomSize: 2.0)) == 5000.0)
    }

    // MARK: - The diffusion filter

    @Test("the kernel radius matches upstream's formula and clamp")
    func kernelRadius() throws {
        var params = DiffusionFilterParams()
        params.active = true

        // ceil(8 * 950 / 400) = 19, under the 48x60 bound of 23.
        params.family = .blackProMist
        #expect(
            Diffusion.diffusionKernelRadius(
                params, pixelSizeMicrons: 400.0, imageHeight: 48, imageWidth: 60) == 19)
        // glimmerglass wants 77 at a 512 px long edge and clamps to 23.
        params.family = .glimmerglass
        #expect(
            Diffusion.diffusionKernelRadius(
                params, pixelSizeMicrons: Self.pixelSize512, imageHeight: 48, imageWidth: 60) == 23)
        // The floor of 5 applies when the bloom is tiny.
        #expect(
            Diffusion.diffusionKernelRadius(
                params, pixelSizeMicrons: 100_000.0, imageHeight: 48, imageWidth: 60) == 5)
        // The 6000x4000 clamp upstream measures: pro_mist wants 2229 and cinebloom 3429, and both
        // clamp to 1999.
        params.family = .proMist
        #expect(
            Diffusion.diffusionKernelRadius(
                params, pixelSizeMicrons: 35_000.0 / 6000.0, imageHeight: 4000, imageWidth: 6000)
                == 1999)
        params.family = .cinebloom
        #expect(
            Diffusion.diffusionKernelRadius(
                params, pixelSizeMicrons: 35_000.0 / 6000.0, imageHeight: 4000, imageWidth: 6000)
                == 1999)
    }

    @Test(
        "the diffusion filter matches",
        arguments: [
            ("difffilter_bpm_48x60_px400_s1", "diffusion_rand_48x60x3"),
            ("difffilter_glimmerglass_48x60_px400_s0p0625", "diffusion_rand_48x60x3"),
            ("difffilter_glimmerglass_48x60_px512edge_s2", "diffusion_rand_48x60x3"),
            ("difffilter_cinebloom_48x60_px400_warmth", "diffusion_rand_48x60x3"),
            ("difffilter_promist_48x60_px400_overrides", "diffusion_rand_48x60x3"),
            ("difffilter_bpm_step_edge_px200", "diffusion_step_edge_64x64x3"),
            ("difffilter_bpm_120x160_px100", "diffusion_wide_120x160x3"),
        ]
    )
    func diffusionFilter(golden: String, inputName: String) throws {
        let (params, pixelSize) = Self.diffusionCase(golden)
        let input = try image(inputName)
        let result = try Diffusion.applyDiffusionFilter(
            input, params, pixelSizeMicrons: pixelSize)
        try expectParity(result.values, matches: golden)
    }

    /// Gates a direct mirror-boundary correlation against the same golden as the FFT path, so a wrong
    /// boundary fold cannot pass on the golden alone. The *other* "reflect" convention is 4.7e-4 away
    /// on this fixture and could pass a looser tolerance on a smoother one.
    @Test("the diffusion filter's boundary is the mirror fold, not the duplicated-edge reflection")
    func diffusionFilterBoundary() throws {
        let golden = "difffilter_bpm_48x60_px400_s1"
        let (params, pixelSize) = Self.diffusionCase(golden)
        let input = try image("diffusion_rand_48x60x3")

        let radius = Diffusion.diffusionKernelRadius(
            params, pixelSizeMicrons: pixelSize, imageHeight: input.height,
            imageWidth: input.width)
        let side = 2 * radius + 1
        let psf = Diffusion.diffusionFilterPSF(
            kernelHeight: side, kernelWidth: side, family: params.family,
            spatialScale: params.spatialScale, pixelSizeMicrons: pixelSize,
            haloWarmth: params.haloWarmth, overrides: Diffusion.overrides(from: params))
        let scatter = Diffusion.strengthToScatter(params.strength, family: params.family)

        let mirror = Self.directCorrelate(
            input, psf: psf, radius: radius, scatter: scatter,
            fold: BoundaryIndex.mirrorEdgeShared)
        try expectParity(mirror, matches: golden)

        // The same reconstruction with the FIR blur's fold misses the golden by more than the gate,
        // so this test catches the swap.
        let duplicated = Self.directCorrelate(
            input, psf: psf, radius: radius, scatter: scatter,
            fold: BoundaryIndex.reflectEdgeDuplicated)
        let expected = try Golden(golden).values
        var worst = 0.0
        for i in duplicated.indices { worst = max(worst, abs(duplicated[i] - expected[i])) }
        #expect(worst > 1e-4, "the two folds should differ above the gate; max difference \(worst)")
    }

    @Test("the diffusion filter's early returns pass the input through")
    func diffusionFilterEarlyReturns() throws {
        let input = try image("diffusion_rand_48x60x3")
        var params = DiffusionFilterParams()
        var result = try Diffusion.applyDiffusionFilter(input, params, pixelSizeMicrons: 400.0)
        #expect(result.values == input.values, "inactive")

        params.active = true
        params.strength = 0.0
        result = try Diffusion.applyDiffusionFilter(input, params, pixelSizeMicrons: 400.0)
        #expect(result.values == input.values, "zero strength")

        params.strength = 0.5
        params.spatialScale = 0.0
        result = try Diffusion.applyDiffusionFilter(input, params, pixelSizeMicrons: 400.0)
        #expect(result.values == input.values, "zero spatial scale")
    }

    /// A 6000 px long edge clamps the radius to 1999, a 3999x3999x3 PSF over an 8000x7998 padded
    /// plane, which peaks near 3.8 GB. No FFT arrangement fits that on a phone, and overlap-save does
    /// not help because a tile cannot be smaller than the kernel. The operator throws before
    /// allocating and reports how much memory it needed.
    @Test("an oversized radius is refused rather than attempted")
    func diffusionFilterMemoryGuard() throws {
        var params = DiffusionFilterParams()
        params.active = true
        params.family = .cinebloom

        let peak = Diffusion.diffusionFilterPeakBytes(
            imageHeight: 4000, imageWidth: 6000,
            radius: Diffusion.diffusionKernelRadius(
                params, pixelSizeMicrons: 35_000.0 / 6000.0, imageHeight: 4000, imageWidth: 6000))
        #expect(peak >> 20 > 3000, "the 6000 px case should want gigabytes, got \(peak >> 20) MB")

        let image = try self.image("diffusion_rand_48x60x3")
        #expect(throws: SpektraError.self) {
            _ = try Diffusion.applyDiffusionFilter(
                image, params, pixelSizeMicrons: 400.0, memoryBudgetBytes: 1 << 10)
        }
        // The same call goes through under the default budget.
        _ = try Diffusion.applyDiffusionFilter(image, params, pixelSizeMicrons: 400.0)
    }

    // MARK: - FFT convolution

    @Test("the transform length is one vDSP accepts and is never short")
    func supportedTransformLength() {
        for n in [1, 7, 8, 9, 100, 124, 136, 1022, 1192, 3999] {
            let length = FFTConvolve2D.supportedLength(atLeast: n)
            #expect(length >= n, "\(n) rounded down to \(length)")
            var residue = length
            while residue % 2 == 0 { residue /= 2 }
            #expect([1, 3, 5, 15].contains(residue), "\(length) is not f * 2^k")
            #expect(length >= 8, "\(length) is below the minimum")
        }
        #expect(FFTConvolve2D.supportedLength(atLeast: 1192) == 1280)
    }

    /// Any transform size at or above the linear length gives the same result, because
    /// `fftconvolve` computes the full linear convolution and no output sample wraps. The whole
    /// diffusion filter depends on this, so the FFT path is checked against a direct sum.
    @Test("the FFT convolution equals a direct sum")
    func fftAgainstDirect() throws {
        let input = try plane("blur_tiny_3x11")
        let kernelSide = 3
        let kernel = (0..<(kernelSide * kernelSide)).map { Double($0 + 1) / 45.0 }
        let fft = FFTConvolve2D.convolveValid(
            image: input.values, height: input.height, width: input.width, kernel: kernel,
            kernelHeight: kernelSide, kernelWidth: kernelSide)

        let outHeight = input.height - kernelSide + 1
        let outWidth = input.width - kernelSide + 1
        var direct = [Double](repeating: 0, count: outHeight * outWidth)
        for y in 0..<outHeight {
            for x in 0..<outWidth {
                var sum = 0.0
                for ky in 0..<kernelSide {
                    for kx in 0..<kernelSide {
                        sum +=
                            input.values[(y + ky) * input.width + x + kx]
                            * kernel[(kernelSide - 1 - ky) * kernelSide + kernelSide - 1 - kx]
                    }
                }
                direct[y * outWidth + x] = sum
            }
        }
        var worst = 0.0
        for i in direct.indices { worst = max(worst, abs(direct[i] - fft[i])) }
        #expect(worst < 1e-13, "FFT and direct convolution differ by \(worst)")
    }

    // MARK: - Fixtures

    /// Parameters for each `difffilter_*` golden, keyed by its name so the fixture module and the
    /// test cannot diverge unnoticed.
    static func diffusionCase(_ golden: String) -> (DiffusionFilterParams, Double) {
        var params = DiffusionFilterParams()
        params.active = true
        switch golden {
        case "difffilter_bpm_48x60_px400_s1":
            params.family = .blackProMist
            params.strength = 1.0
            return (params, 400.0)
        case "difffilter_glimmerglass_48x60_px400_s0p0625":
            params.family = .glimmerglass
            params.strength = 0.0625
            return (params, 400.0)
        case "difffilter_glimmerglass_48x60_px512edge_s2":
            params.family = .glimmerglass
            params.strength = 2.0
            return (params, pixelSize512)
        case "difffilter_cinebloom_48x60_px400_warmth":
            params.family = .cinebloom
            params.strength = 0.75
            params.haloWarmth = 0.6
            params.spatialScale = 0.5
            return (params, 400.0)
        case "difffilter_promist_48x60_px400_overrides":
            params.family = .proMist
            params.strength = 1.5
            params.coreIntensity = 2.0
            params.haloSize = 0.5
            params.bloomIntensity = 0.1
            return (params, 400.0)
        case "difffilter_bpm_step_edge_px200":
            params.family = .blackProMist
            params.strength = 1.0
            return (params, 200.0)
        case "difffilter_bpm_120x160_px100":
            params.family = .blackProMist
            params.strength = 0.5
            return (params, 100.0)
        default:
            preconditionFailure("no parameters recorded for golden '\(golden)'")
        }
    }

    /// The whole diffusion-filter operator by direct correlation, with the boundary fold supplied by
    /// the caller. Slow, and shares no code with the FFT path by design.
    static func directCorrelate(
        _ image: ImageBuffer, psf: ImageBuffer, radius: Int, scatter: Double,
        fold: (Int, Int) -> Int
    ) -> [Double] {
        let side = 2 * radius + 1
        var out = image.values
        for y in 0..<image.height {
            for x in 0..<image.width {
                for c in 0..<3 {
                    var sum = 0.0
                    for ky in 0..<side {
                        let sy = fold(y + ky - radius, image.height)
                        for kx in 0..<side {
                            let sx = fold(x + kx - radius, image.width)
                            sum += image[sy, sx, c] * psf[ky, kx, c]
                        }
                    }
                    out[(y * image.width + x) * 3 + c] =
                        (1.0 - scatter) * image[y, x, c] + scatter * sum
                }
            }
        }
        return out
    }
}

/// `numpy.geomspace`, for the boost curve fixture.
private func geomspace(_ start: Double, _ stop: Double, count: Int) -> [Double] {
    linspace(log10(start), log10(stop), count: count).map { pow(10.0, $0) }
}

import Foundation
import Testing

@testable import SpektraFilm

// MARK: - Support

/// Biased (divide by n) sample moments, matching what `Tools/parity/fixtures/grain.py` measures
/// through `numpy.std` and `scipy.stats.skew`.
private struct SampleMoments {
    let mean: Double
    let standardDeviation: Double
    let skewness: Double

    init(_ samples: [Double]) {
        let n = Double(samples.count)
        var sum = 0.0
        for x in samples { sum += x }
        mean = sum / n
        var m2 = 0.0
        var m3 = 0.0
        for x in samples {
            let d = x - mean
            m2 += d * d
            m3 += d * d * d
        }
        m2 /= n
        m3 /= n
        standardDeviation = m2.squareRoot()
        skewness = m2 > 0 ? m3 / (m2 * m2.squareRoot()) : 0
    }
}

/// Gates a measured statistic against its closed form at five times the oracle's measured sampling
/// standard deviation.
///
/// The sampling standard deviations come from `grain_*_moment_sd_*`, which the oracle measures over
/// 48 independent realisations of exactly ``statisticalSamples`` samples drawn from the closed-form
/// distribution. They are measured rather than assumed because `sqrt(6/n)`, the normal-theory
/// standard error of the skewness estimator, understates the real spread by up to 25 percent at the
/// skewness levels grain produces.
///
/// Five measured sigma is a per-assertion false-failure rate of 5.7e-7, so about 1 in 13000 runs
/// across the roughly 135 seed-dependent assertions here. Every seed is fixed, so in practice a run
/// either always passes or always fails; the rate is the risk taken on when a seed or the sampler
/// changes. Measured over 12 independent realisations of the layered gate, 648 assertions, the worst
/// miss was 3.79 sigma and nothing exceeded 4. Raise ``statisticalSamples`` if that ever bites; do
/// not widen this.
private func expectStatistic(
    _ measured: Double,
    closedForm: Double,
    samplingSD: Double,
    _ label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let miss = abs(measured - closedForm)
    #expect(
        miss <= 5.0 * samplingSD,
        Comment(
            rawValue: "\(label): \(measured) vs closed form \(closedForm), off by "
                + "\(miss / samplingSD) sampling sigma (gate 5)"),
        sourceLocation: sourceLocation)
}

/// Sample count every calibrated tolerance in this file assumes. Changing it invalidates every
/// `grain_*_moment_sd_*` golden.
private let statisticalSamples = 512 * 512
private let statisticalSide = 512

/// The FIR half of `utils/fast_gaussian_filter.fast_gaussian_filter`, as a test double.
///
/// The production filter belongs to the diffusion subsystem and is injected through
/// ``SpatialFilter``. Grain and glare need one to exercise the blur gates, the dye-cloud radius-0
/// identity and the variance retention, so this reimplements the small-sigma path: `truncate = 3.0`,
/// `radius = int(truncate * sigma + 0.5)`, an unnormalised `exp(-0.5 * (x/sigma)^2)` kernel scaled to
/// sum 1, separable, with `scipy.ndimage`-style reflect boundaries. Grain's sigmas never reach the
/// `sigma >= 3` IIR crossover.
private struct FIRGaussianFilter: SpatialFilter {
    static let truncate = 3.0

    static func kernel(sigma: Double) -> [Double] {
        let radius = Int(truncate * sigma + 0.5)
        var taps = (-radius...radius).map {
            Foundation.exp(-0.5 * (Double($0) / sigma) * (Double($0) / sigma))
        }
        let total = taps.reduce(0, +)
        for i in taps.indices { taps[i] /= total }
        return taps
    }

    /// `Var_out / Var_in` for a white-noise field through the separable 2D kernel.
    static func varianceRetention(sigma: Double) -> Double {
        guard sigma > 0 else { return 1.0 }
        let sumSquares = kernel(sigma: sigma).reduce(0) { $0 + $1 * $1 }
        return sumSquares * sumSquares
    }

    func gaussian(_ image: ImageBuffer, sigma: Double) -> ImageBuffer {
        guard sigma > 0 else { return image }
        let taps = Self.kernel(sigma: sigma)
        let radius = taps.count / 2
        // A radius-0 kernel is exactly [1.0], so the convolution is the identity.
        if radius == 0 { return image }

        let channels = image.channels
        var vertical = image
        for y in 0..<image.height {
            for x in 0..<image.width {
                for c in 0..<channels {
                    var sum = 0.0
                    for k in -radius...radius {
                        let yy = BoundaryIndex.reflectEdgeDuplicated(y + k, count: image.height)
                        sum += image[yy, x, c] * taps[k + radius]
                    }
                    vertical[y, x, c] = sum
                }
            }
        }
        var out = vertical
        for y in 0..<image.height {
            for x in 0..<image.width {
                for c in 0..<channels {
                    var sum = 0.0
                    for k in -radius...radius {
                        let xx = BoundaryIndex.reflectEdgeDuplicated(x + k, count: image.width)
                        sum += vertical[y, xx, c] * taps[k + radius]
                    }
                    out[y, x, c] = sum
                }
            }
        }
        return out
    }

    func exponential(_ image: ImageBuffer, decay: Double) -> ImageBuffer {
        fatalError("grain and glare never call the exponential filter")
    }
}

/// Portra 400, the stock every grain golden is generated against.
private struct GrainStock {
    let profile: Profile
    /// Fog-normalised, as `develop` hands them to `apply_grain`.
    let curves: [Double]
    /// The raw profile array, which the reference does not normalise.
    let layers: [Double]

    init(_ stock: String = "kodak_portra_400") throws {
        profile = try ProfileLibrary.load(stock)
        curves = DensityCurves.normalized(
            curves: profile.data.densityCurves, minima: profile.data.densityCurveMinima)
        layers = profile.data.densityCurvesLayers
    }

    var positive: Bool { profile.isPositive }
    var densityMaxCurves: [Double] { nanMax(curves, channels: 3) }
    var densityMaxLayers: [Double] { nanMax(layers, channels: 9) }

    func layered(_ params: GrainParams, pixelSizeMicrons: Double) -> Grain.LayeredParameters {
        Grain.LayeredParameters(
            params: params, pixelSizeMicrons: pixelSizeMicrons,
            densityMaxLayers: densityMaxLayers)
    }

    func single(_ params: GrainParams, pixelSizeMicrons: Double) -> Grain.SingleLayerParameters {
        Grain.SingleLayerParameters(
            params: params, pixelSizeMicrons: pixelSizeMicrons,
            densityMaxCurves: densityMaxCurves)
    }
}

/// 35 mm at 4000 px wide, the pitch every derived-parameter golden uses.
private let pixelSize = 8.75

/// `sqrt((48/2)^2 * pi)`, the pixel pitch whose area equals a 48 um densitometer aperture.
private let densitometerPixelSize = 42.538892421642245

/// Defaults with every spatial and clumping stage off, so the samples are i.i.d.
private func quietParams(_ edit: (inout GrainParams) -> Void = { _ in }) -> GrainParams {
    var params = GrainParams()
    params.blur = 0
    params.blurDyeCloudsMicrons = 0
    params.microStructure = (0, 0)
    edit(&params)
    return params
}

private func flatDensity(_ value: Double, side: Int) -> ImageBuffer {
    ImageBuffer(height: side, width: side, channels: 3, repeating: value)
}

private func channel(_ image: ImageBuffer, _ c: Int) -> [Double] {
    (0..<image.pixelCount).map { image.values[$0 * image.channels + c] }
}

// MARK: - Deterministic parity

/// Everything grain computes before the first random draw, gated at 1e-12.
///
/// These catch every index-order, unit and broadcast error in the model, which is most of the file.
/// The sampled stages are gated separately, statistically, in `GrainStatisticsTests`.
@Suite("Grain derived parameters")
struct GrainDerivedParameterTests {

    @Test("the single-layer tables match the oracle")
    func singleLayerTables() throws {
        let stock = try GrainStock()
        for (repeats, golden) in [
            (1, "grain_derived_single_kodak_portra_400"),
            (3, "grain_derived_single_sub3_kodak_portra_400"),
        ] {
            var params = GrainParams()
            params.subLayerCount = repeats
            let derived = stock.single(params, pixelSizeMicrons: pixelSize)
            let table =
                derived.densityMaxCurves + derived.densityMin + derived.densityMax
                + derived.particleArea + derived.particlesPerPixel
            try expectParity(
                table, matches: golden, maxAbsolute: 1e-12, rootMeanSquare: 1e-12)
            #expect(derived.pixelArea == pixelSize * pixelSize)
        }
    }

    @Test("the layered tables match the oracle")
    func layeredTables() throws {
        let stock = try GrainStock()
        let derived = stock.layered(GrainParams(), pixelSizeMicrons: pixelSize)
        let tables =
            derived.densityMaxLayersRaw + derived.densityMaxFractions + derived.densityMinLayers
            + derived.densityMaxLayers + derived.particleAreaLayers + derived.particlesPerPixel
            + derived.odParticle + derived.dyeCloudSigmaPixels
        try expectParity(
            tables, matches: "grain_derived_layers_kodak_portra_400",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-12)
        try expectParity(
            derived.densityMaxTotal, matches: "grain_derived_layers_total_kodak_portra_400",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-12)
    }

    /// The fractions are what make the final `-= density_min` balance, so they get their own check.
    @Test("each channel's sublayer fractions sum to one")
    func fractionsSumToOne() throws {
        let stock = try GrainStock()
        let derived = stock.layered(GrainParams(), pixelSizeMicrons: pixelSize)
        for channel in 0..<3 {
            var fractionSum = 0.0
            var minimumSum = 0.0
            for sublayer in 0..<3 {
                fractionSum += derived.densityMaxFractions[sublayer * 3 + channel]
                minimumSum += derived.densityMinLayers[sublayer * 3 + channel]
            }
            #expect(abs(fractionSum - 1.0) <= 1e-15, "channel \(channel) fractions sum \(fractionSum)")
            #expect(abs(minimumSum - derived.densityMin[channel]) <= 1e-17)
        }
    }

    /// Both micro-structure gates are false at every resolution a render reaches. The table pins the
    /// thresholds, because they move with output resolution.
    @Test("the micro-structure gates match the oracle across pixel pitches")
    func microStructureGates() throws {
        let golden = try Golden("grain_derived_micro_gates")
        var computed: [Double] = []
        var anyGateOpen = false
        for row in 0..<golden.shape[0] {
            let pitch = golden.values[row * 5]
            let micro = (golden.values[row * 5 + 1], golden.values[row * 5 + 2])
            var params = GrainParams()
            params.microStructure = micro
            let derived = try GrainStock().layered(params, pixelSizeMicrons: pitch)
            computed += [
                pitch, micro.0, micro.1, derived.microStructureBlurPixels,
                derived.microStructureSigma,
            ]
            // At the shipped `micro_structure`, the clumping gate stays shut at every pitch a real
            // render lands on. It needs under 0.6 um, a 35 mm frame at 58333 px wide.
            if micro.1 == 30.0 && pitch >= 1.0 {
                #expect(
                    derived.microStructureSigma <= 0.05,
                    "clumping fires at \(pitch) um per pixel, which no render reaches")
            }
            if derived.microStructureSigma > 0.05 { anyGateOpen = true }
        }
        try expectParity(
            computed, matches: "grain_derived_micro_gates",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-12)
        #expect(anyGateOpen, "the table has to include a pitch that opens the gate")
    }

    @Test("the development probability and the saturation match the oracle")
    func probabilityAndSaturation() throws {
        let plane = try Golden("grain_density_plane_input").values
        let cases = try Golden("grain_p_sat_params")
        var computed: [Double] = []
        for row in 0..<cases.shape[0] {
            let densityMax = cases.values[row * 2]
            let uniformity = cases.values[row * 2 + 1]
            var probabilities: [Double] = []
            var saturations: [Double] = []
            for density in plane {
                let p = Grain.probabilityOfDevelopment(density: density, densityMax: densityMax)
                probabilities.append(p)
                saturations.append(Grain.saturation(probability: p, uniformity: uniformity))
            }
            computed += probabilities + saturations
        }
        try expectParity(
            computed, matches: "grain_p_sat", maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    /// The clip floors every non-positive density to 1e-6 and saturates everything above
    /// `density_max` at `1 - 1e-6`.
    @Test("the probability clip has the reference's bounds")
    func probabilityClipBounds() {
        for density in [-1e9, -0.5, -0.01, 0.0, 1e-12] {
            #expect(
                Grain.probabilityOfDevelopment(density: density, densityMax: 2.2)
                    == Grain.probabilityFloor, "\(density)")
        }
        for density in [2.2, 2.3, 1e9] {
            #expect(
                Grain.probabilityOfDevelopment(density: density, densityMax: 2.2)
                    == Grain.probabilityCeiling, "\(density)")
        }
        #expect(Grain.probabilityOfDevelopment(density: .nan, densityMax: 2.2).isNaN)
        // sat stays positive at the ceiling even when uniformity is 1, which is what keeps lambda
        // finite. 2e-6 is the floor grain.md section 4.1 records.
        let sat = Grain.saturation(probability: Grain.probabilityCeiling, uniformity: 1.0)
        #expect(sat > 0)
        #expect(abs(sat - 2e-6) < 1e-11, "saturation floor \(sat)")
    }

    @Test("the sublayer split matches the oracle for a negative and a positive stock")
    func sublayerSplit() throws {
        let input = try Golden("grain_sublayer_input").imageBuffer()
        for name in ["kodak_portra_400", "fujifilm_velvia_100"] {
            let stock = try GrainStock(name)
            let split = Grain.sublayerDensities(
                input, densityCurves: stock.curves, densityCurvesLayers: stock.layers,
                positive: stock.positive)
            // The golden is [H][W][sublayer][channel]; the split is one buffer per channel, three
            // sublayers deep.
            var flattened = [Double](repeating: 0, count: input.pixelCount * 9)
            for pixel in 0..<input.pixelCount {
                for sublayer in 0..<3 {
                    for channel in 0..<3 {
                        flattened[pixel * 9 + sublayer * 3 + channel] =
                            split[channel].values[pixel * 3 + sublayer]
                    }
                }
            }
            try expectParity(
                flattened, matches: "grain_sublayer_split_\(name)",
                maxAbsolute: 1e-12, rootMeanSquare: 1e-12)
        }
    }

    /// The closed form is the whole model, so gating it against the oracle exactly gates the derived
    /// parameter chain, the clip, the saturation term and the Poisson-thinning identity in one go.
    @Test("the closed-form moments match the oracle")
    func closedFormMoments() throws {
        let stock = try GrainStock()
        let params = quietParams()
        let derived = stock.layered(params, pixelSizeMicrons: pixelSize)

        let levels = try Golden("grain_closed_form_levels").values
        var computed: [Double] = []
        for density in levels {
            let split = Grain.sublayerDensities(
                flatDensity(density, side: 1), densityCurves: stock.curves,
                densityCurvesLayers: stock.layers, positive: stock.positive)
            for channel in 0..<3 {
                let moments = derived.closedFormMoments(
                    sublayerDensities: Array(split[channel].values[0..<3]), channel: channel)
                computed += [
                    moments.mean, moments.standardDeviation, moments.skewness,
                    moments.excessKurtosis,
                ]
            }
        }
        try expectParity(
            computed, matches: "grain_closed_form_layers_kodak_portra_400",
            maxAbsolute: 1e-12, rootMeanSquare: 1e-12)

        let singleLevels = try Golden("grain_single_layer_levels").values
        for repeats in [1, 3] {
            let single = stock.single(
                quietParams {
                    $0.sublayersActive = false; $0.subLayerCount = repeats
                },
                pixelSizeMicrons: pixelSize)
            var rows: [Double] = []
            for density in singleLevels {
                for channel in 0..<3 {
                    let moments = single.closedFormMoments(density: density, channel: channel)
                    rows += [
                        moments.mean, moments.standardDeviation, moments.skewness,
                        moments.excessKurtosis,
                    ]
                }
            }
            try expectParity(
                rows, matches: "grain_closed_form_single_sub\(repeats)_kodak_portra_400",
                maxAbsolute: 1e-12, rootMeanSquare: 1e-12)
        }
    }

    /// `n_sub_layers` is a complete distributional no-op, not just a mean- and variance-preserving
    /// one.
    ///
    /// `grain.md` section 5.1 and trap 15 say it flattens the skewness and that a skewness test would
    /// catch a port that dropped it. It does not. `od_particle` is derived from the *already divided*
    /// `n_particles_per_pixel`, so each repeat's step grows by `n_sub_layers` and the final division
    /// by `n_sub_layers` cancels it exactly. The sum of `S` independent `Poisson(lambda/S)` variates
    /// is `Poisson(lambda)`, so the output is the same scaled Poisson for every `S`: mean, variance,
    /// skewness and kurtosis all identical. Confirmed in the oracle at 512x512 for `S` in
    /// `{1, 2, 3, 5}`, where all four moments agree within sampling noise and the closed forms are
    /// bit-identical. The only cost is `S` times the Poisson draws.
    @Test("n_sub_layers leaves the whole distribution alone")
    func subLayerCountIsADistributionalNoOp() throws {
        let stock = try GrainStock()
        let reference = stock.single(
            quietParams { $0.sublayersActive = false }, pixelSizeMicrons: pixelSize
        ).closedFormMoments(density: 1.0, channel: 0)
        for repeats in [2, 3, 5] {
            let moments = stock.single(
                quietParams {
                    $0.sublayersActive = false; $0.subLayerCount = repeats
                },
                pixelSizeMicrons: pixelSize
            ).closedFormMoments(density: 1.0, channel: 0)
            #expect(abs(moments.mean - reference.mean) <= 1e-15, "\(repeats) repeats moved the mean")
            #expect(abs(moments.variance / reference.variance - 1.0) <= 1e-14)
            #expect(abs(moments.skewness / reference.skewness - 1.0) <= 1e-14)
            #expect(abs(moments.excessKurtosis / reference.excessKurtosis - 1.0) <= 1e-14)
            // The step each repeat contributes after the division is unchanged, which is why the
            // moments are. `od_particle` itself grows by the repeat count.
            let population = moments.populations[0]
            #expect(abs(population.step / reference.populations[0].step - 1.0) <= 1e-14)
            #expect(
                abs(
                    population.odTimesSaturation
                        / reference.populations[0].odTimesSaturation - Double(repeats)) <= 1e-12)
            #expect(
                abs(population.lambda * Double(repeats) / reference.populations[0].lambda - 1.0)
                    <= 1e-12)
        }
    }
}

// MARK: - Structural invariants

/// Properties any correct sampler has, gated exactly. None of them depend on the RNG.
@Suite("Grain structural invariants")
struct GrainInvariantTests {

    private static let side = 24

    @Test("an inactive or bypassed stage returns the input untouched")
    func bypassIsExact() throws {
        let stock = try GrainStock()
        let input = try Golden("grain_sublayer_input").imageBuffer()
        for (label, params, bypass) in [
            ("active = false", quietParams { $0.active = false }, false),
            ("bypass", quietParams(), true),
        ] {
            let out = Grain.apply(
                input, pixelSizeMicrons: pixelSize, params: params, densityCurves: stock.curves,
                densityCurvesLayers: stock.layers, positive: stock.positive, bypass: bypass,
                spatial: NoSpatialFilter())
            #expect(out.values == input.values, "\(label) changed the image")
        }
    }

    /// The output of one population is an exact integer multiple of `od_particle * sat`. A wrong
    /// `od`, a wrong `sat` or a continuous approximation all break this.
    @Test("the output lies on the particle lattice")
    func integerLattice() throws {
        let stock = try GrainStock()
        let derived = stock.layered(quietParams(), pixelSizeMicrons: pixelSize)
        // A ramp, so `sat` varies per pixel and the lattice spacing varies with it.
        var plane = ImageBuffer(height: Self.side, width: Self.side, channels: 1)
        for i in plane.values.indices {
            plane.values[i] = -0.2 + 2.6 * Double(i) / Double(plane.values.count - 1)
        }

        for sublayer in 0..<3 {
            for channel in 0..<3 {
                let i = sublayer * 3 + channel
                let densityMax = derived.densityMaxLayers[i]
                let particles = derived.particlesPerPixel[i]
                let grain = Grain.layerParticleModel(
                    plane, densityMax: densityMax, particlesPerPixel: particles,
                    uniformity: derived.uniformity[channel],
                    key: PhiloxKey(seed: 0, channel: channel, sublayer: sublayer),
                    spatial: NoSpatialFilter())

                var worst = 0.0
                var sawSomething = false
                for pixel in plane.values.indices {
                    let p = Grain.probabilityOfDevelopment(
                        density: plane.values[pixel], densityMax: densityMax)
                    let step =
                        densityMax / particles
                        * Grain.saturation(probability: p, uniformity: derived.uniformity[channel])
                    let counts = grain.values[pixel] / step
                    #expect(counts >= -1e-9, "negative particle count \(counts)")
                    worst = max(worst, abs(counts - counts.rounded()))
                    if counts > 0.5 { sawSomething = true }
                }
                #expect(worst <= 1e-9, "sublayer \(sublayer) channel \(channel): off lattice by \(worst)")
                #expect(sawSomething, "sublayer \(sublayer) channel \(channel) developed nothing")
            }
        }
    }

    /// A pixel's value must be fixed by the stream key and the linear pixel index alone. That is what
    /// makes a render reproducible across machines with different core counts, and it is the property
    /// upstream's fast path does not have.
    @Test("every pixel is a function of its key and index only")
    func perPixelDeterminism() throws {
        let stock = try GrainStock()
        let derived = stock.layered(quietParams(), pixelSizeMicrons: pixelSize)
        let plane = ImageBuffer(height: 9, width: 11, channels: 1, repeating: 0.8)
        let key = PhiloxKey(seed: 7, channel: 1, sublayer: 2)
        let i = 2 * 3 + 1
        let densityMax = derived.densityMaxLayers[i]
        let particles = derived.particlesPerPixel[i]
        let uniformity = derived.uniformity[1]

        let grain = Grain.layerParticleModel(
            plane, densityMax: densityMax, particlesPerPixel: particles, uniformity: uniformity,
            key: key, spatial: NoSpatialFilter())

        let p = Grain.probabilityOfDevelopment(density: 0.8, densityMax: densityMax)
        let sat = Grain.saturation(probability: p, uniformity: uniformity)
        let odParticle = densityMax / particles
        for pixel in plane.values.indices {
            let drawn = Distributions.poisson(
                lambda: particles * p / sat, key: key, counter: UInt64(pixel))
            #expect(grain.values[pixel] == Double(drawn) * odParticle * sat)
        }
    }

    @Test("the same seed repeats exactly and a different seed does not")
    func seedSelectsTheRealisation() throws {
        let stock = try GrainStock()
        let input = flatDensity(0.6, side: Self.side)
        func render(seed: UInt64) -> ImageBuffer {
            Grain.apply(
                input, pixelSizeMicrons: pixelSize, params: quietParams(),
                densityCurves: stock.curves, densityCurvesLayers: stock.layers,
                positive: stock.positive, seed: seed, spatial: NoSpatialFilter())
        }
        let first = render(seed: 3)
        #expect(render(seed: 3).values == first.values)
        #expect(render(seed: 4).values != first.values)
    }

    /// Non-positive densities all clip to the same probability, so they produce the same plane. A
    /// density above `density_max` saturates.
    @Test("non-positive densities are indistinguishable and high densities saturate")
    func clippingBehaviour() throws {
        let stock = try GrainStock()
        let derived = stock.layered(quietParams(), pixelSizeMicrons: pixelSize)
        let densityMax = derived.densityMaxLayers[0]
        let particles = derived.particlesPerPixel[0]
        let key = PhiloxKey(seed: 0)

        func sample(_ density: Double) -> [Double] {
            Grain.layerParticleModel(
                ImageBuffer(height: 16, width: 16, channels: 1, repeating: density),
                densityMax: densityMax, particlesPerPixel: particles,
                uniformity: derived.uniformity[0], key: key, spatial: NoSpatialFilter()
            ).values
        }
        let floored = sample(0.0)
        for density in [-0.5, -0.01, 0.0] {
            #expect(sample(density) == floored, "density \(density) differs from 0.0")
        }
        // p = 1e-6, so the mean is 1e-6 * density_max, which at these particle counts is almost
        // always zero developed particles.
        #expect(floored.allSatisfy { $0 >= 0 })

        // Above density_max the mean is density_max * (1 - 1e-6).
        let saturated = SampleMoments(
            Grain.layerParticleModel(
                ImageBuffer(height: 256, width: 256, channels: 1, repeating: densityMax * 3),
                densityMax: densityMax, particlesPerPixel: particles,
                uniformity: derived.uniformity[0], key: key, spatial: NoSpatialFilter()
            ).values)
        let expected = densityMax * Grain.probabilityCeiling
        let closedForm = Grain.ParticleMoments(
            summing: [
                Grain.ParticlePopulation(
                    density: densityMax * 3, densityMax: densityMax,
                    particlesPerPixel: particles, uniformity: derived.uniformity[0], weight: 1)
            ], offset: 0)
        let standardError = closedForm.standardDeviation / 256.0
        #expect(
            abs(saturated.mean - expected) <= 5 * standardError,
            "saturated mean \(saturated.mean), want \(expected) within \(5 * standardError)")
    }

    /// Grain legitimately goes negative, down to about `-density_min`, because the subtraction at the
    /// end is unconditional. The printing and scanning LUT wires reserve headroom for it, so nothing
    /// downstream may clamp at zero.
    @Test("a black frame produces densities near minus density_min")
    func outputGoesNegative() throws {
        let stock = try GrainStock()
        let params = quietParams()
        let out = Grain.apply(
            flatDensity(0.0, side: 64), pixelSizeMicrons: pixelSize, params: params,
            densityCurves: stock.curves, densityCurvesLayers: stock.layers,
            positive: stock.positive, spatial: NoSpatialFilter())
        let smallest = out.values.min() ?? 0
        #expect(smallest < 0, "nothing went negative; smallest \(smallest)")
        #expect(smallest >= -params.densityMin.0 - 1e-12, "overshot below -density_min: \(smallest)")
    }
}

// MARK: - Statistics

/// The sampled stages, gated against the closed form of `grain.md` section 4.2 at five measured
/// sampling sigma.
///
/// Gating against the analytic moments takes the oracle's RNG out of the loop: byte parity with
/// SciPy on NumPy's legacy MT19937 is unreachable at acceptable cost, and reproducing it would buy
/// nothing. Each test also puts the oracle's own realisation through the same gate, so a
/// miscalibrated tolerance shows up as the oracle failing rather than as a false pass.
@Suite("Grain statistics")
struct GrainStatisticsTests {

    @Test("the layered path's moments match the closed form at six density levels")
    func layeredMoments() throws {
        let stock = try GrainStock()
        let params = quietParams()
        let derived = stock.layered(params, pixelSizeMicrons: pixelSize)
        let levels = try Golden("grain_closed_form_levels").values
        let samplingSD = try Golden("grain_layered_moment_sd_kodak_portra_400").values
        let oracle = try Golden("grain_oracle_layered_moments_kodak_portra_400").values

        for (row, density) in levels.enumerated() {
            let out = Grain.apply(
                flatDensity(density, side: statisticalSide), pixelSizeMicrons: pixelSize,
                params: params, densityCurves: stock.curves, densityCurvesLayers: stock.layers,
                positive: stock.positive, seed: 0x6011 &+ UInt64(row),
                spatial: NoSpatialFilter())
            #expect(out.pixelCount == statisticalSamples)

            let split = Grain.sublayerDensities(
                flatDensity(density, side: 1), densityCurves: stock.curves,
                densityCurvesLayers: stock.layers, positive: stock.positive)

            for c in 0..<3 {
                let closedForm = derived.closedFormMoments(
                    sublayerDensities: Array(split[c].values[0..<3]), channel: c)
                let measured = SampleMoments(channel(out, c))
                let base = (row * 3 + c) * 3
                let label = "layered D=\(density) channel \(c)"
                let targets = [closedForm.mean, closedForm.standardDeviation, closedForm.skewness]
                let names = ["mean", "sd", "skewness"]
                let values = [measured.mean, measured.standardDeviation, measured.skewness]
                for k in 0..<3 {
                    expectStatistic(
                        values[k], closedForm: targets[k], samplingSD: samplingSD[base + k],
                        "\(label) \(names[k])")
                    expectStatistic(
                        oracle[base + k], closedForm: targets[k],
                        samplingSD: samplingSD[base + k], "oracle \(label) \(names[k])")
                }
            }
        }
    }

    /// The gate that separates the real sampler from a Gaussian with the right first two moments.
    /// At low density the Poisson asymmetry is large and a normal approximation returns nearly zero.
    @Test("the measured skewness is far from a Gaussian's")
    func skewnessWouldCatchAGaussian() throws {
        let stock = try GrainStock()
        let derived = stock.layered(quietParams(), pixelSizeMicrons: pixelSize)
        let samplingSD = try Golden("grain_layered_moment_sd_kodak_portra_400").values
        let levels = try Golden("grain_closed_form_levels").values

        // Row 1 is D = 0.2, the case grain.md section 9.3 names: true skew 0.209, a normal
        // approximation returns about 0.046.
        let row = 1
        let out = Grain.apply(
            flatDensity(levels[row], side: statisticalSide), pixelSizeMicrons: pixelSize,
            params: quietParams(), densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive, seed: 0x9E57,
            spatial: NoSpatialFilter())
        let split = Grain.sublayerDensities(
            flatDensity(levels[row], side: 1), densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive)

        for c in 0..<3 {
            let closedForm = derived.closedFormMoments(
                sublayerDensities: Array(split[c].values[0..<3]), channel: c)
            let measured = SampleMoments(channel(out, c)).skewness
            let sd = samplingSD[(row * 3 + c) * 3 + 2]
            expectStatistic(measured, closedForm: closedForm.skewness, samplingSD: sd, "skewness")
            #expect(
                measured / sd >= 20.0,
                Comment(
                    rawValue: "channel \(c): skewness \(measured) is only \(measured / sd) sampling "
                        + "sigma from a Gaussian's 0, so this gate would not catch one"))
        }
    }

    @Test("the single-layer path's moments match the closed form, at one and three repeats")
    func singleLayerMoments() throws {
        let stock = try GrainStock()
        let levels = try Golden("grain_single_layer_levels").values

        for repeats in [1, 3] {
            let params = quietParams {
                $0.sublayersActive = false; $0.subLayerCount = repeats
            }
            let derived = stock.single(params, pixelSizeMicrons: pixelSize)
            let samplingSD = try Golden(
                "grain_single_moment_sd_sub\(repeats)_kodak_portra_400"
            ).values
            let oracle = try Golden(
                "grain_oracle_single_moments_sub\(repeats)_kodak_portra_400"
            ).values

            for (row, density) in levels.enumerated() {
                let out = Grain.apply(
                    flatDensity(density, side: statisticalSide), pixelSizeMicrons: pixelSize,
                    params: params, densityCurves: stock.curves,
                    densityCurvesLayers: stock.layers, positive: stock.positive,
                    seed: 0x5117 &+ UInt64(row), spatial: NoSpatialFilter())

                for c in 0..<3 {
                    let closedForm = derived.closedFormMoments(density: density, channel: c)
                    let measured = SampleMoments(channel(out, c))
                    let base = (row * 3 + c) * 3
                    let label = "single sub\(repeats) D=\(density) channel \(c)"
                    let targets = [
                        closedForm.mean, closedForm.standardDeviation, closedForm.skewness,
                    ]
                    let values = [measured.mean, measured.standardDeviation, measured.skewness]
                    let names = ["mean", "sd", "skewness"]
                    for k in 0..<3 {
                        expectStatistic(
                            values[k], closedForm: targets[k], samplingSD: samplingSD[base + k],
                            "\(label) \(names[k])")
                        expectStatistic(
                            oracle[base + k], closedForm: targets[k],
                            samplingSD: samplingSD[base + k], "oracle \(label) \(names[k])")
                    }
                }
            }
        }
    }

    /// RMS granularity through a 48 um densitometer aperture, the figure a photographer recognises.
    /// Also the only test that drives `pixel_size_um` five times larger than any real render.
    ///
    /// 128x128 samples, so `SE(sd)/sd` is 0.55 percent against the oracle's 0.14 percent at 512x512.
    /// The 5 percent gate is about nine combined sigma.
    @Test("RMS granularity lands within 5 percent of the oracle")
    func rmsGranularity() throws {
        let stock = try GrainStock()
        let levels = try Golden("grain_closed_form_levels").values

        for (label, params, crop, filter) in [
            ("quiet", quietParams(), 0, AnySpatialFilter(NoSpatialFilter())),
            ("default", GrainParams(), 8, AnySpatialFilter(FIRGaussianFilter())),
        ] {
            let golden = try Golden("grain_rms_granularity_\(label)_kodak_portra_400").values
            for (row, density) in levels.enumerated() {
                let out = Grain.apply(
                    flatDensity(density, side: 128), pixelSizeMicrons: densitometerPixelSize,
                    params: params, densityCurves: stock.curves,
                    densityCurvesLayers: stock.layers, positive: stock.positive,
                    seed: 0x624D &+ UInt64(row), spatial: filter)
                for c in 0..<3 {
                    var interior: [Double] = []
                    for y in crop..<(out.height - crop) {
                        for x in crop..<(out.width - crop) { interior.append(out[y, x, c]) }
                    }
                    let measured = SampleMoments(interior).standardDeviation * 1000
                    let expected = golden[row * 3 + c]
                    #expect(
                        abs(measured / expected - 1.0) <= 0.05,
                        Comment(
                            rawValue: "\(label) RMS granularity D=\(density) channel \(c): "
                                + "\(measured) vs oracle \(expected)"))
                }
            }
        }
    }
}

// MARK: - Blur wiring

/// The three blur sites, their gates and what they do to the variance. The filter itself belongs to
/// the diffusion subsystem; these check that grain drives it with the right sigma at the right time.
@Suite("Grain blur wiring")
struct GrainBlurTests {

    private static let side = 64

    private func render(
        _ params: GrainParams, filter: some SpatialFilter, density: Double = 1.0
    )
        throws -> ImageBuffer
    {
        let stock = try GrainStock()
        return Grain.apply(
            flatDensity(density, side: Self.side), pixelSizeMicrons: pixelSize, params: params,
            densityCurves: stock.curves, densityCurvesLayers: stock.layers,
            positive: stock.positive, seed: 11, spatial: filter)
    }

    /// `apply_grain_to_density` gates the final blur on `> 0.4` and
    /// `apply_grain_to_density_layers` on `> 0`. At sigma 0.3 the single-layer path skips the filter
    /// and the layered path runs it with radius 1.
    @Test("the two topologies gate the final blur differently")
    func blurGateAsymmetry() throws {
        let filter = FIRGaussianFilter()
        /// True when the requested blur actually reached the filter and changed something.
        func blurred(_ params: GrainParams) throws -> Bool {
            let filtered = try render(params, filter: filter).values
            let unfiltered = try render(params, filter: NoSpatialFilter()).values
            return filtered != unfiltered
        }

        var single = quietParams()
        single.sublayersActive = false
        single.blur = 0.3
        #expect(
            try !blurred(single),
            "the single-layer path blurred at sigma 0.3, which its > 0.4 gate excludes")
        single.blur = 0.5
        #expect(try blurred(single), "the single-layer path skipped the blur at sigma 0.5")

        var layered = quietParams()
        layered.blur = 0.3
        #expect(
            try blurred(layered),
            "the layered path skipped the blur at sigma 0.3, which its > 0 gate admits")

        // Its gate admits 0.1 too, and the filter turns that into an exact identity because the
        // radius rounds to 0. Both halves of the asymmetry matter.
        layered.blur = 0.1
        #expect(FIRGaussianFilter.kernel(sigma: 0.1) == [1.0])
        #expect(try !blurred(layered))
    }

    /// `radius = int(3 * sigma + 0.5)` puts the identity boundary just *below* sigma = 1/6, not at
    /// it: `3 * (1.0/6.0) + 0.5` rounds to exactly 1.0 in float64, so 1/6 itself gives radius 1.
    /// `grain.md` section 6.1 says "the radius-0 boundary is sigma = 1/6"; verified in the oracle,
    /// 1/6 is the first sigma that is not an identity.
    @Test("the kernel's radius-0 boundary sits just below one sixth")
    func radiusZeroBoundary() {
        for sigma in [0.05, 0.13, 0.1666, 0.16666] {
            #expect(FIRGaussianFilter.kernel(sigma: sigma) == [1.0], "sigma \(sigma)")
        }
        for sigma in [1.0 / 6.0, 0.166667, 0.167] {
            #expect(FIRGaussianFilter.kernel(sigma: sigma).count == 3, "sigma \(sigma)")
        }
        #expect(FIRGaussianFilter.kernel(sigma: 0.65).count == 5)
        #expect(FIRGaussianFilter.kernel(sigma: 1.0).count == 7)
    }

    /// Grain is white noise before the final blur, so the standard deviation drops by exactly the
    /// kernel's `sum(k^2)` per axis. Boundary pixels have inflated variance under reflect, hence the
    /// interior crop.
    @Test("the final blur retains the variance the kernel predicts")
    func blurVarianceRetention() throws {
        var params = quietParams()
        let unblurred = try render(params, filter: NoSpatialFilter())
        params.blur = 0.65
        let blurred = try render(params, filter: FIRGaussianFilter())

        let expected = FIRGaussianFilter.varianceRetention(sigma: 0.65).squareRoot()
        #expect(abs(expected - 0.447004481618) < 1e-9, "retention factor \(expected)")

        let crop = 8
        for c in 0..<3 {
            var before: [Double] = []
            var after: [Double] = []
            for y in crop..<(Self.side - crop) {
                for x in crop..<(Self.side - crop) {
                    before.append(unblurred[y, x, c])
                    after.append(blurred[y, x, c])
                }
            }
            let ratio =
                SampleMoments(after).standardDeviation / SampleMoments(before).standardDeviation
            // SE(sd)/sd is 1 percent at (64 - 16)^2 samples, twice over, so 4 percent is 3 sigma.
            #expect(
                abs(ratio / expected - 1.0) <= 0.04,
                "channel \(c): std ratio \(ratio), kernel predicts \(expected)")
        }
    }

    /// `blur_dye_clouds_um` is dimensionless: it multiplies `sqrt(od_particle)` to give a sigma in
    /// pixels. The physical sigma is therefore resolution invariant and the pixel sigma is not, so
    /// which sublayers the stage touches depends on output resolution.
    ///
    /// `grain.md` section 4.3 says the stage is a no-op at 8.75 um per pixel and that sublayer 0
    /// first gets radius 1 at 5.83 um. Its own sigma table contradicts that: sublayer 0, blue is
    /// 0.1915 px at 8.75 um, above the radius-0 boundary, so eight of the nine planes are exact
    /// identities and that one is not. Its kernel is `[1.19e-6, 0.9999976, 1.19e-6]`, three orders
    /// below the parity gate but not zero.
    @Test("the dye-cloud blur touches one plane at 8.75 um and more at 5.83 um")
    func dyeCloudResolutionDependence() throws {
        let stock = try GrainStock()
        var params = quietParams()
        params.blurDyeCloudsMicrons = 1.0

        let coarse = stock.layered(params, pixelSizeMicrons: pixelSize)
        let identities = coarse.dyeCloudSigmaPixels.filter {
            FIRGaussianFilter.kernel(sigma: $0) == [1.0]
        }
        #expect(identities.count == 8, "\(identities.count) of 9 planes are identities at 8.75 um")
        #expect(FIRGaussianFilter.kernel(sigma: coarse.dyeCloudSigmaPixels[2]).count == 3)

        let filtered = try render(params, filter: FIRGaussianFilter()).values
        let unfiltered = try render(params, filter: NoSpatialFilter()).values
        let worst = zip(filtered, unfiltered).map { abs($0 - $1) }.max() ?? 0
        #expect(worst > 0, "not one plane was blurred at 8.75 um per pixel")
        #expect(worst < 1e-6, "the dye-cloud blur moved a pixel by \(worst) at 8.75 um")

        // 35 mm at 6000 px wide, where three more planes cross the boundary.
        let fine = stock.layered(params, pixelSizeMicrons: 35_000.0 / 6000.0)
        let fineIdentities = fine.dyeCloudSigmaPixels.filter {
            FIRGaussianFilter.kernel(sigma: $0) == [1.0]
        }
        #expect(fineIdentities.count < identities.count)

        // The physical sigma does not move with resolution, which is the property that makes the
        // pixel threshold resolution dependent in the first place.
        for i in 0..<9 {
            let coarseMicrons = coarse.dyeCloudSigmaPixels[i] * pixelSize
            let fineMicrons = fine.dyeCloudSigmaPixels[i] * 35_000.0 / 6000.0
            #expect(abs(coarseMicrons / fineMicrons - 1.0) <= 1e-12)
        }
    }
}

// MARK: - Micro-structure

@Suite("Grain micro-structure")
struct GrainMicroStructureTests {

    /// Neither gate opens at any realistic pixel pitch, so with default parameters the stage is dead
    /// code in every production render.
    @Test("the default parameters leave the grain untouched")
    func defaultsNeverFire() throws {
        let stock = try GrainStock()
        let input = flatDensity(1.0, side: 32)
        func render(_ micro: (Double, Double)) -> ImageBuffer {
            var params = quietParams()
            params.microStructure = micro
            return Grain.apply(
                input, pixelSizeMicrons: pixelSize, params: params, densityCurves: stock.curves,
                densityCurvesLayers: stock.layers, positive: stock.positive,
                spatial: NoSpatialFilter())
        }
        #expect(render((0.2, 30)).values == render((0, 0)).values)
    }

    /// The clumping field is unit-mean by construction, so the stage preserves the mean before the
    /// blur and after it.
    @Test("the clumping field has the mean and the standard deviation it was asked for")
    func clumpingMoments() throws {
        let golden = try Golden("grain_clumping_moments")
        let side = 512
        let samples = Double(side * side)

        for row in 0..<golden.shape[0] {
            let sigma = golden.values[row * 3]
            let oracleMean = golden.values[row * 3 + 1]
            let oracleSD = golden.values[row * 3 + 2]

            // `micro_structure[1] * 0.001 / pixel_size_um` has to come out as `sigma`.
            var params = quietParams()
            params.microStructure = (0.0, sigma * 1000.0)
            let ones = ImageBuffer(height: side, width: side, channels: 3, repeating: 1.0)
            let clumping = Grain.addMicroStructure(
                ones, microStructure: params.microStructure, pixelSizeMicrons: 1.0, seed: 5,
                spatial: NoSpatialFilter())

            for c in 0..<3 {
                let measured = SampleMoments(channel(clumping, c))
                // A lognormal's sample mean has SE = sigma / sqrt(n); its sample sd has a heavier
                // tail, so allow the lognormal factor sqrt(exp(sigma^2) + 1) on the sd's SE.
                let meanSE = sigma / samples.squareRoot()
                let sdSE =
                    sigma * (Foundation.exp(sigma * sigma) + 1.0).squareRoot()
                    / (2.0 * samples).squareRoot()
                expectStatistic(
                    measured.mean, closedForm: 1.0, samplingSD: meanSE,
                    "clumping sigma \(sigma) channel \(c) mean")
                expectStatistic(
                    measured.standardDeviation, closedForm: sigma, samplingSD: sdSE,
                    "clumping sigma \(sigma) channel \(c) sd")
            }
            // The oracle's own realisation, at 1024x1024, has to clear the same closed form.
            expectStatistic(
                oracleMean, closedForm: 1.0, samplingSD: sigma / 1024.0,
                "oracle clumping sigma \(sigma) mean")
            #expect(abs(oracleSD / sigma - 1.0) <= 0.01, "oracle clumping sd \(oracleSD)")
        }
    }

    /// Forced on with the fixture from `grain.md` section 9.6: `pixel_size_um = 0.3` and
    /// `micro_structure = (0.2, 300)` open both gates.
    @Test("forced on, the stage preserves the mean and adds the variance it should")
    func forcedOn() throws {
        let stock = try GrainStock()
        let side = 128
        var params = quietParams()
        params.microStructure = (0.2, 300)
        let pitch = 0.3
        let derived = stock.layered(params, pixelSizeMicrons: pitch)
        #expect(abs(derived.microStructureSigma - 1.0) < 1e-12)
        #expect(abs(derived.microStructureBlurPixels - 2.0 / 3.0) < 1e-12)

        let input = flatDensity(1.0, side: side)
        let filter = FIRGaussianFilter()
        let withClumping = Grain.apply(
            input, pixelSizeMicrons: pitch, params: params, densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive, seed: 17,
            spatial: filter)
        var quiet = params
        quiet.microStructure = (0, 0)
        let without = Grain.apply(
            input, pixelSizeMicrons: pitch, params: quiet, densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive, seed: 17,
            spatial: filter)
        #expect(withClumping.values != without.values, "the clumping stage did not fire")

        // Blurred clumping has mean 1 and variance sigma^2 * (sum k^2)^2, and it multiplies a field
        // whose own moments the closed form gives. Var(XY) = E[X]^2 Var(Y) + E[Y]^2 Var(X) +
        // Var(X) Var(Y) for independent X and Y, with the grain shifted back up by density_min
        // because the subtraction happens after the multiply.
        let clumpingVariance =
            derived.microStructureSigma * derived.microStructureSigma
            * FIRGaussianFilter.varianceRetention(sigma: derived.microStructureBlurPixels)
        let split = Grain.sublayerDensities(
            flatDensity(1.0, side: 1), densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive)

        let crop = 4
        for c in 0..<3 {
            var interior: [Double] = []
            for y in crop..<(side - crop) {
                for x in crop..<(side - crop) { interior.append(withClumping[y, x, c]) }
            }
            let grain = derived.closedFormMoments(
                sublayerDensities: Array(split[c].values[0..<3]), channel: c)
            let beforeSubtraction = grain.mean + derived.densityMin[c]
            let expectedMean = beforeSubtraction - derived.densityMin[c]
            let expectedVariance =
                beforeSubtraction * beforeSubtraction * clumpingVariance
                + grain.variance * (1.0 + clumpingVariance)

            let measured = SampleMoments(interior)
            let n = Double(interior.count)
            expectStatistic(
                measured.mean, closedForm: expectedMean,
                samplingSD: expectedVariance.squareRoot() / n.squareRoot(),
                "forced clumping channel \(c) mean")
            // The blurred field is spatially correlated, so the sd estimator's spread is wider than
            // the i.i.d. 1/sqrt(2n). Three times that, which still rejects a missing or
            // double-counted clumping term by a wide margin.
            let sdSE = 3.0 * expectedVariance.squareRoot() / (2.0 * n).squareRoot()
            expectStatistic(
                measured.standardDeviation, closedForm: expectedVariance.squareRoot(),
                samplingSD: sdSE, "forced clumping channel \(c) sd")
        }
    }
}

// MARK: - Glare

/// `model/glare.py`: a unit-mean lognormal flare field, blurred, divided by 100, added as a multiple
/// of the viewing illuminant.
@Suite("Glare")
struct GlareTests {

    private static let illuminant = [0.96422, 1.0, 0.82521]

    @Test("the flare field's moments match what was asked for, and the oracle's")
    func fieldMoments() throws {
        let params = try Golden("glare_moments_params")
        let oracle = try Golden("glare_moments").values
        let samplingSD = try Golden("glare_moment_sd").values
        let side = statisticalSide

        for row in 0..<params.shape[0] {
            let percent = params.values[row * 2]
            let roughness = params.values[row * 2 + 1]
            let field = Glare.randomAmount(
                amount: percent, roughness: roughness, blur: 0, height: side, width: side,
                seed: 0xF1A4 &+ UInt64(row), spatial: NoSpatialFilter())
            #expect(field.channels == 1)
            #expect(field.values.count == statisticalSamples)

            // The division by 100 is what turns `percent` into a fraction.
            let expectedMean = percent / 100
            let expectedSD = roughness * percent / 100
            let measured = SampleMoments(field.values)
            let label = "glare percent \(percent) roughness \(roughness)"

            if expectedSD == 0 {
                // sigma_log falls under 1e-6, so the reference returns exp(mu) and draws nothing.
                #expect(field.values.allSatisfy { $0 == expectedMean })
                #expect(oracle[row * 2] == expectedMean)
                continue
            }
            expectStatistic(
                measured.mean, closedForm: expectedMean, samplingSD: samplingSD[row * 2],
                "\(label) mean")
            expectStatistic(
                measured.standardDeviation, closedForm: expectedSD,
                samplingSD: samplingSD[row * 2 + 1], "\(label) sd")
            expectStatistic(
                oracle[row * 2], closedForm: expectedMean, samplingSD: samplingSD[row * 2],
                "oracle \(label) mean")
            expectStatistic(
                oracle[row * 2 + 1], closedForm: expectedSD,
                samplingSD: samplingSD[row * 2 + 1], "oracle \(label) sd")
        }
    }

    /// `blur` is a pixel sigma, so at the shipped 0.5 it takes the FIR path with radius 2. The mean
    /// survives and the standard deviation drops by the kernel's factor.
    @Test("the blur leaves the mean alone and scales the standard deviation")
    func blurMovesOnlyTheSpread() throws {
        let side = 256
        let crop = 8
        let unblurred = Glare.randomAmount(
            amount: 0.03, roughness: 0.7, blur: 0, height: side, width: side, seed: 9,
            spatial: NoSpatialFilter())
        let blurred = Glare.randomAmount(
            amount: 0.03, roughness: 0.7, blur: 0.5, height: side, width: side, seed: 9,
            spatial: FIRGaussianFilter())

        var before: [Double] = []
        var after: [Double] = []
        for y in crop..<(side - crop) {
            for x in crop..<(side - crop) {
                before.append(unblurred[y, x, 0])
                after.append(blurred[y, x, 0])
            }
        }
        let expected = FIRGaussianFilter.varianceRetention(sigma: 0.5).squareRoot()
        let ratio = SampleMoments(after).standardDeviation / SampleMoments(before).standardDeviation
        #expect(abs(ratio / expected - 1.0) <= 0.04, "std ratio \(ratio), kernel predicts \(expected)")

        let meanSE = 0.7 * 0.03 / 100 / Double(before.count).squareRoot()
        #expect(abs(SampleMoments(after).mean - 0.0003) <= 5 * meanSE)
    }

    @Test("the three off switches return the input untouched")
    func offSwitches() throws {
        let xyz = ImageBuffer(height: 8, width: 9, channels: 3, repeating: 0.4)
        var inactive = GlareParams()
        inactive.active = false
        var zero = GlareParams()
        zero.percent = 0
        for (label, params) in [
            ("nil, the scan-film branch", GlareParams?.none),
            ("active = false, what deactivateStochasticEffects sets", inactive),
            ("percent = 0", zero),
        ] {
            let out = Glare.add(
                xyz, illuminantXYZ: Self.illuminant, glare: params, spatial: NoSpatialFilter())
            #expect(out.values == xyz.values, "\(label) changed the image")
        }
    }

    @Test("the added term is the field times the illuminant")
    func additiveTerm() throws {
        let side = 16
        let xyz = ImageBuffer(height: side, width: side, channels: 3, repeating: 0.4)
        let params = GlareParams()
        let out = Glare.add(
            xyz, illuminantXYZ: Self.illuminant, glare: params, seed: 21,
            spatial: NoSpatialFilter())
        let field = Glare.randomAmount(
            amount: params.percent, roughness: params.roughness, blur: params.blur,
            height: side, width: side, seed: 21, spatial: NoSpatialFilter())

        for pixel in 0..<xyz.pixelCount {
            for c in 0..<3 {
                #expect(out.values[pixel * 3 + c] == 0.4 + field.values[pixel] * Self.illuminant[c])
            }
        }
        // The shipped defaults add 3e-4 times the illuminant on average.
        let mean = SampleMoments(field.values).mean
        #expect(abs(mean - 3e-4) <= 5 * 0.7 * 3e-4 / Double(side * side).squareRoot())
    }

    @Test("the same seed repeats exactly and a different seed does not")
    func seedSelectsTheRealisation() {
        func field(seed: UInt64) -> [Double] {
            Glare.randomAmount(
                amount: 0.03, roughness: 0.7, blur: 0, height: 16, width: 16, seed: seed,
                spatial: NoSpatialFilter()
            ).values
        }
        #expect(field(seed: 2) == field(seed: 2))
        #expect(field(seed: 2) != field(seed: 3))
    }
}

// MARK: - Filter erasure

/// Lets one test table hold two different ``SpatialFilter`` implementations.
private struct AnySpatialFilter: SpatialFilter {
    private let blur: @Sendable (ImageBuffer, Double) -> ImageBuffer
    private let decay: @Sendable (ImageBuffer, Double) -> ImageBuffer

    init(_ filter: some SpatialFilter) {
        blur = { filter.gaussian($0, sigma: $1) }
        decay = { filter.exponential($0, decay: $1) }
    }

    func gaussian(_ image: ImageBuffer, sigma: Double) -> ImageBuffer { blur(image, sigma) }
    func exponential(_ image: ImageBuffer, decay: Double) -> ImageBuffer { self.decay(image, decay) }
}

// MARK: - ZZ REVIEW PERF PROBE (temporary)

@Suite("ZZ perf probe")
struct ZZPerfProbe {
    @Test("perf")
    func perf() throws {
        let stock = try GrainStock()
        let params = GrainParams()
        for side in [1000] {
            let input = flatDensity(0.8, side: side)
            let t0 = Date()
            let out = Grain.apply(
                input, pixelSizeMicrons: 8.75, params: params, densityCurves: stock.curves,
                densityCurvesLayers: stock.layers, positive: stock.positive, seed: 1,
                spatial: FastSpatialFilter())
            let dt = Date().timeIntervalSince(t0)
            print(
                "PERF layered \(side)x\(side) = \(dt) s, \(dt / Double(side * side) * 1e9) ns/pixel,"
                    + " check \(out.values[0])")
        }
        var single = GrainParams()
        single.sublayersActive = false
        let input = flatDensity(0.8, side: 1000)
        let t1 = Date()
        _ = Grain.apply(
            input, pixelSizeMicrons: 8.75, params: single, densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive, seed: 1,
            spatial: FastSpatialFilter())
        let dt1 = Date().timeIntervalSince(t1)
        print("PERF single 1000x1000 = \(dt1) s, \(dt1 / 1e6 * 1e9) ns/pixel")
    }
}

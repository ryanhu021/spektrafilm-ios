#if canImport(Metal)
import Foundation
import Metal
import Testing

@testable import SpektraFilm

// MARK: - Support

/// Biased (divide by n) sample moments, as the CPU grain and random tests measure them.
private struct GPUSampleMoments {
    let mean: Double
    let variance: Double
    let skewness: Double
    let excessKurtosis: Double

    var standardDeviation: Double { variance.squareRoot() }

    /// `shift` is subtracted before the sums, which keeps float64 exact for counts near 1e8.
    init(_ samples: [Double], shift: Double = 0) {
        let n = Double(samples.count)
        var sum = 0.0
        for x in samples { sum += x - shift }
        let centre = sum / n
        var m2 = 0.0
        var m3 = 0.0
        var m4 = 0.0
        for x in samples {
            let d = x - shift - centre
            let d2 = d * d
            m2 += d2
            m3 += d2 * d
            m4 += d2 * d2
        }
        m2 /= n
        m3 /= n
        m4 /= n
        mean = centre + shift
        variance = m2
        skewness = m2 > 0 ? m3 / (m2 * m2.squareRoot()) : 0
        excessKurtosis = m2 > 0 ? m4 / (m2 * m2) - 3.0 : 0
    }
}

/// Five sampling sigma, the gate every statistical test of the CPU sampler uses.
private func expectWithinFiveSigma(
    _ measured: Double, closedForm: Double, samplingSD: Double, _ label: String,
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

/// Standard errors of the sample mean, variance, skewness and excess kurtosis of `n` Poisson
/// draws, by the delta method.
///
/// Each statistic's influence function is a polynomial in `y = x - lambda`, so its variance is a
/// sum of central moments. A Poisson's cumulants are all `lambda`, which gives the central moments
/// up to the eighth in closed form.
private func poissonStandardErrors(lambda l: Double, n: Int) -> [Double] {
    let mu: [Double] = [
        1, 0, l, l, l + 3 * l * l, l + 10 * l * l, l + 25 * l * l + 15 * l * l * l,
        l + 56 * l * l + 105 * l * l * l,
        l + 119 * l * l + 490 * l * l * l + 105 * l * l * l * l,
    ]
    func variance(_ c: [Double]) -> Double {
        var sum = 0.0
        for j in c.indices {
            for k in c.indices { sum += c[j] * c[k] * mu[j + k] }
        }
        return sum / Double(n)
    }
    let m2 = mu[2]
    let m3 = mu[3]
    let m4 = mu[4]
    // Influence functions of m2, m3 and m4, coefficients of y^0 ... y^4.
    let if2: [Double] = [-m2, 0, 1, 0, 0]
    let if3: [Double] = [-m3, -3 * m2, 0, 1, 0]
    let if4: [Double] = [-m4, -4 * m3, 0, 0, 1]
    let skew = (0..<5).map { if3[$0] / pow(m2, 1.5) - 1.5 * m3 / pow(m2, 2.5) * if2[$0] }
    let kurt = (0..<5).map { if4[$0] / (m2 * m2) - 2 * m4 / pow(m2, 3) * if2[$0] }
    return [variance([0, 1]), variance(if2), variance(skew), variance(kurt)].map {
        $0.squareRoot()
    }
}

/// Portra 400, the stock every CPU grain golden is generated against.
private struct Stock {
    let profile: Profile
    let curves: [Double]
    let layers: [Double]

    init(_ name: String = "kodak_portra_400") throws {
        profile = try ProfileLibrary.load(name)
        curves = DensityCurves.normalized(
            curves: profile.data.densityCurves, minima: profile.data.densityCurveMinima)
        layers = profile.data.densityCurvesLayers
    }

    var positive: Bool { profile.isPositive }

    func layered(_ params: GrainParams, pixelSizeMicrons: Double) -> Grain.LayeredParameters {
        Grain.LayeredParameters(
            params: params, pixelSizeMicrons: pixelSizeMicrons,
            densityMaxLayers: nanMax(layers, channels: 9))
    }

    func split(_ density: Double) -> [[Double]] {
        let split = Grain.sublayerDensities(
            ImageBuffer(height: 1, width: 1, channels: 3, repeating: density),
            densityCurves: curves, densityCurvesLayers: layers, positive: positive)
        return (0..<3).map { Array(split[$0].values[0..<3]) }
    }
}

/// Defaults with every spatial and clumping stage off, as the CPU statistics tests use.
private func quietParams(_ edit: (inout GrainParams) -> Void = { _ in }) -> GrainParams {
    var params = GrainParams()
    params.blur = 0
    params.blurDyeCloudsMicrons = 0
    params.microStructure = (0, 0)
    edit(&params)
    return params
}

private let pixelSize = 8.75
private let side = 512

private func channel(_ image: ImageBuffer, _ c: Int) -> [Double] {
    (0..<image.pixelCount).map { image.values[$0 * image.channels + c] }
}

private func gpuGrain(
    _ c: MetalContext, _ input: ImageBuffer, stock: Stock, params: GrainParams,
    pixelSizeMicrons: Double = pixelSize, seed: UInt64
) throws -> ImageBuffer {
    let frame = try GPUFrame(c, uploading: input)
    return try MetalGrain.apply(
        c, frame, pixelSizeMicrons: pixelSizeMicrons, params: params,
        densityCurves: stock.curves, densityCurvesLayers: stock.layers,
        positive: stock.positive, seed: seed
    ).download()
}

/// The input as the GPU sees it, rounded to float32, so the CPU reference starts from the same
/// numbers.
private func roundedToFloat(_ image: ImageBuffer) -> ImageBuffer {
    ImageBuffer(
        height: image.height, width: image.width, channels: image.channels,
        values: image.values.map { Double(Float($0)) })
}

// MARK: - Tests

/// The Metal grain against ``Grain``: the Poisson sampler and the whole operator against the closed
/// form, the deterministic stages against the CPU at float32 precision, and a real frame against
/// the CPU's mean at the same seed.
@Suite("Metal grain", .enabled(if: MetalContext.shared != nil))
struct MetalGrainTests {

    private static let poissonLambdas = [4.5, 9.9, 10, 20, 200, 1e4, 1e6, 1.2e8]
    private static let poissonSamples = 1 << 20

    private func poissonDraws(
        _ c: MetalContext, lambda: Double, seed: UInt64, count: Int = poissonSamples
    ) throws -> [Double] {
        let out = try #require(
            c.device.makeBuffer(length: count * 8, options: .storageModeShared))
        var l = MetalGrain.split(lambda)
        var s = seed
        var n = UInt32(count)
        try c.dispatch("grain_poisson_samples", count: count) { e in
            e.setBuffer(out, offset: 0, index: 0)
            e.setBytes(&l, length: 8, index: 1)
            e.setBytes(&s, length: 8, index: 2)
            e.setBytes(&n, length: 4, index: 3)
        }
        let p = out.contents().bindMemory(to: Int64.self, capacity: count)
        return (0..<count).map { Double(p[$0]) }
    }

    @Test("the Poisson sampler's four moments match the closed form from 4.5 to 1.2e8")
    func poissonMoments() throws {
        let c = try #require(MetalContext.shared)
        for (row, lambda) in Self.poissonLambdas.enumerated() {
            let draws = try poissonDraws(c, lambda: lambda, seed: 0x3100_0000 + UInt64(row))
            let m = GPUSampleMoments(draws, shift: lambda.rounded())
            let se = poissonStandardErrors(lambda: lambda, n: draws.count)
            let closedForm = [lambda, lambda, 1 / lambda.squareRoot(), 1 / lambda]
            let measured = [m.mean, m.variance, m.skewness, m.excessKurtosis]
            let names = ["mean", "variance", "skewness", "excess kurtosis"]
            for k in 0..<4 {
                expectWithinFiveSigma(
                    measured[k], closedForm: closedForm[k], samplingSD: se[k],
                    "Poisson(\(lambda)) \(names[k])")
            }
        }
    }

    /// The gate a Gaussian with the right mean and variance fails: at lambda 20 the skewness is
    /// 0.224, about 90 standard errors from a Gaussian's 0.
    @Test("the Poisson skewness at lambda 20 is far from a Gaussian's")
    func poissonSkewnessIsNotGaussian() throws {
        let c = try #require(MetalContext.shared)
        let lambda = 20.0
        let draws = try poissonDraws(c, lambda: lambda, seed: 0x4100_0000)
        let measured = GPUSampleMoments(draws).skewness
        let se = poissonStandardErrors(lambda: lambda, n: draws.count)[2]
        expectWithinFiveSigma(
            measured, closedForm: 1 / lambda.squareRoot(), samplingSD: se, "skewness")
        #expect(measured / se >= 20, "skewness \(measured) is only \(measured / se) SE from 0")
    }

    /// The GPU draws from the CPU's stream with a 24-bit truncation of its uniforms, so the two
    /// samplers take the same branch and return the same count almost everywhere. A misaligned
    /// stream would agree at about the rate two independent draws do, at most 14 percent at these
    /// lambdas.
    @Test("the Poisson draws equal the CPU's at nearly every counter")
    func poissonDrawsTrackTheCPU() throws {
        let c = try #require(MetalContext.shared)
        let count = 1 << 16
        for lambda in Self.poissonLambdas {
            let seed: UInt64 = 0x5100_0000
            let gpu = try poissonDraws(c, lambda: lambda, seed: seed, count: count)
            var source = Philox4x32(key: PhiloxKey(seed: seed))
            var equal = 0
            for i in 0..<count {
                source.reset(counter: UInt64(i))
                if Double(Distributions.poisson(lambda: lambda, &source)) == gpu[i] { equal += 1 }
            }
            let fraction = Double(equal) / Double(count)
            #expect(fraction >= 0.99, "Poisson(\(lambda)): only \(fraction) of the draws agree")
        }
    }

    // MARK: Deterministic stages

    /// The sublayer interpolation, lambda and the lattice step for all nine planes of a real
    /// negative, before any draw.
    @Test("the sublayer planes, lambda and the step match the CPU")
    func deterministicStages() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let input = roundedToFloat(try Golden("photo_portra400_endura_negative").imageBuffer())
        let derived = stock.layered(GrainParams(), pixelSizeMicrons: pixelSize)
        let tables = try MetalGrain.SublayerTables(
            c, densityCurves: stock.curves, densityCurvesLayers: stock.layers,
            positive: stock.positive)
        let frame = try GPUFrame(c, uploading: input)
        let out = try c.buffer(floats: input.pixelCount * 4)

        var worstPlane = 0.0
        var worstLambda = 0.0
        var worstLambdaFromPlane = 0.0
        var worstStep = 0.0
        for ch in 0..<3 {
            for sublayer in 0..<3 {
                let i = sublayer * 3 + ch
                var layer = MetalGrain.layer(
                    derived, tables: tables, positive: stock.positive, channel: ch,
                    sublayer: sublayer, pixels: input.pixelCount, seed: 0)
                try MetalGrain.encodeLayer(
                    c, "grain_layer_setup", frame, out, layer: &layer, tables: tables)
                let gpu = out.contents().bindMemory(to: Float.self, capacity: input.pixelCount * 4)
                let plane = Grain.sublayerPlane(
                    input, channel: ch, sublayer: sublayer, densityCurves: stock.curves,
                    densityCurvesLayers: stock.layers, positive: stock.positive)
                for p in 0..<input.pixelCount {
                    let gpuPlane = Double(gpu[p * 4])
                    let gpuLambda = Double(gpu[p * 4 + 1]) + Double(gpu[p * 4 + 2])
                    worstPlane = max(worstPlane, abs(gpuPlane - plane.values[p]))
                    func population(_ density: Double) -> Grain.ParticlePopulation {
                        Grain.ParticlePopulation(
                            density: density + derived.densityMinLayers[i],
                            densityMax: derived.densityMaxLayers[i],
                            particlesPerPixel: derived.particlesPerPixel[i],
                            uniformity: derived.uniformity[ch], weight: 1)
                    }
                    let cpu = population(plane.values[p])
                    let same = population(gpuPlane)
                    worstLambda = max(worstLambda, abs(gpuLambda / cpu.lambda - 1))
                    worstLambdaFromPlane = max(
                        worstLambdaFromPlane, abs(gpuLambda / same.lambda - 1))
                    worstStep = max(
                        worstStep, abs(Double(gpu[p * 4 + 3]) / same.odTimesSaturation - 1))
                }
            }
        }
        // Measured on an M4 Pro: 7.4e-8, 7.5e-6, 4.3e-13 and 5.9e-8. Lambda against the CPU's own
        // plane carries the plane's float32 rounding, which is relative to densities near 0.01.
        #expect(worstPlane < 5e-7, "sublayer plane max abs difference \(worstPlane)")
        #expect(worstLambda < 5e-5, "lambda relative difference \(worstLambda)")
        #expect(
            worstLambdaFromPlane < 1e-11,
            "lambda from the same plane, relative difference \(worstLambdaFromPlane)")
        #expect(worstStep < 1.2e-7, "step relative difference \(worstStep)")
    }

    /// At `uniformity = 1` and a saturated density, `sat` is 2e-6 and lambda is at the top of the
    /// model's range. float32 alone would lose about two digits of `sat` here.
    @Test("lambda keeps its precision where sat nearly cancels")
    func saturatedLambda() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let params = quietParams { $0.uniformity = (1, 1, 1) }
        let derived = stock.layered(params, pixelSizeMicrons: pixelSize)
        let tables = try MetalGrain.SublayerTables(
            c, densityCurves: stock.curves, densityCurvesLayers: stock.layers,
            positive: stock.positive)
        let input = ImageBuffer(
            height: 1, width: 4, channels: 3,
            values: [Double](
                repeating: 10, count: 12))
        let frame = try GPUFrame(c, uploading: input)
        let out = try c.buffer(floats: 16)
        for i in 0..<9 {
            let ch = i % 3
            let sublayer = i / 3
            var layer = MetalGrain.layer(
                derived, tables: tables, positive: stock.positive, channel: ch,
                sublayer: sublayer, pixels: 4, seed: 0)
            try MetalGrain.encodeLayer(
                c, "grain_layer_setup", frame, out, layer: &layer, tables: tables)
            let gpu = out.contents().bindMemory(to: Float.self, capacity: 16)
            let cpu = Grain.ParticlePopulation(
                density: Double(gpu[0]) + derived.densityMinLayers[i],
                densityMax: derived.densityMaxLayers[i],
                particlesPerPixel: derived.particlesPerPixel[i], uniformity: 1, weight: 1)
            #expect(cpu.saturation < 3e-6, "sat \(cpu.saturation)")
            let lambda = Double(gpu[1]) + Double(gpu[2])
            #expect(abs(lambda / cpu.lambda - 1) < 1e-9, "lambda \(lambda) vs \(cpu.lambda)")
        }
    }

    // MARK: The operator

    /// The same configuration, seeds of the same form and the same measured-sigma gate as the CPU's
    /// `GrainStatisticsTests.layeredMoments`.
    @Test("the layered operator's moments match the closed form at six density levels")
    func layeredMoments() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let params = quietParams()
        let derived = stock.layered(params, pixelSizeMicrons: pixelSize)
        let levels = try Golden("grain_closed_form_levels").values
        let samplingSD = try Golden("grain_layered_moment_sd_kodak_portra_400").values

        for (row, density) in levels.enumerated() {
            let out = try gpuGrain(
                c, ImageBuffer(height: side, width: side, channels: 3, repeating: density),
                stock: stock, params: params, seed: 0x7011 &+ UInt64(row))
            let split = stock.split(density)
            for ch in 0..<3 {
                let closedForm = derived.closedFormMoments(
                    sublayerDensities: split[ch], channel: ch)
                let m = GPUSampleMoments(channel(out, ch))
                let targets = [closedForm.mean, closedForm.standardDeviation, closedForm.skewness]
                let values = [m.mean, m.standardDeviation, m.skewness]
                let names = ["mean", "sd", "skewness"]
                for k in 0..<3 {
                    expectWithinFiveSigma(
                        values[k], closedForm: targets[k],
                        samplingSD: samplingSD[(row * 3 + ch) * 3 + k],
                        "layered D=\(density) channel \(ch) \(names[k])")
                }
            }
        }
    }

    @Test("the single-layer operator's moments match the closed form, at one and three repeats")
    func singleLayerMoments() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let levels = try Golden("grain_single_layer_levels").values
        for repeats in [1, 3] {
            let params = quietParams {
                $0.sublayersActive = false
                $0.subLayerCount = repeats
            }
            let derived = Grain.SingleLayerParameters(
                params: params, pixelSizeMicrons: pixelSize,
                densityMaxCurves: nanMax(stock.curves, channels: 3))
            let samplingSD = try Golden(
                "grain_single_moment_sd_sub\(repeats)_kodak_portra_400"
            ).values
            for (row, density) in levels.enumerated() {
                let out = try gpuGrain(
                    c, ImageBuffer(height: side, width: side, channels: 3, repeating: density),
                    stock: stock, params: params, seed: 0x6117 &+ UInt64(row))
                for ch in 0..<3 {
                    let closedForm = derived.closedFormMoments(density: density, channel: ch)
                    let m = GPUSampleMoments(channel(out, ch))
                    let targets = [
                        closedForm.mean, closedForm.standardDeviation, closedForm.skewness,
                    ]
                    let values = [m.mean, m.standardDeviation, m.skewness]
                    let names = ["mean", "sd", "skewness"]
                    for k in 0..<3 {
                        expectWithinFiveSigma(
                            values[k], closedForm: targets[k],
                            samplingSD: samplingSD[(row * 3 + ch) * 3 + k],
                            "single sub\(repeats) D=\(density) channel \(ch) \(names[k])")
                    }
                }
            }
        }
    }

    /// A real negative, each pixel its own density. The tolerance on the mean is five standard
    /// errors of the difference of two independent realisations, from the closed-form variance at
    /// each pixel. The two share a stream, so the real difference is far smaller.
    @Test("a real frame's grain has the CPU's mean at the same seed")
    func realFrameMean() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let params = quietParams()
        let derived = stock.layered(params, pixelSizeMicrons: pixelSize)
        let input = roundedToFloat(try Golden("photo_portra400_endura_negative").imageBuffer())
        let seed: UInt64 = 0xC0FFEE
        let gpu = try gpuGrain(c, input, stock: stock, params: params, seed: seed)
        let cpu = Grain.apply(
            input, pixelSizeMicrons: pixelSize, params: params, densityCurves: stock.curves,
            densityCurvesLayers: stock.layers, positive: stock.positive, seed: seed,
            spatial: NoSpatialFilter())
        let split = Grain.sublayerDensities(
            input, densityCurves: stock.curves, densityCurvesLayers: stock.layers,
            positive: stock.positive)

        let n = Double(input.pixelCount)
        for ch in 0..<3 {
            var variance = 0.0
            for p in 0..<input.pixelCount {
                let sublayers = (0..<3).map { split[ch].values[p * 3 + $0] }
                variance +=
                    derived.closedFormMoments(
                        sublayerDensities: sublayers, channel: ch
                    ).variance
            }
            let se = (2 * variance).squareRoot() / n
            let gpuMean = channel(gpu, ch).reduce(0, +) / n
            let cpuMean = channel(cpu, ch).reduce(0, +) / n
            #expect(
                abs(gpuMean - cpuMean) <= 5 * se,
                "channel \(ch): GPU mean \(gpuMean), CPU \(cpuMean), SE \(se)")
        }

        var agree = 0
        for (a, b) in zip(gpu.values, cpu.values) where abs(a - b) <= 1e-6 * max(1, abs(b)) {
            agree += 1
        }
        let fraction = Double(agree) / Double(gpu.count)
        #expect(fraction >= 0.99, "only \(fraction) of the values agree with the CPU's")
    }

    /// Defaults, so the dye-cloud blur and the closing blur run, and a forced clumping field. The
    /// GPU and CPU share streams, so the frames agree value for value to float32 blur precision
    /// except where a draw differs.
    @Test(
        "the blurs and the clumping field track the CPU pixel for pixel",
        arguments: [(8.75, (0.2, 30.0)), (1.0, (0.5, 300.0))])
    func spatialStagesTrackTheCPU(pixelSizeMicrons: Double, micro: (Double, Double)) throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        var params = GrainParams()
        params.microStructure = micro
        let input = roundedToFloat(try Golden("photo_portra400_endura_negative").imageBuffer())
        let seed: UInt64 = 0xBEEF
        let gpu = try gpuGrain(
            c, input, stock: stock, params: params, pixelSizeMicrons: pixelSizeMicrons, seed: seed)
        let cpu = Grain.apply(
            input, pixelSizeMicrons: pixelSizeMicrons, params: params,
            densityCurves: stock.curves, densityCurvesLayers: stock.layers,
            positive: stock.positive, seed: seed, spatial: FastSpatialFilter())

        var agree = 0
        for (a, b) in zip(gpu.values, cpu.values) where abs(a - b) <= 2e-5 * max(1, abs(b)) {
            agree += 1
        }
        let fraction = Double(agree) / Double(gpu.count)
        #expect(fraction >= 0.98, "only \(fraction) of the values agree with the CPU's")
        let n = Double(input.pixelCount)
        for ch in 0..<3 {
            let gpuMean = channel(gpu, ch).reduce(0, +) / n
            let cpuMean = channel(cpu, ch).reduce(0, +) / n
            #expect(abs(gpuMean - cpuMean) < 1e-4, "channel \(ch): \(gpuMean) vs \(cpuMean)")
        }
    }

    /// The counter is the linear pixel index, so reshaping the frame leaves every value alone.
    @Test("the result depends only on the pixel index")
    func independentOfShape() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let input = try Golden("photo_portra400_endura_negative").imageBuffer()
        let flat = ImageBuffer(
            height: 1, width: input.pixelCount, channels: 3, values: input.values)
        let a = try gpuGrain(c, input, stock: stock, params: quietParams(), seed: 9)
        let b = try gpuGrain(c, flat, stock: stock, params: quietParams(), seed: 9)
        let again = try gpuGrain(c, input, stock: stock, params: quietParams(), seed: 9)
        #expect(a.values == b.values)
        #expect(a.values == again.values)
    }

    @Test("inactive or bypassed grain returns the input frame")
    func bypass() throws {
        let c = try #require(MetalContext.shared)
        let stock = try Stock()
        let frame = try GPUFrame(c, height: 2, width: 2, channels: 3)
        var off = GrainParams()
        off.active = false
        for (params, bypass) in [(off, false), (GrainParams(), true)] {
            let out = try MetalGrain.apply(
                c, frame, pixelSizeMicrons: pixelSize, params: params,
                densityCurves: stock.curves, densityCurvesLayers: stock.layers,
                positive: stock.positive, bypass: bypass)
            #expect(out === frame)
        }
    }
}
#endif

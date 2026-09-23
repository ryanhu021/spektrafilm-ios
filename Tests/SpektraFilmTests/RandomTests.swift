import Foundation
import Testing

@testable import SpektraFilm

// MARK: - Shared statistics

/// Sample mean, variance, skewness and excess kurtosis, using the same biased (divide by n)
/// estimators as `Tools/parity/fixtures/random.py`.
private struct Moments {
    let mean: Double
    let variance: Double
    let skewness: Double
    let excessKurtosis: Double

    init(_ samples: [Double]) {
        let n = Double(samples.count)
        var sum = 0.0
        for x in samples { sum += x }
        let mean = sum / n
        var m2 = 0.0
        var m3 = 0.0
        var m4 = 0.0
        for x in samples {
            let d = x - mean
            let d2 = d * d
            m2 += d2
            m3 += d2 * d
            m4 += d2 * d2
        }
        m2 /= n
        m3 /= n
        m4 /= n
        self.mean = mean
        variance = m2
        skewness = m2 > 0 ? m3 / (m2 * m2.squareRoot()) : 0
        excessKurtosis = m2 > 0 ? m4 / (m2 * m2) - 3.0 : 0
    }

    subscript(index: Int) -> Double {
        switch index {
        case 0: return mean
        case 1: return variance
        case 2: return skewness
        default: return excessKurtosis
        }
    }
}

private let momentNames = ["mean", "variance", "skewness", "excess kurtosis"]

/// Sample count every calibrated tolerance in this file assumes.
///
/// `random_*_moment_sd` holds the standard deviation of each statistic across 256 independent
/// oracle realisations of this many samples, so changing this number invalidates every gate below.
private let samples = 512 * 512

/// Gates a measured statistic against its closed form at five times the oracle's measured sampling
/// standard deviation.
///
/// Five sigma with an estimated sigma puts the false-failure rate for the whole file near 1e-4, so a
/// failure almost always means a real defect. The gate still catches a Gaussian approximation in
/// place of the Poisson sampler: at lambda 20 the correct skewness is 0.224 and a Gaussian returns
/// 0, a 44 sigma miss.
private func expectMoment(
    _ measured: Double,
    closedForm: Double,
    samplingSD: Double,
    _ label: String,
    sourceLocation: SourceLocation = #_sourceLocation
) {
    let gate = 5.0 * samplingSD
    let miss = abs(measured - closedForm)
    #expect(
        miss <= gate,
        Comment(
            rawValue: "\(label): \(measured) vs closed form \(closedForm), off by "
                + "\(miss / samplingSD) sampling sigma (gate 5)"),
        sourceLocation: sourceLocation)
}

/// Standard normal CDF, for the Kolmogorov-Smirnov test.
private func normalCDF(_ z: Double) -> Double {
    0.5 * erfc(-z / 2.0.squareRoot())
}

// MARK: - Philox

/// Checks the generator itself: the published test vectors, then the two properties the render
/// relies on, uniformity and diffusion.
@Suite("Philox4x32-10")
struct PhiloxTests {

    /// Known-answer vectors from Random123's `kat_vectors`, in its `counter key -> output` order.
    ///
    /// The round function, the key schedule and the output word order were also cross-checked
    /// against `numpy.random.Philox`, the 4x64 member of the same family. Philox4x64-10 built with
    /// this structure reproduces `Philox(key:counter:).random_raw(4)` bit for bit, once you account
    /// for NumPy incrementing its counter before it fills a block.
    @Test("published Random123 test vectors")
    func knownAnswerVectors() {
        let vectors: [(counter: SIMD4<UInt32>, key: (UInt32, UInt32), expected: SIMD4<UInt32>)] = [
            (
                SIMD4(0, 0, 0, 0), (0, 0),
                SIMD4(0x6627_E8D5, 0xE169_C58D, 0xBC57_AC4C, 0x9B00_DBD8)
            ),
            (
                SIMD4(repeating: 0xFFFF_FFFF), (0xFFFF_FFFF, 0xFFFF_FFFF),
                SIMD4(0x408F_276D, 0x41C8_3B0E, 0xA20B_C7C6, 0x6D54_51FD)
            ),
            (
                SIMD4(0x243F_6A88, 0x85A3_08D3, 0x1319_8A2E, 0x0370_7344),
                (0xA409_3822, 0x299F_31D0),
                SIMD4(0xD16C_FE09, 0x94FD_CCEB, 0x5001_E420, 0x2412_6EA1)
            ),
        ]

        for vector in vectors {
            let got = Philox4x32.generate(counter: vector.counter, key: vector.key)
            #expect(got == vector.expected, "counter \(vector.counter) key \(vector.key)")
        }
    }

    @Test("the buffered stream yields the raw blocks in order")
    func bufferedStreamMatchesRawBlocks() {
        let key = PhiloxKey(seed: 0x0123_4567_89AB_CDEF, channel: 2, sublayer: 1)
        var source = Philox4x32(key: key, counter: 7)
        let drawn = (0..<12).map { _ in source.nextBits() }

        let keyWords: (UInt32, UInt32) = (0x89AB_CDEF, 0x0123_4567)
        let streamWord: UInt32 = 2 | (1 << 16)
        var expected: [UInt32] = []
        for block in 0..<3 {
            let out = Philox4x32.generate(
                counter: SIMD4(7, 0, UInt32(block), streamWord), key: keyWords)
            for word in 0..<4 { expected.append(out[word]) }
        }
        #expect(drawn == expected)
    }

    @Test("uniforms are flat across 128 bins")
    func uniformsAreFlat() throws {
        let critical = try Golden("random_chi2_critical_1em3").values
        let bins = 128
        var counts = [Int](repeating: 0, count: bins)
        var source = Philox4x32(key: PhiloxKey(seed: 0xA5A5_5A5A))
        for _ in 0..<samples {
            let u = source.nextUniform()
            counts[min(bins - 1, Int(u * Double(bins)))] += 1
        }
        let expectedPerBin = Double(samples) / Double(bins)
        var chiSquare = 0.0
        for count in counts {
            let d = Double(count) - expectedPerBin
            chiSquare += d * d / expectedPerBin
        }
        #expect(
            chiSquare <= critical[bins - 2],
            "chi2 = \(chiSquare) on \(bins - 1) dof, critical \(critical[bins - 2])")
    }

    @Test("uniforms stay inside their intervals")
    func uniformRanges() {
        var source = Philox4x32(key: PhiloxKey(seed: 11))
        for _ in 0..<100_000 {
            let half = source.nextUniform()
            #expect(half >= 0.0 && half < 1.0)
            let open = source.nextOpenUniform()
            #expect(open > 0.0 && open < 1.0)
        }
    }

    /// Flipping one counter bit must scramble the whole output, otherwise neighbouring pixels share
    /// visible structure.
    @Test("one counter bit flips about half of every output bit")
    func avalanche() throws {
        let critical = try Golden("random_chi2_critical_1em3").values
        let trials = 4096
        let key: (UInt32, UInt32) = (0x1234_5678, 0x9ABC_DEF0)

        for flipped in [0, 31, 64, 127] {
            var perBitFlips = [Int](repeating: 0, count: 128)
            var hammingTotal = 0
            for trial in 0..<trials {
                // Spread the base counters over the whole 128-bit space so the test is not
                // measuring diffusion from small integers only.
                let base = Philox4x32.generate(
                    counter: SIMD4(UInt32(trial), 0, 0, 0), key: (0xDEAD_BEEF, 0xFEED_FACE))
                var perturbed = base
                perturbed[flipped / 32] ^= 1 << UInt32(flipped % 32)

                let a = Philox4x32.generate(counter: base, key: key)
                let b = Philox4x32.generate(counter: perturbed, key: key)
                for word in 0..<4 {
                    let diff = a[word] ^ b[word]
                    hammingTotal += diff.nonzeroBitCount
                    for bit in 0..<32 where (diff >> UInt32(bit)) & 1 == 1 {
                        perBitFlips[word * 32 + bit] += 1
                    }
                }
            }

            let meanHamming = Double(hammingTotal) / Double(trials)
            #expect(
                abs(meanHamming - 64.0) <= 0.5,
                "counter bit \(flipped): mean Hamming distance \(meanHamming), want 64")

            let expectedFlips = Double(trials) / 2.0
            let variancePerBit = Double(trials) / 4.0
            var chiSquare = 0.0
            for count in perBitFlips {
                let d = Double(count) - expectedFlips
                chiSquare += d * d / variancePerBit
            }
            #expect(
                chiSquare <= critical[127],
                Comment(
                    rawValue: "counter bit \(flipped): per-bit chi2 = \(chiSquare) on 128 dof, "
                        + "critical \(critical[127])"))
        }
    }

    /// The nine grain streams are `(channel, sublayer)` pairs on one seed. They have to be
    /// unrelated, since they are summed into the same pixel.
    @Test("the nine grain streams are distinct and uncorrelated")
    func grainStreamsAreIndependent() {
        let length = 4096
        var streams: [[Double]] = []
        for channel in 0..<3 {
            for sublayer in 0..<3 {
                var source = Philox4x32(
                    key: PhiloxKey(seed: 1, channel: channel, sublayer: sublayer))
                streams.append((0..<length).map { _ in source.nextUniform() })
            }
        }

        for i in 0..<streams.count {
            for j in (i + 1)..<streams.count {
                #expect(streams[i] != streams[j], "streams \(i) and \(j) are identical")
                var covariance = 0.0
                for k in 0..<length {
                    covariance += (streams[i][k] - 0.5) * (streams[j][k] - 0.5)
                }
                // Var(U) = 1/12, so the correlation estimate has SE = 1/sqrt(length).
                let correlation = covariance / (Double(length) / 12.0)
                #expect(
                    abs(correlation) <= 5.0 / Double(length).squareRoot(),
                    "streams \(i) and \(j) correlate at \(correlation)")
            }
        }
    }
}

// MARK: - Decomposition independence

/// The generator is counter-based so that a pixel's noise does not depend on how the frame is split
/// up. Upstream's fast path fails this, and its output changes with the thread count (`grain.md`
/// section 8.2).
@Suite("Counter-based reproducibility")
struct CounterIndependenceTests {

    private static let width = 64
    private static let height = 48
    private static let key = PhiloxKey(seed: 0xC0FF_EE00, channel: 2, sublayer: 1)

    /// Spans both sampler branches, so the test covers Knuth and transformed rejection together.
    private static func lambda(at index: Int) -> Double {
        let table = [0.4, 4.5, 9.0, 9.999, 10.0, 31.0, 1.0e3, 1.2e8]
        return table[index % table.count]
    }

    private static func draw(visiting order: [Int], reusingSource: Bool) -> [Int] {
        var plane = [Int](repeating: -1, count: width * height)
        if reusingSource {
            var source = Philox4x32(key: key)
            for index in order {
                source.reset(counter: UInt64(index))
                plane[index] = Distributions.poisson(lambda: lambda(at: index), &source)
            }
        } else {
            for index in order {
                plane[index] = Distributions.poisson(
                    lambda: lambda(at: index), key: key, counter: UInt64(index))
            }
        }
        return plane
    }

    /// Tiles of `rows` rows, even tiles before odd ones, so the order differs from a raster scan.
    private static func tiledOrder(rows: Int) -> [Int] {
        var order: [Int] = []
        for parity in 0..<2 {
            var top = parity * rows
            while top < height {
                for y in top..<min(top + rows, height) {
                    for x in 0..<width { order.append(y * width + x) }
                }
                top += 2 * rows
            }
        }
        return order
    }

    @Test("visit order and tile size do not change a single pixel")
    func orderAndTileIndependence() {
        let raster = Array(0..<(Self.width * Self.height))
        let reference = Self.draw(visiting: raster, reusingSource: false)
        #expect(!reference.contains(-1))
        #expect(reference.contains { $0 > 0 })

        var orders: [(String, [Int])] = [
            ("reverse raster", raster.reversed()),
            // A stride coprime with the pixel count touches every pixel in scattered order.
            ("stride 997", (0..<raster.count).map { ($0 * 997) % raster.count }),
        ]
        for rows in [1, 3, 7, 16, Self.height] {
            orders.append(("tiles of \(rows) rows", Self.tiledOrder(rows: rows)))
        }

        for (name, order) in orders {
            #expect(Set(order).count == raster.count, "\(name) does not cover every pixel once")
            #expect(Self.draw(visiting: order, reusingSource: false) == reference, "\(name)")
            #expect(
                Self.draw(visiting: order, reusingSource: true) == reference,
                "\(name), reusing one source through reset(counter:)")
        }
    }

    @Test("reset discards a partly consumed block")
    func resetClearsTheBuffer() {
        let key = PhiloxKey(seed: 42, channel: 1, sublayer: 2)
        var fresh = Philox4x32(key: key, counter: 9)
        let expected = (0..<7).map { _ in fresh.nextBits() }

        var reused = Philox4x32(key: key, counter: 3)
        _ = reused.nextBits()
        _ = reused.nextBits()
        _ = reused.nextBits()
        reused.reset(counter: 9)
        #expect((0..<7).map { _ in reused.nextBits() } == expected)
    }

    @Test("the seed selects a different realisation")
    func seedChangesTheField() {
        let count = 2048
        let a = (0..<count).map {
            Distributions.poisson(lambda: 12.0, key: PhiloxKey(seed: 0), counter: UInt64($0))
        }
        let b = (0..<count).map {
            Distributions.poisson(lambda: 12.0, key: PhiloxKey(seed: 1), counter: UInt64($0))
        }
        #expect(a != b)
        var agreements = 0
        for i in 0..<count where a[i] == b[i] { agreements += 1 }
        // Two independent Poisson(12) draws agree about 11.5 percent of the time.
        #expect(agreements < count / 3, "\(agreements) of \(count) pixels agree")
    }
}

// MARK: - Distributions

@Suite("Distributions")
struct DistributionsTests {

    // MARK: Lognormal

    /// The log-space inversion is pure arithmetic, so its golden is gated at 1e-15. It should differ
    /// from the oracle only by rounding.
    @Test("the log-space inversion matches fast_lognormal_from_mean_std")
    func lognormalLogParameters() throws {
        let golden = try Golden("random_lognormal_log_params")
        let rows = golden.shape[0]
        var computed = [Double](repeating: 0, count: rows * 4)
        for row in 0..<rows {
            let mean = golden.values[row * 4]
            let std = golden.values[row * 4 + 1]
            let (mu, sigma) = Distributions.lognormalLogParameters(mean: mean, std: std)
            computed[row * 4] = mean
            computed[row * 4 + 1] = std
            computed[row * 4 + 2] = mu
            computed[row * 4 + 3] = sigma
        }
        try expectParity(
            computed, matches: "random_lognormal_log_params",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    @Test("the degenerate branches draw nothing")
    func lognormalDegenerateBranches() {
        let key = PhiloxKey(seed: 3)
        // mean <= 0 forces mu = sigma = 0 in the reference, so every pixel comes back as 1.
        for mean in [0.0, -1.0, -1e9] {
            #expect(
                Distributions.lognormalFromMeanStd(mean: mean, std: 0.5, key: key, counter: 0)
                    == 1.0)
        }
        // A sigma under 1e-6 skips the normal draw, so the field is exactly its mean.
        #expect(
            Distributions.lognormalFromMeanStd(mean: 2.0, std: 0.0, key: key, counter: 0) == 2.0)
        let tiny = Distributions.lognormalFromMeanStd(mean: 2.0, std: 1e-9, key: key, counter: 0)
        #expect(tiny == 2.0)
        // Every counter gives the same constant, since nothing is consumed.
        for counter in UInt64(0)..<16 {
            #expect(
                Distributions.lognormalFromMeanStd(mean: 2.0, std: 1e-9, key: key, counter: counter)
                    == 2.0)
        }
    }

    /// `m <= 0` is false for NaN, so the reference propagates a NaN mean through its arithmetic, and
    /// `fast_lognormal_from_mean_std(nan, 0.5)` returns NaN (measured). A guard written `mean > 0`
    /// would return the `mean <= 0` constant 1.0 and hide a NaN density behind a clumping field of
    /// all ones.
    @Test("a NaN mean or std stays NaN")
    func lognormalCarriesNaN() {
        let key = PhiloxKey(seed: 3)
        #expect(
            Distributions.lognormalFromMeanStd(mean: .nan, std: 0.5, key: key, counter: 0).isNaN)
        #expect(
            Distributions.lognormalFromMeanStd(mean: 1.0, std: .nan, key: key, counter: 0).isNaN)
        let (mu, sigma) = Distributions.lognormalLogParameters(mean: .nan, std: 0.5)
        #expect(mu.isNaN && sigma.isNaN)
        // An infinite mean has sigma = 0 and mu = inf, so the sigma floor short-circuits to exp(mu).
        #expect(
            Distributions.lognormalFromMeanStd(mean: .infinity, std: 0.5, key: key, counter: 0)
                == .infinity)
        // -inf takes the `mean <= 0` branch like any other non-positive mean.
        #expect(
            Distributions.lognormalFromMeanStd(mean: -.infinity, std: 0.5, key: key, counter: 0)
                == 1.0)
    }

    @Test("lognormal mean and variance match the oracle")
    func lognormalMoments() throws {
        let params = try Golden("random_lognormal_params_sampled")
        let expected = try Golden("random_lognormal_moments")
        let samplingSD = try Golden("random_lognormal_moment_sd")

        for row in 0..<params.shape[0] {
            let mean = params.values[row * 2]
            let std = params.values[row * 2 + 1]
            let key = PhiloxKey(seed: 0x10_0000 + UInt64(row))
            var source = Philox4x32(key: key)
            var drawn = [Double](repeating: 0, count: samples)
            for i in 0..<samples {
                source.reset(counter: UInt64(i))
                drawn[i] = Distributions.lognormalFromMeanStd(mean: mean, std: std, &source)
            }
            let moments = Moments(drawn)

            // The closed form is the requested (mean, std): the inversion exists to make the
            // linear-space moments equal the requested ones.
            let label = "lognormal(mean \(mean), std \(std))"
            expectMoment(
                moments.mean, closedForm: mean, samplingSD: samplingSD.values[row * 4],
                "\(label) mean")
            expectMoment(
                moments.variance, closedForm: std * std,
                samplingSD: samplingSD.values[row * 4 + 1], "\(label) variance")
            // The oracle's own realisation must pass the same gate, or the gate is wrong.
            expectMoment(
                expected.values[row * 4], closedForm: mean,
                samplingSD: samplingSD.values[row * 4], "oracle \(label) mean")
        }
    }

    // MARK: Normal

    @Test("standard normal moments match the oracle")
    func standardNormalMoments() throws {
        let expected = try Golden("random_normal_moments").values
        let samplingSD = try Golden("random_normal_moment_sd").values

        var source = Philox4x32(key: PhiloxKey(seed: 0x2000_0001))
        var drawn = [Double](repeating: 0, count: samples)
        for i in 0..<samples {
            source.reset(counter: UInt64(i))
            drawn[i] = Distributions.standardNormal(&source)
        }
        let moments = Moments(drawn)
        let closedForm = [0.0, 1.0, 0.0, 0.0]
        for index in 0..<4 {
            expectMoment(
                moments[index], closedForm: closedForm[index], samplingSD: samplingSD[index],
                "standard normal \(momentNames[index])")
            expectMoment(
                expected[index], closedForm: closedForm[index], samplingSD: samplingSD[index],
                "oracle standard normal \(momentNames[index])")
        }
    }

    @Test("standard normal passes Kolmogorov-Smirnov")
    func standardNormalGoodnessOfFit() throws {
        let critical = try Golden("random_ks_critical_1em3").values[0]
        var source = Philox4x32(key: PhiloxKey(seed: 0x2000_0002))
        var drawn = [Double](repeating: 0, count: samples)
        for i in 0..<samples {
            source.reset(counter: UInt64(i))
            drawn[i] = Distributions.standardNormal(&source)
        }
        drawn.sort()

        var statistic = 0.0
        let n = Double(samples)
        for i in 0..<samples {
            let cdf = normalCDF(drawn[i])
            statistic = max(statistic, max(cdf - Double(i) / n, Double(i + 1) / n - cdf))
        }
        let gate = critical / n.squareRoot()
        #expect(statistic <= gate, "KS D = \(statistic), gate \(gate)")
    }

    // MARK: Poisson

    @Test("lambda at or below zero, and non-finite lambda, yield zero")
    func poissonEdgeCases() {
        let key = PhiloxKey(seed: 5)
        for lambda in [0.0, -0.0, -1.0, -1e9, Double.nan, -Double.infinity] {
            #expect(Distributions.poisson(lambda: lambda, key: key, counter: 0) == 0, "\(lambda)")
        }
        // The reference has no usable answer for infinity: the exact path raises and
        // `fast_poisson` returns Int64.max. Returning 0 keeps `Int(Double)` from trapping.
        #expect(Distributions.poisson(lambda: .infinity, key: key, counter: 0) == 0)
    }

    /// A finite lambda past `Int64` range must not trap. The reference raises above its own
    /// `POISSON_LAM_MAX`; here lambda clamps to it, which keeps every accepted candidate inside
    /// `Int`. Grain can reach this: `sat = 1 - p * u * (1 - 1e-6)` is one ulp above zero just past
    /// `uniformity = 1`, and `lambda = N * p / sat` then reaches 2.8e19.
    @Test("a lambda past the Int64 range clamps instead of trapping")
    func poissonClampsHugeLambda() {
        let key = PhiloxKey(seed: 8)
        let saturation = 1.0 - (1.0 - 1e-6) * 1.0000020000029999 * (1.0 - 1e-6)
        #expect(saturation > 0 && saturation < 1e-15, "saturation \(saturation)")
        let grainLambda = 6125.0 * (1.0 - 1e-6) / saturation
        #expect(grainLambda > 1e19, "lambda \(grainLambda)")

        for lambda in [1e19, grainLambda, 1e300, Double.greatestFiniteMagnitude] {
            let drawn = Distributions.poisson(lambda: lambda, key: key, counter: 0)
            // Within 10 sqrt(lambda) of the clamp, the width NumPy's bound leaves for the tail.
            let miss = abs(Double(drawn) - Distributions.poissonLambdaMax)
            #expect(
                miss <= 10.0 * Distributions.poissonLambdaMax.squareRoot(),
                "lambda \(lambda) drew \(drawn), off the clamp by \(miss)")
        }
        // Just under the clamp the draws are untouched, so the bound does not cut into the range the
        // grain model uses.
        var source = Philox4x32(key: key)
        for i in 0..<64 {
            source.reset(counter: UInt64(i))
            let drawn = Double(Distributions.poisson(lambda: 9.0e18, &source))
            #expect(abs(drawn - 9.0e18) <= 10.0 * 9.0e18.squareRoot(), "\(drawn)")
        }
    }

    @Test("Poisson moments match the closed form on both branches")
    func poissonMoments() throws {
        let lambdas = try Golden("random_poisson_lambdas").values
        let oracle = try Golden("random_poisson_moments")
        let samplingSD = try Golden("random_poisson_moment_sd")

        for (row, lambda) in lambdas.enumerated() {
            let key = PhiloxKey(seed: 0x3000_0000 + UInt64(row))
            var source = Philox4x32(key: key)
            var drawn = [Double](repeating: 0, count: samples)
            for i in 0..<samples {
                source.reset(counter: UInt64(i))
                drawn[i] = Double(Distributions.poisson(lambda: lambda, &source))
            }
            let moments = Moments(drawn)
            let closedForm = [lambda, lambda, 1.0 / lambda.squareRoot(), 1.0 / lambda]

            for index in 0..<4 {
                let sd = samplingSD.values[row * 4 + index]
                expectMoment(
                    moments[index], closedForm: closedForm[index], samplingSD: sd,
                    "Poisson(\(lambda)) \(momentNames[index])")
                expectMoment(
                    oracle.values[row * 4 + index], closedForm: closedForm[index], samplingSD: sd,
                    "oracle Poisson(\(lambda)) \(momentNames[index])")
            }
        }
    }

    /// A Gaussian with the right mean and variance passes the first two moment gates above and fails
    /// this one, so do not drop the skewness gate.
    @Test("Poisson skewness is nowhere near a Gaussian's")
    func poissonSkewnessSeparatesTheSamplerFromAGaussian() throws {
        let samplingSD = try Golden("random_poisson_moment_sd")
        let lambdas = try Golden("random_poisson_lambdas").values

        // lambda 20 is in the transformed-rejection branch, which a Gaussian approximation would
        // replace.
        let row = try #require(lambdas.firstIndex(of: 20.0))
        let lambda = lambdas[row]
        var source = Philox4x32(key: PhiloxKey(seed: 0x4000_0000))
        var drawn = [Double](repeating: 0, count: samples)
        for i in 0..<samples {
            source.reset(counter: UInt64(i))
            drawn[i] = Double(Distributions.poisson(lambda: lambda, &source))
        }
        let measured = Moments(drawn).skewness
        let sd = samplingSD.values[row * 4 + 2]
        let expected = 1.0 / lambda.squareRoot()

        #expect(abs(measured - expected) <= 5.0 * sd, "skewness \(measured), want \(expected)")
        #expect(
            measured / sd >= 20.0,
            Comment(
                rawValue: "skewness \(measured) is only \(measured / sd) sampling sigma from a "
                    + "Gaussian's 0, so this gate would not catch one"))
    }

    @Test("Poisson counts pass a chi-square goodness of fit at small lambda")
    func poissonGoodnessOfFit() throws {
        let lambdas = try Golden("random_poisson_pmf_lambdas").values
        let pmf = try Golden("random_poisson_pmf")
        let critical = try Golden("random_chi2_critical_1em3").values
        let bins = pmf.shape[1] - 1
        // Independent of the calibrated moment tolerances, so it can use fewer samples.
        let count = 65536

        for (row, lambda) in lambdas.enumerated() {
            let perCount = (0...bins).map { Double(count) * pmf.values[row * (bins + 1) + $0] }

            // Pool neighbouring counts until every bin expects at least 5. Below 5 the chi-square
            // approximation to the statistic's null distribution breaks down, and at lambda 20 it
            // is the low counts that are sparse as well as the tail.
            var expected: [Double] = []
            var binOfCount = [Int](repeating: 0, count: bins + 1)
            var running = 0.0
            for k in 0...bins {
                binOfCount[k] = expected.count
                running += perCount[k]
                if running >= 5.0 {
                    expected.append(running)
                    running = 0.0
                }
            }
            if running > 0 {
                // The leftover tail is under 5, so it joins the last full bin.
                let last = expected.count - 1
                expected[last] += running
                for k in 0...bins where binOfCount[k] > last { binOfCount[k] = last }
            }

            var observed = [Double](repeating: 0, count: expected.count)
            var source = Philox4x32(key: PhiloxKey(seed: 0x5000_0000 + UInt64(row)))
            for i in 0..<count {
                source.reset(counter: UInt64(i))
                let drawn = Distributions.poisson(lambda: lambda, &source)
                observed[binOfCount[min(drawn, bins)]] += 1
            }

            var chiSquare = 0.0
            for k in expected.indices {
                let d = observed[k] - expected[k]
                chiSquare += d * d / expected[k]
            }
            let dof = expected.count - 1
            #expect(dof >= 5, "Poisson(\(lambda)): only \(dof) dof, too coarse to be a test")
            #expect(
                chiSquare <= critical[dof - 1],
                Comment(
                    rawValue: "Poisson(\(lambda)): chi2 = \(chiSquare) on \(dof) dof, critical "
                        + "\(critical[dof - 1])"))
        }
    }

    /// The transformed-rejection branch must stay exact at the top of the grain model's range: `sat`
    /// reaches its minimum near 2e-6 at `uniformity = 1`, which puts lambda at 1.2e8.
    @Test("transformed rejection survives lambda 1.2e8")
    func poissonAtTheTopOfTheRange() {
        let lambda = 1.2e8
        let count = 65536
        var source = Philox4x32(key: PhiloxKey(seed: 0x6000_0000))
        var sum = 0.0
        var sumSquares = 0.0
        var smallest = Double.infinity
        var largest = -Double.infinity
        for i in 0..<count {
            source.reset(counter: UInt64(i))
            let drawn = Double(Distributions.poisson(lambda: lambda, &source))
            sum += drawn
            sumSquares += (drawn - lambda) * (drawn - lambda)
            smallest = min(smallest, drawn)
            largest = max(largest, drawn)
        }
        let mean = sum / Double(count)
        let sd = (sumSquares / Double(count)).squareRoot()
        let expectedSD = lambda.squareRoot()

        // SE(mean) = sqrt(lambda / count) = 42.8.
        #expect(abs(mean - lambda) <= 5.0 * (lambda / Double(count)).squareRoot(), "mean \(mean)")
        #expect(abs(sd / expectedSD - 1.0) <= 5.0 * (2.0 / Double(count)).squareRoot(), "sd \(sd)")
        // A sampler that silently clamped or overflowed would collapse the range.
        #expect(largest - smallest > 6.0 * expectedSD, "range \(smallest) to \(largest)")
    }

    @Test("the two branches agree where they meet")
    func poissonBranchesAgreeAtTheCrossover() {
        let count = 262_144
        func moments(lambda: Double, seed: UInt64) -> Moments {
            var source = Philox4x32(key: PhiloxKey(seed: seed))
            var drawn = [Double](repeating: 0, count: count)
            for i in 0..<count {
                source.reset(counter: UInt64(i))
                drawn[i] = Double(Distributions.poisson(lambda: lambda, &source))
            }
            return Moments(drawn)
        }
        // 9.999 takes the Knuth branch and 10.0 takes transformed rejection. Their moments differ
        // by far less than the sampling noise, so a discontinuity at the crossover shows up here.
        let knuth = moments(lambda: 9.999, seed: 0x7000_0001)
        let rejection = moments(lambda: 10.0, seed: 0x7000_0002)
        let meanSE = (10.0 / Double(count)).squareRoot()
        #expect(abs(knuth.mean - rejection.mean) <= 7.0 * meanSE)
        #expect(abs(knuth.variance / rejection.variance - 1.0) <= 0.02)
        #expect(abs(knuth.skewness - rejection.skewness) <= 0.03)
    }
}

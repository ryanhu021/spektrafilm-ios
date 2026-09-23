import Foundation
import Testing

@testable import SpektraFilm

/// Checks the error function, the CDFs built on it, and Brent root finding against the oracle.
///
/// These are leaf primitives, so their errors arrive at the render disguised as someone else's bug.
/// The gate here is 1e-15, eleven orders tighter than the pipeline gate, because nothing downstream
/// can explain a libm that is merely close.
@Suite("Transcendental primitives")
struct TranscendentalTests {

    // MARK: - erf and the CDFs

    @Test("erf matches scipy.special.erf")
    func erfSweep() throws {
        let xs = try Golden("transcendental_sweep_input").values
        try expectParity(
            xs.map(Erf.erf), matches: "transcendental_erf",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    @Test("erfc matches scipy.special.erfc")
    func erfcSweep() throws {
        let xs = try Golden("transcendental_sweep_input").values
        try expectParity(
            xs.map(Erf.erfc), matches: "transcendental_erfc",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    @Test("normalCDF matches scipy.special.ndtr")
    func normalCDFSweep() throws {
        let xs = try Golden("transcendental_sweep_input").values
        try expectParity(
            xs.map(Erf.normalCDF), matches: "transcendental_normal_cdf",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    @Test("gumbelMatchedCDF matches morph_curves")
    func gumbelSweep() throws {
        let xs = try Golden("transcendental_sweep_input").values
        try expectParity(
            xs.map(Erf.gumbelMatchedCDF), matches: "transcendental_gumbel_cdf",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    /// The constants are literals in the source, so this is the only place that checks they are the
    /// closed forms the reference computes at import time.
    @Test("the Gumbel constants are their closed forms")
    func gumbelConstants() {
        #expect(Erf.gumbelLocation == -Foundation.log(Foundation.log(2.0)))
        #expect(Erf.gumbelWidth == 0.5 * Foundation.log(2.0) * (2.0 * Double.pi).squareRoot())
        // Both constants exist to put the Gumbel's median at the normal's.
        #expect(abs(Erf.gumbelMatchedCDF(0.0) - 0.5) < 1e-16)
        #expect(Erf.normalCDF(0.0) == 0.5)
    }

    /// `erfc` is not `1 - erf`: past x = 6 the subtraction has no significant digits left, and the
    /// print-curve morph evaluates layer CDFs out to |z| = 20 on steep sub-layers.
    @Test("erfc keeps precision where 1 - erf has none")
    func erfcTail() {
        #expect(1.0 - Erf.erf(6.0) == 0.0)
        #expect(Erf.erfc(6.0) > 2.1e-17)
        #expect(Erf.normalCDF(-20.0) > 2.7e-89)
    }

    // MARK: - Band-pass filter

    static let bandPassCases:
        [(
            slug: String,
            uv: (amplitude: Double, centre: Double, width: Double),
            ir: (amplitude: Double, centre: Double, width: Double)
        )] = [
            ("default", (0.0, 410.0, 8.0), (0.0, 675.0, 15.0)),
            ("active", (1.0, 410.0, 8.0), (1.0, 675.0, 15.0)),
            ("partial", (0.6, 395.0, 12.0), (0.35, 690.0, 9.0)),
        ]

    @Test("band-pass filter matches compute_band_pass_filter", arguments: bandPassCases)
    func bandPass(
        slug: String,
        uv: (amplitude: Double, centre: Double, width: Double),
        ir: (amplitude: Double, centre: Double, width: Double)
    ) throws {
        let filter = Erf.bandPassFilter(uv: uv, ir: ir)
        #expect(filter.count == 81)
        try expectParity(
            filter, matches: "filter_bandpass_\(slug)",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    /// Amplitude 0 has to leave the spectrum untouched, since the caller only skips the block when
    /// both amplitudes are non-positive and would otherwise scale the sensitivity by this array.
    @Test("amplitude zero passes everything and amplitudes clip to [0, 1]")
    func bandPassAmplitude() {
        let neutral = Erf.bandPassFilter(uv: (0.0, 410.0, 8.0), ir: (0.0, 675.0, 15.0))
        #expect(neutral.allSatisfy { $0 == 1.0 })

        let clipped = Erf.bandPassFilter(uv: (5.0, 410.0, 8.0), ir: (-3.0, 675.0, 15.0))
        let atOne = Erf.bandPassFilter(uv: (1.0, 410.0, 8.0), ir: (0.0, 675.0, 15.0))
        #expect(clipped == atOne)

        // The IR arm's negated width makes it a falling edge; without the negation the product
        // would pass long wavelengths and block the visible band.
        let active = Erf.bandPassFilter(uv: (1.0, 410.0, 8.0), ir: (1.0, 675.0, 15.0))
        #expect(abs(active[6] - 0.5) < 1e-15)  // 410 nm
        #expect(abs(active[59] - 0.5) < 1e-15)  // 675 nm
        #expect(active[40] > 0.999)  // 580 nm, mid band
        #expect(active[80] < 1e-12)  // 780 nm
    }

    // MARK: - Brent

    /// Same residuals as `Tools/parity/fixtures/transcendental.py`, written with explicit
    /// multiplications so both sides evaluate identical float64.
    static let brentCases: [(name: String, f: @Sendable (Double) -> Double, lower: Double, upper: Double)] = [
        ("x^2 - 2", { $0 * $0 - 2.0 }, 0.0, 2.0),
        ("cos(x) - x", { Foundation.cos($0) - $0 }, 0.0, 1.0),
        ("x^3 - 2x - 5", { $0 * $0 * $0 - 2.0 * $0 - 5.0 }, 2.0, 3.0),
        ("exp(x) - 3x", { Foundation.exp($0) - 3.0 * $0 }, 0.0, 1.0),
        ("atan(x) - 0.5", { Foundation.atan($0) - 0.5 }, -1.0, 2.0),
        ("x^3 - 0.027", { $0 * $0 * $0 - 0.027 }, -1.0, 0.5),
        ("tanh(x - 1)", { Foundation.tanh($0 - 1.0) }, -5.0, 5.0),
        ("x^9 - 1e-3", { x in (0..<9).reduce(1.0) { p, _ in p * x } - 1e-3 }, -1.0, 1.3),
        ("1/(x - 0.3) - 2", { 1.0 / ($0 - 0.3) - 2.0 }, 0.35, 5.0),
        ("exp(-x^2) - 0.75", { Foundation.exp(-$0 * $0) - 0.75 }, 0.0, 3.0),
    ]

    @Test("brentq matches scipy.optimize.brentq at both tolerances")
    func brentRoots() throws {
        let expected = try Golden("transcendental_brentq_roots")
        #expect(expected.shape == [2, Self.brentCases.count])
        var found: [Double] = []
        for xTolerance in [RootFind.defaultXTolerance, 1e-10] {
            for testCase in Self.brentCases {
                let root = RootFind.brentq(
                    lower: testCase.lower, upper: testCase.upper, xTolerance: xTolerance,
                    testCase.f)
                #expect(root?.converged == true, "\(testCase.name) did not converge")
                found.append(root?.value ?? .nan)
            }
        }
        try expectParity(
            found, matches: "transcendental_brentq_roots",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    /// Roots outside the reference's initial `[-0.25, 0.25]`, so the doubling search has to fire.
    ///
    /// The last three pin the conventions, in `Tools/parity/fixtures/transcendental.py`'s order: a
    /// residual of exactly zero at `lo`, the same at `hi`, and a root only the twelfth bracket
    /// reaches.
    static let bracketCases: [@Sendable (Double) -> Double] = [
        { Foundation.tanh($0 - 1.7) },
        { 1.0 - Foundation.exp(-($0 + 1.1)) },
        { Foundation.tanh(0.3 * ($0 - 3.8)) },
        { ($0 + 0.25) * ($0 - 3.0) },
        { ($0 - 0.25) * ($0 + 3.0) },
        { Foundation.tanh(0.001 * ($0 - 300.0)) },
    ]

    @Test("the doubling bracket search matches the reference's loop")
    func bracketExpansion() throws {
        // A length mismatch against the fixture traps inside `parity`, so check it here first.
        #expect(try Golden("transcendental_bracket_roots").shape == [Self.bracketCases.count])
        let roots = Self.bracketCases.map { RootFind.expandingBracketRoot($0) ?? .nan }
        try expectParity(
            roots, matches: "transcendental_bracket_roots",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    /// Two roots with the residual negative between them and positive outside: every symmetric
    /// bracket has the same sign at both ends, so the reference gives up after 12 doublings and
    /// substitutes a zero offset. Verified against the reference's loop in the oracle.
    @Test("a bracket that never straddles a sign change gives up")
    func bracketGivesUp() {
        let root = RootFind.expandingBracketRoot { ($0 + 2.65) * ($0 - 3.8) }
        #expect(root == nil)
    }

    /// Three brackets the reference's loop declines, each checked against that loop in the oracle.
    /// None can be a golden, because the reference returns `None` and a `.spkg` carries only doubles.
    @Test("brackets the reference declines")
    func bracketDeclined() {
        // Root at 1024, one doubling past the twelfth bracket.
        #expect(RootFind.expandingBracketRoot { $0 - 1024.0 } == nil)
        // Residuals of ±1e-180 multiply to ±0.0, so the product test never sees the sign change. A
        // sign-bit test would accept the first bracket and return 0.7.
        #expect(RootFind.expandingBracketRoot { 1e-180 * ($0 - 0.7) } == nil)
        // NaN fails every comparison in the loop, so the search doubles out and gives up. Profiles
        // carry NaN for missing datasheet coverage, which is how a NaN residual gets here.
        #expect(RootFind.expandingBracketRoot { _ in Double.nan } == nil)
    }

    /// The reference's `brentq` raises `RuntimeError` on an exhausted budget, so the search reports
    /// no root at all. Unreachable with the default budget: a bracket that
    /// straddles always converges well inside 100 iterations.
    @Test("a straddling bracket that does not converge reports no root")
    func bracketNonConvergence() {
        let f: @Sendable (Double) -> Double = { Foundation.atan($0) - 0.5 }
        #expect(RootFind.expandingBracketRoot(f) != nil)
        #expect(RootFind.expandingBracketRoot(xTolerance: 1e-300, maxIterations: 4, f) == nil)
    }

    @Test("brentq declines a same-sign bracket")
    func sameSignBracket() {
        #expect(RootFind.brentq(lower: 1.0, upper: 2.0) { $0 * $0 + 1.0 } == nil)
        // Both residuals underflow to the same sign; the product would be +0.0 and read as a bracket.
        #expect(RootFind.brentq(lower: -1.0, upper: 2.0) { _ in 1e-200 } == nil)
        // Opposite signs survive the same underflow, so this one is a bracket. SciPy agrees on the
        // root and on the 42 iterations it takes.
        let root = RootFind.brentq(lower: -1.0, upper: 2.0) { $0 < 0.5 ? -1e-200 : 1e-200 }
        #expect(root?.value == 0.49999999999863576)
        #expect(root?.iterations == 42)
    }

    /// SciPy wraps `f` and raises `ValueError` on the first NaN. Nothing here throws, so a NaN at an
    /// endpoint is `nil` and a NaN mid-solve is a non-converged result. Returning it as a converged
    /// root would hand the morph a silent wrong answer.
    @Test("a NaN residual is reported in the return value")
    func nanResidual() {
        #expect(RootFind.brentq(lower: -1.0, upper: 2.0) { $0 > 1.0 ? Double.nan : $0 - 0.7 } == nil)
        #expect(RootFind.brentq(lower: -1.0, upper: 2.0) { _ in Double.nan } == nil)
        let mid = RootFind.brentq(lower: -1.0, upper: 2.0) {
            abs($0 - 0.5) < 0.4 ? Double.nan : $0 - 0.7
        }
        #expect(mid?.converged == false)
    }

    /// An endpoint that is already a root short-circuits, as `brentq.c` does before iterating.
    @Test("an exact root at an endpoint returns immediately")
    func endpointRoot() {
        let low = RootFind.brentq(lower: 0.0, upper: 3.0) { $0 * ($0 - 1.0) }
        #expect(low?.value == 0.0)
        #expect(low?.iterations == 0)
        let high = RootFind.brentq(lower: -3.0, upper: 0.0) { $0 * ($0 - 1.0) }
        #expect(high?.value == 0.0)
    }

    /// The tolerances are invisible in every root above, because `xtol` dominates `rtol * |x|` by
    /// four orders at `xtol = 2e-12`. They still have to be SciPy's, so they are checked directly.
    @Test("the defaults are scipy.optimize.brentq's")
    func brentDefaults() {
        #expect(RootFind.defaultXTolerance == 2e-12)
        #expect(RootFind.defaultRelativeTolerance == 4.0 * Double.ulpOfOne)
        #expect(RootFind.defaultMaxIterations == 100)
    }

    @Test("an exhausted iteration budget reports not converged")
    func iterationBudget() {
        let root = RootFind.brentq(
            lower: 0.0, upper: 2.0, xTolerance: 1e-300, maxIterations: 3
        ) { $0 * $0 - 2.0 }
        #expect(root?.converged == false)
        #expect(root?.iterations == 3)
    }

    // MARK: - The morph composition

    /// `morph_curves._developer_exhaustion_center_offset` end to end: normal CDF, Gumbel CDF, the
    /// doubling bracket search and Brent, on `kodak_portra_endura`'s fitted model.
    ///
    /// The morph itself belongs to `Model/PrintCurvesFit.swift`. This reproduces only the offset
    /// solve, which is where all four primitives meet.
    @Test("the developer-exhaustion offset matches the oracle")
    func exhaustionOffsets() throws {
        let centers = try Golden("transcendental_morph_centers")
        let amplitudes = try Golden("transcendental_morph_amplitudes")
        let sigmas = try Golden("transcendental_morph_sigmas")
        #expect(centers.shape == [3, 3])

        var offsets: [Double] = []
        for positive in [true, false] {
            for exhaustion in [0.1, 0.35, 1.0] {
                for channel in 0..<3 {
                    let row = (channel * 3)..<(channel * 3 + 3)
                    offsets.append(
                        Self.exhaustionOffset(
                            centers: Array(centers.values[row]),
                            amplitudes: Array(amplitudes.values[row]),
                            sigmas: Array(sigmas.values[row]),
                            positive: positive,
                            exhaustion: exhaustion))
                }
            }
        }
        try expectParity(
            offsets, matches: "transcendental_morph_offsets",
            maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
    }

    /// Zero exhaustion is the shipped default and must cost no root solve at all.
    @Test("zero exhaustion short-circuits to a zero offset")
    func exhaustionZero() throws {
        let centers = try Golden("transcendental_morph_centers")
        let amplitudes = try Golden("transcendental_morph_amplitudes")
        let sigmas = try Golden("transcendental_morph_sigmas")
        for exhaustion in [0.0, 1e-9] {
            let offset = Self.exhaustionOffset(
                centers: Array(centers.values[0..<3]),
                amplitudes: Array(amplitudes.values[0..<3]),
                sigmas: Array(sigmas.values[0..<3]),
                positive: false,
                exhaustion: exhaustion)
            #expect(offset == 0.0)
        }
    }

    // MARK: - Helpers

    private static func layerCDF(_ z: Double, positive: Bool, gumbelMix: Double) -> Double {
        let zs = positive ? -z : z
        let cdf = Erf.normalCDF(zs)
        guard gumbelMix > 0.0 else { return cdf }
        return (1.0 - gumbelMix) * cdf + gumbelMix * Erf.gumbelMatchedCDF(zs)
    }

    private static func channelDensity(
        at x: Double, centers: [Double], amplitudes: [Double], sigmas: [Double],
        positive: Bool, gumbelMix: [Double]
    ) -> Double {
        var total = 0.0
        for i in centers.indices {
            let z = (x - centers[i]) / sigmas[i]
            total += amplitudes[i] * layerCDF(z, positive: positive, gumbelMix: gumbelMix[i])
        }
        return total
    }

    private static func exhaustionOffset(
        centers: [Double], amplitudes: [Double], sigmas: [Double],
        positive: Bool, exhaustion: Double
    ) -> Double {
        let mix = [Double](repeating: exhaustion, count: centers.count)
        // np.allclose(mix, 0.0), whose default atol is 1e-8.
        if mix.allSatisfy({ abs($0) <= 1e-8 }) { return 0.0 }
        let zeros = [Double](repeating: 0.0, count: centers.count)
        let target = channelDensity(
            at: 0.0, centers: centers, amplitudes: amplitudes, sigmas: sigmas,
            positive: positive, gumbelMix: zeros)
        let residual: (Double) -> Double = { offset in
            channelDensity(
                at: 0.0, centers: centers.map { $0 + offset }, amplitudes: amplitudes,
                sigmas: sigmas, positive: positive, gumbelMix: mix) - target
        }
        if abs(residual(0.0)) <= 1e-12 { return 0.0 }
        return RootFind.expandingBracketRoot(residual) ?? 0.0
    }
}

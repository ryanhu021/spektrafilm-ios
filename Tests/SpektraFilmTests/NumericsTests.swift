import Testing

@testable import SpektraFilm

/// Gates the scalar primitives every stage shares against NumPy and colour-science.
///
/// These are closed form, so the gate is bit equality wherever the oracle allows it: every fixture
/// here matches at `max_abs = 0` except the two sums where NumPy's pairwise accumulation differs
/// from a sequential one, and those are gated at 1e-15. The 1e-4 render budget belongs to the
/// stages, and anything spent here is spent twice: `linspace(-3, 4, 256)` is an interpolation
/// domain, so a last-bit error in a breakpoint moves every query that lands near it.
@Suite("Numerics parity")
struct NumericsTests {

    @Test("spow is sign(a) * |a|^p")
    func spowMatchesColourScience() throws {
        let cases = try Golden("num_spow_input")
        #expect(cases.shape == [20, 2])
        let result = (0..<cases.shape[0]).map {
            spow(cases.values[$0 * 2], cases.values[$0 * 2 + 1])
        }
        try expectParity(result, matches: "num_spow", maxAbsolute: 0, rootMeanSquare: 0)
    }

    /// colour-science's `spow` maps a NaN *result* to 0 for a 0-d input only
    /// (`0 if a_p.ndim == 0 and np.isnan(a_p) else a_p`). Every engine call site passes an array, so
    /// the array behaviour is the one to match: NaN propagates. Values measured in the oracle.
    @Test("spow on the non-finite cases follows the array path")
    func spowNonFinite() {
        #expect(spow(.nan, 2).isNaN)
        #expect(spow(0, -1).isNaN)
        #expect(spow(.infinity, 2) == .infinity)
        #expect(spow(-.infinity, 2) == -.infinity)
        #expect(spow(.infinity, 0) == 1)
        #expect(spow(1, .infinity) == 1)
        #expect(spow(-1, .infinity) == -1)
        // np.sign(-0.0) is +0.0, so a negative zero base loses its sign.
        #expect(spow(-0.0, 2.2).sign == .plus)
    }

    @Test("npFmax drops NaN the way np.fmax does")
    func fmaxMatchesNumpy() throws {
        let cases = try Golden("num_fmax_input")
        let result = (0..<cases.shape[0]).map {
            npFmax(cases.values[$0 * 2], cases.values[$0 * 2 + 1])
        }
        try expectParity(result, matches: "num_fmax", maxAbsolute: 0, rootMeanSquare: 0)
    }

    /// The reason `npFmax` exists. Swift's `max` is `y >= x ? y : x`, and `NaN >= x` is false, so it
    /// returns whichever operand came first when that operand is NaN. The pipeline needs the NaN
    /// dropped: `log10Guard` would otherwise carry it into the density lookup.
    @Test("npFmax and Swift max disagree about NaN")
    func fmaxIsNotMax() {
        #expect(npFmax(.nan, 3) == 3)
        #expect(npFmax(3, .nan) == 3)
        #expect(npFmax(.nan, .nan).isNaN)
        #expect(max(Double.nan, 3).isNaN)
    }

    /// `num_fmax` carries both signed-zero orders, and the golden cannot check them: `parity()`
    /// compares `abs(a - e)`, which is 0 for `-0.0` against `+0.0`. `np.fmax` returns `+0.0` for the
    /// tie in either order, measured over array lengths 1 to 100. `Double.maximum` returns the second
    /// operand instead, so this is what stops a substitution.
    @Test("npFmax breaks the signed-zero tie the way np.fmax does")
    func fmaxTieOnSignedZero() {
        #expect(npFmax(0.0, -0.0).sign == .plus)
        #expect(npFmax(-0.0, 0.0).sign == .plus)
        #expect(npFmax(0.0, 0.0).sign == .plus)
        #expect(npFmax(-0.0, -0.0).sign == .minus)
        #expect(Double.maximum(0.0, -0.0).sign == .minus)
    }

    @Test("nanToNum substitutes NumPy's default values")
    func nanToNumMatchesNumpy() throws {
        let input = try Golden("num_nan_to_num_input")
        try expectParity(
            nanToNum(input.values), matches: "num_nan_to_num", maxAbsolute: 0, rootMeanSquare: 0)
        // The substitutions themselves, so a golden regenerated on a different platform cannot hide
        // a wrong constant.
        #expect(nanToNum(Double.infinity) == .greatestFiniteMagnitude)
        #expect(nanToNum(-Double.infinity) == -.greatestFiniteMagnitude)
        #expect(nanToNum(Double.nan) == 0)
        #expect(Double.greatestFiniteMagnitude == 1.7976931348623157e308)
        // np.nan_to_num leaves a negative zero alone, and the golden comparison cannot see that.
        #expect(nanToNum(-0.0).sign == .minus)
    }

    @Test("log10Guard floors at -10")
    func log10GuardMatchesNumpy() throws {
        let input = try Golden("num_log10_guard_input")
        try expectParity(
            log10Guard(input.values), matches: "num_log10_guard", maxAbsolute: 0,
            rootMeanSquare: 0)
        #expect(log10Guard(.nan) == log10Guard(0))
        #expect(log10Guard(-1) == log10Guard(0))
    }

    @Test("linspace reproduces numpy.linspace bit for bit")
    func linspaceMatchesNumpy() throws {
        let logExposure = linspace(-3, 4, count: 256)
        try expectParity(
            logExposure, matches: "num_linspace_log_exposure", maxAbsolute: 0, rootMeanSquare: 0)
        #expect(logExposure[255] == 4.0)

        let closed = try Golden("num_linspace_cases")
        let bounds: [(Double, Double)] = [
            (0, 1), (1, 0), (-3, 4), (2, 2), (-Double.pi, Double.pi), (0.002, 0.18),
        ]
        var rows: [Double] = []
        for (start, stop) in bounds { rows += linspace(start, stop, count: closed.shape[1]) }
        try expectParity(rows, matches: "num_linspace_cases", maxAbsolute: 0, rootMeanSquare: 0)

        let open = try Golden("num_linspace_open_cases")
        var openRows: [Double] = []
        for (start, stop) in [(-Double.pi, Double.pi), (0.0, 1.0)] {
            openRows += linspace(start, stop, count: open.shape[1], endpoint: false)
        }
        try expectParity(
            openRows, matches: "num_linspace_open_cases", maxAbsolute: 0, rootMeanSquare: 0)

        try expectParity(
            linspace(5, 9, count: 1), matches: "num_linspace_single", maxAbsolute: 0,
            rootMeanSquare: 0)
        #expect(linspace(0, 1, count: 0).isEmpty)
    }

    /// Forming the step once and multiplying is not the same as multiplying then dividing. The
    /// second form misses `linspace(-3, 4, 256)` by 8.9e-16, measured against the oracle.
    @Test("the linspace step is formed before the multiply")
    func linspaceStepOrder() throws {
        let golden = try Golden("num_linspace_log_exposure").values
        var twoRoundings = (0..<256).map { -3.0 + Double($0) * 7.0 / 255.0 }
        twoRoundings[255] = 4.0
        let worst = zip(golden, twoRoundings).map { abs($0 - $1) }.max() ?? 0
        #expect(worst > 0, "the two orderings agree here, so this test no longer proves anything")
        #expect(linspace(-3, 4, count: 256) == golden)
    }

    /// NumPy overwrites the last sample with `stop`. None of the counts in the goldens above needs
    /// it, so without this test the overwrite can be deleted and everything stays green. These three
    /// counts do need it: the accumulated last sample is 0.9999999999999999, 0.9999999999999999 and
    /// 3.999999999999999, all measured in the oracle. `linspace(0, 1, L)` is the LUT axis at seven
    /// reference call sites, and 505 of the 4095 counts in `2...4096` land here.
    @Test("linspace writes the final sample as stop exactly")
    func linspaceEndpointIsWritten() {
        #expect(linspace(0, 1, count: 50)[49] == 1.0)
        #expect(linspace(0.05, 1.0, count: 16)[15] == 1.0)
        #expect(linspace(-3, 4, count: 441)[440] == 4.0)
        #expect(49 * (1.0 / 49) != 1.0)
        // endpoint=False has no final sample to pin, so the accumulation stands.
        #expect(linspace(0, 1, count: 50, endpoint: false)[49] == 49 * (1.0 / 50))
    }

    @Test("the NaN-skipping reductions match numpy, all-NaN slices included")
    func nanReductions() throws {
        let input = try Golden("num_nan_reduce_input")
        #expect(input.shape == [12, 3])
        let flat = input.values

        try expectParity(
            nanMin(flat, channels: 3), matches: "num_nan_min_axis0", maxAbsolute: 0,
            rootMeanSquare: 0)
        try expectParity(
            nanMax(flat, channels: 3), matches: "num_nan_max_axis0", maxAbsolute: 0,
            rootMeanSquare: 0)
        try expectParity(
            nanMean(flat, channels: 3), matches: "num_nan_mean_axis0", maxAbsolute: 0,
            rootMeanSquare: 0)
        try expectParity(
            nanMeanPerSample(flat, channels: 3), matches: "num_nan_mean_axis1",
            maxAbsolute: 0, rootMeanSquare: 0)
        // 2.2e-16 off: numpy sums 36 elements pairwise, this sums them in order.
        try expectParity(
            [nanMean(flat)], matches: "num_nan_mean_flat", maxAbsolute: 1e-15,
            rootMeanSquare: 1e-15)

        // Column 2 is entirely NaN and row 3 is entirely NaN. NumPy warns and returns NaN for both,
        // and `expectParity` reports a NaN that appears or disappears as a mismatch.
        #expect(nanMin(flat, channels: 3)[2].isNaN)
        #expect(nanMax(flat, channels: 3)[2].isNaN)
        #expect(nanMean(flat, channels: 3)[2].isNaN)
        #expect(nanMeanPerSample(flat, channels: 3)[3].isNaN)
        #expect(nanMean([Double.nan, .nan]).isNaN)
    }

    /// The same reductions on the one bundled profile array with missing datasheet coverage:
    /// `fujifilm_c200.channel_density` has 8/7/7 NaN per channel and 7 all-NaN wavelengths.
    @Test("the reductions match numpy on a real profile array")
    func nanReductionsOnProfile() throws {
        let profile = try ProfileLibrary.load("fujifilm_c200")
        let density = profile.data.channelDensity
        #expect(density.count == 81 * 3)
        #expect(density.contains { $0.isNaN })

        try expectParity(
            nanMin(density, channels: 3), matches: "num_nan_min_c200_channel_density",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            nanMax(density, channels: 3), matches: "num_nan_max_c200_channel_density",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            nanMean(density, channels: 3), matches: "num_nan_mean_c200_channel_density",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            nanMeanPerSample(density, channels: 3), matches: "num_nan_mean_c200_per_wavelength",
            maxAbsolute: 0, rootMeanSquare: 0)
    }

    @Test("the buffer overloads apply the scalar rule elementwise")
    func bufferOverloads() {
        let image = ImageBuffer(
            height: 1, width: 2, channels: 3,
            values: [.nan, .infinity, -.infinity, 0, -1, 1])
        let cleaned = nanToNum(image)
        #expect(cleaned.values[0] == 0)
        #expect(cleaned.values[1] == .greatestFiniteMagnitude)
        #expect(cleaned.values[2] == -.greatestFiniteMagnitude)
        let logged = log10Guard(image)
        #expect(logged.values[0] == log10Guard(0))
        #expect(logged.values[4] == log10Guard(0))
        #expect(logged.values[5] == log10Guard(1))
    }
}

/// Gates the three boundary folds against the padding they have to match.
///
/// `numpy.pad` and `scipy.ndimage` both use the word "reflect", for opposite conventions, and four
/// call sites in the engine split across them. Each golden pads a ramp by more than one period, so a
/// map that is only right one step outside the range fails here.
@Suite("Boundary index parity")
struct BoundaryIndexTests {

    private static let counts = [1, 5, 8]

    @Test("reflectEdgeDuplicated is numpy.pad symmetric, several periods out")
    func reflectEdgeDuplicated() throws {
        for n in Self.counts {
            let pad = 3 * n + 2
            let golden = try Golden("boundary_reflect_edge_duplicated_n\(n)")
            let mapped = (0..<golden.count).map {
                Double(BoundaryIndex.reflectEdgeDuplicated($0 - pad, count: n))
            }
            try expectParity(
                mapped, matches: "boundary_reflect_edge_duplicated_n\(n)", maxAbsolute: 0,
                rootMeanSquare: 0)
        }
    }

    @Test("mirrorEdgeShared is numpy.pad reflect, several periods out")
    func mirrorEdgeShared() throws {
        for n in Self.counts {
            let pad = 3 * n + 2
            let golden = try Golden("boundary_mirror_edge_shared_n\(n)")
            let mapped = (0..<golden.count).map {
                Double(BoundaryIndex.mirrorEdgeShared($0 - pad, count: n))
            }
            try expectParity(
                mapped, matches: "boundary_mirror_edge_shared_n\(n)", maxAbsolute: 0,
                rootMeanSquare: 0)
        }
    }

    @Test("clampEdge is numpy.pad edge")
    func clampEdge() throws {
        for n in Self.counts {
            let pad = 3 * n + 2
            let golden = try Golden("boundary_clamp_edge_n\(n)")
            let mapped = (0..<golden.count).map {
                Double(BoundaryIndex.clampEdge($0 - pad, count: n))
            }
            try expectParity(
                mapped, matches: "boundary_clamp_edge_n\(n)", maxAbsolute: 0, rootMeanSquare: 0)
        }
    }

    /// `scipy.ndimage`'s names for the same three maps, which are not `numpy.pad`'s names:
    /// ndimage `reflect` is pad `symmetric`, and ndimage `mirror` is pad `reflect`. The fixture
    /// reads each map off a single-tap `correlate1d`, so `table[t][i] == map(i + t - pad)`.
    @Test("the scipy.ndimage mode names map onto the same three folds")
    func ndimageModeNames() throws {
        let n = 5
        let pad = 3 * n + 2
        let maps: [(String, (Int, Int) -> Int)] = [
            ("reflect_edge_duplicated", BoundaryIndex.reflectEdgeDuplicated),
            ("mirror_edge_shared", BoundaryIndex.mirrorEdgeShared),
            ("clamp_edge", BoundaryIndex.clampEdge),
        ]
        for (name, map) in maps {
            let golden = try Golden("boundary_ndimage_\(name)_n\(n)")
            #expect(golden.shape == [2 * pad + 1, n])
            var table: [Double] = []
            for t in 0...(2 * pad) {
                for i in 0..<n { table.append(Double(map(i + t - pad, n))) }
            }
            try expectParity(
                table, matches: "boundary_ndimage_\(name)_n\(n)", maxAbsolute: 0, rootMeanSquare: 0)
        }
    }

    /// The two folds agree inside the range and differ immediately outside it, which is the whole
    /// reason they have separate names.
    @Test("the two folds differ one step outside the range")
    func foldsDisagreeAtTheEdge() {
        #expect(BoundaryIndex.reflectEdgeDuplicated(-1, count: 5) == 0)
        #expect(BoundaryIndex.mirrorEdgeShared(-1, count: 5) == 1)
        #expect(BoundaryIndex.reflectEdgeDuplicated(5, count: 5) == 4)
        #expect(BoundaryIndex.mirrorEdgeShared(5, count: 5) == 3)
        #expect(BoundaryIndex.clampEdge(-1, count: 5) == 0)
        #expect(BoundaryIndex.clampEdge(5, count: 5) == 4)
        for i in 0..<5 {
            #expect(BoundaryIndex.reflectEdgeDuplicated(i, count: 5) == i)
            #expect(BoundaryIndex.mirrorEdgeShared(i, count: 5) == i)
            #expect(BoundaryIndex.clampEdge(i, count: 5) == i)
        }
    }

    /// A one-sample axis has no period. `numpy.pad(mode: "reflect")` returns a constant, and the FIR
    /// blur on a one-row image reaches the modulo path, where Swift's truncating `%` needs the
    /// negative fixup that is dead code in Python.
    @Test("a one-sample axis degenerates to a constant")
    func singleSampleAxis() {
        for i in -20...20 {
            #expect(BoundaryIndex.reflectEdgeDuplicated(i, count: 1) == 0)
            #expect(BoundaryIndex.mirrorEdgeShared(i, count: 1) == 0)
            #expect(BoundaryIndex.clampEdge(i, count: 1) == 0)
        }
        // The modulo path, well past the explicit ranges, on both sides.
        for n in [3, 5, 8] {
            for i in -6 * n...6 * n {
                let r = BoundaryIndex.reflectEdgeDuplicated(i, count: n)
                let m = BoundaryIndex.mirrorEdgeShared(i, count: n)
                #expect(r >= 0 && r < n)
                #expect(m >= 0 && m < n)
            }
        }
    }
}

/// Gates the spectral grid and the two contraction directions.
@Suite("Spectral shape parity")
struct SpectralShapeTests {

    @Test("the grid is colour.SpectralShape(380, 780, 5)")
    func grid() throws {
        #expect(SpectralShape.count == 81)
        #expect(SpectralShape.startNm == 380)
        #expect(SpectralShape.endNm == 780)
        #expect(SpectralShape.intervalNm == 5)
        try expectParity(
            SpectralShape.wavelengthsNm, matches: "spectralshape_wavelengths_nm", maxAbsolute: 0,
            rootMeanSquare: 0)
        try expectParity(
            SpectralShape.wavelengthsMetres, matches: "spectralshape_wavelengths_m",
            maxAbsolute: 0, rootMeanSquare: 0)
        #expect(SpectralShape.wavelengthNm(0) == 380)
        #expect(SpectralShape.wavelengthMetres(80) == 780e-9)
    }

    @Test("the contractions run along the axis they claim to")
    func contractions() throws {
        let cmfs = SpectralMatrix(ColourTables.cie1931_2deg)
        let probe = Spectrum(SpectralShape.wavelengthsNm.map { $0 * 1e-3 })

        try expectParity(
            cmfs.contracted(with: probe), matches: "spectralshape_contract_cmfs",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            cmfs.columnSums(), matches: "spectralshape_cmfs_column_sums", maxAbsolute: 0,
            rootMeanSquare: 0)
        // 1.1e-16 off: einsum reassociates the three-term row sum, this one runs left to right.
        try expectParity(
            cmfs.spectrum(weightedBy: [0.2, 0.5, 0.3]).values,
            matches: "spectralshape_cmfs_weighted_rows", maxAbsolute: 1e-15, rootMeanSquare: 1e-15)
        try expectParity(
            cmfs.multipliedPerWavelength(by: probe).values,
            matches: "spectralshape_cmfs_times_probe", maxAbsolute: 0, rootMeanSquare: 0)

        // `multiplied(by:)` has no golden of its own. Squaring the CMFs reuses one: the elementwise
        // product of the table with itself is the same as scaling each row by its own value.
        let squared = cmfs.multiplied(by: cmfs)
        for i in 0..<SpectralShape.count {
            for c in 0..<SpectralMatrix.channels {
                let x = cmfs[wavelength: i, channel: c]
                #expect(squared[wavelength: i, channel: c] == x * x)
            }
        }
    }

    @Test("element access is [wavelength][channel]")
    func elementOrder() {
        let cmfs = SpectralMatrix(ColourTables.cie1931_2deg)
        for i in [0, 1, 40, 80] {
            for c in 0..<3 {
                #expect(cmfs[wavelength: i, channel: c] == ColourTables.cie1931_2deg[i * 3 + c])
            }
        }
        // ȳ peaks at 555 nm, index 35. colour-science's 5 nm table gives 1.0000000000000004 there,
        // not 1, so this doubles as a check that the embedded table was not rounded.
        #expect(cmfs[wavelength: 35, channel: 1] == 1.0000000000000004)
        #expect(cmfs.channel(1)[35] == 1.0000000000000004)
        #expect(SpectralMatrix(channels: [cmfs.channel(0), cmfs.channel(1), cmfs.channel(2)]) == cmfs)
    }

    @Test("Spectrum arithmetic is elementwise")
    func spectrumArithmetic() {
        let ones = Spectrum.ones
        #expect(ones.sum == 81)
        #expect(ones.mean == 1)
        let ramp = Spectrum(SpectralShape.wavelengthsNm)
        #expect((ramp * ones) == ramp)
        #expect((ones * 2).values.allSatisfy { $0 == 2 })
        #expect((ramp * ramp)[10] == ramp[10] * ramp[10])
        #expect(Spectrum.zeros.sum == 0)
    }
}

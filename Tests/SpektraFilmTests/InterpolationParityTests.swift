import Testing

@testable import SpektraFilm

/// Gates both interpolators against NumPy.
///
/// The non-monotonic case is the one that matters. `compute_density_curves_before_dir_couplers`
/// interpolates over `log_exposure - couplers_amount_curves`, which for positive stocks steps
/// backwards; NumPy's answer there depends on its guess-threaded search path, so a textbook
/// bisection silently renders slide film differently.
@Suite("Interpolation parity")
struct InterpolationParityTests {

    @Test("np.interp on an ascending axis (negative stock)")
    func npInterpMonotonic() throws {
        try checkCouplerCurves(stock: "kodak_portra_400", positive: false)
    }

    @Test("np.interp on a non-monotonic axis (positive stock)")
    func npInterpNonMonotonic() throws {
        try checkCouplerCurves(stock: "fujifilm_velvia_100", positive: true)
    }

    /// Reproduces `compute_density_curves_before_dir_couplers` from committed inputs, so this
    /// exercises `npInterp` without needing the profile loader or the coupler model yet.
    private func checkCouplerCurves(stock: String, positive: Bool) throws {
        let axis = try Golden("interp_axis_\(stock)")
        let expected = try Golden("interp_npinterp_\(stock)")
        let points = axis.shape[0]

        let logExposure = try Golden("interp_log_exposure").values
        let curves = try Golden("interp_curves_\(stock)").values

        var result = [Double](repeating: 0, count: points * 3)
        for c in 0..<3 {
            let xp = (0..<points).map { axis.values[$0 * 3 + c] }
            let fp = (0..<points).map { curves[$0 * 3 + c] }
            let queried =
                positive
                ? Interpolation.npInterp(query: logExposure, xp: xp, fp: fp.map { -$0 }).map { -$0 }
                : Interpolation.npInterp(query: logExposure, xp: xp, fp: fp)
            for i in 0..<points { result[i * 3 + c] = queried[i] }
        }
        try expectParity(result, matches: "interp_npinterp_\(stock)")
        #expect(expected.shape == [points, 3])
    }

    @Test("the positive-stock axis really is non-monotonic")
    func axisIsNonMonotonic() throws {
        let velvia = try Golden("interp_axis_fujifilm_velvia_100")
        let portra = try Golden("interp_axis_kodak_portra_400")

        func minimumStep(_ g: Golden) -> Double {
            var smallest = Double.infinity
            for c in 0..<3 {
                for i in 1..<g.shape[0] {
                    smallest = min(smallest, g.values[i * 3 + c] - g.values[(i - 1) * 3 + c])
                }
            }
            return smallest
        }

        #expect(minimumStep(velvia) < 0, "Velvia's coupler axis should step backwards")
        #expect(minimumStep(portra) > 0, "Portra's coupler axis should stay ascending")
    }

    @Test("fast_interp with one axis shared by all channels")
    func fastInterpSharedAxis() throws {
        let query = try Golden("interp_fast_query").imageBuffer()
        let axis = try Golden("interp_log_exposure").values
        let curves = try Golden("interp_curves_kodak_portra_400").values
        let result = Interpolation.fastInterp(query, axis: axis, values: curves)
        try expectParity(result.values, matches: "interp_fast_shared_axis")
    }

    @Test("fast_interp with a per-channel axis (the gamma_factor case)")
    func fastInterpPerChannelAxis() throws {
        let query = try Golden("interp_fast_query").imageBuffer()
        let axis = try Golden("interp_log_exposure").values
        let curves = try Golden("interp_curves_kodak_portra_400").values
        let gamma = [0.9, 1.0, 1.15]
        var perChannel = [Double](repeating: 0, count: axis.count * 3)
        for i in axis.indices {
            for c in 0..<3 { perChannel[i * 3 + c] = axis[i] / gamma[c] }
        }
        let result = Interpolation.fastInterp(query, axis: perChannel, values: curves)
        try expectParity(result.values, matches: "interp_fast_perchannel_axis")
    }

    @Test("fast_interp clamps to the endpoint values outside the axis")
    func fastInterpClamping() {
        let axis = [0.0, 1.0, 2.0]
        let values = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0]
        let query = ImageBuffer(
            height: 1, width: 4, channels: 3,
            values: [-5, -5, -5, 0, 0, 0, 2, 2, 2, 99, 99, 99])
        let out = Interpolation.fastInterp(query, axis: axis, values: values)
        #expect(out.values[0..<3].allSatisfy { $0 == 10 })
        #expect(out.values[3..<6].allSatisfy { $0 == 10 })
        #expect(out.values[6..<9].allSatisfy { $0 == 30 })
        #expect(out.values[9..<12].allSatisfy { $0 == 30 })
    }

    /// Upstream stores a reciprocal of zero for a repeated x, which makes the weight zero and
    /// returns the lower y rather than dividing by zero.
    @Test("fast_interp survives repeated axis values")
    func fastInterpRepeatedAxis() {
        let axis = [0.0, 1.0, 1.0, 2.0]
        let values = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 9.0, 9.0, 9.0]
        let query = ImageBuffer(height: 1, width: 1, channels: 3, values: [1.0, 1.0, 1.0])
        let out = Interpolation.fastInterp(query, axis: axis, values: values)
        #expect(out.values.allSatisfy { $0.isFinite })
    }

    @Test("np.interp propagates NaN queries")
    func npInterpNaN() {
        let out = Interpolation.npInterp(
            query: [.nan, 0.5], xp: [0, 1, 2, 3, 4, 5], fp: [0, 1, 2, 3, 4, 5])
        #expect(out[0].isNaN)
        #expect(abs(out[1] - 0.5) < 1e-15)
    }

    @Test("np.interp clamps outside the range to the endpoint values")
    func npInterpClamping() {
        let xp = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        let fp = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        let out = Interpolation.npInterp(query: [-1, 0, 5, 6], xp: xp, fp: fp)
        #expect(out == [10, 10, 15, 15])
    }

    /// Below five samples NumPy linear-scans instead of using the guess machinery, and the two
    /// paths must agree where they overlap.
    @Test("np.interp short-array path agrees with the general path")
    func npInterpShortArrays() {
        for count in 2...8 {
            let xp = (0..<count).map { Double($0) }
            let fp = (0..<count).map { Double($0) * 2 }
            for q in stride(from: -0.5, through: Double(count) - 0.5, by: 0.125) {
                let got = Interpolation.npInterp(query: [q], xp: xp, fp: fp)[0]
                let clamped = min(max(q, 0), Double(count - 1))
                #expect(
                    abs(got - clamped * 2) < 1e-12,
                    "count=\(count) q=\(q) gave \(got), expected \(clamped * 2)")
            }
        }
    }
}

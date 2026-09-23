import Foundation

// MARK: - Spectral blur

/// `scipy.ndimage.gaussian_filter` along the wavelength axis, as
/// `compute_hanatos2025_tc_lut` calls it.
///
/// **The sigma is in array samples, not nanometres.** The reference's dataclass comment and the GUI
/// tooltip both say nm, but the code passes the value straight to `gaussian_filter`, whose axis has
/// a 5 nm step. `spectralGaussianBlur = 4` is therefore 20 nm of blur. Verified against the oracle.
/// Porting the docstring instead of the code would divide every non-zero blur by 5.
///
/// `truncate` is `gaussian_filter`'s default 4.0. The spatial blur in `fast_gaussian_filter.py`
/// uses 3.0. The boundary is half-sample `reflect`, which is
/// ``BoundaryIndex/reflectEdgeDuplicated(_:count:)``. Both differ from the spatial filter, so this
/// stays a separate function.
public struct HanatosSpectralBlur: Sendable {
    public let sigma: Double
    public let radius: Int
    /// `2 * radius + 1` normalised weights, centre at `radius`.
    public let weights: [Double]

    /// Returns nil for a sigma that `gaussian_filter` would skip, i.e. not strictly positive.
    public init?(sigma: Double) {
        guard sigma > 0 else { return nil }
        self.sigma = sigma
        // scipy: radius = int(truncate * sigma + 0.5) with truncate = 4.0.
        let radius = Int(4.0 * sigma + 0.5)
        self.radius = radius
        var kernel = (-radius...radius).map { x -> Double in
            let t = Double(x) / sigma
            return exp(-0.5 * t * t)
        }
        let total = kernel.reduce(0, +)
        for i in kernel.indices { kernel[i] /= total }
        self.weights = kernel
    }

    /// Correlates `values` with the kernel, reflecting at both ends. `out` must be the same length.
    public func apply(_ values: [Double], into out: inout [Double]) {
        let n = values.count
        for i in 0..<n {
            var acc = 0.0
            for k in -radius...radius {
                let index = BoundaryIndex.reflectEdgeDuplicated(i + k, count: n)
                acc += weights[k + radius] * values[index]
            }
            out[i] = acc
        }
    }
}

// MARK: - Band-pass window

/// The spectral band-pass window folded into the film sensitivities before the LUT contraction.
///
/// `eval_spectral_bandpass_window`. `compute_hanatos2025_tc_lut` never passes a model, so
/// ``erf4`` is the only reachable one; ``logiflex8`` is ported because the reference unit-tests it
/// and it is eight lines.
public enum SpectralBandpassWindow: String, Sendable, CaseIterable {
    /// `(c_uv, sigma_uv, c_ir, sigma_ir)`, one band-pass shared by all three channels.
    case erf4
    /// `(c_uv_base, sigma_uv, c_ir_base, sigma_ir, c_uv_b, c_ir_r, nu_uv, nu_ir)`.
    case logiflex8

    /// The `(81, 3)` window.
    public func evaluate(params: [Double]) throws -> SpectralMatrix {
        switch self {
        case .erf4:
            guard params.count == 4 else {
                throw SpektraError.unsupportedSetting(
                    "hanatos2025_adaptation.window_params", value: "\(params.count) values, erf4 needs 4")
            }
            return Self.erf4(params)
        case .logiflex8:
            guard params.count == 8 else {
                throw SpektraError.unsupportedSetting(
                    "hanatos2025_adaptation.window_params",
                    value: "\(params.count) values, logiflex8 needs 8")
            }
            return Self.logiflex8(params)
        }
    }

    /// `eval_erf4_spectral_bandpass`. A rising erf edge in the UV times a falling one in the IR,
    /// repeated across the three channels.
    static func erf4(_ params: [Double]) -> SpectralMatrix {
        let sqrt2 = 2.0.squareRoot()
        let (cUV, sigmaUV, cIR, sigmaIR) = (params[0], params[1], params[2], params[3])
        var values = [Double](repeating: 0, count: SpectralShape.count * 3)
        for i in 0..<SpectralShape.count {
            let lambda = SpectralShape.wavelengthsNm[i]
            let edgeUV = 0.5 * (1.0 + Erf.erf((lambda - cUV) / (sigmaUV * sqrt2)))
            let edgeIR = 0.5 * (1.0 - Erf.erf((lambda - cIR) / (sigmaIR * sqrt2)))
            let common = edgeUV * edgeIR
            for c in 0..<3 { values[i * 3 + c] = common }
        }
        return SpectralMatrix(values)
    }

    /// `eval_logiflex8_spectral_bandpass`. The blue channel gets its own UV edge and the red its own
    /// IR edge, so the three columns differ.
    static func logiflex8(_ params: [Double]) -> SpectralMatrix {
        let (cUVBase, sigmaUV, cIRBase, sigmaIR) = (params[0], params[1], params[2], params[3])
        let (cUVB, cIRR, nuUV, nuIR) = (params[4], params[5], params[6], params[7])
        let cuv = [cUVBase, cUVBase, cUVB]
        let cir = [cIRR, cIRBase, cIRBase]
        var values = [Double](repeating: 0, count: SpectralShape.count * 3)
        for i in 0..<SpectralShape.count {
            let lambda = SpectralShape.wavelengthsNm[i]
            for c in 0..<3 {
                let edgeUV = lockedLogisticRising(lambda, mu: cuv[c], sigma: sigmaUV, nu: nuUV)
                let edgeIR = 1.0 - lockedLogisticRising(lambda, mu: cir[c], sigma: sigmaIR, nu: nuIR)
                values[i * 3 + c] = edgeUV * edgeIR
            }
        }
        return SpectralMatrix(values)
    }

    /// `locked_logistic_rising`: a logistic locked through `(mu, 1/2)` with maximum slope
    /// `1 / (sigma * sqrt(2 pi))`, with `nu` bending the tails.
    ///
    /// Written in logs because the direct form overflows for the `±500` exponent the reference
    /// clamps to.
    static func lockedLogisticRising(_ x: Double, mu: Double, sigma: Double, nu: Double) -> Double {
        let nu = Swift.max(nu, 1e-5)
        // sqrt(2 * pi), the reference's _SQRT2PI.
        let s = 1.0 / (sigma * (2.0 * Double.pi).squareRoot())
        let k = (2.0 * nu * s) / (1.0 - Foundation.pow(2.0, -nu))
        let exponent = Swift.min(Swift.max(-k * (x - mu), -500.0), 500.0)
        let nuLog2 = nu * Foundation.log(2.0)
        let logQ =
            nuLog2 > 50.0
            ? nuLog2 + Foundation.log1p(-Foundation.exp(-nuLog2)) : Foundation.log(Foundation.expm1(nuLog2))
        return Foundation.exp(-(1.0 / nu) * logAddExp(0.0, logQ + exponent))
    }

    /// `np.logaddexp`, term for term.
    static func logAddExp(_ x: Double, _ y: Double) -> Double {
        if x == y { return x + Foundation.log(2.0) }
        let difference = x - y
        if difference > 0 { return x + Foundation.log1p(Foundation.exp(-difference)) }
        if difference <= 0 { return y + Foundation.log1p(Foundation.exp(difference)) }
        return difference  // NaN
    }
}

// MARK: - Log-exposure correction surface

/// The per-chromaticity log-exposure correction multiplied into the contracted LUT.
///
/// `eval_log_exposure_correction_surface`. `compute_hanatos2025_tc_lut` never passes a model, so
/// ``poly4`` is the only reachable one. ``poly4WarpXY`` needs 16 coefficients per channel, and no
/// shipped profile has them. It is ported because it is the only other fitted model.
public enum LogExposureCorrectionSurface: String, Sendable, CaseIterable {
    case poly4
    case poly4WarpXY = "poly4_warp_xy"

    /// The fit's bound, `_HANATOS2025_MAX_CORRECTION_STOPS`, so the surface is in `(-2, 2)` stops
    /// and `exp2(surface)` in `(0.25, 4)`.
    public static let maxCorrectionStops = 2.0

    /// Coefficients per channel: 15 for ``poly4``, 16 for ``poly4WarpXY`` (the extra one is the warp
    /// strength).
    public var coefficientCount: Int { self == .poly4 ? 15 : 16 }

    /// The `[gridSize][gridSize][3]` surface in stops.
    ///
    /// `gridSize` comes from the irradiance LUT's first axis, not from the contracted `raw_lut`, so
    /// the two cannot drift apart if a future LUT changes size.
    public func evaluate(
        params: [Double], illuminantXY: Chromaticity, gridSize: Int
    ) throws -> ImageBuffer {
        let stride = coefficientCount
        guard params.count == 3 * stride else {
            throw SpektraError.unsupportedSetting(
                "hanatos2025_adaptation.surface_params",
                value: "\(params.count) values, \(rawValue) needs \(3 * stride)")
        }

        let base = linspace(0, 1, count: gridSize)
        let centre = ChromaticityCoordinates.triToQuad(illuminantXY)
        var out = ImageBuffer(height: gridSize, width: gridSize, channels: 3)

        for channel in 0..<3 {
            let coefficients = Array(params[(channel * stride)..<((channel + 1) * stride)])
            let alpha = self == .poly4WarpXY ? coefficients[stride - 1] : 0
            let polynomial = self == .poly4WarpXY ? Array(coefficients[0..<(stride - 1)]) : coefficients

            for i in 0..<gridSize {
                for j in 0..<gridSize {
                    var tc = TCCoordinate(x: base[i], y: base[j])
                    if self == .poly4WarpXY {
                        let xy = ChromaticityCoordinates.quadToTri(tc)
                        tc = ChromaticityCoordinates.triToQuad(
                            Self.radialMobiusWarp(xy, centre: illuminantXY, alpha: alpha))
                    }
                    let raw = Self.poly2dDeg4(tc, params: polynomial, centre: centre)
                    out[i, j, channel] = Self.hanikaSigmoid(raw, maxValue: Self.maxCorrectionStops)
                }
            }
        }
        return out
    }

    /// `poly2d_deg4`. `params[0]` is unused. Dropping the constant term forces zero correction at
    /// the illuminant chromaticity, which keeps white in place.
    static func poly2dDeg4(_ tc: TCCoordinate, params p: [Double], centre: TCCoordinate) -> Double {
        let x = tc.x - centre.x
        let y = tc.y - centre.y
        let x2 = x * x
        let y2 = y * y
        let xy = x * y
        let x3 = x2 * x
        let y3 = y2 * y
        return p[1] * x + p[2] * y + p[3] * x2 + p[4] * y2 + p[5] * xy
            + p[6] * x3 + p[7] * y3 + p[8] * (x2 * y) + p[9] * (x * y2)
            + p[10] * (x2 * x2) + p[11] * (y2 * y2) + p[12] * (x3 * y) + p[13] * (x2 * y2)
            + p[14] * (x * y3)
    }

    /// `hanika_sigmoid`: the Jakob & Hanika algebraic sigmoid, bounded to `±maxValue`.
    @inlinable
    public static func hanikaSigmoid(_ z: Double, maxValue: Double) -> Double {
        let r = z / maxValue
        return z / (1.0 + r * r).squareRoot()
    }

    /// `_radial_mobius_warp_xy`: radial compression toward `centre`.
    static func radialMobiusWarp(
        _ xy: Chromaticity, centre: Chromaticity, alpha: Double
    ) -> Chromaticity {
        let dx = xy.x - centre.x
        let dy = xy.y - centre.y
        let scale = 1.0 / (1.0 + alpha * (dx * dx + dy * dy).squareRoot())
        return Chromaticity(x: centre.x + dx * scale, y: centre.y + dy * scale)
    }
}

// MARK: - Adaptation record

extension Hanatos2025SensitivityAdaptation {

    /// `HANATOS2025_NO_ADAPTATION`: the bare contraction, used by `rgb_to_raw_hanatos2025` when no
    /// `tc_lut` is supplied.
    ///
    /// The reference leaves `reference_illuminant` as `None` here. Nothing reads it with both flags
    /// off, so the Swift record holds D55 to fill the non-optional field. Changing it cannot change
    /// a result.
    public static let noAdaptation = Hanatos2025SensitivityAdaptation(
        windowParams: [],
        surfaceParams: [],
        spectralGaussianBlur: 0,
        referenceIlluminant: .cie("D55"),
        applyWindow: false,
        applySurface: false
    )

    /// The window, normalised so the reference illuminant's raw response is unchanged.
    ///
    /// ```
    /// normalization[m] = sum_lambda S[lambda,m] illu[lambda] window[lambda,m]
    ///                  / sum_lambda S[lambda,m] illu[lambda]
    /// ```
    ///
    /// `illu` is the **mean-normalised** SPD (`sum == 81`), which is what
    /// ``Illuminant/spectrum`` returns. It cancels between numerator and denominator only if the
    /// same SPD is used in both, so do not mix in the unnormalised ``ColourTables`` rows here.
    public func normalisedWindow(
        sensitivity: SpectralMatrix, model: SpectralBandpassWindow = .erf4
    ) throws -> SpectralMatrix {
        let window = try model.evaluate(params: windowParams)
        let normalisation = Self.windowNormalisation(
            window: window, sensitivity: sensitivity, illuminant: referenceIlluminant.spectrum)
        var out = window
        for i in 0..<SpectralShape.count {
            for c in 0..<3 { out[wavelength: i, channel: c] /= normalisation[c] }
        }
        return out
    }

    /// The three per-channel divisors of ``normalisedWindow(sensitivity:model:)``.
    public static func windowNormalisation(
        window: SpectralMatrix, sensitivity: SpectralMatrix, illuminant: [Double]
    ) -> [Double] {
        var numerator = [Double](repeating: 0, count: 3)
        var denominator = [Double](repeating: 0, count: 3)
        for i in 0..<SpectralShape.count {
            let illu = illuminant[i]
            for c in 0..<3 {
                let weighted = sensitivity[wavelength: i, channel: c] * illu
                numerator[c] += weighted * window[wavelength: i, channel: c]
                denominator[c] += weighted
            }
        }
        return (0..<3).map { numerator[$0] / denominator[$0] }
    }
}

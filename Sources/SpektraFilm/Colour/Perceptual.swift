import Foundation

// The four perceptual spaces the gamut compressors work in: Oklab, Oklab with Ottosson's rebased
// lightness, JzAzBz and CAM16-UCS, forward and inverse.
//
// Transcribed from `Tools/parity/reference/gamut_compression_reference.py`, a flat NumPy
// transcription of the colour-science call graph. The oracle reports it bit-identical for Oklab
// and JzAzBz and equal to within 1.3e-13 for CAM16-UCS. colour-science reaches these transforms
// through five layers of generic Iab/CAM plumbing with domain-range scaling. The arithmetic here
// follows the flat form term by term, in the same order.
//
// CAM16 is the exception: there the flat form is not faithful enough, and this code follows
// colour-science directly. One divergence is behavioural: the flat form selects the inverse's
// opponent axes with a single mask, which sends a NaN hue down the wrong branch. Five more are
// rounding order, each marked at its line. Together they account for the flat form's 1.3e-13.
//
// Every power is `spow`, colour-science's signed power, because `spow_enable` is on by default
// there. `pow(-0.5, 1.0/3.0)` is NaN in C, and these transforms see negative operands on ordinary
// out-of-gamut pixels.

// MARK: - Oklab

/// Björn Ottosson's Oklab, on XYZ adapted to D65.
///
/// The compressors feed it XYZ straight from the output colour space with no chromatic adaptation,
/// so a ProPhoto or ACES output is not adapted to D65 first. That is reference behaviour, not an
/// oversight.
public enum Oklab {
    static let xyzToLMS = Matrix3(rows: ColourTables.oklabXYZToLMS)
    static let lmsToXYZ = Matrix3(rows: ColourTables.oklabLMSToXYZ)
    static let lmsPrimeToLab = Matrix3(rows: ColourTables.oklabLMSPToLab)
    static let labToLMSPrime = Matrix3(rows: ColourTables.oklabLabToLMSP)

    /// XYZ (Y = 1 at white) to `(L, a, b)`.
    public static func fromXYZ(_ xyz: (Double, Double, Double)) -> (Double, Double, Double) {
        let lms = xyzToLMS.apply(xyz)
        let third = 1.0 / 3.0
        return lmsPrimeToLab.apply(
            (spow(lms.0, third), spow(lms.1, third), spow(lms.2, third)))
    }

    /// `(L, a, b)` back to XYZ.
    public static func toXYZ(_ lab: (Double, Double, Double)) -> (Double, Double, Double) {
        let p = labToLMSPrime.apply(lab)
        return lmsToXYZ.apply((spow(p.0, 3.0), spow(p.1, 3.0), spow(p.2, 3.0)))
    }

    // Ottosson 2023's rebased lightness, https://bottosson.github.io/posts/colorpicker/.
    static let k1 = 0.206
    static let k2 = 0.03
    static let k3 = (1.0 + k1) / (1.0 + k2)

    /// `Lr`, a monotonic remap of `L` whose scale tracks CIELAB `L*`. `Lr(0) = 0`, `Lr(1) = 1`.
    public static func lightnessLr(_ L: Double) -> Double {
        let t = k3 * L - k1
        return 0.5 * (t + (t * t + 4.0 * k2 * k3 * L).squareRoot())
    }

    /// Inverse of ``lightnessLr(_:)``.
    public static func lightnessFromLr(_ Lr: Double) -> Double {
        (Lr * (Lr + k1)) / (k3 * (Lr + k2))
    }
}

// MARK: - JzAzBz

/// Safdar et al. 2017 JzAzBz, on **absolute** XYZ in cd/m².
///
/// The PQ curve is ST 2084's with JzAzBz's re-optimised `m_2 = 1.7 * 2523 / 32`, which is not the
/// broadcast 78.84375. `ColourTables` stores it as a literal because `134.034375` is a different
/// double from the product.
public enum JzAzBz {
    static let xyzToLMS = Matrix3(rows: ColourTables.jzazbzXYZToLMS)
    static let lmsToXYZ = Matrix3(rows: ColourTables.jzazbzLMSToXYZ)
    static let lmsPrimeToIzAzBz = Matrix3(rows: ColourTables.jzazbzLMSPToIzAzBz)
    static let izAzBzToLMSPrime = Matrix3(rows: ColourTables.jzazbzIzAzBzToLMSP)

    /// PQ peak luminance in cd/m², fixed by the transform.
    static let peakLuminance = 10_000.0

    /// `colour.models.eotf_inverse_ST2084` with the JzAzBz `m_2`: luminance to a PQ code value.
    static func pqEncode(_ luminance: Double) -> Double {
        let yp = spow(luminance / peakLuminance, ColourTables.jzazbz_m_1)
        return spow(
            (ColourTables.jzazbz_c_1 + ColourTables.jzazbz_c_2 * yp)
                / (ColourTables.jzazbz_c_3 * yp + 1.0), ColourTables.jzazbz_m_2)
    }

    /// `colour.models.eotf_ST2084` with the JzAzBz `m_2`: a PQ code value back to luminance.
    static func pqDecode(_ code: Double) -> Double {
        let vp = spow(code, 1.0 / ColourTables.jzazbz_m_2)
        let n = npFmax(0.0, vp - ColourTables.jzazbz_c_1)
        return peakLuminance
            * spow(
                n / (ColourTables.jzazbz_c_2 - ColourTables.jzazbz_c_3 * vp),
                1.0 / ColourTables.jzazbz_m_1)
    }

    /// Absolute XYZ in cd/m² to `(Jz, az, bz)`.
    public static func fromXYZ(_ xyz: (Double, Double, Double)) -> (Double, Double, Double) {
        let b = ColourTables.jzazbz_b
        let g = ColourTables.jzazbz_g
        let xp = b * xyz.0 - (b - 1.0) * xyz.2
        let yp = g * xyz.1 - (g - 1.0) * xyz.0
        let lms = xyzToLMS.apply((xp, yp, xyz.2))
        let iab = lmsPrimeToIzAzBz.apply((pqEncode(lms.0), pqEncode(lms.1), pqEncode(lms.2)))
        let d = ColourTables.jzazbz_d
        let jz = ((1.0 + d) * iab.0) / (1.0 + d * iab.0) - ColourTables.jzazbz_d_0
        return (jz, iab.1, iab.2)
    }

    /// `(Jz, az, bz)` back to absolute XYZ in cd/m².
    public static func toXYZ(_ jab: (Double, Double, Double)) -> (Double, Double, Double) {
        let b = ColourTables.jzazbz_b
        let g = ColourTables.jzazbz_g
        let d = ColourTables.jzazbz_d
        let shifted = jab.0 + ColourTables.jzazbz_d_0
        let iz = shifted / (1.0 + d - d * shifted)
        let lmsPrime = izAzBzToLMSPrime.apply((iz, jab.1, jab.2))
        let lms = (pqDecode(lmsPrime.0), pqDecode(lmsPrime.1), pqDecode(lmsPrime.2))
        let p = lmsToXYZ.apply(lms)
        let X = (p.0 + (b - 1.0) * p.2) / b
        // The Y line uses the reconstructed X, not X'. Using X' is a silent hue shift.
        let Y = (p.1 + (g - 1.0) * X) / g
        return (X, Y, p.2)
    }
}

// MARK: - CAM16

/// The CAM16 scalars that depend only on the adapting whitepoint, `L_A`, `Y_b` and the surround.
///
/// Hoisted out of the pixel loop: the default output compressor runs the CAM16 forward and inverse
/// per pixel, and recomputing these would double its cost.
public struct CAM16ViewingConditions: Sendable, Equatable {
    /// Typical display review, per CIE 159. Fixed for the whole module.
    public static let displayReviewLuminance = 64.0
    public static let displayReviewBackground = 20.0

    let dRGB: (Double, Double, Double)
    let n: Double
    let F_L: Double
    let N_bb: Double
    let N_cb: Double
    let z: Double
    let A_w: Double
    let c: Double
    let N_c: Double
    /// `spow(1.64 - 0.29^n, 0.73)`, which both directions need.
    let chromaExponentTerm: Double

    /// `whitepointXYZ` is the adapting white at Y = 1. `colour.XYZ_to_CAM16UCS` scales both the
    /// sample and the whitepoint by 100 under its default "reference" domain-range scale, and leaves
    /// `L_A` and `Y_b` alone, so the ×100 happens here.
    ///
    /// The surround defaults are colour-science's Average: `F = 1`, `c = 0.69`, `N_c = 1`.
    public init(
        whitepointXYZ: (Double, Double, Double),
        L_A: Double = CAM16ViewingConditions.displayReviewLuminance,
        Y_b: Double = CAM16ViewingConditions.displayReviewBackground,
        F: Double = 1.0,
        c: Double = 0.69,
        N_c: Double = 1.0
    ) {
        let white100 = (whitepointXYZ.0 * 100.0, whitepointXYZ.1 * 100.0, whitepointXYZ.2 * 100.0)
        let rgbW = CAM16UCS.cat16.apply(white100)
        let D = min(max(F * (1.0 - (1.0 / 3.6) * exp((-L_A - 42.0) / 92.0)), 0.0), 1.0)
        let yW = white100.1
        let nn = Y_b / yW
        let k = 1.0 / (5.0 * L_A + 1.0)
        let k4 = k * k * k * k
        let kComplement = 1.0 - k4
        self.F_L =
            0.2 * k4 * (5.0 * L_A) + 0.1 * (kComplement * kComplement) * spow(5.0 * L_A, 1.0 / 3.0)
        self.n = nn
        self.N_bb = 0.725 * spow(1.0 / nn, 0.2)
        self.N_cb = self.N_bb
        self.z = 1.48 + nn.squareRoot()
        self.c = c
        self.N_c = N_c
        self.dRGB = (
            D * yW / rgbW.0 + 1.0 - D, D * yW / rgbW.1 + 1.0 - D, D * yW / rgbW.2 + 1.0 - D
        )
        let adaptedWhite = CAM16UCS.postAdaptation(
            (dRGB.0 * rgbW.0, dRGB.1 * rgbW.1, dRGB.2 * rgbW.2), F_L: F_L)
        self.A_w = CAM16UCS.achromaticResponse(adaptedWhite, N_bb: self.N_bb)
        self.chromaExponentTerm = spow(1.64 - pow(0.29, nn), 0.73)
    }

    public static func == (a: CAM16ViewingConditions, b: CAM16ViewingConditions) -> Bool {
        a.dRGB == b.dRGB && a.n == b.n && a.F_L == b.F_L && a.N_bb == b.N_bb && a.N_cb == b.N_cb
            && a.z == b.z && a.A_w == b.A_w && a.c == b.c && a.N_c == b.N_c
    }
}

/// CAM16-UCS: CAM16 (Li & Luo 2017) through the Luo 2006 CAM02-UCS `J'M'h'` compression.
///
/// `Jp` is on a 0 to 100 scale, and white sits at exactly 100 for every colour space, because
/// `J = 100 * (A_w / A_w)^cz` and `Jp = 170 / 1.7`.
public enum CAM16UCS {
    static let cat16 = Matrix3(rows: ColourTables.cam16)
    static let cat16Inverse = Matrix3(rows: ColourTables.cam16Inverse)

    /// The `P_2, a, b` to post-adaptation cone response matrix of the CAM16 inverse.
    ///
    /// Integer entries, with the `/ 1403` applied to the product. colour-science divides the result
    /// vector, so scaling the matrix entries instead would round differently.
    static let inverseResponse = Matrix3(
        460, 451, 288,
        460, -891, -261,
        460, -220, -6300)

    /// `(2R + G + (1/20)B - 0.305) * N_bb`. The blue term is a multiplication by the double nearest
    /// 0.05, which is not the same as dividing by 20.
    static func achromaticResponse(_ rgb: (Double, Double, Double), N_bb: Double) -> Double {
        (2.0 * rgb.0 + rgb.1 + (1.0 / 20.0) * rgb.2 - 0.305) * N_bb
    }

    static func postAdaptation(
        _ rgb: (Double, Double, Double), F_L: Double
    ) -> (Double, Double, Double) {
        func f(_ v: Double) -> Double {
            let t = spow(F_L * abs(v) / 100.0, 0.42)
            return (400.0 * signum(v) * t) / (27.13 + t) + 0.1
        }
        return (f(rgb.0), f(rgb.1), f(rgb.2))
    }

    static func postAdaptationInverse(
        _ rgb: (Double, Double, Double), F_L: Double
    ) -> (Double, Double, Double) {
        // `sign * 100 / F_L * spow(...)`, left to right, which is how colour-science groups it.
        // `sign * (100 / F_L) * spow(...)` rounds differently.
        func f(_ v: Double) -> Double {
            let a = abs(v - 0.1)
            return signum(v - 0.1) * 100.0 / F_L * spow((27.13 * a) / (400.0 - a), 1.0 / 0.42)
        }
        return (f(rgb.0), f(rgb.1), f(rgb.2))
    }

    /// XYZ at Y = 1 to `(Jp, ap, bp)`.
    ///
    /// Undefined for negative luminance: `J < 0` feeds `spow(J/100, 0.5)` and `ap`, `bp` come back
    /// NaN. The scanning stage guarantees `Y > 0`, and the compressors pass the NaN through.
    public static func forward(
        _ xyz: (Double, Double, Double), _ vc: CAM16ViewingConditions
    ) -> (Double, Double, Double) {
        let cone = cat16.apply((xyz.0 * 100.0, xyz.1 * 100.0, xyz.2 * 100.0))
        let adapted = postAdaptation(
            (vc.dRGB.0 * cone.0, vc.dRGB.1 * cone.1, vc.dRGB.2 * cone.2), F_L: vc.F_L)
        let (R, G, B) = adapted
        let a = R - 12.0 * G / 11.0 + B / 11.0
        let b = (R + G - 2.0 * B) / 9.0
        let h = degreesMod360(atan2(b, a))
        let e_t = eccentricity(h)
        let A = achromaticResponse(adapted, N_bb: vc.N_bb)
        // `colour.appearance.ciecam02.lightness_correlate` divides through `sdiv`, so a non-finite
        // `A` gives J = 0 instead of NaN. Reachable from a NaN or infinite input channel, which
        // turns the post-adaptation response into NaN.
        let J = 100.0 * spow(safeDivide(A, vc.A_w), vc.c * vc.z)
        // The constant factor multiplies the quotient; it is not folded into the numerator. The
        // quotient is an `sdiv`, so a zero denominator gives 0.
        let t =
            ((50000.0 / 13.0) * vc.N_c * vc.N_cb)
            * safeDivide(e_t * spow(a * a + b * b, 0.5), R + G + 21.0 * B / 20.0)
        let C = spow(t, 0.9) * spow(J / 100.0, 0.5) * vc.chromaExponentTerm
        let M = C * spow(vc.F_L, 0.25)
        let c1 = ColourTables.cam16ucs_c_1
        let c2 = ColourTables.cam16ucs_c_2
        let Jp = ((1.0 + 100.0 * c1) * J) / (1.0 + c1 * J)
        let Mp = (1.0 / c2) * log1p(c2 * M)
        let hr = h * (Double.pi / 180.0)
        return (Jp, Mp * cos(hr), Mp * sin(hr))
    }

    /// `(Jp, ap, bp)` back to XYZ at Y = 1.
    ///
    /// Like the reference, this routes UCS to `JMh` and derives `C` from `M` as `CAM16_to_XYZ`
    /// does. Five divisions go through colour-science's `sdiv` in its default
    /// "ignore zero conversion" mode, where a non-finite quotient becomes 0. Without it the four
    /// hue axes (0°, 90°, 180°, 270°) come back NaN.
    public static func inverse(
        _ jab: (Double, Double, Double), _ vc: CAM16ViewingConditions
    ) -> (Double, Double, Double) {
        let c1 = ColourTables.cam16ucs_c_1
        let c2 = ColourTables.cam16ucs_c_2
        let J = -jab.0 / (c1 * jab.0 - 1.0 - 100.0 * c1)
        let Mp = hypot(jab.1, jab.2)
        let h = degreesMod360(atan2(jab.2, jab.1))
        // `expm1(M' / (1 / c_2))`, as colour-science writes it. Dividing by the reciprocal is not the
        // same double as multiplying by `c_2`.
        let M = expm1(Mp / (1.0 / c2)) / c2
        let C = M / spow(vc.F_L, 0.25)
        let jSafe = max(J, 2.2204460492503131e-16)
        let t = spow(C / ((jSafe / 100.0).squareRoot() * vc.chromaExponentTerm), 1.0 / 0.9)
        let e_t = eccentricity(h)
        let A = vc.A_w * spow(J / 100.0, 1.0 / (vc.c * vc.z))
        let P_1 = safeDivide((50000.0 / 13.0) * vc.N_c * vc.N_cb * e_t, t)
        let P_2 = A / vc.N_bb + 0.305
        let P_3 = 21.0 / 20.0

        let hr = h * (Double.pi / 180.0)
        let sinH = sin(hr)
        let cosH = cos(hr)
        let cs = safeDivide(cosH, sinH)
        let sc = safeDivide(sinH, cosH)
        let P_4 = safeDivide(P_1, sinH)
        let P_5 = safeDivide(P_1, cosH)
        let nn = P_2 * (2.0 + P_3) * (460.0 / 1403.0)

        // Three-way, not if/else. colour-science seeds `a` and `b` with zeros and fills them from two
        // separate masks, `|sin| >= |cos|` and `|sin| < |cos|`, so a NaN hue satisfies neither and the
        // opponent axes stay at 0. An if/else would take the second branch instead and move the pixel
        // by whole units. Reachable: any pixel with negative luminance has a NaN hue here.
        var a = 0.0
        var b = 0.0
        if abs(sinH) >= abs(cosH) {
            b =
                nn
                / (P_4 + (2.0 + P_3) * (220.0 / 1403.0) * cs - (27.0 / 1403.0)
                    + P_3 * (6300.0 / 1403.0))
            a = b * cs
        } else if abs(sinH) < abs(cosH) {
            a =
                nn
                / (P_5 + (2.0 + P_3) * (220.0 / 1403.0)
                    - ((27.0 / 1403.0) - P_3 * (6300.0 / 1403.0)) * sc)
            b = a * sc
        }
        if t == 0 {
            a = 0
            b = 0
        }

        let product = inverseResponse.apply((P_2, a, b))
        let response = (product.0 / 1403.0, product.1 / 1403.0, product.2 / 1403.0)
        let linear = postAdaptationInverse(response, F_L: vc.F_L)
        let cone = (linear.0 / vc.dRGB.0, linear.1 / vc.dRGB.1, linear.2 / vc.dRGB.2)
        let xyz = cat16Inverse.apply(cone)
        return (xyz.0 / 100.0, xyz.1 / 100.0, xyz.2 / 100.0)
    }

    /// `0.25 * (cos(2 + h_radians) + 3.8)`. The constant 2 is in radians while `h` arrives in
    /// degrees, which is the published CIECAM02 expression; it is not `cos(radians(h + 2))`.
    static func eccentricity(_ hDegrees: Double) -> Double {
        0.25 * (cos(2.0 + hDegrees * Double.pi / 180.0) + 3.8)
    }
}

// MARK: - Shared scalar helpers

/// xy to XYZ at Y = 1, the gamut module's `_xy_to_xyz_unit_y`.
///
/// Do not replace with ``Chromaticity/XYZ``, which multiplies by the reciprocal of `y` and forms
/// `1 - (x + y)`. Both spellings differ from these divisions in the last bit, and the `C_max`
/// bisection depends on an inside/outside predicate that a last-bit difference can flip.
func xyToXYZUnitY(x: Double, y: Double) -> (Double, Double, Double) {
    let safeY = npFmax(y, 1e-12)
    return (x / safeY, 1.0, (1.0 - x - y) / safeY)
}

/// `np.sign`: zero at zero, and NaN for NaN.
@inlinable
func signum(_ x: Double) -> Double {
    if x > 0 { return 1 }
    if x < 0 { return -1 }
    return x.isNaN ? Double.nan : 0
}

/// colour-science's `sdiv` in its default "ignore zero conversion" mode: divide, then map a
/// non-finite quotient to 0. NumPy raises a warning and `nan_to_num`s the result; only the value is
/// ported.
@inlinable
func safeDivide(_ a: Double, _ b: Double) -> Double {
    let q = a / b
    return q.isFinite ? q : 0
}

/// `np.degrees(radians) % 360`, with NumPy's floor modulo so a negative angle lands in `[0, 360)`.
@inlinable
func degreesMod360(_ radians: Double) -> Double {
    let degrees = radians * (180.0 / Double.pi)
    var m = fmod(degrees, 360.0)
    if m != 0, m < 0 { m += 360.0 }
    return m
}

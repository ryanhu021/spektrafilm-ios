import Foundation

/// A light source, identified by the same string labels the profile JSONs use.
///
/// Mirrors `spektrafilm.model.illuminants.standard_illuminant`. Every spectrum is normalised so
/// its mean over the 81 samples is 1, which is what makes exposure factors comparable between
/// illuminants.
public enum Illuminant: Sendable, Hashable {
    /// A CIE standard illuminant tabulated in ``ColourTables``: `"D50"`, `"D55"` or `"D65"`.
    case cie(String)
    /// colour-science's `Incandescent` light source, profile label `"T"`.
    case incandescent
    /// colour-science's `Kinoton 75P` cinema projector lamp, profile label `"K75P"`.
    case kinoton75P
    /// Planckian radiator at the given temperature in kelvin, profile label `"BB<temp>"`.
    case blackbody(Double)
    /// 3400 K halogen through Schott KG3 heat-absorbing glass, the default enlarger lamp.
    case tungstenHalogenKG3
    /// As ``tungstenHalogenKG3``, additionally through the measured lens transmission.
    case tungstenHalogenKG3Lens

    public init(label: String) throws {
        switch label {
        case "T": self = .incandescent
        case "K75P": self = .kinoton75P
        case "TH-KG3": self = .tungstenHalogenKG3
        case "TH-KG3-L": self = .tungstenHalogenKG3Lens
        default:
            if label.hasPrefix("BB"), let t = Double(label.dropFirst(2)) {
                self = .blackbody(t)
            } else if Self.cieTables[label] != nil {
                self = .cie(label)
            } else {
                throw SpektraError.unknownIlluminant(label)
            }
        }
    }

    public var label: String {
        switch self {
        case .cie(let name): return name
        case .incandescent: return "T"
        case .kinoton75P: return "K75P"
        case .blackbody(let t): return "BB\(Int(t))"
        case .tungstenHalogenKG3: return "TH-KG3"
        case .tungstenHalogenKG3Lens: return "TH-KG3-L"
        }
    }

    private static let cieTables: [String: [Double]] = [
        "D50": ColourTables.illuminantD50,
        "D55": ColourTables.illuminantD55,
        "D65": ColourTables.illuminantD65,
    ]

    /// Relative spectral power on the engine's 81-sample grid, normalised to unit mean.
    public var spectrum: [Double] {
        var values: [Double]
        switch self {
        case .cie(let name):
            values = Self.cieTables[name] ?? []
        case .incandescent:
            values = ColourTables.lightSourceT
        case .kinoton75P:
            values = ColourTables.lightSourceK75P
        case .blackbody(let temperature):
            values = Self.blackbodySpectrum(temperature: temperature)
        case .tungstenHalogenKG3:
            values = Self.blackbodySpectrum(temperature: 3400)
            Self.multiply(&values, by: FilterTables.schottKG3)
        case .tungstenHalogenKG3Lens:
            values = Self.blackbodySpectrum(temperature: 3400)
            Self.multiply(&values, by: FilterTables.schottKG3)
            Self.multiply(&values, by: FilterTables.canonLens)
        }
        // standard_illuminant normalises by the mean, not by the peak or the integral.
        let mean = values.reduce(0, +) / Double(values.count)
        for i in values.indices { values[i] /= mean }
        return values
    }

    /// The chromaticity of this illuminant under the CIE 1931 2° observer.
    ///
    /// Matches `spectral_upsampling._illuminant_to_xy`: sum the normalised spectrum against each
    /// colour-matching function, then divide by the sum of the three.
    public var chromaticity: Chromaticity {
        let spd = spectrum
        var xyz = (0.0, 0.0, 0.0)
        for i in 0..<ColourTables.wavelengthCount {
            xyz.0 += spd[i] * ColourTables.cie1931_2deg[i * 3]
            xyz.1 += spd[i] * ColourTables.cie1931_2deg[i * 3 + 1]
            xyz.2 += spd[i] * ColourTables.cie1931_2deg[i * 3 + 2]
        }
        return Colour.XYZToxy(xyz)
    }

    /// `GenericFilter.apply` with `value == 1`, i.e. full-strength filtering.
    private static func multiply(_ values: inout [Double], by transmittance: [Double]) {
        for i in values.indices { values[i] *= transmittance[i] }
    }

    /// `colour.colorimetry.blackbody_spectral_radiance`, i.e. Planck's law with colour-science's
    /// constants, evaluated at wavelengths in metres.
    static func blackbodySpectrum(temperature: Double) -> [Double] {
        let c1 = 3.741771e-16
        let c2 = 0.014388
        let n = 1.0
        return (0..<ColourTables.wavelengthCount).map { i in
            let lambda =
                (ColourTables.wavelengthStart + Double(i) * ColourTables.wavelengthInterval) * 1e-9
            let d = 1.0 / expm1(c2 / (n * lambda * temperature))
            return ((c1 * Foundation.pow(n, -2) * Foundation.pow(lambda, -5)) / Double.pi) * d
        }
    }
}

/// The CIE 1931 2° standard observer, and the cone fundamentals the spectral-upsampling stage uses.
public enum Observer {
    public static let sampleCount = ColourTables.wavelengthCount

    /// Wavelengths in nanometres.
    public static let wavelengths: [Double] = (0..<sampleCount).map {
        ColourTables.wavelengthStart + Double($0) * ColourTables.wavelengthInterval
    }

    /// x̄ȳz̄, flattened `[wavelength][xyz]`.
    public static let cmfs = ColourTables.cie1931_2deg

    /// Stockman & Sharpe 2° cone fundamentals, flattened `[wavelength][lms]`.
    public static let lms = ColourTables.stockmanSharpe2degLMS

    /// `sum(illuminant * ȳ)`, the normalisation the scanning stage divides XYZ by.
    public static func luminanceNormalisation(illuminant: [Double]) -> Double {
        var total = 0.0
        for i in 0..<sampleCount { total += illuminant[i] * cmfs[i * 3 + 1] }
        return total
    }

    /// `contract("k,kl->l", illuminant, cmfs) / normalisation`.
    public static func illuminantXYZ(_ illuminant: [Double]) -> (Double, Double, Double) {
        var xyz = (0.0, 0.0, 0.0)
        for i in 0..<sampleCount {
            xyz.0 += illuminant[i] * cmfs[i * 3]
            xyz.1 += illuminant[i] * cmfs[i * 3 + 1]
            xyz.2 += illuminant[i] * cmfs[i * 3 + 2]
        }
        let norm = luminanceNormalisation(illuminant: illuminant)
        return (xyz.0 / norm, xyz.1 / norm, xyz.2 / norm)
    }
}

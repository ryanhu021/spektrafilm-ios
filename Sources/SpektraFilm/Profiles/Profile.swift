import Foundation

/// A film or print-paper profile: the measured data the simulation is driven by.
///
/// Mirrors `spektrafilm.profiles.io.Profile` one-to-one, including field names on the wire, so the
/// 28 upstream JSON files load unmodified and stay redistributable under their own CC BY-SA 4.0
/// terms (see docs/LICENSING.md).
///
/// Arrays are stored flattened with shapes derived from `wavelengths` and `logExposure`; see each
/// property for its logical shape. Missing datasheet coverage is `null` in JSON and NaN here, and
/// it must stay NaN, because the reference reduces with `nanmin`/`nanmax` and relies on it propagating
/// through the density lookups.
public struct Profile: Sendable, Equatable {
    public var metadata: ProfileMetadata
    public var info: ProfileInfo
    public var data: ProfileData

    public init(metadata: ProfileMetadata, info: ProfileInfo, data: ProfileData) {
        self.metadata = metadata
        self.info = info
        self.data = data
    }

    public var isPositive: Bool { info.type == .positive }
    public var isNegative: Bool { info.type == .negative }
    public var isFilm: Bool { info.support == .film }
    public var isPaper: Bool { info.support == .paper }
    public var isColour: Bool { info.channelModel == .colour }
    public var isMonochrome: Bool { info.channelModel == .blackAndWhite }

    /// The sensitivity-adaptation record the spectral-upsampling LUT needs.
    public func hanatos2025Adaptation() throws -> Hanatos2025SensitivityAdaptation {
        Hanatos2025SensitivityAdaptation(
            windowParams: data.hanatos2025AdaptationWindowParams,
            surfaceParams: data.hanatos2025AdaptationSurfaceParams,
            referenceIlluminant: try Illuminant(label: info.referenceIlluminant)
        )
    }
}

/// Redistribution terms and provenance, carried verbatim from the upstream JSON.
///
/// Decoded and preserved rather than dropped: each profile's `license` and `citation` are the
/// attribution the CC BY-SA 4.0 terms require, and the app surfaces them.
public struct ProfileMetadata: Sendable, Equatable, Codable {
    public var version: String?
    public var copyright: String?
    public var created: String?
    public var license: String?
    public var citation: String?
    public var datasource: String?
}

public struct ProfileInfo: Sendable, Equatable, Codable {
    public enum Kind: String, Sendable, Codable, CaseIterable {
        case negative
        case positive
    }

    public enum Support: String, Sendable, Codable, CaseIterable {
        case film
        case paper
    }

    public enum Stage: String, Sendable, Codable, CaseIterable {
        case filming
        case printing
    }

    public enum Use: String, Sendable, Codable, CaseIterable {
        case still
        case cine
    }

    public enum Antihalation: String, Sendable, Codable, CaseIterable {
        case strong
        case weak
        case no
    }

    public enum ChannelModel: String, Sendable, Codable, CaseIterable {
        case colour = "color"
        case blackAndWhite = "bw"
    }

    public var stock: String
    public var name: String?
    public var type: Kind
    public var support: Support
    public var stage: Stage
    public var use: Use
    public var antihalation: Antihalation
    public var targetPrint: String?
    public var channelModel: ChannelModel
    public var densitometer: String
    public var logSensitivityDensityOverMin: Double
    /// Label of the illuminant the stock's sensitivities were measured under.
    public var referenceIlluminant: String
    /// Label of the illuminant the developed result is viewed under.
    public var viewingIlluminant: String

    /// Display name, falling back to the slug.
    public var displayName: String { name ?? stock }

    enum CodingKeys: String, CodingKey {
        case stock
        case name
        case type
        case support
        case stage
        case use
        case antihalation
        case targetPrint = "target_print"
        case channelModel = "channel_model"
        case densitometer
        case logSensitivityDensityOverMin = "log_sensitivity_density_over_min"
        case referenceIlluminant = "reference_illuminant"
        case viewingIlluminant = "viewing_illuminant"
    }
}

/// Parametric model of the density curves, used by the print-curve morph.
///
/// `centers`, `amplitudes` and `sigmas` are `[channel][layer]`, flattened. The layer count comes
/// from the data rather than being fixed at 3.
public struct DensityCurvesModel: Sendable, Equatable {
    public var modelType: String
    public var channelCount: Int
    public var layerCount: Int
    public var centers: [Double]
    public var amplitudes: [Double]
    public var sigmas: [Double]

    public init(
        modelType: String = "cdfs",
        channelCount: Int = 0,
        layerCount: Int = 0,
        centers: [Double] = [],
        amplitudes: [Double] = [],
        sigmas: [Double] = []
    ) {
        self.modelType = modelType
        self.channelCount = channelCount
        self.layerCount = layerCount
        self.centers = centers
        self.amplitudes = amplitudes
        self.sigmas = sigmas
    }

    public var isEmpty: Bool { channelCount == 0 || layerCount == 0 }

    @inlinable
    public func center(channel: Int, layer: Int) -> Double { centers[channel * layerCount + layer] }
    @inlinable
    public func amplitude(channel: Int, layer: Int) -> Double {
        amplitudes[channel * layerCount + layer]
    }
    @inlinable
    public func sigma(channel: Int, layer: Int) -> Double { sigmas[channel * layerCount + layer] }
}

public struct ProfileData: Sendable, Equatable {
    /// Spectral sample points in nm; 81 values on the engine's 380–780 / 5 nm grid.
    public var wavelengths: [Double]
    /// log10 spectral sensitivity, `[wavelength][rgb]` flattened.
    public var logSensitivity: [Double]
    /// Hanatos-2025 sensitivity-adaptation window fit, 4 coefficients.
    public var hanatos2025AdaptationWindowParams: [Double]
    /// Hanatos-2025 surface fit, `[rgb][coefficient]` flattened (3 × 15).
    public var hanatos2025AdaptationSurfaceParams: [Double]
    /// Spectral absorption of each dye at unit density, `[wavelength][cmy]` flattened.
    public var channelDensity: [Double]
    /// Spectral density of the unexposed developed medium, per wavelength.
    public var baseDensity: [Double]
    /// Spectral density at a neutral mid-scale exposure, per wavelength.
    public var midscaleNeutralDensity: [Double]
    /// log10 exposure axis of the characteristic curves; 256 points spanning −3…4.
    public var logExposure: [Double]
    /// Characteristic curves, `[exposure][cmy]` flattened.
    public var densityCurves: [Double]
    /// Per-sublayer curves, `[exposure][layer][channel]` flattened.
    public var densityCurvesLayers: [Double]
    public var densityCurvesModel: DensityCurvesModel

    public var wavelengthCount: Int { wavelengths.count }
    public var exposureCount: Int { logExposure.count }

    @inlinable
    public func densityCurve(exposure i: Int, channel c: Int) -> Double {
        densityCurves[i * 3 + c]
    }

    /// Per-channel minimum over the exposure axis, ignoring NaN (`np.nanmin(..., axis=0)`).
    public var densityCurveMinima: [Double] {
        channelReduce(densityCurves) { min($0, $1) }
    }

    /// Per-channel maximum over the exposure axis, ignoring NaN (`np.nanmax(..., axis=0)`).
    public var densityCurveMaxima: [Double] {
        channelReduce(densityCurves) { max($0, $1) }
    }

    private func channelReduce(_ flat: [Double], _ combine: (Double, Double) -> Double) -> [Double] {
        var out = [Double](repeating: .nan, count: 3)
        for c in 0..<3 {
            var acc = Double.nan
            for i in 0..<exposureCount {
                let v = flat[i * 3 + c]
                if v.isNaN { continue }
                acc = acc.isNaN ? v : combine(acc, v)
            }
            out[c] = acc
        }
        return out
    }
}

/// Sensitivity adaptation applied when upsampling RGB to spectra.
///
/// The window and surface fits are per-stock corrections that keep the Hanatos-2025 reconstruction
/// consistent with the stock's measured sensitivities; `applyWindow` defaults on and
/// `applySurface` off, matching `SettingsParams`.
public struct Hanatos2025SensitivityAdaptation: Sendable, Equatable {
    public var windowParams: [Double]
    public var surfaceParams: [Double]
    /// Gaussian blur applied to the spectra, sigma in nm. 0 disables it.
    public var spectralGaussianBlur: Double = 0
    public var referenceIlluminant: Illuminant
    public var applyWindow: Bool = true
    public var applySurface: Bool = false

    public init(
        windowParams: [Double],
        surfaceParams: [Double],
        spectralGaussianBlur: Double = 0,
        referenceIlluminant: Illuminant,
        applyWindow: Bool = true,
        applySurface: Bool = false
    ) {
        self.windowParams = windowParams
        self.surfaceParams = surfaceParams
        self.spectralGaussianBlur = spectralGaussianBlur
        self.referenceIlluminant = referenceIlluminant
        self.applyWindow = applyWindow
        self.applySurface = applySurface
    }
}

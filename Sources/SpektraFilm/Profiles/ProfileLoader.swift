import Foundation

// MARK: - Decoding

/// JSON `null` means "the datasheet does not cover this wavelength", and the reference loads it as
/// NaN and keeps it: `nanmin`/`nanmax` skip it, and it propagates through the spectral products so
/// unmeasured bands contribute nothing. Decoding it as 0 would invent absorption that is not there.
private func flatten(_ values: [Double?]) -> [Double] {
    values.map { $0 ?? .nan }
}

extension ProfileData: Decodable {
    enum CodingKeys: String, CodingKey {
        case wavelengths
        case logSensitivity = "log_sensitivity"
        case hanatos2025AdaptationWindowParams = "hanatos2025_adaptation_window_params"
        case hanatos2025AdaptationSurfaceParams = "hanatos2025_adaptation_surface_params"
        case channelDensity = "channel_density"
        case baseDensity = "base_density"
        case midscaleNeutralDensity = "midscale_neutral_density"
        case logExposure = "log_exposure"
        case densityCurves = "density_curves"
        case densityCurvesLayers = "density_curves_layers"
        case densityCurvesModel = "density_curves_model"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        func vector(_ key: CodingKeys) throws -> [Double] {
            flatten(try c.decodeIfPresent([Double?].self, forKey: key) ?? [])
        }

        func matrix(_ key: CodingKeys) throws -> [Double] {
            let rows = try c.decodeIfPresent([[Double?]].self, forKey: key) ?? []
            return rows.flatMap(flatten)
        }

        func tensor(_ key: CodingKeys) throws -> [Double] {
            let outer = try c.decodeIfPresent([[[Double?]]].self, forKey: key) ?? []
            return outer.flatMap { $0.flatMap(flatten) }
        }

        wavelengths = try vector(.wavelengths)
        logSensitivity = try matrix(.logSensitivity)
        hanatos2025AdaptationWindowParams = try vector(.hanatos2025AdaptationWindowParams)
        hanatos2025AdaptationSurfaceParams = try matrix(.hanatos2025AdaptationSurfaceParams)
        channelDensity = try matrix(.channelDensity)
        baseDensity = try vector(.baseDensity)
        midscaleNeutralDensity = try vector(.midscaleNeutralDensity)
        logExposure = try vector(.logExposure)
        densityCurves = try matrix(.densityCurves)
        densityCurvesLayers = try tensor(.densityCurvesLayers)
        densityCurvesModel =
            try c.decodeIfPresent(DensityCurvesModel.self, forKey: .densityCurvesModel)
            ?? DensityCurvesModel()
    }
}

extension DensityCurvesModel: Decodable {
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case centers
        case amplitudes
        case sigmas
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cdfs"
        let centreRows = try c.decodeIfPresent([[Double?]].self, forKey: .centers) ?? []
        channelCount = centreRows.count
        layerCount = centreRows.first?.count ?? 0
        centers = centreRows.flatMap(flatten)
        amplitudes = (try c.decodeIfPresent([[Double?]].self, forKey: .amplitudes) ?? [])
            .flatMap(flatten)
        sigmas = (try c.decodeIfPresent([[Double?]].self, forKey: .sigmas) ?? []).flatMap(flatten)
    }
}

extension Profile: Decodable {
    enum CodingKeys: String, CodingKey {
        case metadata
        case info
        case data
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        metadata = try c.decodeIfPresent(ProfileMetadata.self, forKey: .metadata) ?? .init()
        info = try c.decode(ProfileInfo.self, forKey: .info)
        data = try c.decode(ProfileData.self, forKey: .data)
    }
}

// MARK: - Loading

/// Loads the bundled film and print-paper profiles.
///
/// The 28 JSON files are upstream's, shipped verbatim. Loading validates array shapes the way
/// `_validate_profile` does, because a profile whose curves and exposure axis disagree would
/// otherwise fail deep inside the pipeline with a bounds trap.
public enum ProfileLibrary {

    /// Where the profiles live inside the package bundle.
    static let subdirectory = "Resources/profiles"

    private static let cache = Cache()

    private final class Cache: @unchecked Sendable {
        private let lock = NSLock()
        private var profiles: [String: Profile] = [:]

        func value(for key: String, build: () throws -> Profile) throws -> Profile {
            lock.lock()
            if let cached = profiles[key] {
                lock.unlock()
                return cached
            }
            lock.unlock()
            let built = try build()
            lock.lock()
            profiles[key] = built
            lock.unlock()
            return built
        }
    }

    /// Slugs of every bundled profile, sorted, matching `list_profiles()`.
    public static let available: [String] = {
        guard
            let urls = Bundle.module.urls(
                forResourcesWithExtension: "json", subdirectory: subdirectory)
        else { return [] }
        return urls.map { $0.deletingPathExtension().lastPathComponent }.sorted()
    }()

    /// Profiles valid as the camera negative, i.e. `stage == .filming`, which is 20 of the 28.
    ///
    /// Keyed on `stage`, not `support`: Kodak 2383 and 2393 are cine print *films*, so they carry
    /// `support: film` while belonging on the print side. Filtering by support would offer them as
    /// camera stocks and hide two print media.
    public static var filmStocks: [Profile] {
        get throws { try available.map { try load($0) }.filter { $0.info.stage == .filming } }
    }

    /// Profiles valid as the print medium, i.e. `stage == .printing`: 6 papers plus the 2 cine
    /// print films.
    public static var printMedia: [Profile] {
        get throws { try available.map { try load($0) }.filter { $0.info.stage == .printing } }
    }

    /// Loads a profile by slug, e.g. `"kodak_portra_400"`. Cached after the first load.
    public static func load(_ stock: String) throws -> Profile {
        try cache.value(for: stock) {
            guard
                let url = Bundle.module.url(
                    forResource: stock, withExtension: "json", subdirectory: subdirectory)
            else {
                throw SpektraError.unknownProfile(stock, known: available)
            }
            let profile: Profile
            do {
                profile = try JSONDecoder().decode(Profile.self, from: try Data(contentsOf: url))
            } catch let error as DecodingError {
                throw SpektraError.malformedResource(
                    "\(subdirectory)/\(stock).json", reason: String(describing: error))
            }
            try validate(profile, stock: stock)
            return profile
        }
    }

    /// The shape checks from `_validate_profile`, plus resolvable illuminant labels.
    ///
    /// Illuminants are checked here rather than at first use so an unknown label fails profile
    /// loading instead of silently becoming D50 halfway through a render.
    static func validate(_ profile: Profile, stock: String) throws {
        let d = profile.data
        let wavelengths = d.wavelengthCount
        let exposures = d.exposureCount

        func require(_ condition: Bool, _ reason: @autoclosure () -> String) throws {
            if !condition { throw SpektraError.invalidProfile(stock, reason: reason()) }
        }

        try require(wavelengths > 0, "no wavelengths")
        try require(exposures > 0, "no log_exposure axis")
        try require(
            d.densityCurves.count == exposures * 3,
            "density_curves has \(d.densityCurves.count) values, expected \(exposures * 3)")
        try require(
            d.logSensitivity.count == wavelengths * 3,
            "log_sensitivity has \(d.logSensitivity.count) values, expected \(wavelengths * 3)")
        try require(
            d.channelDensity.count == wavelengths * 3,
            "channel_density has \(d.channelDensity.count) values, expected \(wavelengths * 3)")
        try require(
            d.baseDensity.count == wavelengths,
            "base_density has \(d.baseDensity.count) values, expected \(wavelengths)")
        try require(
            d.midscaleNeutralDensity.count == wavelengths,
            "midscale_neutral_density has \(d.midscaleNeutralDensity.count) values, expected \(wavelengths)"
        )
        try require(
            wavelengths == ColourTables.wavelengthCount,
            "profile is sampled at \(wavelengths) wavelengths, engine grid has \(ColourTables.wavelengthCount)"
        )
        _ = try Illuminant(label: profile.info.referenceIlluminant)
        _ = try Illuminant(label: profile.info.viewingIlluminant)
    }
}

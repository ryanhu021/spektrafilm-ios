import Foundation

/// Failures the engine reports to callers.
///
/// Every case is a configuration or resource problem caught before rendering starts. Numeric
/// problems inside the pipeline never throw. The reference propagates NaN and clamps, and the port
/// does the same, so a render never stops partway.
public enum SpektraError: Error, Equatable, Sendable {
    case unknownColourSpace(String, known: [String])
    case unknownIlluminant(String)
    case unknownProfile(String, known: [String])
    case invalidProfile(String, reason: String)
    case missingResource(String)
    case malformedResource(String, reason: String)
    case unsupportedSetting(String, value: String)
    case noPathToTap(from: String, to: String)
}

extension SpektraError: CustomStringConvertible {
    public var description: String {
        switch self {
        case .unknownColourSpace(let name, let known):
            return "unknown colour space '\(name)'; registered: \(known.joined(separator: ", "))"
        case .unknownIlluminant(let name):
            return
                "unknown illuminant '\(name)'; expected D50/D55/D65, T, K75P, TH-KG3, TH-KG3-L or BB<kelvin>"
        case .unknownProfile(let name, let known):
            return "unknown profile '\(name)'; \(known.count) bundled"
        case .invalidProfile(let name, let reason):
            return "profile '\(name)' is invalid: \(reason)"
        case .missingResource(let path):
            return "bundled resource missing: \(path)"
        case .malformedResource(let path, let reason):
            return "bundled resource '\(path)' is malformed: \(reason)"
        case .unsupportedSetting(let key, let value):
            return "unsupported value '\(value)' for setting '\(key)'"
        case .noPathToTap(let from, let to):
            return "no node path reaches tap '\(to)' from '\(from)'"
        }
    }
}

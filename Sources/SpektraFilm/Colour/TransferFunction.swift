import Foundation

/// The colour-component transfer functions of the supported RGB colourspaces.
///
/// Reimplemented from colour-science's definitions rather than tabulated, and gated against it by
/// `TransferFunctionParityTests`, which sweeps the domain including negatives and the piecewise
/// breakpoints. Each case documents the colour-science function it mirrors, because the details
/// that matter here are not the ones a textbook states:
///
/// - `spow` is a *signed* power (`sign(a) * |a|^p`), so the sRGB and BT.2020 segments are odd
///   functions rather than undefined below zero.
/// - `gamma_function`'s default negative handling is "Indeterminate", a plain `a ** p`, which is
///   NaN for a negative base and a fractional exponent. DCI-P3 and Adobe RGB therefore return NaN
///   below zero, and the engine must not quietly clamp that away.
/// - ROMM RGB (ProPhoto) scales through an 8-bit integer range and back, so its arithmetic is
///   `X * 16 * 255 / 255`, not `X * 16`.
public enum TransferFunction: String, Sendable, CaseIterable {
    case linear
    case sRGB
    case gamma26
    case adobeRGB
    case bt2020
    case proPhoto

    /// `sign(a) * |a|^p`, colour-science's `colour.algebra.spow`.
    @inlinable
    static func spow(_ a: Double, _ p: Double) -> Double {
        let s: Double = a < 0 ? -1 : (a > 0 ? 1 : 0)
        return s * Foundation.pow(abs(a), p)
    }

    /// Scene-linear → encoded.
    @inlinable
    public func encode(_ value: Double) -> Double {
        switch self {
        case .linear:
            return value

        case .sRGB:
            // colour.models.rgb.transfer_functions.eotf_inverse_sRGB
            return value <= 0.0031308
                ? value * 12.92
                : 1.055 * Self.spow(value, 1.0 / 2.4) - 0.055

        case .gamma26:
            // gamma_function(exponent: 1 / 2.6), "Indeterminate"
            return Foundation.pow(value, 1.0 / 2.6)

        case .adobeRGB:
            // gamma_function(exponent: 256 / 563), "Indeterminate"
            return Foundation.pow(value, 1.0 / (563.0 / 256.0))

        case .bt2020:
            // oetf_BT2020, 10-bit constants
            return Self.bt2020Beta > value
                ? value * 4.5
                : Self.bt2020Alpha * Self.spow(value, 0.45) - (Self.bt2020Alpha - 1)

        case .proPhoto:
            // cctf_encoding_ROMMRGB, bit_depth 8, out_int false
            let iMax = 255.0
            let xp =
                Self.rommEt > value
                ? value * 16 * iMax
                : Self.spow(value, 1.0 / 1.8) * iMax
            return xp / iMax
        }
    }

    /// Encoded → scene-linear.
    @inlinable
    public func decode(_ value: Double) -> Double {
        switch self {
        case .linear:
            return value

        case .sRGB:
            // eotf_sRGB compares against eotf_inverse_sRGB(0.0031308) rather than the
            // customary rounded 0.04045.
            return Self.sRGBDecodeThreshold >= value
                ? value / 12.92
                : Self.spow((value + 0.055) / 1.055, 2.4)

        case .gamma26:
            return Foundation.pow(value, 2.6)

        case .adobeRGB:
            return Foundation.pow(value, 563.0 / 256.0)

        case .bt2020:
            // oetf_inverse_BT2020 compares against oetf_BT2020(beta).
            return value < Self.bt2020DecodeThreshold
                ? value / 4.5
                : Self.spow((value + (Self.bt2020Alpha - 1)) / Self.bt2020Alpha, 1.0 / 0.45)

        case .proPhoto:
            // cctf_decoding_ROMMRGB, bit_depth 8, in_int false
            let iMax = 255.0
            let xp = value * iMax
            return xp < 16 * Self.rommEt * iMax
                ? xp / (16 * iMax)
                : Self.spow(xp / iMax, 1.8)
        }
    }

    public func encode(_ buffer: inout ImageBuffer) {
        if self == .linear { return }
        for i in buffer.values.indices { buffer.values[i] = encode(buffer.values[i]) }
    }

    public func decode(_ buffer: inout ImageBuffer) {
        if self == .linear { return }
        for i in buffer.values.indices { buffer.values[i] = decode(buffer.values[i]) }
    }

    // MARK: - Constants

    /// `CONSTANTS_BT2020.alpha(is_12_bits_system: False)`. colour-science uses the rounded
    /// Rec.2020 values, not the exact 1.09929682680944 / 0.018053968510807 pair.
    @usableFromInline static let bt2020Alpha = 1.099
    @usableFromInline static let bt2020Beta = 0.018
    /// `oetf_BT2020(0.018)` = 0.018 * 4.5.
    @usableFromInline static let bt2020DecodeThreshold = 0.018 * 4.5
    /// `eotf_inverse_sRGB(0.0031308)` = 0.0031308 * 12.92.
    @usableFromInline static let sRGBDecodeThreshold = 0.0031308 * 12.92
    /// `16 ** (1.8 / (1 - 1.8))`, the ROMM RGB linear-segment breakpoint.
    @usableFromInline static let rommEt = Foundation.pow(16.0, 1.8 / (1 - 1.8))
}

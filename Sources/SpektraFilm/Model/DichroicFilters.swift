import Foundation

/// The cyan, magenta and yellow dichroic filters of a colour enlarger head.
///
/// Ports the `DichroicFilters` class and `color_enlarger` from `model/color_filters.py`.
/// Transmittances are baked into ``FilterTables``; see the extractor for why they are not resampled
/// on device.
public struct DichroicFilters: Sendable {
    /// Which measured or modelled filter set to use.
    public enum Set: String, Sendable, CaseIterable {
        /// The analytic erf model. `color_enlarger` defaults to it, so the runtime renders with
        /// this set.
        case custom
        case thorlabs
        case edmundOptics
        case durstDigitalLight

        var table: [Double] {
            switch self {
            case .custom: return FilterTables.Dichroic.custom
            case .thorlabs: return FilterTables.Dichroic.thorlabs
            case .edmundOptics: return FilterTables.Dichroic.edmundOptics
            case .durstDigitalLight: return FilterTables.Dichroic.durstDigitalLight
            }
        }
    }

    /// Pure filter transmittances, flattened `[wavelength][cmy]`.
    public let transmittances: [Double]

    public init(_ set: Set = .custom) {
        transmittances = set.table
    }

    /// `DichroicFilters.apply`.
    ///
    /// Each filter is dialled back toward fully open by its own transmittance value, then the three
    /// are multiplied together and applied to the light. A value of 1 leaves the filter fully out of
    /// the beam, 0 puts it fully in.
    public func apply(
        to illuminant: [Double], transmittance: (c: Double, m: Double, y: Double)
    ) -> [Double] {
        let count = ColourTables.wavelengthCount
        precondition(illuminant.count == count, "illuminant must have \(count) samples")

        let dial = [transmittance.c, transmittance.m, transmittance.y]
        var out = illuminant
        for l in 0..<count {
            var total = 1.0
            for c in 0..<3 {
                total *= 1 - (1 - transmittances[l * 3 + c]) * (1 - dial[c])
            }
            out[l] *= total
        }
        return out
    }

    /// `DichroicFilters.apply_cc`.
    ///
    /// Kodak CC units are proportional to density: 100 units is 1.0 density, so a 90% cut in
    /// transmittance.
    public func applyCC(
        to illuminant: [Double], cc: (c: Double, m: Double, y: Double)
    ) -> [Double] {
        apply(
            to: illuminant,
            transmittance: (
                c: Foundation.pow(10.0, -cc.c / 100.0),
                m: Foundation.pow(10.0, -cc.m / 100.0),
                y: Foundation.pow(10.0, -cc.y / 100.0)
            ))
    }
}

/// The enlarger head: which illuminant reaches the paper, under the current filter settings.
///
/// Ports `runtime/services/filter_enlarger_source.py`. It also holds the midgray density
/// references. Those depend on the film profile, so the filming stage computes them and sets them
/// here.
public struct EnlargerService: Sendable {
    private let params: EnlargerParams
    private let filters: DichroicFilters

    /// Spectral density of an 18% grey patch through the negative, used to balance the print.
    /// Set by the filming stage.
    public var densitySpectralMidgray: ImageBuffer?
    /// The same, with the camera's exposure compensation applied. `nil` when
    /// `printExposureCompensation` is off.
    public var densitySpectralMidgrayCompensated: ImageBuffer?

    public var printExposureCompensation: Bool { params.printExposureCompensation }

    public init(_ params: EnlargerParams, filters: DichroicFilters = DichroicFilters()) {
        self.params = params
        self.filters = filters
    }

    /// The illuminant reaching the paper with the current filter shifts applied.
    public func filteredIlluminant(_ lightSource: [Double]) -> [Double] {
        filters.applyCC(
            to: lightSource,
            cc: (
                c: params.cFilterNeutral,
                m: params.mFilterNeutral + params.mFilterShift,
                y: params.yFilterNeutral + params.yFilterShift
            ))
    }

    /// The illuminant at the neutral filter positions, ignoring the shifts.
    public func neutralIlluminant(_ lightSource: [Double]) -> [Double] {
        filters.applyCC(
            to: lightSource,
            cc: (c: params.cFilterNeutral, m: params.mFilterNeutral, y: params.yFilterNeutral))
    }

    /// The illuminant used for the pre-flash exposure, which holds highlights on the print.
    public func preflashIlluminant(_ lightSource: [Double]) -> [Double] {
        filters.applyCC(
            to: lightSource,
            cc: (
                c: params.cFilterNeutral,
                m: params.mFilterNeutral + params.preflashMFilterShift,
                y: params.yFilterNeutral + params.preflashYFilterShift
            ))
    }
}

import Testing

@testable import SpektraFilm

/// Gates the colour foundations against colour-science.
///
/// These run first in spirit: the whole render sits on top of transfer functions, illuminant
/// spectra and the XYZ↔RGB matrices, so a defect here would show up as a diffuse "the colours are
/// slightly off" in the end-to-end fixtures and be miserable to localise.
@Suite("Colour parity")
struct ColourParityTests {

    // MARK: - Transfer functions

    /// Slug matching `generate_goldens.py`'s, so a renamed fixture fails loudly.
    static let transferCases: [(space: String, slug: String)] = [
        ("sRGB", "srgb"),
        ("DCI-P3", "dci_p3"),
        ("Display P3", "display_p3"),
        ("Adobe RGB (1998)", "adobe_rgb_1998"),
        ("ITU-R BT.2020", "itu_r_bt2020"),
        ("ProPhoto RGB", "prophoto_rgb"),
        ("ACES2065-1", "aces2065_1"),
    ]

    @Test("encode matches colour-science", arguments: transferCases)
    func transferEncode(space: String, slug: String) throws {
        let input = try Golden("transfer_sweep_input").values
        let transfer = try ColourSpace.named(space).transfer
        try expectParity(input.map(transfer.encode), matches: "transfer_\(slug)_encode")
    }

    @Test("decode matches colour-science", arguments: transferCases)
    func transferDecode(space: String, slug: String) throws {
        let input = try Golden("transfer_sweep_input").values
        let transfer = try ColourSpace.named(space).transfer
        try expectParity(input.map(transfer.decode), matches: "transfer_\(slug)_decode")
    }

    /// The sweep deliberately includes negatives. sRGB and BT.2020 use a signed power and stay
    /// finite; DCI-P3 and Adobe RGB use colour-science's "Indeterminate" gamma and go NaN. If that
    /// ever silently changed to a clamp, the parity test above would still pass on the positive
    /// half, so assert the shape of the behaviour directly.
    @Test("negative inputs keep colour-science's divergent handling")
    func negativeHandling() throws {
        #expect(TransferFunction.sRGB.encode(-0.5).isFinite)
        #expect(TransferFunction.bt2020.encode(-0.5).isFinite)
        #expect(TransferFunction.proPhoto.encode(-0.5).isFinite)
        #expect(TransferFunction.gamma26.encode(-0.5).isNaN)
        #expect(TransferFunction.adobeRGB.encode(-0.5).isNaN)
        #expect(TransferFunction.linear.encode(-0.5) == -0.5)
    }

    @Test("encode and decode round-trip in the positive domain")
    func roundTrip() throws {
        for transfer in TransferFunction.allCases {
            for v in stride(from: 0.0, through: 1.0, by: 0.05) {
                let back = transfer.decode(transfer.encode(v))
                #expect(
                    abs(back - v) < 1e-12,
                    "\(transfer.rawValue) round trip at \(v) gave \(back)")
            }
        }
    }

    // MARK: - Observer and illuminants

    @Test("colour-matching functions match the aligned dataset")
    func observerCMFs() throws {
        try expectParity(Observer.cmfs, matches: "observer_cmfs", maxAbsolute: 0, rootMeanSquare: 0)
    }

    @Test("cone fundamentals match the aligned dataset")
    func observerLMS() throws {
        try expectParity(Observer.lms, matches: "observer_lms", maxAbsolute: 0, rootMeanSquare: 0)
    }

    static let illuminantLabels = [
        "D50", "D55", "D65", "T", "K75P", "TH-KG3", "TH-KG3-L", "BB3400", "BB5500",
    ]

    @Test("illuminant spectra match standard_illuminant")
    func illuminantSpectra() throws {
        var flattened: [Double] = []
        for label in Self.illuminantLabels {
            flattened.append(contentsOf: try Illuminant(label: label).spectrum)
        }
        try expectParity(flattened, matches: "illuminant_spectra")
    }

    @Test("illuminant chromaticities match _illuminant_to_xy")
    func illuminantChromaticity() throws {
        var flattened: [Double] = []
        for label in Self.illuminantLabels {
            let xy = try Illuminant(label: label).chromaticity
            flattened.append(contentsOf: [xy.x, xy.y])
        }
        try expectParity(flattened, matches: "illuminant_xy")
    }

    @Test("an unknown illuminant label is rejected rather than defaulted")
    func unknownIlluminant() {
        #expect(throws: SpektraError.unknownIlluminant("D93")) {
            try Illuminant(label: "D93")
        }
    }

    // MARK: - Filters

    @Test("resampled filter transmittances match the SciPy Akima result")
    func filterTables() throws {
        try expectParity(
            FilterTables.schottKG3, matches: "filter_schott_kg3",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            FilterTables.canonLens, matches: "filter_canon_lens",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            FilterTables.Dichroic.custom, matches: "filter_dichroic_custom",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            FilterTables.Dichroic.thorlabs, matches: "filter_dichroic_thorlabs",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            FilterTables.Dichroic.edmundOptics, matches: "filter_dichroic_edmund",
            maxAbsolute: 0, rootMeanSquare: 0)
        try expectParity(
            FilterTables.Dichroic.durstDigitalLight, matches: "filter_dichroic_durst",
            maxAbsolute: 0, rootMeanSquare: 0)
    }

    // MARK: - Conversions

    static let conversionCases: [(space: String, slug: String)] = [
        ("sRGB", "srgb"),
        ("Display P3", "display_p3"),
        ("ITU-R BT.2020", "itu_r_bt2020"),
        ("ProPhoto RGB", "prophoto_rgb"),
    ]

    @Test(
        "XYZ→RGB adapts from the viewing illuminant the way colour-science does",
        arguments: conversionCases, ["D50", "K75P"]
    )
    func xyzToRGB(space: (space: String, slug: String), illuminantLabel: String) throws {
        let illuminant = try Illuminant(label: illuminantLabel)
        let illuminantXY = Colour.XYZToxy(Observer.illuminantXYZ(illuminant.spectrum))
        var buffer = try Golden("colour_xyz_input").imageBuffer()
        Colour.XYZToRGB(
            &buffer, colourspace: try ColourSpace.named(space.space), illuminant: illuminantXY)
        try expectParity(
            buffer.values,
            matches: "colour_xyz_to_rgb_\(space.slug)_\(illuminantLabel.lowercased())")
    }

    /// The scanning stage encodes its output via `RGB_to_RGB(rgb, cs, cs, ...)`, which applies
    /// `fromXYZ · (CAT · toXYZ)` before the transfer function. That product is close to identity
    /// but not identity, so skipping it would show up here.
    @Test(
        "same-space RGB_to_RGB keeps its near-identity matrix",
        arguments: [("sRGB", "srgb"), ("Display P3", "display_p3"), ("ProPhoto RGB", "prophoto_rgb")]
    )
    func sameSpaceEncode(space: String, slug: String) throws {
        let cs = try ColourSpace.named(space)
        var buffer = try Golden("colour_rgb_input").imageBuffer()
        Colour.RGBToRGB(&buffer, from: cs, to: cs, applyEncoding: true)
        try expectParity(buffer.values, matches: "colour_same_space_encode_\(slug)")
    }

    @Test("all seven colourspaces the reference GUI exposes are registered")
    func registry() throws {
        #expect(ColourSpace.all.count == 7)
        for name in Self.transferCases.map(\.space) {
            #expect(throws: Never.self) { try ColourSpace.named(name) }
        }
        #expect(throws: (any Error).self) { try ColourSpace.named("Rec. 709") }
    }

    @Test("XYZ→RGB and RGB→XYZ matrices invert each other")
    func matrixConsistency() {
        for cs in ColourSpace.all {
            let product = cs.matrixXYZToRGB * cs.matrixRGBToXYZ
            for r in 0..<3 {
                for c in 0..<3 {
                    let expected = r == c ? 1.0 : 0.0
                    #expect(
                        abs(product[r, c] - expected) < 1e-4,
                        "\(cs.name) matrices are not inverses at [\(r),\(c)]: \(product[r, c])")
                }
            }
        }
    }
}

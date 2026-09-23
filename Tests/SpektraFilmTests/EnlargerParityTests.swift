import Testing

@testable import SpektraFilm

/// Checks the colour enlarger head against the reference.
@Suite("Enlarger parity")
struct EnlargerParityTests {

    @Test("the lamp spectrum matches TH-KG3")
    func lamp() throws {
        try expectParity(
            try Illuminant(label: "TH-KG3").spectrum, matches: "enlarger_lamp_th_kg3")
    }

    @Test(
        "dichroic filtering in CC units",
        arguments: [
            ("neutral", (c: 0.0, m: 65.0, y: 55.0)),
            ("open", (c: 0.0, m: 0.0, y: 0.0)),
            ("heavy", (c: 20.0, m: 90.0, y: 80.0)),
        ]
    )
    func ccFiltering(label: String, cc: (c: Double, m: Double, y: Double)) throws {
        let lamp = try Illuminant(label: "TH-KG3").spectrum
        let filtered = DichroicFilters().applyCC(to: lamp, cc: cc)
        try expectParity(filtered, matches: "enlarger_cc_\(label)")
    }

    @Test(
        "every filter set",
        arguments: [
            (DichroicFilters.Set.custom, "custom"),
            (.thorlabs, "thorlabs"),
            (.edmundOptics, "edmund"),
            (.durstDigitalLight, "durst"),
        ]
    )
    func filterSets(set: DichroicFilters.Set, slug: String) throws {
        let lamp = try Illuminant(label: "TH-KG3").spectrum
        let filtered = DichroicFilters(set).applyCC(to: lamp, cc: (c: 0, m: 65, y: 55))
        try expectParity(filtered, matches: "enlarger_set_\(slug)")
    }

    @Test("the service applies shifts to the filtered and preflash beams only")
    func service() throws {
        var params = EnlargerParams()
        params.mFilterShift = 12.0
        params.yFilterShift = -8.0
        params.preflashMFilterShift = 5.0
        params.preflashYFilterShift = 3.0
        let service = EnlargerService(params)
        let lamp = try Illuminant(label: "TH-KG3").spectrum

        try expectParity(service.filteredIlluminant(lamp), matches: "enlarger_service_filtered")
        try expectParity(service.neutralIlluminant(lamp), matches: "enlarger_service_neutral")
        try expectParity(service.preflashIlluminant(lamp), matches: "enlarger_service_preflash")
    }

    /// CC units are density, so 100 units should cut transmittance by a factor of ten. A swapped
    /// sign or scale would still produce a plausible looking spectrum, so this checks the unit
    /// convention directly.
    @Test("100 CC units is one density unit")
    func ccIsDensity() {
        let flat = [Double](repeating: 1.0, count: ColourTables.wavelengthCount)
        let filters = DichroicFilters()
        let open = filters.applyCC(to: flat, cc: (c: 0, m: 0, y: 0))
        #expect(open.allSatisfy { abs($0 - 1.0) < 1e-12 }, "zero CC should leave the beam alone")

        // At 100 CC the dial is 0.1, so each filter's transmittance moves a tenth of the way from 1
        // toward its pure value.
        let hundred = filters.applyCC(to: flat, cc: (c: 100, m: 0, y: 0))
        for l in 0..<ColourTables.wavelengthCount {
            let pure = FilterTables.Dichroic.custom[l * 3]
            let expected = 1 - (1 - pure) * (1 - 0.1)
            #expect(abs(hundred[l] - expected) < 1e-12, "wavelength index \(l)")
        }
    }
}

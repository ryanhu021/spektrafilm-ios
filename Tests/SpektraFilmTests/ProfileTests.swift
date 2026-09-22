import Testing

@testable import SpektraFilm

/// Checks the bundled profiles load, validate, and preserve the values the pipeline reduces over.
///
/// Not parity tests against goldens — the JSON is upstream's, byte for byte, so what needs proving
/// is that the Swift decoder reads it faithfully. The NaN cases are the interesting ones: profiles
/// use JSON `null` for wavelengths the datasheet does not cover, and the reference relies on those
/// staying NaN.
@Suite("Profiles")
struct ProfileTests {

    @Test("all 28 upstream profiles are bundled and load")
    func loadsEveryProfile() throws {
        #expect(ProfileLibrary.available.count == 28)
        for stock in ProfileLibrary.available {
            let profile = try ProfileLibrary.load(stock)
            #expect(profile.info.stock == stock, "\(stock) declares stock '\(profile.info.stock)'")
        }
    }

    /// Stage, not support, is what makes a profile selectable as the negative or the print. Kodak
    /// 2383 and 2393 are cine print films: `support: film`, `stage: printing`. Keying the pickers
    /// off support would offer them as camera stocks and hide two print media.
    @Test("the library splits by stage into 20 camera stocks and 8 print media")
    func stageSplit() throws {
        let profiles = try ProfileLibrary.available.map { try ProfileLibrary.load($0) }
        #expect(try ProfileLibrary.filmStocks.count == 20)
        #expect(try ProfileLibrary.printMedia.count == 8)
        #expect(profiles.filter(\.isPositive).count == 4)

        // Support and stage genuinely disagree for exactly these two.
        let printFilms = profiles.filter { $0.isFilm && $0.info.stage == .printing }
        #expect(printFilms.map(\.info.stock).sorted() == ["kodak_2383", "kodak_2393"])
        #expect(profiles.filter(\.isPaper).count == 6)
    }

    @Test("array shapes line up with the engine's spectral and exposure grids")
    func shapes() throws {
        for stock in ProfileLibrary.available {
            let d = try ProfileLibrary.load(stock).data
            #expect(d.wavelengthCount == 81, "\(stock)")
            #expect(d.exposureCount == 256, "\(stock)")
            #expect(d.logSensitivity.count == 81 * 3, "\(stock)")
            #expect(d.channelDensity.count == 81 * 3, "\(stock)")
            #expect(d.densityCurves.count == 256 * 3, "\(stock)")
            #expect(d.densityCurvesLayers.count == 256 * 3 * 3, "\(stock)")
        }
    }

    /// Portra 400 has 22 nulls in `channel_density` and 20 in `base_density`; those are bands the
    /// datasheet leaves out, and the reference's `nanmin`/`nanmax` skip them.
    @Test("JSON null decodes to NaN, not zero")
    func nullsBecomeNaN() throws {
        let d = try ProfileLibrary.load("kodak_portra_400").data
        #expect(d.channelDensity.filter(\.isNaN).count == 22)
        #expect(d.baseDensity.filter(\.isNaN).count == 20)
        #expect(d.midscaleNeutralDensity.filter(\.isNaN).count == 20)
        #expect(d.logSensitivity.allSatisfy { !$0.isNaN })
    }

    @Test("NaN-skipping channel reductions match numpy's nanmin/nanmax")
    func channelReductions() throws {
        let d = try ProfileLibrary.load("kodak_portra_400").data
        let minima = d.densityCurveMinima
        let maxima = d.densityCurveMaxima
        for c in 0..<3 {
            var lo = Double.infinity
            var hi = -Double.infinity
            for i in 0..<d.exposureCount {
                let v = d.densityCurves[i * 3 + c]
                if v.isNaN { continue }
                lo = min(lo, v)
                hi = max(hi, v)
            }
            #expect(minima[c] == lo)
            #expect(maxima[c] == hi)
        }
    }

    @Test("every profile's illuminant labels resolve")
    func illuminantsResolve() throws {
        var referenced = Set<String>()
        var viewed = Set<String>()
        for stock in ProfileLibrary.available {
            let info = try ProfileLibrary.load(stock).info
            referenced.insert(info.referenceIlluminant)
            viewed.insert(info.viewingIlluminant)
            #expect(throws: Never.self) { try Illuminant(label: info.referenceIlluminant) }
            #expect(throws: Never.self) { try Illuminant(label: info.viewingIlluminant) }
        }
        #expect(referenced == ["D55", "TH-KG3", "T"])
        #expect(viewed == ["D50", "K75P"])
    }

    @Test("the parametric curve model decodes with its layer shape")
    func curveModel() throws {
        let model = try ProfileLibrary.load("kodak_portra_endura").data.densityCurvesModel
        #expect(model.modelType == "cdfs")
        #expect(model.channelCount == 3)
        #expect(model.layerCount > 0)
        #expect(model.centers.count == model.channelCount * model.layerCount)
        #expect(model.amplitudes.count == model.centers.count)
        #expect(model.sigmas.count == model.centers.count)
    }

    @Test("CC BY-SA attribution survives decoding")
    func metadataPreserved() throws {
        let metadata = try ProfileLibrary.load("kodak_portra_400").metadata
        #expect(metadata.license?.contains("CC BY-SA 4.0") == true)
        #expect(metadata.citation?.contains("spektrafilm") == true)
        #expect(metadata.datasource?.isEmpty == false)
    }

    @Test("an unknown slug is an error, not an empty profile")
    func unknownProfile() {
        #expect(throws: (any Error).self) { try ProfileLibrary.load("ilford_hp5") }
    }

    @Test("loading twice returns the cached profile")
    func caching() throws {
        let first = try ProfileLibrary.load("kodak_gold_200")
        let second = try ProfileLibrary.load("kodak_gold_200")
        #expect(first == second)
    }
}

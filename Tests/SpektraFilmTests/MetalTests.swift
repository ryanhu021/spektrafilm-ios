#if canImport(Metal)
import Foundation
import Testing

@testable import SpektraFilm

/// The Metal operators against the float64 CPU engine.
///
/// Apple GPUs compute in float32, so these compare at float32 tolerances rather than bit for bit.
/// The bound is on the log10 of the result, since every caller takes the log next: an error there
/// is what the downstream density lookup sees. Skipped where no Metal device exists.
@Suite("Metal", .enabled(if: MetalContext.shared != nil))
struct MetalTests {

    /// Real negative densities, from the photograph fixture, through each film's own dyes.
    /// Fujifilm C200's profile has missing wavelengths, so its NaN handling is covered too.
    @Test(
        "the spectral contraction matches the CPU",
        arguments: ["kodak_portra_400", "fujifilm_c200", "fujifilm_velvia_100"])
    func spectralContraction(film: String) throws {
        let context = try #require(MetalContext.shared)
        let profile = try ProfileLibrary.load(film)
        let cmy = try Golden("photo_portra400_endura_negative").imageBuffer()
        let illuminant = try Illuminant(label: "TH-KG3").spectrum

        let cpu = SpectralContraction.project(
            cmy: cmy, channelDensity: profile.data.channelDensity,
            baseDensity: profile.data.baseDensity, illuminant: illuminant,
            response: Observer.cmfs, scale: 0.01)
        let gpu = try MetalSpectralContraction.project(
            context, cmy: cmy, channelDensity: profile.data.channelDensity,
            baseDensity: profile.data.baseDensity, illuminant: illuminant,
            response: Observer.cmfs, scale: 0.01)

        var worst = 0.0
        for (a, b) in zip(cpu.values, gpu.values) {
            worst = max(worst, abs(log10Guard(a) - log10Guard(b)))
        }
        // Measured at 2.8e-7 to 3.8e-7 across these three films on an M4 Pro, 1/260 of the 1e-4
        // parity tolerance.
        #expect(worst < 2e-6, "worst log10 difference \(worst)")
    }

    /// A band boundary must not show: a frame larger than one band gives the same result as the
    /// per-pixel CPU path at every pixel, including the ones either side of the seam.
    @Test("banding is invisible")
    func banding() throws {
        let context = try #require(MetalContext.shared)
        let profile = try ProfileLibrary.load("kodak_portra_400")
        let illuminant = try Illuminant(label: "D55").spectrum
        let pixels = MetalSpectralContraction.bandPixels + 1000
        var values = [Double](repeating: 0, count: pixels * 3)
        for i in values.indices { values[i] = Double((i * 7919) % 1000) / 400.0 }
        let cmy = ImageBuffer(height: 1, width: pixels, channels: 3, values: values)

        let cpu = SpectralContraction.project(
            cmy: cmy, channelDensity: profile.data.channelDensity,
            baseDensity: profile.data.baseDensity, illuminant: illuminant,
            response: Observer.cmfs)
        let gpu = try MetalSpectralContraction.project(
            context, cmy: cmy, channelDensity: profile.data.channelDensity,
            baseDensity: profile.data.baseDensity, illuminant: illuminant,
            response: Observer.cmfs)
        var worst = 0.0
        for (a, b) in zip(cpu.values, gpu.values) {
            worst = max(worst, abs(log10Guard(a) - log10Guard(b)))
        }
        #expect(worst < 2e-6, "worst log10 difference \(worst)")
    }

    /// The GPU contraction in a whole render, compared with the oracle at the same parity tolerance
    /// as the CPU engine. Measured at 4.7e-7 to 6.6e-7 max_abs and 2.6e-8 to 9.9e-8 RMS across the
    /// four combinations on an M4 Pro, well inside 1e-4 and 1e-5.
    @Test(
        "a Metal render of the photograph meets the parity tolerance",
        arguments: PhotographParityTests.combinations)
    func photograph(combination: (label: String, film: String, paper: String)) throws {
        var params = try RuntimePhotoParams.make(film: combination.film, print: combination.paper)
        params.camera.autoExposure = false
        params.debug.lutMode = true
        let input = try Golden("photo_input").imageBuffer()
        let out = try Simulator(params, backend: .metal).process(input)
        try expectParity(out.values, matches: "photo_\(combination.label)")
    }

    /// Every tap across the four film, paper and output-space cases, through the GPU pipeline,
    /// against the oracle at the CPU's tolerance.
    @Test(
        "every tap of a Metal render matches the oracle",
        arguments: EndToEndTests.cases, EndToEndTests.taps)
    func tapParity(
        testCase: (label: String, film: String, paper: String, output: String),
        tap: (tap: Tap, slug: String)
    ) throws {
        var params = try RuntimePhotoParams.make(film: testCase.film, print: testCase.paper)
        params.camera.autoExposure = false
        params.debug.lutMode = true
        params.io.outputColourSpace = testCase.output
        let input = try Golden("pipeline_ramp_input").imageBuffer()
        let out = try Simulator(params, backend: .metal).process(input, inject: nil, collect: tap.tap)
        try expectParity(out.values, matches: "pipeline_\(testCase.label)_\(tap.slug)")
    }

    @Test(
        "scanning the negative on Metal matches the oracle",
        arguments: [(Tap.logExposureFilm, "log_e_film"), (.cmyFilm, "cmy_film"), (.rgbOut, "rgb_out")])
    func scanFilmParity(tap: Tap, slug: String) throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true
        params.io.scanFilm = true
        let input = try Golden("pipeline_ramp_input").imageBuffer()
        let out = try Simulator(params, backend: .metal).process(input, inject: nil, collect: tap)
        try expectParity(out.values, matches: "pipeline_scanfilm_portra400_\(slug)")
    }

    /// The spatial effects and glare are off in lut_mode, so the oracle fixtures do not reach
    /// them. This renders with everything but grain on, and compares the GPU against the CPU.
    @Test("a full Metal render with the spatial effects matches the CPU")
    func spatialRender() throws {
        var params = try RuntimePhotoParams.make(film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.filmRender.grain.active = false
        let input = try Golden("photo_input").imageBuffer()
        let cpu = try Simulator(params, backend: .cpu).process(input)
        let gpu = try Simulator(params, backend: .metal).process(input)
        // Measured at 2.1e-6 max_abs and 8.2e-8 RMS.
        let report = parity(gpu.values, cpu.values)
        #expect(report.maxAbsolute < 1e-4 && report.rootMeanSquare < 1e-5, "\(report)")
    }

    /// Grain on. The GPU's Poisson draws equal the CPU's at nearly every counter, so the two
    /// renders should agree at nearly every pixel, and on average everywhere.
    @Test("a Metal render with grain tracks the CPU")
    func grainRender() throws {
        var params = try RuntimePhotoParams.make(film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        let input = try Golden("photo_input").imageBuffer()
        let cpu = try Simulator(params, backend: .cpu).process(input)
        let gpu = try Simulator(params, backend: .metal).process(input)
        let differences = zip(gpu.values, cpu.values).map { abs($0 - $1) }
        let far = differences.filter { $0 > 1e-4 }.count
        let meanDifference =
            zip(gpu.values, cpu.values).map { $0 - $1 }.reduce(0, +)
            / Double(cpu.values.count)
        // Measured: 11 of 31 500 values differ by over 1e-4, and the mean difference is 9.4e-8.
        #expect(Double(far) / Double(cpu.values.count) < 0.01)
        #expect(abs(meanDifference) < 1e-5)
    }
}
#endif

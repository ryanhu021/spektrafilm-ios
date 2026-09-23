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
}
#endif

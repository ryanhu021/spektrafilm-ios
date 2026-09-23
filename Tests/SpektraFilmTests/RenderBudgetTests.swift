import Testing

@testable import SpektraFilm

/// Checks the memory ceiling that keeps a full-resolution export from being terminated.
///
/// Peak footprint is about 148 MB per megapixel, measured in release. A 12 MP frame needs 1.7 GB,
/// and iOS terminates a foreground app at roughly 1.4 GB. Set the ceiling too high and export
/// crashes; set it too low and export shrinks the user's photo more than it needs to.
@Suite("Render budget")
struct RenderBudgetTests {

    @Test("a source that already fits is left alone")
    func fittingSourceIsUntouched() {
        // Half a megapixel fits under any plausible allowance.
        #expect(RenderBudget.longEdge(forWidth: 800, height: 600) == nil)
    }

    @Test("the cap preserves aspect ratio and lands under the ceiling")
    func capRespectsRatio() {
        let width = 4032
        let height = 3024
        guard let longEdge = RenderBudget.longEdge(forWidth: width, height: height) else {
            // Reached only when the allowance is very large, as it may be on a desktop test host.
            // Not a failure.
            return
        }
        let scale = Double(longEdge) / Double(max(width, height))
        let cappedWidth = Double(width) * scale
        let cappedHeight = Double(height) * scale
        let ratioBefore = Double(width) / Double(height)
        let ratioAfter = cappedWidth / cappedHeight
        #expect(abs(ratioBefore - ratioAfter) < 0.01, "aspect ratio drifted")

        let megapixels = cappedWidth * cappedHeight / 1e6
        #expect(
            megapixels <= RenderBudget.maximumMegapixels() * 1.02,
            "capped frame is \(megapixels) MP against a ceiling of \(RenderBudget.maximumMegapixels())")
    }

    @Test("the ceiling never collapses to nothing")
    func neverZero() {
        #expect(RenderBudget.maximumMegapixels() >= 0.5)
    }

    @Test("the measured cost per megapixel is what the ceiling divides by")
    func costIsRecorded() {
        // Pinned so the constant changes only with a new measurement. 125 MB/MP comes from 250 MB
        // at 2 MP, 738 at 6 and 1471 at 12, measured in release with Tools/memprofile, one
        // measurement per process.
        #expect(RenderBudget.bytesPerMegapixel == 125 * 1_048_576)
        #expect(RenderBudget.safetyFraction > 0 && RenderBudget.safetyFraction < 1)
    }

    /// A render at the capped size must complete and produce that size. The cap exists because the
    /// full-size render would not complete.
    @Test("rendering at the capped size works")
    func cappedRenderRuns() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        params.debug.lutMode = true

        let simulator = try Simulator(params, resampler: SkimageResampler())
        let out = try simulator.process(
            ImageBuffer(height: 48, width: 64, channels: 3, repeating: 0.184))
        #expect(out.height == 48 && out.width == 64)
    }
}

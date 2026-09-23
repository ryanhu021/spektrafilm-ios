import Foundation
import Testing

@testable import SpektraFilm

/// The engine splits its per-pixel, per-row and per-column loops across cores. These check that the
/// split never shows in the output.
@Suite("Parallel")
struct ParallelTests {

    @Test("chunks cover every index exactly once", arguments: [0, 1, 7, 2047, 2048, 100_003])
    func coverage(count: Int) {
        let hits = UnsafeMutableBufferPointer<Int>.allocate(capacity: max(1, count))
        defer { hits.deallocate() }
        hits.initialize(repeating: 0)
        let base = hits.baseAddress!
        Parallel.forEachChunk(of: count) { range in
            for i in range { base[i] += 1 }
        }
        #expect(hits.prefix(count).allSatisfy { $0 == 1 })
    }

    /// Grain, glare, halation and the blurs all on, so every parallel loop runs. The frame is large
    /// enough that each loop splits into several chunks.
    @Test("a parallel render is bit-identical to a serial one")
    func renderMatchesSerial() throws {
        var params = try RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params.camera.autoExposure = false
        let simulator = try Simulator(params, resampler: SkimageResampler())

        let height = 192
        let width = 256
        var values = [Double](repeating: 0, count: height * width * 3)
        for i in values.indices { values[i] = Double((i * 7919) % 1000) / 1000.0 * 1.2 }
        let image = ImageBuffer(height: height, width: width, channels: 3, values: values)

        let parallel = try simulator.process(image)
        let serial = try Parallel.$forceSerial.withValue(true) { try simulator.process(image) }

        #expect(parallel.values.count == serial.values.count)
        let differing = zip(parallel.values, serial.values).filter {
            $0.bitPattern != $1.bitPattern
        }.count
        #expect(differing == 0, "\(differing) of \(serial.values.count) values differ")
    }
}

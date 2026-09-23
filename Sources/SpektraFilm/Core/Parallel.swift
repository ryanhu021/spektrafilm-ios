import Foundation

/// Splits a loop of independent iterations across the CPU cores.
///
/// Callers use this only where each output element is computed from inputs the split cannot
/// change: a pixel, a row or a column. The result is then bit-identical to a sequential loop at
/// any core count, which the parity fixtures and `ParallelTests` both check. Reductions over a
/// whole frame stay sequential.
public enum Parallel {

    /// Runs every split loop on the calling thread when bound to `true`, for tests that compare a
    /// parallel render against a sequential one.
    @TaskLocal public static var forceSerial = false

    /// Below this many iterations per chunk, dispatch costs more than it saves.
    static let minimumIterationsPerChunk = 2048

    /// Calls `body` once per contiguous chunk of `0..<count`, concurrently.
    ///
    /// The chunks cover the range exactly once, in no particular order. `cost` is the work per
    /// iteration relative to one pixel, so a loop over rows passes the row width and gets fewer,
    /// larger chunks.
    @usableFromInline
    static func forEachChunk(
        of count: Int, cost: Int = 1, _ body: (Range<Int>) -> Void
    ) {
        guard count > 0 else { return }
        let perChunk = max(1, minimumIterationsPerChunk / max(1, cost))
        // Four chunks per core, so the performance and efficiency cores finish close together.
        let cores = ProcessInfo.processInfo.activeProcessorCount
        let chunks = min(cores * 4, count / perChunk)
        if chunks <= 1 || forceSerial {
            body(0..<count)
            return
        }
        // The body is not `Sendable`: it captures the caller's raw buffer pointers. That is sound
        // because each chunk writes a disjoint range and `concurrentPerform` returns only after
        // every chunk has finished, so nothing escapes the call.
        withoutActuallyEscaping(body) { body in
            nonisolated(unsafe) let body = body
            DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
                body((count * chunk / chunks)..<(count * (chunk + 1) / chunks))
            }
        }
    }
}

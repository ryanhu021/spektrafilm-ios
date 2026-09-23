import Foundation

/// A named boundary in the pipeline where data can be injected or collected.
///
/// Taps identify data, not steps. `runtime/topology.py` uses the lowercase string values as wire
/// identifiers, so they are preserved here for cross-referencing the reference.
public enum Tap: String, Sendable, CaseIterable {
    case rgbIn = "rgb_in"
    case rgbPre = "rgb_pre"
    case logExposureFilm = "log_e_film"
    case cmyFilm = "cmy_film"
    case logExposurePrint = "log_e_print"
    case cmyPrint = "cmy_print"
    case rgbOut = "rgb_out"
}

/// One unit of computation in the pipeline.
///
/// A node declares the taps it reads and writes. The dispatcher fires it once every tap it reads is
/// present in the state.
///
/// Not `Sendable`: a node closes over the stage that runs it, and the stages hold caches and the
/// references the print balance needs. One render is single-threaded, and parallelism lives inside
/// the operators.
public struct Node {
    public let reads: [Tap]
    public let writes: [Tap]
    public let label: String
    public let run: ([ImageBuffer]) throws -> [ImageBuffer]

    public init(
        reads: [Tap],
        writes: [Tap],
        label: String,
        run: @escaping ([ImageBuffer]) throws -> [ImageBuffer]
    ) {
        self.reads = reads
        self.writes = writes
        self.label = label
        self.run = run
    }

    /// Convenience for the single-input, single-output case, which is every node the pipeline
    /// currently declares.
    public init(
        from input: Tap,
        to output: Tap,
        label: String,
        run: @escaping (ImageBuffer) throws -> ImageBuffer
    ) {
        self.init(reads: [input], writes: [output], label: label) { inputs in
            [try run(inputs[0])]
        }
    }
}

/// Walks `topology` in declared order, firing every node whose reads are satisfied, and returns the
/// value at `collect` as soon as it appears.
///
/// `onFire` reports each node's wall-clock time, which the pipeline uses for its timing breakdown.
public func runTopology(
    _ topology: [Node],
    inject: Tap,
    collect: Tap,
    image: ImageBuffer,
    onFire: ((Node, TimeInterval) -> Void)? = nil
) throws -> ImageBuffer {
    var state: [Tap: ImageBuffer] = [inject: image]

    for node in topology {
        guard node.reads.allSatisfy({ state[$0] != nil }) else { continue }

        let start = DispatchTime.now().uptimeNanoseconds
        let outputs = try node.run(node.reads.map { state[$0]! })
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9

        guard outputs.count == node.writes.count else {
            throw SpektraError.unsupportedSetting(
                "node \(node.label)",
                value: "declares \(node.writes.count) writes but produced \(outputs.count)")
        }
        for (tap, value) in zip(node.writes, outputs) { state[tap] = value }
        onFire?(node, elapsed)

        if let result = state[collect] { return result }
    }

    throw SpektraError.noPathToTap(from: inject.rawValue, to: collect.rawValue)
}

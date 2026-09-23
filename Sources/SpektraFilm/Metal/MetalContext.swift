#if canImport(Metal)
import Foundation
import Metal

/// The GPU device, its command queue and the compiled kernels, shared by every Metal operator.
///
/// Apple GPUs have no float64, so the Metal operators compute in float32. The CPU engine stays the
/// float64 reference that the parity fixtures check; each Metal operator is tested against it at a
/// float32 tolerance.
///
/// The kernels are compiled at first use from ``MetalKernels/source``. A `.metal` file would need
/// Xcode's build system, and the package also builds with `swift build`. Fast math is off, so
/// `exp10`, `log10` and `pow` keep full float32 precision.
public final class MetalContext: @unchecked Sendable {

    /// `nil` where no Metal device exists, such as some CI virtual machines.
    public static let shared: MetalContext? = try? MetalContext()

    let device: MTLDevice
    let queue: MTLCommandQueue
    private let library: MTLLibrary
    private let lock = NSLock()
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var staging: (input: MTLBuffer, output: MTLBuffer)?
    /// Serialises the operators that share the staging buffers.
    let stagingLock = NSLock()

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
        else { throw SpektraError.missingResource("Metal device") }
        let options = MTLCompileOptions()
        if #available(iOS 18, macOS 15, *) {
            options.mathMode = .safe
        } else {
            options.fastMathEnabled = false
        }
        self.device = device
        self.queue = queue
        library = try device.makeLibrary(source: MetalKernels.source, options: options)
    }

    /// The pipeline for a kernel function, built once.
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock()
        defer { lock.unlock() }
        if let cached = pipelines[name] { return cached }
        guard let function = library.makeFunction(name: name) else {
            throw SpektraError.missingResource("Metal kernel \(name)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }

    /// A shared-storage buffer, which CPU and GPU address directly on Apple silicon.
    func buffer(floats count: Int) throws -> MTLBuffer {
        guard
            let buffer = device.makeBuffer(
                length: max(1, count) * MemoryLayout<Float>.stride, options: .storageModeShared)
        else { throw SpektraError.missingResource("Metal buffer of \(count) floats") }
        return buffer
    }

    /// An input and an output buffer of at least `floats` each, reused across calls. Allocating a
    /// fresh pair per call leaves the driver holding the old ones for a while, and the footprint
    /// grows. Hold ``stagingLock`` while using them.
    func stagingBuffers(floats: Int) throws -> (input: MTLBuffer, output: MTLBuffer) {
        let bytes = max(1, floats) * MemoryLayout<Float>.stride
        if let staging, staging.input.length >= bytes { return staging }
        let pair = (try buffer(floats: floats), try buffer(floats: floats))
        staging = pair
        return pair
    }

    /// Runs `kernel` over `count` threads and waits for it.
    ///
    /// Command buffers are autoreleased, so the pool is drained here. A thread without a run loop
    /// would otherwise keep every command buffer, and the buffers it references, alive.
    func dispatch(
        _ kernel: String, count: Int, _ encode: (MTLComputeCommandEncoder) -> Void
    ) throws {
        let pipeline = try pipeline(kernel)
        try autoreleasepool {
            guard let commands = queue.makeCommandBuffer(),
                let encoder = commands.makeComputeCommandEncoder()
            else { throw SpektraError.missingResource("Metal command buffer") }
            encoder.setComputePipelineState(pipeline)
            encode(encoder)
            let width = pipeline.threadExecutionWidth
            encoder.dispatchThreads(
                MTLSize(width: count, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            encoder.endEncoding()
            commands.commit()
            commands.waitUntilCompleted()
            if let error = commands.error { throw error }
        }
    }
}
#endif

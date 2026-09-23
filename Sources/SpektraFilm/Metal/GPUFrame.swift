#if canImport(Metal)
import Foundation
import Metal

/// A float32 image in shared memory, which the CPU and GPU address directly on Apple silicon.
///
/// Interleaved like ``ImageBuffer``: `[height][width][channels]`. Half the size of the float64
/// frame, which is what lets the Metal pipeline hold a larger export under the same memory limit.
final class GPUFrame: @unchecked Sendable {
    let buffer: MTLBuffer
    let height: Int
    let width: Int
    let channels: Int

    var pixelCount: Int { height * width }
    var count: Int { height * width * channels }
    var floats: UnsafeMutablePointer<Float> {
        buffer.contents().bindMemory(to: Float.self, capacity: max(1, count))
    }

    init(_ context: MetalContext, height: Int, width: Int, channels: Int) throws {
        self.height = height
        self.width = width
        self.channels = channels
        buffer = try context.buffer(floats: height * width * channels)
    }

    /// Converts an ``ImageBuffer`` to float32, across cores.
    convenience init(_ context: MetalContext, uploading image: ImageBuffer) throws {
        try self.init(context, height: image.height, width: image.width, channels: image.channels)
        let destination = floats
        image.values.withUnsafeBufferPointer { source in
            let s = source.baseAddress!
            Parallel.forEachChunk(of: image.count) { range in
                for i in range { destination[i] = Float(s[i]) }
            }
        }
    }

    /// Converts back to a float64 ``ImageBuffer``, across cores.
    func download() -> ImageBuffer {
        var out = ImageBuffer(height: height, width: width, channels: channels)
        let source = floats
        out.values.withUnsafeMutableBufferPointer { destination in
            let d = destination.baseAddress!
            Parallel.forEachChunk(of: count) { range in
                for i in range { d[i] = Double(source[i]) }
            }
        }
        return out
    }
}
#endif

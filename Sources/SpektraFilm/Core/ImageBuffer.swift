import Foundation

/// A dense, row-major `[height][width][channels]` buffer of `Double`.
///
/// Every stage passes these where the reference passes NumPy arrays. `channels` is 3 for RGB, CMY
/// density and XYZ, and 81 for spectral quantities on the engine's 380 to 780 nm grid at 5 nm
/// steps.
///
/// `Double` throughout. The reference computes in float64, and the spectral contractions sum 81
/// terms per pixel per channel, which would drift past the 1e-4 gate in Float.
public struct ImageBuffer: Sendable, Equatable {
    public let height: Int
    public let width: Int
    public let channels: Int
    /// `count == height * width * channels`. Index of channel `c` at `(y, x)` is
    /// `(y * width + x) * channels + c`.
    public var values: [Double]

    public init(height: Int, width: Int, channels: Int, values: [Double]) {
        precondition(
            values.count == height * width * channels,
            "ImageBuffer shape \(height)x\(width)x\(channels) does not match \(values.count) values"
        )
        self.height = height
        self.width = width
        self.channels = channels
        self.values = values
    }

    public init(height: Int, width: Int, channels: Int, repeating value: Double = 0) {
        self.init(
            height: height, width: width, channels: channels,
            values: [Double](repeating: value, count: height * width * channels)
        )
    }

    public var pixelCount: Int { height * width }
    public var count: Int { values.count }

    /// A buffer with the same spatial extent but a different channel count.
    public func reshapedChannels(_ channels: Int, repeating value: Double = 0) -> ImageBuffer {
        ImageBuffer(height: height, width: width, channels: channels, repeating: value)
    }

    @inlinable
    public subscript(y: Int, x: Int, c: Int) -> Double {
        get { values[(y * width + x) * channels + c] }
        set { values[(y * width + x) * channels + c] = newValue }
    }

    /// The `channels` values at `(y, x)`, as a slice into the backing storage.
    @inlinable
    public func pixel(y: Int, x: Int) -> ArraySlice<Double> {
        let base = (y * width + x) * channels
        return values[base..<(base + channels)]
    }

    // MARK: - Row blocks

    /// A contiguous band of rows, `[rows][width][channels]`.
    ///
    /// Spectral stages expand 3 channels to 81, so a whole frame will not fit in one allocation:
    /// 12 MP would need 7.8 GB. Per-pixel stages work band by band through
    /// ``mapPerPixel(bandRows:channelsOut:transform:)``.
    public func rowBand(from start: Int, count rows: Int) -> ImageBuffer {
        let lo = start * width * channels
        let hi = (start + rows) * width * channels
        return ImageBuffer(
            height: rows, width: width, channels: channels, values: Array(values[lo..<hi])
        )
    }

    public mutating func writeRowBand(_ band: ImageBuffer, at start: Int) {
        precondition(band.width == width && band.channels == channels, "row band shape mismatch")
        let lo = start * width * channels
        values.replaceSubrange(lo..<(lo + band.count), with: band.values)
    }

    /// Applies a per-pixel transform in row bands to keep intermediates small.
    ///
    /// Only valid when each output pixel depends on the input pixel at the same position. The
    /// spectral maps (density to radiance to XYZ, film CMY to print exposure) qualify. Blur,
    /// grain and coupler diffusion do not.
    public func mapPerPixel(
        bandRows: Int,
        channelsOut: Int,
        transform: (ImageBuffer) throws -> ImageBuffer
    ) rethrows -> ImageBuffer {
        var out = ImageBuffer(height: height, width: width, channels: channelsOut)
        var start = 0
        while start < height {
            let rows = min(bandRows, height - start)
            let result = try transform(rowBand(from: start, count: rows))
            precondition(
                result.height == rows && result.width == width && result.channels == channelsOut,
                "per-pixel transform changed shape"
            )
            out.writeRowBand(result, at: start)
            start += rows
        }
        return out
    }

    // MARK: - In-place transforms

    /// Applies `transform` to every value, reusing the storage.
    ///
    /// Peak memory caps export size on iOS. The peak is set by how many full-frame buffers are live
    /// at once, not by how many are allocated over the run. A `map` that returns a new buffer holds
    /// two frames while it runs; this holds one.
    ///
    /// Runs across cores. `transform` is `@Sendable` so it cannot capture mutable state, which is
    /// what makes the split invisible in the result.
    @inlinable
    public mutating func transformInPlace(_ transform: @Sendable (Double) -> Double) {
        values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            Parallel.forEachChunk(of: buffer.count) { range in
                for i in range { p[i] = transform(p[i]) }
            }
        }
    }

    /// Applies `transform` per channel, reusing the storage. The closure receives the channel index.
    @inlinable
    public mutating func transformInPlace(_ transform: @Sendable (Int, Double) -> Double) {
        let channels = self.channels
        values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            Parallel.forEachChunk(of: buffer.count) { range in
                for i in range { p[i] = transform(i % channels, p[i]) }
            }
        }
    }

    /// Scales every value by a per-channel factor, reusing the storage.
    @inlinable
    public mutating func scaleInPlace(_ factors: (Double, Double, Double)) {
        precondition(channels == 3, "per-channel scale expects 3 channels")
        let f = [factors.0, factors.1, factors.2]
        values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for i in stride(from: 0, to: buffer.count, by: 3) {
                p[i] *= f[0]
                p[i + 1] *= f[1]
                p[i + 2] *= f[2]
            }
        }
    }

    /// `self = combine(self, other)` elementwise, reusing this buffer's storage.
    ///
    /// Lets a two-buffer blend finish with two live frames instead of three.
    @inlinable
    public mutating func combineInPlace(
        with other: ImageBuffer, _ combine: @Sendable (Double, Double) -> Double
    ) {
        precondition(
            other.count == count, "combineInPlace needs matching shapes")
        other.values.withUnsafeBufferPointer { source in
            guard let s = source.baseAddress else { return }
            values.withUnsafeMutableBufferPointer { buffer in
                guard let p = buffer.baseAddress else { return }
                Parallel.forEachChunk(of: buffer.count) { range in
                    for i in range { p[i] = combine(p[i], s[i]) }
                }
            }
        }
    }

    /// Row-band height that keeps a `channels`-deep intermediate under `budgetBytes`.
    ///
    /// Always at least one row, even when a single row exceeds the budget. A very wide panorama
    /// gets one-row bands.
    public static func bandRows(
        width: Int, channels: Int, budgetBytes: Int = 64 << 20
    ) -> Int {
        let perRow = max(1, width * channels * MemoryLayout<Double>.size)
        return max(1, budgetBytes / perRow)
    }
}

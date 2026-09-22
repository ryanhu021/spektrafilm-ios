import Foundation

/// A dense, row-major `[height][width][channels]` buffer of `Double`.
///
/// This is the single interchange type inside the engine, standing in for the NumPy arrays the
/// reference implementation passes between stages. `channels` is 3 for RGB, CMY density and XYZ,
/// and 81 for spectral quantities sampled on the engine's 380–780 nm / 5 nm grid.
///
/// `Double` throughout, deliberately: the reference computes in float64 and the parity gate is
/// tight enough (1e-4 absolute) that accumulating in Float would fail it in the spectral
/// contractions, where 81 terms are summed per pixel per channel.
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
    /// Spectral stages expand 3 channels to 81, so a full-resolution frame cannot be converted in
    /// one allocation — 12 MP would need 7.8 GB. Per-pixel stages therefore run band by band; see
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

    /// Applies a per-pixel transform in row bands, so intermediates stay bounded.
    ///
    /// Only valid for transforms where each output pixel depends solely on the co-located input
    /// pixel — the spectral maps (`density → radiance → XYZ`, `film CMY → print exposure`) qualify,
    /// spatial effects (blur, grain, coupler diffusion) do not.
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

    /// Row-band height that keeps a `channels`-deep intermediate under `budgetBytes`.
    ///
    /// At least one row always, even if a single row exceeds the budget: correctness before
    /// footprint, and a one-row band is what a very wide panorama will get.
    public static func bandRows(
        width: Int, channels: Int, budgetBytes: Int = 64 << 20
    ) -> Int {
        let perRow = max(1, width * channels * MemoryLayout<Double>.size)
        return max(1, budgetBytes / perRow)
    }
}

import Foundation

/// Bakes input gamut compression into the per-film tc LUT.
///
/// Ports `gamut_compression.remap_tc_lut_for_compression`. The runtime lookup is
/// RGB, then CIE xy, then tc, then a sample of the LUT. Compressing per pixel would put a
/// ray-polygon intersection in that path, so the LUT is remapped once when it is built:
///
///     new[xy] = old[compress(xy)]
///
/// Each cell of the new LUT decodes its own tc index back to xy, compresses that, re-encodes to tc,
/// and samples the original LUT there. The per-pixel path stays compression-agnostic.
///
/// This implements ``InputGamutCompressionBake``. It lives with the compressor because the
/// compression is the part that varies. The sampling is fixed.
public struct TCLUTCompressionBake: InputGamutCompressionBake {
    public init() {}

    public func remap(
        tcLUT: ImageBuffer, referenceIlluminantXY: Chromaticity, spec: InputGamutCompressSpec
    ) throws -> ImageBuffer {
        guard spec.active else { return tcLUT }
        try spec.validate()
        precondition(tcLUT.channels == 3, "a tc LUT has 3 channels, got \(tcLUT.channels)")

        let height = tcLUT.height
        let width = tcLUT.width
        var out = ImageBuffer(height: height, width: width, channels: 3)

        out.values.withUnsafeMutableBufferPointer { buffer in
            let o = buffer.baseAddress!
            Parallel.forEachChunk(of: height, cost: width * 64) { rows in
                for i in rows {
                    for j in 0..<width {
                        // Cell (i, j) sits at tc = (i / (H-1), j / (W-1)).
                        let tc = TCCoordinate(
                            x: Double(i) / Double(height - 1), y: Double(j) / Double(width - 1))
                        let xy = ChromaticityCoordinates.quadToTri(tc)
                        let compressed = InputGamutCompression.compress(
                            (x: xy.x, y: xy.y), white: referenceIlluminantXY, spec: spec)
                        let source = ChromaticityCoordinates.triToQuad(
                            x: compressed.x, y: compressed.y)

                        let sampled = Self.bilinearNearestEdge(
                            tcLUT,
                            row: source.x * Double(height - 1),
                            column: source.y * Double(width - 1))
                        for c in 0..<3 { o[(i * width + j) * 3 + c] = sampled[c] }
                    }
                }
            }
        }
        return out
    }

    /// `scipy.ndimage.map_coordinates(order: 1, mode: "nearest")`.
    ///
    /// Out-of-grid coordinates take the closest edge value. Wrapping or extrapolating would push raw
    /// RGB past the LUT's own range, which the reference notes and avoids.
    static func bilinearNearestEdge(
        _ lut: ImageBuffer, row: Double, column: Double
    ) -> [Double] {
        let maxRow = lut.height - 1
        let maxColumn = lut.width - 1
        let r = min(max(row, 0), Double(maxRow))
        let c = min(max(column, 0), Double(maxColumn))

        let r0 = min(Int(r.rounded(.down)), maxRow)
        let c0 = min(Int(c.rounded(.down)), maxColumn)
        let r1 = min(r0 + 1, maxRow)
        let c1 = min(c0 + 1, maxColumn)
        let fr = r - Double(r0)
        let fc = c - Double(c0)

        var out = [Double](repeating: 0, count: 3)
        for channel in 0..<3 {
            let v00 = lut[r0, c0, channel]
            let v01 = lut[r0, c1, channel]
            let v10 = lut[r1, c0, channel]
            let v11 = lut[r1, c1, channel]
            let top = v00 + (v01 - v00) * fc
            let bottom = v10 + (v11 - v10) * fc
            out[channel] = top + (bottom - top) * fr
        }
        return out
    }
}

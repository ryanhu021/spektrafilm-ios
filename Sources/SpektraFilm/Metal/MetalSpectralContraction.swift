#if canImport(Metal)
import Foundation
import Metal

/// ``SpectralContraction/project(cmy:channelDensity:baseDensity:illuminant:response:scale:)`` on
/// the GPU, in float32.
///
/// Works in bands of ``bandPixels`` so the float32 staging buffers stay small. Converting a whole
/// 12 MP frame at once would add two 144 MB buffers on top of the float64 input and output, and
/// peak memory is what caps export size.
enum MetalSpectralContraction {
    static let bandPixels = 1 << 20

    /// The contraction of a whole ``GPUFrame``, for the Metal pipeline, where the frame is already
    /// float32 and no staging is needed.
    static func project(
        _ context: MetalContext,
        frame cmy: GPUFrame,
        channelDensity: [Double],
        baseDensity: [Double],
        illuminant: [Double],
        response: [Double],
        scale: Double = 1.0
    ) throws -> GPUFrame {
        var table = Self.table(channelDensity, baseDensity, illuminant, response)
        let out = try GPUFrame(context, height: cmy.height, width: cmy.width, channels: 3)
        var count = UInt32(cmy.pixelCount)
        var wavelengths = UInt32(ColourTables.wavelengthCount)
        var floatScale = Float(scale)
        try context.dispatch("spectral_project", count: cmy.pixelCount) { encoder in
            encoder.setBuffer(cmy.buffer, offset: 0, index: 0)
            encoder.setBytes(&table, length: table.count * 4, index: 1)
            encoder.setBuffer(out.buffer, offset: 0, index: 2)
            encoder.setBytes(&count, length: 4, index: 3)
            encoder.setBytes(&wavelengths, length: 4, index: 4)
            encoder.setBytes(&floatScale, length: 4, index: 5)
        }
        return out
    }

    /// Per wavelength: the three dye weights, the base density, the illuminant and the three
    /// response values.
    static func table(
        _ channelDensity: [Double], _ baseDensity: [Double], _ illuminant: [Double],
        _ response: [Double]
    ) -> [Float] {
        let wavelengths = ColourTables.wavelengthCount
        precondition(channelDensity.count == wavelengths * 3 && baseDensity.count == wavelengths)
        precondition(illuminant.count == wavelengths && response.count == wavelengths * 3)
        var table = [Float](repeating: 0, count: wavelengths * 8)
        for l in 0..<wavelengths {
            table[l * 8] = Float(channelDensity[l * 3])
            table[l * 8 + 1] = Float(channelDensity[l * 3 + 1])
            table[l * 8 + 2] = Float(channelDensity[l * 3 + 2])
            table[l * 8 + 3] = Float(baseDensity[l])
            table[l * 8 + 4] = Float(illuminant[l])
            table[l * 8 + 5] = Float(response[l * 3])
            table[l * 8 + 6] = Float(response[l * 3 + 1])
            table[l * 8 + 7] = Float(response[l * 3 + 2])
        }
        return table
    }

    static func project(
        _ context: MetalContext,
        cmy: ImageBuffer,
        channelDensity: [Double],
        baseDensity: [Double],
        illuminant: [Double],
        response: [Double],
        scale: Double = 1.0
    ) throws -> ImageBuffer {
        let wavelengths = ColourTables.wavelengthCount
        precondition(cmy.channels == 3, "density buffer must have 3 channels")
        let table = Self.table(channelDensity, baseDensity, illuminant, response)

        let band = min(bandPixels, max(1, cmy.pixelCount))
        context.stagingLock.lock()
        defer { context.stagingLock.unlock() }
        let (input, output) = try context.stagingBuffers(floats: band * 3)
        var out = ImageBuffer(height: cmy.height, width: cmy.width, channels: 3)
        var wavelengthCount = UInt32(wavelengths)
        var floatScale = Float(scale)

        try cmy.values.withUnsafeBufferPointer { source in
            try out.values.withUnsafeMutableBufferPointer { destination in
                let src = source.baseAddress!
                let dst = destination.baseAddress!
                let staged = input.contents().bindMemory(to: Float.self, capacity: band * 3)
                let result = output.contents().bindMemory(to: Float.self, capacity: band * 3)

                var start = 0
                while start < cmy.pixelCount {
                    let pixels = min(band, cmy.pixelCount - start)
                    let offset = start * 3
                    Parallel.forEachChunk(of: pixels * 3) { range in
                        for i in range { staged[i] = Float(src[offset + i]) }
                    }
                    var count = UInt32(pixels)
                    try context.dispatch("spectral_project", count: pixels) { encoder in
                        encoder.setBuffer(input, offset: 0, index: 0)
                        table.withUnsafeBytes {
                            encoder.setBytes($0.baseAddress!, length: $0.count, index: 1)
                        }
                        encoder.setBuffer(output, offset: 0, index: 2)
                        encoder.setBytes(&count, length: MemoryLayout<UInt32>.size, index: 3)
                        encoder.setBytes(
                            &wavelengthCount, length: MemoryLayout<UInt32>.size, index: 4)
                        encoder.setBytes(&floatScale, length: MemoryLayout<Float>.size, index: 5)
                    }
                    Parallel.forEachChunk(of: pixels * 3) { range in
                        for i in range { dst[offset + i] = Double(result[i]) }
                    }
                    start += pixels
                }
            }
        }
        return out
    }
}
#endif

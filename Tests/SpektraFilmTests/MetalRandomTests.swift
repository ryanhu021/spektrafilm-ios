#if canImport(Metal)
import Foundation
import Metal
import Testing

@testable import SpektraFilm

/// The GPU Philox against ``Philox4x32``, and its normal sampler against the normal moments.
@Suite("Metal random", .enabled(if: MetalContext.shared != nil))
struct MetalRandomTests {

    /// Six words per counter crosses a block boundary, so the block increment is covered too.
    @Test("Philox words match the CPU bit for bit")
    func words() throws {
        let c = try #require(MetalContext.shared)
        let n = 4096
        let seed: UInt64 = 0x1234_5678_9ABC_DEF0
        let out = try #require(c.device.makeBuffer(length: n * 6 * 4, options: .storageModeShared))
        var s = seed
        var channel: UInt32 = 2
        var sublayer: UInt32 = 7
        var count = UInt32(n)
        try c.dispatch("philox_words", count: n) { e in
            e.setBuffer(out, offset: 0, index: 0)
            e.setBytes(&s, length: 8, index: 1)
            e.setBytes(&channel, length: 4, index: 2)
            e.setBytes(&sublayer, length: 4, index: 3)
            e.setBytes(&count, length: 4, index: 4)
        }
        let gpu = out.contents().bindMemory(to: UInt32.self, capacity: n * 6)
        var mismatches = 0
        for i in 0..<n {
            var cpu = Philox4x32(key: PhiloxKey(seed: seed, channel: 2, sublayer: 7))
            cpu.reset(counter: UInt64(i))
            for w in 0..<6 where cpu.nextBits() != gpu[i * 6 + w] { mismatches += 1 }
        }
        #expect(mismatches == 0)
    }

    /// Mean 0, variance 1, skewness 0 and excess kurtosis 0, each within five standard errors.
    @Test("the normal sampler has normal moments")
    func normals() throws {
        let c = try #require(MetalContext.shared)
        let n = 1 << 20
        let frame = try GPUFrame(c, height: 1, width: n, channels: 1)
        var seed: UInt64 = 42
        var count = UInt32(n)
        try c.dispatch("philox_normals", count: n) { e in
            e.setBuffer(frame.buffer, offset: 0, index: 0)
            e.setBytes(&seed, length: 8, index: 1)
            e.setBytes(&count, length: 4, index: 2)
        }
        let x = frame.download().values
        let mean = x.reduce(0, +) / Double(n)
        let m2 = x.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(n)
        let m3 = x.map { pow($0 - mean, 3) }.reduce(0, +) / Double(n)
        let m4 = x.map { pow($0 - mean, 4) }.reduce(0, +) / Double(n)
        let skew = m3 / pow(m2, 1.5)
        let kurtosis = m4 / (m2 * m2) - 3
        let root = Double(n).squareRoot()
        #expect(abs(mean) < 5 / root, "mean \(mean)")
        #expect(abs(m2 - 1) < 5 * 2.0.squareRoot() / root, "variance \(m2)")
        #expect(abs(skew) < 5 * 6.0.squareRoot() / root, "skewness \(skew)")
        #expect(abs(kurtosis) < 5 * 24.0.squareRoot() / root, "excess kurtosis \(kurtosis)")
    }
}
#endif

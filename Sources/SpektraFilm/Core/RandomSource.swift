import Foundation

/// Identity of one random stream.
///
/// Philox4x32 takes a 64-bit key and a 128-bit counter, enough room for
/// `(seed, channel, sublayer, pixel)` without hashing. The seed fills the key. The pixel index and
/// a block index fill the low counter words, and `channel` and `sublayer` are packed into the top
/// counter word. Two distinct streams never collide.
public struct PhiloxKey: Hashable, Sendable {
    public let seed: UInt64
    public let channel: UInt32
    public let sublayer: UInt32

    /// - Parameters:
    ///   - seed: the user-visible grain or glare seed.
    ///   - channel: `0 ..< 65536`.
    ///   - sublayer: `0 ..< 65536`.
    public init(seed: UInt64, channel: Int = 0, sublayer: Int = 0) {
        precondition(
            channel >= 0 && channel < 0x1_0000, "channel must fit in 16 bits, got \(channel)")
        precondition(
            sublayer >= 0 && sublayer < 0x1_0000, "sublayer must fit in 16 bits, got \(sublayer)")
        self.seed = seed
        self.channel = UInt32(channel)
        self.sublayer = UInt32(sublayer)
    }
}

/// A counter-based random stream.
///
/// The values at a given counter depend only on the key and the counter, so a render split into
/// tiles across any number of threads produces the same noise as a single-threaded pass over the
/// whole frame. Exports must be reproducible, and a sequential generator cannot guarantee that:
/// upstream's `use_fast_stats` path draws from Numba's thread-local state, and its output changes
/// with the thread count (`grain.md` section 8.2).
///
/// Grain and glare use this protocol so the planned Metal path can supply a GPU generator with the
/// same key and counter contract and produce the same values.
public protocol RandomSource {
    /// Restarts the stream at `counter`, discarding any partly consumed block.
    mutating func reset(counter: UInt64)

    /// The next 32 raw bits.
    mutating func nextBits() -> UInt32
}

extension RandomSource {
    /// Uniform on `[0, 1)` with 53 significant bits, the same as NumPy's `next_double`, which the
    /// ported samplers consume.
    @inlinable
    public mutating func nextUniform() -> Double {
        let high = UInt64(nextBits())
        let low = UInt64(nextBits())
        return Double((high << 32 | low) >> 11) * 0x1p-53
    }

    /// Uniform on `(0, 1)`, for callers that take a logarithm.
    @inlinable
    public mutating func nextOpenUniform() -> Double {
        let high = UInt64(nextBits())
        let low = UInt64(nextBits())
        return (Double((high << 32 | low) >> 11) + 0.5) * 0x1p-53
    }
}

/// Philox4x32-10, the counter-based generator from Salmon et al., "Parallel Random Numbers: As Easy
/// as 1, 2, 3" (SC11), as implemented in Random123.
///
/// Ten rounds of a two-word Feistel network over four 32-bit counter words. It passes BigCrush
/// and costs about twenty integer operations per word. The render relies on it needing no state
/// beyond the key and the counter.
public struct Philox4x32: RandomSource, Sendable {

    /// Round multipliers and key bumps from Random123's `philox.h`. The bumps are the fractional
    /// parts of the golden ratio and of sqrt(3), scaled to 32 bits.
    @usableFromInline static let multiplier0: UInt32 = 0xD251_1F53
    @usableFromInline static let multiplier1: UInt32 = 0xCD9E_8D57
    @usableFromInline static let keyBump0: UInt32 = 0x9E37_79B9
    @usableFromInline static let keyBump1: UInt32 = 0xBB67_AE85
    @usableFromInline static let rounds = 10

    /// Words a single `generate` call yields.
    public static let blockSize = 4

    // Storage is `@usableFromInline` so the sampler loops in `Distributions` can inline the whole
    // draw into a caller in another module.
    @usableFromInline let key0: UInt32
    @usableFromInline let key1: UInt32
    /// Top counter word: `channel | sublayer << 16`.
    @usableFromInline let streamWord: UInt32

    @usableFromInline var counterLow: UInt32
    @usableFromInline var counterHigh: UInt32
    @usableFromInline var block: UInt32
    @usableFromInline var buffer: SIMD4<UInt32>
    @usableFromInline var consumed: Int

    public init(key: PhiloxKey, counter: UInt64 = 0) {
        key0 = UInt32(truncatingIfNeeded: key.seed)
        key1 = UInt32(truncatingIfNeeded: key.seed >> 32)
        streamWord = key.channel | (key.sublayer << 16)
        counterLow = UInt32(truncatingIfNeeded: counter)
        counterHigh = UInt32(truncatingIfNeeded: counter >> 32)
        block = 0
        buffer = .zero
        consumed = Philox4x32.blockSize
    }

    @inlinable
    public mutating func reset(counter: UInt64) {
        counterLow = UInt32(truncatingIfNeeded: counter)
        counterHigh = UInt32(truncatingIfNeeded: counter >> 32)
        block = 0
        consumed = Philox4x32.blockSize
    }

    @inlinable
    public mutating func nextBits() -> UInt32 {
        if consumed == Philox4x32.blockSize {
            buffer = Philox4x32.generate(
                counter: SIMD4(counterLow, counterHigh, block, streamWord),
                key: (key0, key1))
            // 2^32 blocks of 4 words per pixel is more than any sampler here can consume, so the
            // block counter wraps instead of trapping: a pathological rejection loop should not
            // crash a render.
            block &+= 1
            consumed = 0
        }
        let word = buffer[consumed]
        consumed += 1
        return word
    }

    /// The raw ten-round permutation, exposed so a GPU implementation can be checked against it and
    /// so the published Random123 test vectors can be applied directly.
    @inlinable
    public static func generate(counter: SIMD4<UInt32>, key: (UInt32, UInt32)) -> SIMD4<UInt32> {
        var counter = counter
        var key0 = key.0
        var key1 = key.1
        for round in 0..<rounds {
            if round > 0 {
                key0 &+= keyBump0
                key1 &+= keyBump1
            }
            let product0 = multiplier0.multipliedFullWidth(by: counter[0])
            let product1 = multiplier1.multipliedFullWidth(by: counter[2])
            counter = SIMD4(
                product1.high ^ counter[1] ^ key0,
                product1.low,
                product0.high ^ counter[3] ^ key1,
                product0.low)
        }
        return counter
    }
}

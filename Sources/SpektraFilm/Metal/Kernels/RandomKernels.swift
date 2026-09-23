extension MetalKernels {
    /// Philox4x32-10 and the samplers built on it, keyed exactly as ``Philox4x32``: the seed is the
    /// key, and the counter is `(pixel low, pixel high, block, channel | sublayer << 16)`. The raw
    /// words match the CPU bit for bit. Uniforms carry 24 bits, where the CPU's carry 53, so the
    /// samples agree in distribution rather than value.
    static let random = #"""
        struct Philox {
            uint key0;
            uint key1;
            uint stream;
            uint counterLow;
            uint counterHigh;
            uint block;
            uint4 buffer;
            uint consumed;
        };

        static inline uint4 philox_generate(uint4 counter, uint key0, uint key1) {
            for (int round = 0; round < 10; ++round) {
                if (round > 0) { key0 += 0x9E3779B9u; key1 += 0xBB67AE85u; }
                uint hi0 = mulhi(0xD2511F53u, counter.x);
                uint lo0 = 0xD2511F53u * counter.x;
                uint hi1 = mulhi(0xCD9E8D57u, counter.z);
                uint lo1 = 0xCD9E8D57u * counter.z;
                counter = uint4(hi1 ^ counter.y ^ key0, lo1, hi0 ^ counter.w ^ key1, lo0);
            }
            return counter;
        }

        // PhiloxKey(seed:channel:sublayer:) and Philox4x32.reset(counter:)
        static inline Philox philox_make(ulong seed, uint channel, uint sublayer, ulong counter) {
            Philox r;
            r.key0 = uint(seed & 0xFFFFFFFFul);
            r.key1 = uint(seed >> 32);
            r.stream = channel | (sublayer << 16);
            r.counterLow = uint(counter & 0xFFFFFFFFul);
            r.counterHigh = uint(counter >> 32);
            r.block = 0;
            r.buffer = uint4(0);
            r.consumed = 4;
            return r;
        }

        static inline uint philox_next(thread Philox &r) {
            if (r.consumed == 4) {
                r.buffer = philox_generate(
                    uint4(r.counterLow, r.counterHigh, r.block, r.stream), r.key0, r.key1);
                r.block += 1;
                r.consumed = 0;
            }
            uint word = r.buffer[r.consumed];
            r.consumed += 1;
            return word;
        }

        // RandomSource.nextUniform and nextOpenUniform, from the top 24 bits of the first word.
        // Two words are consumed, as on the CPU, so the stream stays in step.
        static inline float philox_uniform(thread Philox &r) {
            uint high = philox_next(r);
            philox_next(r);
            return float(high >> 8) * 0x1p-24f;
        }

        static inline float philox_open_uniform(thread Philox &r) {
            uint high = philox_next(r);
            philox_next(r);
            return (float(high >> 8) + 0.5f) * 0x1p-24f;
        }

        // Distributions.standardNormal, Box-Muller.
        static inline float philox_normal(thread Philox &r) {
            float radial = philox_open_uniform(r);
            float angular = philox_uniform(r);
            return sqrt(-2.0f * log(radial)) * cos(2.0f * M_PI_F * angular);
        }

        // Test hook: the first four raw words at each counter, to check against the CPU.
        kernel void philox_words(
            device uint *out [[buffer(0)]], constant ulong &seed [[buffer(1)]],
            constant uint &channel [[buffer(2)]], constant uint &sublayer [[buffer(3)]],
            constant uint &n [[buffer(4)]], uint i [[thread_position_in_grid]])
        {
            if (i >= n) { return; }
            Philox r = philox_make(seed, channel, sublayer, ulong(i));
            for (int w = 0; w < 6; ++w) { out[i * 6 + w] = philox_next(r); }
        }

        // Test hook: one standard normal per counter.
        kernel void philox_normals(
            device float *out [[buffer(0)]], constant ulong &seed [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i >= n) { return; }
            Philox r = philox_make(seed, 0, 0, ulong(i));
            out[i] = philox_normal(r);
        }
        """#
}

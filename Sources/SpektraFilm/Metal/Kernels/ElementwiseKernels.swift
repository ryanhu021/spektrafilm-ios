extension MetalKernels {
    /// Per-value and per-pixel maps. Each mirrors the CPU expression named above it.
    static let elementwise = #"""
        // x * scale
        kernel void scale(
            device float *x [[buffer(0)]], constant float &s [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i < n) { x[i] = x[i] * s; }
        }

        // log10Guard(x * scale): log10(fmax(v, 0) + 1e-10). fmax drops a NaN, as npFmax does.
        kernel void scale_log10_guard(
            device float *x [[buffer(0)]], constant float &s [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i < n) { x[i] = log10(fmax(x[i] * s, 0.0f) + 1e-10f); }
        }

        // pow(10, x) * scale
        kernel void exp10_scale(
            device float *x [[buffer(0)]], constant float &s [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i < n) { x[i] = exp10(x[i]) * s; }
        }

        // value * factor[channel] + offset[channel], on a 3-channel frame
        kernel void affine3(
            device float *x [[buffer(0)]], constant float *factor [[buffer(1)]],
            constant float *offset [[buffer(2)]], constant uint &n [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i < n) { uint c = i % 3; x[i] = x[i] * factor[c] + offset[c]; }
        }

        // Matrix3.apply(to:), row-major m[9]
        kernel void matrix3(
            device float *x [[buffer(0)]], constant float *m [[buffer(1)]],
            constant uint &pixels [[buffer(2)]], uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            float r = x[p * 3], g = x[p * 3 + 1], b = x[p * 3 + 2];
            x[p * 3] = m[0] * r + m[1] * g + m[2] * b;
            x[p * 3 + 1] = m[3] * r + m[4] * g + m[5] * b;
            x[p * 3 + 2] = m[6] * r + m[7] * g + m[8] * b;
        }

        // a = a + amount * (a - b), the unsharp mask
        kernel void unsharp(
            device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            constant float &amount [[buffer(2)]], constant uint &n [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i < n) { a[i] = a[i] + amount * (a[i] - b[i]); }
        }

        // a = (1 - w) * a + w * b
        kernel void mix_weighted(
            device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            constant float &w [[buffer(2)]], constant uint &n [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i < n) { a[i] = (1.0f - w) * a[i] + w * b[i]; }
        }

        // a = b - a
        kernel void reverse_subtract(
            device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i < n) { a[i] = b[i] - a[i]; }
        }

        // C's pow: NaN for a negative base with a fractional exponent. Metal's pow returns
        // pow(|x|, y) there instead, and the reference's "Indeterminate" gammas depend on the NaN.
        static inline float c_pow(float x, float y) {
            return x < 0.0f ? NAN : pow(x, y);
        }

        // TransferFunction.spow
        static inline float spow(float a, float p) {
            float s = a < 0.0f ? -1.0f : (a > 0.0f ? 1.0f : 0.0f);
            return s * pow(fabs(a), p);
        }

        // TransferFunction.encode and decode. `code` is the case's position in
        // TransferFunction.allCases; `k` carries the constants from the Swift enum:
        // [bt2020Alpha, bt2020Beta, bt2020DecodeThreshold, sRGBDecodeThreshold, rommEt]
        static inline float transfer_encode_value(float v, uint code, constant float *k) {
            switch (code) {
                case 1: return v <= 0.0031308f ? v * 12.92f : 1.055f * spow(v, 1.0f / 2.4f) - 0.055f;
                case 2: return c_pow(v, 1.0f / 2.6f);
                case 3: return c_pow(v, 1.0f / (563.0f / 256.0f));
                case 4: return k[1] > v ? v * 4.5f : k[0] * spow(v, 0.45f) - (k[0] - 1.0f);
                case 5: {
                    float iMax = 255.0f;
                    float xp = k[4] > v ? v * 16.0f * iMax : spow(v, 1.0f / 1.8f) * iMax;
                    return xp / iMax;
                }
                default: return v;
            }
        }

        static inline float transfer_decode_value(float v, uint code, constant float *k) {
            switch (code) {
                case 1: return k[3] >= v ? v / 12.92f : spow((v + 0.055f) / 1.055f, 2.4f);
                case 2: return c_pow(v, 2.6f);
                case 3: return c_pow(v, 563.0f / 256.0f);
                case 4: return v < k[2] ? v / 4.5f : spow((v + (k[0] - 1.0f)) / k[0], 1.0f / 0.45f);
                case 5: {
                    float iMax = 255.0f;
                    float xp = v * iMax;
                    return xp < 16.0f * k[4] * iMax ? xp / (16.0f * iMax) : spow(xp / iMax, 1.8f);
                }
                default: return v;
            }
        }

        kernel void transfer(
            device float *x [[buffer(0)]], constant uint &code [[buffer(1)]],
            constant float *k [[buffer(2)]], constant uint &encode [[buffer(3)]],
            constant uint &n [[buffer(4)]], uint i [[thread_position_in_grid]])
        {
            if (i >= n) { return; }
            x[i] = encode != 0 ? transfer_encode_value(x[i], code, k)
                               : transfer_decode_value(x[i], code, k);
        }
        """#
}

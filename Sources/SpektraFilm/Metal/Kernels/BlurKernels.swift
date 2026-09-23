extension MetalKernels {
    /// GaussianFilter's FIR and IIR paths. Every kernel reads and writes one channel of an
    /// interleaved frame through a (stride, offset) pair, so a pass can go frame to plane or plane
    /// to frame. Sums run in the CPU's order.
    static let blur = #"""
        struct Plane {
            uint height;
            uint width;
            uint srcStride;
            uint srcOffset;
            uint dstStride;
            uint dstOffset;
        };

        // BoundaryIndex.reflectEdgeDuplicated
        static inline int reflect_edge_duplicated(int i, int n) {
            if (i >= 0 && i < n) { return i; }
            if (i >= -n && i < 0) { return -i - 1; }
            if (i >= n && i < 2 * n) { return 2 * n - 1 - i; }
            int period = 2 * n;
            int k = i % period;
            if (k < 0) { k += period; }
            if (k >= n) { k = period - 1 - k; }
            return k;
        }

        // GaussianFilter.firPlane, the vertical pass.
        kernel void fir_vertical(
            device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
            constant float *w [[buffer(2)]], constant int &radius [[buffer(3)]],
            constant Plane &p [[buffer(4)]], uint2 g [[thread_position_in_grid]])
        {
            if (g.x >= p.width || g.y >= p.height) { return; }
            int n = int(p.height);
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                uint row = uint(reflect_edge_duplicated(int(g.y) + k, n));
                sum += src[(row * p.width + g.x) * p.srcStride + p.srcOffset] * w[k + radius];
            }
            dst[(g.y * p.width + g.x) * p.dstStride + p.dstOffset] = sum;
        }

        // GaussianFilter.firPlane, the horizontal pass.
        kernel void fir_horizontal(
            device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
            constant float *w [[buffer(2)]], constant int &radius [[buffer(3)]],
            constant Plane &p [[buffer(4)]], uint2 g [[thread_position_in_grid]])
        {
            if (g.x >= p.width || g.y >= p.height) { return; }
            int m = int(p.width);
            float sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                uint column = uint(reflect_edge_duplicated(int(g.x) + k, m));
                sum += src[(g.y * p.width + column) * p.srcStride + p.srcOffset] * w[k + radius];
            }
            dst[(g.y * p.width + g.x) * p.dstStride + p.dstOffset] = sum;
        }

        // Double-float arithmetic: a value is hi + lo, about 48 bits. The IIR recursions need it.
        // At large sigma their feedback coefficients nearly cancel (b1 + b2 + b3 = 1 - b with b
        // near 1e-5), which amplifies float32 rounding to 3e-3 at sigma 65. Fast math is off, so
        // the compiler keeps these error terms.
        struct df { float hi; float lo; };

        static inline df quick_two_sum(float a, float b) {
            float s = a + b;
            return df{ s, b - (s - a) };
        }

        static inline df two_sum(float a, float b) {
            float s = a + b;
            float bb = s - a;
            return df{ s, (a - (s - bb)) + (b - bb) };
        }

        static inline df df_add(df a, df b) {
            df s = two_sum(a.hi, b.hi);
            return quick_two_sum(s.hi, s.lo + a.lo + b.lo);
        }

        static inline df df_mul(df a, df b) {
            float p = a.hi * b.hi;
            float e = fma(a.hi, b.hi, -p);
            return quick_two_sum(p, e + a.hi * b.lo + a.lo * b.hi);
        }

        // One step of the recursion: b * x + b1 * s1 + b2 * s2 + b3 * s3, in the CPU's order.
        // `k` holds each coefficient as a (hi, lo) pair: [b, b1, b2, b3].
        static inline df iir_step(float x, df s1, df s2, df s3, constant float2 *k) {
            df v = df_mul(df{ k[0].x, k[0].y }, df{ x, 0.0f });
            v = df_add(v, df_mul(df{ k[1].x, k[1].y }, s1));
            v = df_add(v, df_mul(df{ k[2].x, k[2].y }, s2));
            v = df_add(v, df_mul(df{ k[3].x, k[3].y }, s3));
            return v;
        }

        // GaussianFilter.iirHorizontal: one thread per row, forward then back, the recursion
        // state starting from the edge sample.
        kernel void iir_rows(
            device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
            constant float2 *k [[buffer(2)]], constant Plane &p [[buffer(3)]],
            uint y [[thread_position_in_grid]])
        {
            if (y >= p.height) { return; }
            uint base = y * p.width;
            df w1 = df{ src[base * p.srcStride + p.srcOffset], 0.0f };
            df w2 = w1, w3 = w1;
            for (uint j = 0; j < p.width; ++j) {
                df w = iir_step(src[(base + j) * p.srcStride + p.srcOffset], w1, w2, w3, k);
                dst[(base + j) * p.dstStride + p.dstOffset] = w.hi;
                w3 = w2; w2 = w1; w1 = w;
            }
            // The backward pass starts from the last forward output at full precision.
            df y1 = w1, y2 = w1, y3 = w1;
            for (int j = int(p.width) - 1; j >= 0; --j) {
                uint i = (base + uint(j)) * p.dstStride + p.dstOffset;
                df v = iir_step(dst[i], y1, y2, y3, k);
                dst[i] = v.hi;
                y3 = y2; y2 = y1; y1 = v;
            }
        }

        // GaussianFilter.iirVertical: one thread per column.
        kernel void iir_columns(
            device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
            constant float2 *k [[buffer(2)]], constant Plane &p [[buffer(3)]],
            uint x [[thread_position_in_grid]])
        {
            if (x >= p.width) { return; }
            df a = df{ src[x * p.srcStride + p.srcOffset], 0.0f };
            df b = a, d = a;
            for (uint i = 0; i < p.height; ++i) {
                df w = iir_step(src[(i * p.width + x) * p.srcStride + p.srcOffset], a, b, d, k);
                dst[(i * p.width + x) * p.dstStride + p.dstOffset] = w.hi;
                d = b; b = a; a = w;
            }
            b = a; d = a;
            for (int i = int(p.height) - 1; i >= 0; --i) {
                uint j = (uint(i) * p.width + x) * p.dstStride + p.dstOffset;
                df v = iir_step(dst[j], a, b, d, k);
                dst[j] = v.hi;
                d = b; b = a; a = v;
            }
        }

        // One channel copied between frames.
        kernel void copy_channel(
            device const float *src [[buffer(0)]], device float *dst [[buffer(1)]],
            constant Plane &p [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i >= p.height * p.width) { return; }
            dst[i * p.dstStride + p.dstOffset] = src[i * p.srcStride + p.srcOffset];
        }

        // a += w * b
        kernel void axpy(
            device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            constant float &w [[buffer(2)]], constant uint &n [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i < n) { a[i] = a[i] + w * b[i]; }
        }
        """#
}

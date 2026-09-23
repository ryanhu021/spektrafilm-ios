extension MetalKernels {
    /// Spectral upsampling and the density-curve lookup.
    static let film = #"""
        // BoundaryIndex.mirrorEdgeShared
        static inline int mirror_edge_shared(int i, int n) {
            if (n == 1) { return 0; }
            if (i >= 0 && i < n) { return i; }
            int period = 2 * (n - 1);
            int k = i % period;
            if (k < 0) { k += period; }
            if (k >= n) { k = period - k; }
            return k;
        }

        // LUTInterpolation.mitchellWeight, B = C = 1/3. NaN fails both comparisons and weighs 0.
        static inline float mitchell(float t) {
            const float b = 1.0f / 3.0f;
            const float c = 1.0f / 3.0f;
            float x = fabs(t);
            float x2 = x * x;
            float x3 = x2 * x;
            if (x < 1.0f) {
                return (1.0f / 6.0f)
                    * ((12.0f - 9.0f * b - 6.0f * c) * x3 + (-18.0f + 12.0f * b + 6.0f * c) * x2
                        + (6.0f - 2.0f * b));
            }
            if (x < 2.0f) {
                return (1.0f / 6.0f)
                    * ((-b - 6.0f * c) * x3 + (6.0f * b + 30.0f * c) * x2
                        + (-12.0f * b - 48.0f * c) * x + (8.0f * b + 24.0f * c));
            }
            return 0.0f;
        }

        // LUTInterpolation.cubicCoordinateBaseFraction
        static inline void base_fraction(float coordinate, int size, thread int &base,
                                         thread float &fraction) {
            if (isnan(coordinate)) { base = 0; fraction = NAN; return; }
            float upper = float(size - 1);
            float clamped = coordinate <= 0.0f ? 0.0f : (coordinate >= upper ? upper : coordinate);
            if (clamped >= upper) { base = size - 2; fraction = 1.0f; return; }
            float f = floor(clamped);
            base = int(f);
            fraction = clamped - f;
        }

        // ChromaticityCoordinates.clipUnit: NaN passes through.
        static inline float clip_unit(float v) { return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v); }

        // Hanatos2025RawConverter.raw: decode, matrix, chromaticity to tc, Mitchell fetch of the
        // tc_lut, times brightness. The LUT is [size][size][3].
        kernel void rgb_to_raw(
            device const float *rgb [[buffer(0)]],
            device const float *lut [[buffer(1)]],
            device float *out [[buffer(2)]],
            constant float *m [[buffer(3)]],
            constant uint &code [[buffer(4)]],
            constant float *k [[buffer(5)]],
            constant uint &decode [[buffer(6)]],
            constant uint &size [[buffer(7)]],
            constant uint &pixels [[buffer(8)]],
            uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            float r = rgb[p * 3], g = rgb[p * 3 + 1], bl = rgb[p * 3 + 2];
            if (decode != 0) {
                r = transfer_decode_value(r, code, k);
                g = transfer_decode_value(g, code, k);
                bl = transfer_decode_value(bl, code, k);
            }
            float x = m[0] * r + m[1] * g + m[2] * bl;
            float y = m[3] * r + m[4] * g + m[5] * bl;
            float z = m[6] * r + m[7] * g + m[8] * bl;
            float brightness = x + y + z;
            float s = fmax(brightness, 1e-10f);
            float cx = x / s;
            float cy = y / s;
            float qy = clip_unit(cy / fmax(1.0f - cx, 1e-10f));
            float qx = clip_unit((1.0f - cx) * (1.0f - cx));
            float gain = isnan(brightness) ? 0.0f
                : (isinf(brightness) ? copysign(FLT_MAX, brightness) : brightness);

            int n = int(size);
            float scale = float(n - 1);
            int xb, yb;
            float xf, yf;
            base_fraction(qx * scale, n, xb, xf);
            base_fraction(qy * scale, n, yb, yf);
            float wx[4] = { mitchell(xf + 1.0f), mitchell(xf), mitchell(xf - 1.0f), mitchell(xf - 2.0f) };
            float wy[4] = { mitchell(yf + 1.0f), mitchell(yf), mitchell(yf - 1.0f), mitchell(yf - 2.0f) };
            float3 acc = float3(0.0f);
            float weightSum = 0.0f;
            for (int i = 0; i < 4; ++i) {
                int row = mirror_edge_shared(xb - 1 + i, n) * n * 3;
                for (int j = 0; j < 4; ++j) {
                    float weight = wx[i] * wy[j];
                    weightSum += weight;
                    int cell = row + mirror_edge_shared(yb - 1 + j, n) * 3;
                    acc += weight * float3(lut[cell], lut[cell + 1], lut[cell + 2]);
                }
            }
            if (weightSum != 0.0f) { acc /= weightSum; }
            acc *= gain;
            out[p * 3] = acc.x;
            out[p * 3 + 1] = acc.y;
            out[p * 3 + 2] = acc.z;
        }

        // Interpolation.upperBound
        static inline int upper_bound(float key, device const float *a, int stride, int count) {
            int lo = 0;
            int hi = count;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (a[mid * stride] <= key) { lo = mid + 1; } else { hi = mid; }
            }
            return lo;
        }

        // Interpolation.fastInterp on a 3-channel frame. With `per_channel`, the axis is
        // interleaved [count][3] like the values; otherwise it is shared. `inv` holds the
        // reciprocal interval widths, laid out like the axis.
        kernel void fast_interp(
            device const float *x [[buffer(0)]],
            device float *out [[buffer(1)]],
            device const float *axis [[buffer(2)]],
            device const float *values [[buffer(3)]],
            device const float *inv [[buffer(4)]],
            constant uint &count [[buffer(5)]],
            constant uint &per_channel [[buffer(6)]],
            constant uint &n [[buffer(7)]],
            uint i [[thread_position_in_grid]])
        {
            if (i >= n) { return; }
            int c = int(i % 3);
            int stride = per_channel != 0 ? 3 : 1;
            device const float *ax = per_channel != 0 ? axis + c : axis;
            device const float *iv = per_channel != 0 ? inv + c : inv;
            int k = int(count);
            float v = x[i];
            float first = ax[0];
            float last = ax[(k - 1) * stride];
            if (isnan(v) || v <= first) {
                out[i] = values[c];
            } else if (v >= last) {
                out[i] = values[(k - 1) * 3 + c];
            } else {
                int low = upper_bound(v, ax, stride, k) - 1;
                float t = (v - ax[low * stride]) * iv[low * stride];
                float y0 = values[low * 3 + c];
                float y1 = values[(low + 1) * 3 + c];
                out[i] = y0 + t * (y1 - y0);
            }
        }
        """#
}

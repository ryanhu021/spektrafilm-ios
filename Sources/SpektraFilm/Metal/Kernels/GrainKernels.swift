import Foundation

extension MetalKernels {
    /// `lgamma(n + 1) - (n + 0.5) log n + n - log(2 pi) / 2` for n = 0 ..< 16, the Stirling-series
    /// error at the counts where the series has not yet converged. Computed in float64 here, since
    /// MSL has no `lgamma`. The n = 0 entry is unused.
    static let stirlingErrors: String = (0..<16).map { n -> String in
        guard n > 0 else { return "0.0f" }
        let x = Double(n)
        let value =
            Foundation.lgamma(x + 1) - (x + 0.5) * Foundation.log(x) + x
            - 0.5 * Foundation.log(2 * Double.pi)
        return "\(Float(value))f"
    }.joined(separator: ", ")

    /// ``Grain``: the sublayer interpolation, the particle model and its Poisson sampler, and the
    /// clumping field.
    ///
    /// Everything before the draw runs in double-float. `sat = 1 - p * u * (1 - 1e-6)` falls to
    /// 2e-6 when `u = 1` and the density saturates, and in float32 alone that subtraction keeps
    /// about two significant digits, which would put a percent error on lambda and on the variance.
    static let grain = #"""
        struct GrainLayer {
            ulong seed;
            // (hi, lo) double-float pairs.
            float2 densityMin;
            float2 densityMax;
            float2 particles;
            float2 uniformity;
            float2 odParticle;
            float2 saturationScale;
            float2 probabilityFloor;
            float2 probabilityCeiling;
            uint channel;
            uint stream;
            uint pixels;
            uint steps;
            uint positive;
            uint interpolate;
            uint dstStride;
            uint dstOffset;
            uint accumulate;
        };

        constant float grain_stirling_table[16] = { \#(stirlingErrors) };

        static inline df grain_df(float2 v) { return df{ v.x, v.y }; }

        static inline df grain_df_neg(df a) { return df{ -a.hi, -a.lo }; }

        static inline df grain_df_div(df a, df b) {
            float q = a.hi / b.hi;
            df r = df_add(a, grain_df_neg(df_mul(df{ q, 0.0f }, b)));
            return quick_two_sum(q, r.hi / b.hi);
        }

        static inline bool grain_df_less(df a, df b) {
            return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
        }

        static inline df grain_df_from_long(long k) {
            float hi = float(k);
            return df{ hi, float(k - long(hi)) };
        }

        // Grain.sublayerPlane, one pixel.
        static inline float grain_interpolate(
            float value, device const float *axis, device const float *curve,
            device const float *inv, int steps, bool positive)
        {
            float x = positive ? -value : value;
            if (isnan(x) || x <= axis[0]) { return curve[0]; }
            if (x >= axis[steps - 1]) { return curve[steps - 1]; }
            int low = upper_bound(x, axis, 1, steps) - 1;
            float t = (x - axis[low]) * inv[low];
            return curve[low] + t * (curve[low + 1] - curve[low]);
        }

        struct GrainPixel {
            float density;
            df lambda;
            df step;
        };

        // Everything Grain.layerParticleModel computes before its draw: the density (interpolated
        // on the layered path), p, sat, lambda = N * p / sat and the lattice step od * sat.
        static inline GrainPixel grain_setup(
            device const float *density, device const float *axis, device const float *curves,
            device const float *inv, constant GrainLayer &L, uint i)
        {
            GrainPixel px;
            float v = density[i * 3 + L.channel];
            if (L.interpolate != 0) {
                int n = int(L.steps);
                v = grain_interpolate(
                    v, axis + L.channel * n, curves + (L.stream * 3 + L.channel) * n,
                    inv + L.channel * n, n, L.positive != 0);
            }
            px.density = v;
            df d = df_add(df{ v, 0.0f }, grain_df(L.densityMin));
            df p = grain_df_div(d, grain_df(L.densityMax));
            // NaN passes the clip, as Grain.probabilityOfDevelopment lets it.
            if (!isnan(p.hi)) {
                df low = grain_df(L.probabilityFloor);
                df high = grain_df(L.probabilityCeiling);
                if (grain_df_less(p, low)) { p = low; }
                if (grain_df_less(high, p)) { p = high; }
            }
            df pu = df_mul(df_mul(p, grain_df(L.uniformity)), grain_df(L.saturationScale));
            df sat = df_add(df{ 1.0f, 0.0f }, grain_df_neg(pu));
            px.lambda = grain_df_div(df_mul(grain_df(L.particles), p), sat);
            px.step = df_mul(grain_df(L.odParticle), sat);
            return px;
        }

        // Loader's stirlerr: lgamma(k + 1) - (k + 0.5) log k + k - log(2 pi) / 2.
        static inline float grain_stirling_error(long k) {
            if (k < 16) { return grain_stirling_table[k]; }
            float n = float(k);
            float nn = n * n;
            const float s0 = 1.0f / 12.0f, s1 = 1.0f / 360.0f, s2 = 1.0f / 1260.0f;
            const float s3 = 1.0f / 1680.0f, s4 = 1.0f / 1188.0f;
            return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / n;
        }

        // log P(K = k) for K ~ Poisson(lambda), which transformed rejection compares against:
        // -lambda + k log lambda - lgamma(k + 1). Written as in Loader, "Fast and accurate
        // computation of binomial probabilities" (2000): -stirlerr(k) - bd0(k, lambda)
        // - log(2 pi k) / 2, with bd0 = k log(k / lambda) + lambda - k. The direct form cancels
        // terms near 2e9 at lambda 1.2e8 down to a result near 1, which float32 cannot hold. Near
        // the mode bd0 comes from a series in v = (k - lambda) / (k + lambda) whose terms are all
        // positive, so it keeps float32's relative precision. It needs k - lambda exactly, which
        // is why the caller passes lambda as an integer base plus a fraction.
        static inline float grain_log_poisson_pmf(long k, df lambda, long base, float fraction) {
            if (k == 0) { return -(lambda.hi + lambda.lo); }
            float kf = float(k);
            float lam = lambda.hi;
            float delta = float(k - base) - fraction;
            float sum = kf + lam;
            float bd0;
            if (fabs(delta) < 0.1f * sum) {
                float v = delta / sum;
                float v2 = v * v;
                float s = delta * v;
                float term = 2.0f * kf * v;
                for (int j = 1; j < 32; ++j) {
                    term *= v2;
                    float next = s + term / float(2 * j + 1);
                    if (next == s) { break; }
                    s = next;
                }
                bd0 = s;
            } else {
                bd0 = kf * log(kf / lam) + lam - kf;
            }
            return -grain_stirling_error(k) - bd0 - 0.5f * log(2.0f * M_PI_F * kf);
        }

        // Distributions.poisson: Knuth below 10, Hormann's PTRS at and above, consuming the same
        // words per attempt as the CPU. The count can pass 2^24, so it is a long.
        static inline long grain_poisson(df lambda, thread Philox &r) {
            if (!isfinite(lambda.hi) || !(lambda.hi > 0.0f)) { return 0; }
            if (lambda.hi < 10.0f || (lambda.hi == 10.0f && lambda.lo < 0.0f)) {
                float threshold = exp(-lambda.hi);
                float product = 1.0f;
                long count = 0;
                while (true) {
                    product *= philox_uniform(r);
                    if (product <= threshold) { return count; }
                    count += 1;
                }
            }
            // The CPU clamps at poissonLambdaMax, which rounds to 2^63 in float32 and would
            // overflow the long. Only a uniformity above 1 gets here.
            if (lambda.hi > 0x1p62f) { lambda = df{ 0x1p62f, 0.0f }; }
            float lam = lambda.hi;
            float b = 0.931f + 2.53f * sqrt(lam);
            float a = -0.059f + 0.02483f * b;
            float inverseAlpha = 1.1239f + 1.1328f / (b - 3.4f);
            float squeeze = 0.9277f - 3.6224f / (b - 2.0f);

            // lambda = base + fraction, with base an integer and fraction in [0, 1).
            float whole = floor(lambda.hi);
            float rest = (lambda.hi - whole) + lambda.lo;
            float carry = floor(rest);
            long base = long(whole) + long(carry);
            float fraction = rest - carry;

            while (true) {
                float u = philox_uniform(r) - 0.5f;
                float v = philox_uniform(r);
                float us = 0.5f - fabs(u);
                // The CPU's candidate is -infinity here, which it rejects as negative.
                if (us == 0.0f) { continue; }
                long k = base + long(floor((2.0f * a / us + b) * u + fraction + 0.43f));
                if (us >= 0.07f && v <= squeeze) { return k; }
                if (k < 0 || (us < 0.013f && v > us)) { continue; }
                float lhs = log(v) + log(inverseAlpha) - log(a / (us * us) + b);
                if (lhs <= grain_log_poisson_pmf(k, lambda, base, fraction)) { return k; }
            }
        }

        // Grain.layerParticleModel's draw for one (channel, sublayer) plane, written to or added
        // into one channel of `dst`.
        kernel void grain_layer(
            device const float *density [[buffer(0)]],
            device float *dst [[buffer(1)]],
            device const float *axis [[buffer(2)]],
            device const float *curves [[buffer(3)]],
            device const float *inv [[buffer(4)]],
            constant GrainLayer &L [[buffer(5)]],
            uint i [[thread_position_in_grid]])
        {
            if (i >= L.pixels) { return; }
            GrainPixel px = grain_setup(density, axis, curves, inv, L, i);
            Philox r = philox_make(L.seed, L.channel, L.stream, ulong(i));
            long k = grain_poisson(px.lambda, r);
            float value = df_mul(grain_df_from_long(k), px.step).hi;
            uint j = i * L.dstStride + L.dstOffset;
            dst[j] = L.accumulate != 0 ? dst[j] + value : value;
        }

        // Test hook: grain_setup's density, lambda and step at each pixel.
        kernel void grain_layer_setup(
            device const float *density [[buffer(0)]],
            device float4 *out [[buffer(1)]],
            device const float *axis [[buffer(2)]],
            device const float *curves [[buffer(3)]],
            device const float *inv [[buffer(4)]],
            constant GrainLayer &L [[buffer(5)]],
            uint i [[thread_position_in_grid]])
        {
            if (i >= L.pixels) { return; }
            GrainPixel px = grain_setup(density, axis, curves, inv, L, i);
            out[i] = float4(px.density, px.lambda.hi, px.lambda.lo, px.step.hi);
        }

        // Test hook: one Poisson draw per counter.
        kernel void grain_poisson_samples(
            device long *out [[buffer(0)]], constant float2 &lambda [[buffer(1)]],
            constant ulong &seed [[buffer(2)]], constant uint &n [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i >= n) { return; }
            Philox r = philox_make(seed, 0, 0, ulong(i));
            out[i] = grain_poisson(grain_df(lambda), r);
        }

        // Grain.addMicroStructure's field: exp(mu + sigma * z), one stream per channel. The host
        // passes sigma as 0 below Distributions.lognormalSigmaFloor, where the CPU draws nothing.
        kernel void grain_clumping(
            device float *out [[buffer(0)]], constant ulong &seed [[buffer(1)]],
            constant float2 &logParameters [[buffer(2)]], constant uint &stream [[buffer(3)]],
            constant uint &pixels [[buffer(4)]], uint i [[thread_position_in_grid]])
        {
            if (i >= pixels * 3) { return; }
            Philox r = philox_make(seed, i % 3, stream, ulong(i / 3));
            out[i] = exp(logParameters.x + logParameters.y * philox_normal(r));
        }

        // dst[pixel][channel] += plane[pixel]
        kernel void grain_accumulate(
            device float *dst [[buffer(0)]], device const float *plane [[buffer(1)]],
            constant uint &channel [[buffer(2)]], constant uint &pixels [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i < pixels) { dst[i * 3 + channel] += plane[i]; }
        }

        // a *= b
        kernel void grain_multiply(
            device float *a [[buffer(0)]], device const float *b [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            if (i < n) { a[i] = a[i] * b[i]; }
        }
        """#
}

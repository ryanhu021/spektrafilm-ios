extension MetalKernels {
    /// Highlight boost, halation and the DIR-coupler inhibitor: the per-pixel parts of the filming
    /// stage between the blurs.
    static let stage = #"""
        // One partial maximum per thread over `per` consecutive values, NaN winning, as np.max.
        kernel void partial_max(
            device const float *x [[buffer(0)]], device float *partial [[buffer(1)]],
            constant uint &n [[buffer(2)]], constant uint &per [[buffer(3)]],
            uint g [[thread_position_in_grid]])
        {
            uint start = g * per;
            uint end = min(start + per, n);
            float m = -INFINITY;
            bool nan = false;
            for (uint i = start; i < end; ++i) {
                float v = x[i];
                if (isnan(v)) { nan = true; } else if (v > m) { m = v; }
            }
            partial[g] = nan ? NAN : m;
        }

        // exp(x) - x - 1 without the cancellation that loses float32's digits for small x.
        static inline float exp_minus_linear(float x) {
            if (fabs(x) < 1e-2f) {
                return x * x * (0.5f + x * (1.0f / 6.0f + x * (1.0f / 24.0f + x / 120.0f)));
            }
            return exp(x) - x - 1.0f;
        }

        // Diffusion.boostHighlights, the per-value curve above rawX0.
        kernel void boost_highlights(
            device float *x [[buffer(0)]], constant float4 &k [[buffer(1)]],
            constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]])
        {
            // k = (rawX0, 1 / maxRaw, boostScale, a)
            if (i >= n) { return; }
            float v = x[i];
            if (v <= k.x) { return; }
            float dx = (v - k.x) * k.y;
            x[i] = v + k.z * exp_minus_linear(k.w * dx);
        }

        // Diffusion.applyHalation, pass 1 for one channel: blend the core and tail blurs, then blend
        // that with the identity by scatterAmount.
        kernel void halation_scatter(
            device float *dst [[buffer(0)]], device const float *core [[buffer(1)]],
            device const float *tail [[buffer(2)]], constant uint &channel [[buffer(3)]],
            constant float2 &k [[buffer(4)]], constant uint &pixels [[buffer(5)]],
            uint p [[thread_position_in_grid]])
        {
            // k = (scatterAmount, tailWeight)
            if (p >= pixels) { return; }
            float scattered = (1.0f - k.y) * core[p] + k.y * tail[p];
            uint i = p * 3 + channel;
            dst[i] = (1.0f - k.x) * dst[i] + k.x * scattered;
        }

        // Diffusion.applyHalation, pass 2 for one channel: add the back-reflection and, optionally,
        // divide by 1 + strength.
        kernel void halation_bounce(
            device float *dst [[buffer(0)]], device const float *accumulated [[buffer(1)]],
            constant uint &channel [[buffer(2)]], constant float &strength [[buffer(3)]],
            constant uint &renormalise [[buffer(4)]], constant uint &pixels [[buffer(5)]],
            uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            uint i = p * 3 + channel;
            float v = dst[i] + strength * accumulated[p];
            if (renormalise != 0) { v = v / (1.0f + strength); }
            dst[i] = v;
        }

        // Couplers.diffuseInPlace for one channel: (1 - w) * core + w * tail, written into it.
        kernel void mix_into_channel(
            device float *dst [[buffer(0)]], device const float *core [[buffer(1)]],
            device const float *tail [[buffer(2)]], constant uint &channel [[buffer(3)]],
            constant float &w [[buffer(4)]], constant uint &pixels [[buffer(5)]],
            uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            dst[p * 3 + channel] = (1.0f - w) * core[p] + w * tail[p];
        }

        // Couplers.correctedLogExposure, the per-pixel part: silver from density, the high-exposure
        // shift, then the inhibition matrix. In place; m is row-major, donor by receiver.
        kernel void coupler_inhibitor(
            device float *d [[buffer(0)]], constant float *m [[buffer(1)]],
            constant float3 &densityMax [[buffer(2)]], constant float &shift [[buffer(3)]],
            constant uint &positive [[buffer(4)]], constant uint &pixels [[buffer(5)]],
            uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            uint i = p * 3;
            float r = positive != 0 ? densityMax.x - d[i] : d[i];
            float g = positive != 0 ? densityMax.y - d[i + 1] : d[i + 1];
            float b = positive != 0 ? densityMax.z - d[i + 2] : d[i + 2];
            r += shift * r * r;
            g += shift * g * g;
            b += shift * b * b;
            for (uint c = 0; c < 3; ++c) {
                d[i + c] = r * m[c] + g * m[3 + c] + b * m[6 + c];
            }
        }
        """#
}

extension MetalKernels {
    /// The scanning stage's per-pixel steps between the spectral contraction and the output
    /// encoding.
    static let scan = #"""
        // ColorReferenceService.correctXYZ: scale each pixel so Y maps through the clamped line
        // m * Y + q.
        kernel void xyz_correct(
            device float *x [[buffer(0)]], constant float2 &k [[buffer(1)]],
            constant uint &pixels [[buffer(2)]], uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            float y = x[p * 3 + 1];
            float corrected = min(max(k.x * y + k.y, 0.0f), 1.0f);
            float s = corrected / (y + 1e-10f);
            x[p * 3] *= s;
            x[p * 3 + 1] *= s;
            x[p * 3 + 2] *= s;
        }

        // Glare.randomAmount before the blur: a lognormal per pixel from its own Philox stream.
        // `k` is (mu, sigma) from Distributions.lognormalLogParameters; sigma below the floor
        // gives the constant exp(mu).
        kernel void glare_field(
            device float *out [[buffer(0)]], constant float2 &k [[buffer(1)]],
            constant ulong &seed [[buffer(2)]], constant uint2 &stream [[buffer(3)]],
            constant uint &pixels [[buffer(4)]], uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            if (k.y < 1e-6f) { out[p] = exp(k.x); return; }
            Philox r = philox_make(seed, stream.x, stream.y, ulong(p));
            out[p] = exp(k.x + k.y * philox_normal(r));
        }

        // ScanningStage.addGlare: the field, in percent, times the viewing illuminant's XYZ.
        kernel void glare_add(
            device float *x [[buffer(0)]], device const float *field [[buffer(1)]],
            constant float3 &illuminant [[buffer(2)]], constant uint &pixels [[buffer(3)]],
            uint p [[thread_position_in_grid]])
        {
            if (p >= pixels) { return; }
            float flare = field[p] / 100.0f;
            x[p * 3] += flare * illuminant.x;
            x[p * 3 + 1] += flare * illuminant.y;
            x[p * 3 + 2] += flare * illuminant.z;
        }
        """#
}

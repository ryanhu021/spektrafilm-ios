/// Metal Shading Language source for every Metal operator, compiled once by ``MetalContext``.
///
/// Each kernel mirrors a CPU operator line for line in float32, so a difference between the two is
/// rounding, never a different formula. The CPU operator each one mirrors is named above it.
enum MetalKernels {
    /// Every kernel, compiled as one library. Each operator family keeps its source in its own file
    /// as an extension on this type.
    static var source: String {
        [header, spectral, elementwise].joined(separator: "\n")
    }

    static let header = #"""
        #include <metal_stdlib>
        using namespace metal;
        """#

    static let spectral = #"""

        // SpectralContraction.project: CMY density to a spectrum, lit by an illuminant, projected
        // onto three response columns. `table` holds, per wavelength, the three dye weights, the
        // base density, the illuminant and the three response values.
        kernel void spectral_project(
            device const float *cmy [[buffer(0)]],
            constant float *table [[buffer(1)]],
            device float *out [[buffer(2)]],
            constant uint &count [[buffer(3)]],
            constant uint &wavelengths [[buffer(4)]],
            constant float &scale [[buffer(5)]],
            uint pixel [[thread_position_in_grid]])
        {
            if (pixel >= count) { return; }
            const float c = cmy[pixel * 3];
            const float m = cmy[pixel * 3 + 1];
            const float y = cmy[pixel * 3 + 2];
            float3 acc = float3(0.0f);
            for (uint l = 0; l < wavelengths; ++l) {
                constant float *t = table + l * 8;
                float density = c * t[0] + m * t[1] + y * t[2];
                density += t[3];
                float light = exp10(-density) * t[4];
                // Wavelengths the datasheet does not cover are NaN and contribute no light.
                if (isnan(light)) { light = 0.0f; }
                acc += light * float3(t[5], t[6], t[7]);
            }
            out[pixel * 3] = acc.x * scale;
            out[pixel * 3 + 1] = acc.y * scale;
            out[pixel * 3 + 2] = acc.z * scale;
        }
        """#
}

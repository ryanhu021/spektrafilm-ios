extension MetalKernels {
    /// OutputGamutCompressor.apply, one thread per pixel. The linear stages on either side of each
    /// perceptual space's non-linearity are folded into one matrix on the CPU, in float64, so the
    /// kernel rounds each of them once.
    static let gamut = #"""
        // Mirrors MetalGamut.Parameters. `mode` is 0 for aces_rgc, then oklch, oklrab, jzazbz and
        // cam16ucs.
        struct GamutParams {
            uint mode;
            uint pixels;
            uint hasLightnessKnee;
            uint lightnessCount;
            uint hueCount;
            float threshold;
            float limit;
            float power;
            float lightnessThreshold;
            float lightnessLimit;
            float lightnessPower;
            float lightnessWhite;
            float lightnessFirst;
            float lightnessLast;
            float hueFirst;
            float hueStep;
            float F_L;
            float N_bb;
            float A_w;
            float cz;
            float inverseCz;
            float chromaScale;
            float chromaTerm;
            float F_L4;
        };

        // The four matrices, row-major, one after the other.
        enum { GAMUT_TO_CONE = 0, GAMUT_CONE_TO_OPPONENT = 9, GAMUT_OPPONENT_TO_CONE = 18,
               GAMUT_CONE_TO_RGB = 27 };

        static inline float3 gamut_apply(constant float *m, float3 v) {
            return float3(
                m[0] * v.x + m[1] * v.y + m[2] * v.z,
                m[3] * v.x + m[4] * v.y + m[5] * v.z,
                m[6] * v.x + m[7] * v.y + m[8] * v.z);
        }

        // C's pow for a negative base, which a knee with limit < threshold reaches: finite only
        // for an integer exponent. Metal's pow would return pow(|x|, y).
        static inline float gamut_c_pow(float x, float y) {
            if (!(x < 0.0f)) { return pow(x, y); }
            float r = pow(-x, y);
            if (trunc(y) != y) { return isinf(x) ? r : NAN; }
            return fmod(y, 2.0f) != 0.0f ? -r : r;
        }

        // reinhardKnee.
        static inline float gamut_knee(float d, float threshold, float limit, float power) {
            if (!(d > threshold)) { return d; }
            float scale = limit - threshold;
            float x = (d - threshold) / scale;
            float y;
            if (isinf(x)) {
                y = NAN;
            } else if (x > 1.0f) {
                // x^power overflows float32 from x = 2.6e6 at the default power, where float64
                // lasts to 1e51. x / (1 + x^p)^(1/p) = 1 / (x^-p + 1)^(1/p) does not overflow.
                // Past float64's range the CPU's quotient is x / inf = 0, and so is this.
                y = power * log2(x) >= 1024.0f ? 0.0f
                                               : 1.0f / pow(pow(x, -power) + 1.0f, 1.0f / power);
            } else {
                y = x / gamut_c_pow(1.0f + gamut_c_pow(x, power), 1.0f / power);
            }
            return threshold + scale * y;
        }

        // np.sign: NaN for NaN.
        static inline float gamut_sign(float v) {
            if (v > 0.0f) { return 1.0f; }
            if (v < 0.0f) { return -1.0f; }
            return isnan(v) ? NAN : 0.0f;
        }

        // safeDivide, colour-science's sdiv.
        static inline float gamut_sdiv(float a, float b) {
            float q = a / b;
            return isfinite(q) ? q : 0.0f;
        }

        // C's hypot, which is +inf when either side is infinite, even against a NaN.
        static inline float gamut_hypot(float a, float b) {
            a = fabs(a);
            b = fabs(b);
            if (isinf(a) || isinf(b)) { return INFINITY; }
            if (isnan(a) || isnan(b)) { return NAN; }
            float hi = max(a, b);
            float lo = min(a, b);
            if (hi == 0.0f) { return 0.0f; }
            float r = lo / hi;
            return hi * sqrt(1.0f + r * r);
        }

        // log(1 + x) for 1 + x in [0.5, 2], where Metal's log is bounded by an absolute rather than
        // a relative error, which costs the PQ decode 2e-5. The series is 2 atanh(s) with
        // s = x / (2 + x), |s| <= 1/3, cut where the next term falls under float32's half ulp.
        static inline float gamut_log1p_near(float x) {
            float s = x / (2.0f + x);
            float s2 = s * s;
            float series = 1.0f / 17.0f;
            series = series * s2 + 1.0f / 15.0f;
            series = series * s2 + 1.0f / 13.0f;
            series = series * s2 + 1.0f / 11.0f;
            series = series * s2 + 1.0f / 9.0f;
            series = series * s2 + 1.0f / 7.0f;
            series = series * s2 + 1.0f / 5.0f;
            series = series * s2 + 1.0f / 3.0f;
            series = series * s2 + 1.0f;
            return 2.0f * s * series;
        }

        static inline float gamut_log(float x) {
            return x >= 0.5f && x <= 2.0f ? gamut_log1p_near(x - 1.0f) : log(x);
        }

        static inline float gamut_log1p(float x) {
            float u = 1.0f + x;
            return u >= 0.5f && u <= 2.0f ? gamut_log1p_near(x) : log(u);
        }

        // Kahan's expm1: the rounding of exp(x) cancels in the quotient.
        static inline float gamut_expm1(float x) {
            float u = exp(x);
            if (u == 1.0f) { return x; }
            if (isinf(u)) { return u; }
            float um1 = u - 1.0f;
            if (um1 == -1.0f) { return -1.0f; }
            return um1 * x / gamut_log(u);
        }

        // C's atan2 and sin and cos. Metal's return numbers for a NaN argument, where the CAM16
        // inverse depends on a NaN hue failing both branch tests, and NaN for atan2(0, 0), where
        // black depends on a zero hue.
        static inline float gamut_atan2(float y, float x) {
            if (isnan(x) || isnan(y)) { return NAN; }
            if (y == 0.0f && x == 0.0f) { return copysign(signbit(x) ? M_PI_F : 0.0f, y); }
            if (isinf(x) && isinf(y)) {
                return copysign(x > 0.0f ? M_PI_F / 4.0f : 3.0f * M_PI_F / 4.0f, y);
            }
            return atan2(y, x);
        }

        static inline float gamut_sin(float x) { return isfinite(x) ? sin(x) : NAN; }

        static inline float gamut_cos(float x) { return isfinite(x) ? cos(x) : NAN; }

        // degreesMod360.
        static inline float gamut_degrees(float radians) {
            float m = fmod(radians * (180.0f / M_PI_F), 360.0f);
            if (m != 0.0f && m < 0.0f) { m += 360.0f; }
            return m;
        }

        // ChromaEnvelope.lookup.
        static inline float gamut_cmax(
            constant float *table, float L, float h, constant GamutParams &p)
        {
            if (!isfinite(L) || !isfinite(h)) { return 0.0f; }
            int nL = int(p.lightnessCount);
            int nH = int(p.hueCount);
            float clampedL = min(max(L, p.lightnessFirst), p.lightnessLast);
            float hIndex = (h - p.hueFirst) / p.hueStep;
            float hFloor = floor(hIndex);
            int hLo = ((int(hFloor) % nH) + nH) % nH;
            int hHi = (hLo + 1) % nH;
            float hFraction = hIndex - hFloor;
            float lIndex = (clampedL - p.lightnessFirst) / (p.lightnessLast - p.lightnessFirst)
                * float(nL - 1);
            int lLo = min(max(int(floor(lIndex)), 0), nL - 2);
            int lHi = lLo + 1;
            float lFraction = lIndex - float(lLo);
            float v00 = table[lLo * nH + hLo];
            float v01 = table[lLo * nH + hHi];
            float v10 = table[lHi * nH + hLo];
            float v11 = table[lHi * nH + hHi];
            return v00 * (1.0f - lFraction) * (1.0f - hFraction)
                + v01 * (1.0f - lFraction) * hFraction + v10 * lFraction * (1.0f - hFraction)
                + v11 * lFraction * hFraction;
        }

        // Oklab.lightnessLr.
        static inline float gamut_lr(float L) {
            const float k1 = 0.206f;
            const float k2 = 0.03f;
            const float k3 = (1.0f + k1) / (1.0f + k2);
            float t = k3 * L - k1;
            return 0.5f * (t + sqrt(t * t + 4.0f * k2 * k3 * L));
        }

        // JzAzBz.pqEncode and pqDecode. The exponent m_2 = 134 turns one float32 rounding of
        // the ratio into 134 of them, so neither side forms the ratio. c_2 - c_3 and 1 - c_1 are
        // both 0.1640625 exactly, which makes ratio - 1 = 0.1640625 (y - 1) / (c_3 y + 1) and
        // c_2 - c_3 v = 0.1640625 + c_3 (1 - v), each free of cancellation.
        constant float GAMUT_PQ_M1 = 0.1593017578125f;
        constant float GAMUT_PQ_M2 = 134.03437499999998f;
        constant float GAMUT_PQ_C3 = 18.6875f;
        constant float GAMUT_PQ_GAP = 0.1640625f;

        static inline float gamut_pq_encode(float luminance) {
            float yp = spow(luminance / 10000.0f, GAMUT_PQ_M1);
            float excess = GAMUT_PQ_GAP * (yp - 1.0f) / (GAMUT_PQ_C3 * yp + 1.0f);
            float ratio = 1.0f + excess;
            if (!(ratio > 0.0f)) { return spow(ratio, GAMUT_PQ_M2); }
            return exp(GAMUT_PQ_M2 * gamut_log1p(excess));
        }

        static inline float gamut_pq_decode(float code) {
            float vp;
            float w;
            if (code > 0.0f && isfinite(code)) {
                w = -gamut_expm1(gamut_log(code) / GAMUT_PQ_M2);
                vp = 1.0f - w;
            } else {
                vp = spow(code, 1.0f / GAMUT_PQ_M2);
                w = 1.0f - vp;
            }
            float n = fmax(0.0f, GAMUT_PQ_GAP - w);
            return 10000.0f * spow(n / (GAMUT_PQ_GAP + GAMUT_PQ_C3 * w), 1.0f / GAMUT_PQ_M1);
        }

        // CAM16UCS.postAdaptation and its inverse, without the + 0.1 offset. The offsets cancel
        // exactly in the opponent axes, in A against the 0.305 and in the inverse's v - 0.1, so
        // leaving them out saves dark pixels a float32 cancellation. The chroma denominator is
        // the one place that keeps them.
        static inline float gamut_cam16_adapt(float v, float F_L) {
            float t = spow(F_L * fabs(v) / 100.0f, 0.42f);
            return (400.0f * gamut_sign(v) * t) / (27.13f + t);
        }

        static inline float gamut_cam16_unadapt(float v, float F_L) {
            float a = fabs(v);
            return gamut_sign(v) * 100.0f / F_L * spow((27.13f * a) / (400.0f - a), 1.0f / 0.42f);
        }

        static inline float gamut_eccentricity(float hDegrees) {
            return 0.25f * (gamut_cos(2.0f + hDegrees * M_PI_F / 180.0f) + 3.8f);
        }

        // CAM16UCS.forward, from the dRGB-adapted cone response.
        static inline float3 gamut_cam16_forward(float3 cone, constant GamutParams &p) {
            float R = gamut_cam16_adapt(cone.x, p.F_L);
            float G = gamut_cam16_adapt(cone.y, p.F_L);
            float B = gamut_cam16_adapt(cone.z, p.F_L);
            float a = R - 12.0f * G / 11.0f + B / 11.0f;
            float b = (R + G - 2.0f * B) / 9.0f;
            float h = gamut_degrees(gamut_atan2(b, a));
            float e_t = gamut_eccentricity(h);
            float A = (2.0f * R + G + (1.0f / 20.0f) * B) * p.N_bb;
            float J = 100.0f * spow(gamut_sdiv(A, p.A_w), p.cz);
            float t = p.chromaScale
                * gamut_sdiv(e_t * spow(a * a + b * b, 0.5f), R + G + 21.0f * B / 20.0f + 0.305f);
            float C = spow(t, 0.9f) * spow(J / 100.0f, 0.5f) * p.chromaTerm;
            float M = C * p.F_L4;
            float Jp = ((1.0f + 100.0f * 0.007f) * J) / (1.0f + 0.007f * J);
            float Mp = (1.0f / 0.0228f) * gamut_log1p(0.0228f * M);
            float hr = h * (M_PI_F / 180.0f);
            return float3(Jp, Mp * gamut_cos(hr), Mp * gamut_sin(hr));
        }

        // CAM16UCS.inverse, to the dRGB-adapted cone response.
        static inline float3 gamut_cam16_inverse(float3 jab, constant GamutParams &p) {
            const float c1 = 0.007f;
            const float c2 = 0.0228f;
            float J = -jab.x / (c1 * jab.x - 1.0f - 100.0f * c1);
            float Mp = gamut_hypot(jab.y, jab.z);
            float h = gamut_degrees(gamut_atan2(jab.z, jab.y));
            float M = gamut_expm1(Mp / (1.0f / c2)) / c2;
            float C = M / p.F_L4;
            // Swift's max(J, eps), which keeps a NaN J.
            float jSafe = 2.2204460492503131e-16f >= J ? 2.2204460492503131e-16f : J;
            float t = spow(C / (sqrt(jSafe / 100.0f) * p.chromaTerm), 1.0f / 0.9f);
            float e_t = gamut_eccentricity(h);
            float A = p.A_w * spow(J / 100.0f, p.inverseCz);
            float P_1 = gamut_sdiv(p.chromaScale * e_t, t);
            float P_2 = A / p.N_bb + 0.305f;
            float achromatic = A / p.N_bb;
            float P_3 = 21.0f / 20.0f;

            float hr = h * (M_PI_F / 180.0f);
            float sinH = gamut_sin(hr);
            float cosH = gamut_cos(hr);
            float cs = gamut_sdiv(cosH, sinH);
            float sc = gamut_sdiv(sinH, cosH);
            float P_4 = gamut_sdiv(P_1, sinH);
            float P_5 = gamut_sdiv(P_1, cosH);
            float nn = P_2 * (2.0f + P_3) * (460.0f / 1403.0f);

            // Three-way, as on the CPU: a NaN hue takes neither branch.
            float a = 0.0f;
            float b = 0.0f;
            if (fabs(sinH) >= fabs(cosH)) {
                b = nn / (P_4 + (2.0f + P_3) * (220.0f / 1403.0f) * cs - (27.0f / 1403.0f)
                    + P_3 * (6300.0f / 1403.0f));
                a = b * cs;
            } else if (fabs(sinH) < fabs(cosH)) {
                a = nn / (P_5 + (2.0f + P_3) * (220.0f / 1403.0f)
                    - ((27.0f / 1403.0f) - P_3 * (6300.0f / 1403.0f)) * sc);
                b = a * sc;
            }
            if (t == 0.0f) {
                a = 0.0f;
                b = 0.0f;
            }

            float3 response = float3(
                (460.0f * achromatic + 451.0f * a + 288.0f * b) / 1403.0f,
                (460.0f * achromatic - 891.0f * a - 261.0f * b) / 1403.0f,
                (460.0f * achromatic - 220.0f * a - 6300.0f * b) / 1403.0f);
            return float3(
                gamut_cam16_unadapt(response.x, p.F_L), gamut_cam16_unadapt(response.y, p.F_L),
                gamut_cam16_unadapt(response.z, p.F_L));
        }

        // OutputGamutCompressor.acesRGC.
        static inline float3 gamut_aces_rgc(float3 rgb, constant GamutParams &p) {
            // np.max propagates NaN, and a NaN achromatic value fails the test below.
            bool anyNaN = isnan(rgb.x) || isnan(rgb.y) || isnan(rgb.z);
            float ach = anyNaN ? NAN : max(max(rgb.x, rgb.y), rgb.z);
            if (!(ach > 1e-12f)) { return rgb; }
            float3 out;
            for (int k = 0; k < 3; ++k) {
                out[k] = ach * (1.0f - gamut_knee((ach - rgb[k]) / ach, p.threshold, p.limit,
                                                  p.power));
            }
            return out;
        }

        // OutputGamutCompressor.perceptual.
        static inline float3 gamut_perceptual(
            float3 rgb, constant float *m, constant float *table, constant GamutParams &p)
        {
            float3 cone = gamut_apply(m + GAMUT_TO_CONE, rgb);
            float3 lab;
            if (p.mode == 1 || p.mode == 2) {
                const float third = 1.0f / 3.0f;
                lab = gamut_apply(m + GAMUT_CONE_TO_OPPONENT,
                    float3(spow(cone.x, third), spow(cone.y, third), spow(cone.z, third)));
            } else if (p.mode == 3) {
                float3 iab = gamut_apply(m + GAMUT_CONE_TO_OPPONENT,
                    float3(gamut_pq_encode(cone.x), gamut_pq_encode(cone.y),
                           gamut_pq_encode(cone.z)));
                const float d = -0.56f;
                float jz = ((1.0f + d) * iab.x) / (1.0f + d * iab.x) - 1.6295499532821565e-11f;
                lab = float3(jz, iab.y, iab.z);
            } else {
                lab = gamut_cam16_forward(cone, p);
            }

            float L = lab.x;
            if (p.hasLightnessKnee != 0) {
                L = gamut_knee(L / p.lightnessWhite, p.lightnessThreshold, p.lightnessLimit,
                               p.lightnessPower) * p.lightnessWhite;
            }

            float chroma = gamut_hypot(lab.y, lab.z);
            float hue = gamut_atan2(lab.z, lab.y);
            float index = p.mode == 2 ? gamut_lr(L) : L;
            float maximum = fmax(gamut_cmax(table, index, hue, p), 1e-9f);
            float compressed = gamut_knee(chroma / maximum, p.threshold, p.limit, p.power)
                * maximum;
            float newA = compressed * gamut_cos(hue);
            float newB = compressed * gamut_sin(hue);

            float3 back;
            if (p.mode == 1 || p.mode == 2) {
                float3 q = gamut_apply(m + GAMUT_OPPONENT_TO_CONE, float3(L, newA, newB));
                back = float3(spow(q.x, 3.0f), spow(q.y, 3.0f), spow(q.z, 3.0f));
            } else if (p.mode == 3) {
                const float d = -0.56f;
                float shifted = L + 1.6295499532821565e-11f;
                float iz = shifted / (1.0f + d - d * shifted);
                float3 q = gamut_apply(m + GAMUT_OPPONENT_TO_CONE, float3(iz, newA, newB));
                back = float3(gamut_pq_decode(q.x), gamut_pq_decode(q.y), gamut_pq_decode(q.z));
            } else {
                back = gamut_cam16_inverse(float3(L, newA, newB), p);
            }
            return gamut_apply(m + GAMUT_CONE_TO_RGB, back);
        }

        kernel void gamut_compress(
            device float *x [[buffer(0)]], constant float *m [[buffer(1)]],
            constant float *table [[buffer(2)]], constant GamutParams &p [[buffer(3)]],
            uint i [[thread_position_in_grid]])
        {
            if (i >= p.pixels) { return; }
            float3 rgb = float3(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]);
            float3 out = p.mode == 0 ? gamut_aces_rgc(rgb, p) : gamut_perceptual(rgb, m, table, p);
            x[i * 3] = out.x;
            x[i * 3 + 1] = out.y;
            x[i * 3 + 2] = out.z;
        }
        """#
}

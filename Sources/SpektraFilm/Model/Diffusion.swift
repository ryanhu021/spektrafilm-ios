import Foundation

/// `model/diffusion.py`, plus `utils/numba_boost_hightlights.py`.
///
/// Four operators, applied in this order by the filming stage and never folded together:
/// highlight boost, the diffusion-filter PSF, lens blur, halation. The boost is re-exported by
/// `diffusion.py` but called one level up, from `FilmingStage.expose`, so it stays a separate
/// operator here. Folding it into ``applyHalation(_:_:pixelSizeMicrons:)`` would change the
/// pipeline order.
///
/// The printing stage runs a second, independent diffusion filter on the print-side exposure, and
/// the scanning stage runs a lens blur and an unsharp mask whose sigmas are in **pixels** while the
/// camera's lens blur is in **micrometres**.
public enum Diffusion {

    // MARK: - Highlight boost

    /// `boost_highlights`, out of place.
    public static func boostHighlights(
        _ image: ImageBuffer, boostEV: Double, boostRange: Double, protectEV: Double,
        midgray: Double = 0.184
    ) -> ImageBuffer {
        var out = image
        boostHighlights(
            &out, boostEV: boostEV, boostRange: boostRange, protectEV: protectEV, midgray: midgray)
        return out
    }

    /// `boost_highlights` with `out` aliased onto the input, which is how the filming stage calls it.
    ///
    /// The curve is frame-global: its shape depends on the maximum over the whole array, all
    /// channels together, so this is not a per-pixel transform and cannot be baked into a 3D LUT.
    /// `debug.lutMode` zeroes `boostEV` for that reason.
    public static func boostHighlights(
        _ image: inout ImageBuffer, boostEV: Double, boostRange: Double, protectEV: Double,
        midgray: Double = 0.184
    ) {
        precondition(boostEV >= 0, "boostEV must be >= 0, got \(boostEV)")
        precondition(
            boostRange >= 0 && boostRange <= 1, "boostRange must be in [0, 1], got \(boostRange)")
        precondition(protectEV >= 0, "protectEV must be >= 0, got \(protectEV)")
        precondition(midgray >= 0, "midgray must be >= 0, got \(midgray)")
        precondition(!image.values.isEmpty, "boostHighlights needs a non-empty image")

        if boostEV == 0 { return }

        let maxRaw = arrayMaximum(image.values)
        if maxRaw == 0 {
            // Upstream fills zeros here instead of copying, which discards any negative value.
            for i in image.values.indices { image.values[i] = 0 }
            return
        }

        let rawX0 = Swift.min(Swift.max(midgray * pow(2.0, protectEV), 0.0), maxRaw)
        if rawX0 == maxRaw { return }

        // 28 is the curvature at boostRange 0; boostRange 1 flattens it to 1.
        let a = pow(28.0, 1.0 - boostRange)
        let x0 = rawX0 / maxRaw
        let denominator = exp(a * (1.0 - x0)) - a * (1.0 - x0) - 1.0
        precondition(denominator > 0, "denominator for k is non-positive")

        // k solves y(maxRaw) == maxRaw * 2 ** boostEV exactly.
        let k = (pow(2.0, boostEV) - 1.0) / denominator
        let inverseMaxRaw = 1.0 / maxRaw
        let boostScale = k * maxRaw

        image.values.withUnsafeMutableBufferPointer { buf in
            for i in 0..<buf.count {
                let value = buf[i]
                if value <= rawX0 { continue }
                let dx = (value - rawX0) * inverseMaxRaw
                buf[i] = value + boostScale * (exp(a * dx) - a * dx - 1.0)
            }
        }
    }

    /// `np.max` over every element and channel, NaN included.
    private static func arrayMaximum(_ values: [Double]) -> Double {
        var maximum = -Double.infinity
        for value in values {
            if value.isNaN { return value }
            if value > maximum { maximum = value }
        }
        return maximum
    }

    // MARK: - Blur and sharpen entry points

    /// `apply_gaussian_blur`. Sigma in pixels, which is what the scanner's `lens_blur` carries.
    public static func applyGaussianBlur(_ image: ImageBuffer, sigmaPixels: Double) -> ImageBuffer {
        sigmaPixels > 0 ? GaussianFilter.apply(image, sigma: sigmaPixels) : image
    }

    /// `apply_gaussian_blur_um`. Sigma in image-plane micrometres, which is the camera's
    /// `lens_blur_um`.
    ///
    /// The zero check comes before the division deliberately: `pixelSizeMicrons` is unknown until
    /// the resize stage has run, and LUT bakes inject past that point with every spatial size
    /// already zeroed.
    public static func applyGaussianBlur(
        _ image: ImageBuffer, sigmaMicrons: Double, pixelSizeMicrons: Double?
    ) -> ImageBuffer {
        guard sigmaMicrons > 0 else { return image }
        guard let pixelSizeMicrons else {
            preconditionFailure("a non-zero lens blur needs a pixel size; run the resize stage first")
        }
        let sigma = sigmaMicrons / pixelSizeMicrons
        return sigma > 0 ? GaussianFilter.apply(image, sigma: sigma) : image
    }

    /// `apply_unsharp_mask`. Sigma in pixels.
    ///
    /// No guard of its own: the scanning stage is what checks `sigma > 0 && amount > 0`. Nothing is
    /// clipped, so ringing can drive the result negative, and does at `amount >= 1`.
    public static func applyUnsharpMask(
        _ image: ImageBuffer, sigma: Double, amount: Double
    ) -> ImageBuffer {
        let blurred = GaussianFilter.apply(image, sigma: sigma)
        var out = image
        out.values.withUnsafeMutableBufferPointer { dst in
            blurred.values.withUnsafeBufferPointer { src in
                for i in 0..<dst.count { dst[i] = dst[i] + amount * (dst[i] - src[i]) }
            }
        }
        return out
    }

    // MARK: - Halation

    /// `apply_halation_um`: in-emulsion scatter, then back-reflection.
    ///
    /// Both passes gate on "any channel has a non-zero size", so a channel whose size is zero still
    /// goes through the filter with its width floored to 1e-6. At that width the FIR radius is 0, a
    /// single unit tap, so the pass is an exact copy for that channel. Short-circuiting per channel
    /// instead would change the arithmetic.
    public static func applyHalation(
        _ raw: ImageBuffer, _ halation: HalationParams, pixelSizeMicrons: Double
    ) -> ImageBuffer {
        guard halation.active else { return raw }
        precondition(raw.channels == 3, "halation needs an RGB image, got \(raw.channels) channels")
        precondition(pixelSizeMicrons > 0, "pixel size must be positive")

        var result = raw

        // Pass 1: an energy-preserving mixture of a Gaussian core and an exponential tail, blended
        // with the identity by scatterAmount.
        let scatterAmount = halation.scatterAmount
        let tailWeight = triple(halation.scatterTailWeight)
        let coreSigma = triple(halation.scatterCoreMicrons).map {
            $0 * halation.scatterSpatialScale / pixelSizeMicrons
        }
        let tailLambda = triple(halation.scatterTailMicrons).map {
            $0 * halation.scatterSpatialScale / pixelSizeMicrons
        }
        if scatterAmount > 0
            && (coreSigma.contains { $0 > 0 } || tailLambda.contains { $0 > 0 })
        {
            let core = GaussianFilter.apply(
                result, sigmaPerChannel: coreSigma.map { Swift.max($0, 1e-6) })
            let tail = ExponentialFilter.apply(
                result, decayPerChannel: tailLambda.map { Swift.max($0, 1e-6) })
            result.values.withUnsafeMutableBufferPointer { dst in
                core.values.withUnsafeBufferPointer { c in
                    tail.values.withUnsafeBufferPointer { t in
                        for i in stride(from: 0, to: dst.count, by: 3) {
                            for channel in 0..<3 {
                                let w = tailWeight[channel]
                                let scattered = (1.0 - w) * c[i + channel] + w * t[i + channel]
                                dst[i + channel] =
                                    (1.0 - scatterAmount) * dst[i + channel]
                                    + scatterAmount * scattered
                            }
                        }
                    }
                }
            }
        }

        // Pass 2: additive multi-bounce back-reflection, widths spaced by sqrt(k).
        let strength = triple(halation.halationStrength).map { $0 * halation.halationAmount }
        let firstSigma = triple(halation.halationFirstSigmaMicrons).map {
            $0 * halation.halationSpatialScale / pixelSizeMicrons
        }
        let bounces = halation.halationBounceCount
        if bounces >= 1 && (strength.contains { $0 > 0 }) && (firstSigma.contains { $0 > 0 }) {
            var decay = (1...bounces).map { pow(halation.halationBounceDecay, Double($0 - 1)) }
            let decayTotal = decay.reduce(0, +)
            for i in decay.indices { decay[i] /= decayTotal }

            var blur = ImageBuffer(height: raw.height, width: raw.width, channels: 3)
            for k in 1...bounces {
                let width = Double(k).squareRoot()
                let component = GaussianFilter.apply(
                    result, sigmaPerChannel: firstSigma.map { Swift.max($0 * width, 1e-6) })
                let weight = decay[k - 1]
                blur.values.withUnsafeMutableBufferPointer { dst in
                    component.values.withUnsafeBufferPointer { src in
                        for i in 0..<dst.count { dst[i] += weight * src[i] }
                    }
                }
            }
            result.values.withUnsafeMutableBufferPointer { dst in
                blur.values.withUnsafeBufferPointer { src in
                    for i in stride(from: 0, to: dst.count, by: 3) {
                        for channel in 0..<3 {
                            dst[i + channel] += strength[channel] * src[i + channel]
                        }
                    }
                }
            }
            if halation.halationRenormalize {
                result.values.withUnsafeMutableBufferPointer { dst in
                    for i in stride(from: 0, to: dst.count, by: 3) {
                        for channel in 0..<3 { dst[i + channel] /= 1.0 + strength[channel] }
                    }
                }
            }
        }

        return result
    }

    // MARK: - Diffusion-filter PSF families

    /// One `{core|halo|bloom}` block of a family shape.
    public struct PSFGroup: Sendable, Equatable {
        /// Centre of the geometric progression, in image-plane micrometres.
        public var lambdaMicrons: Double
        /// Sub-component lambdas span `[lambdaMicrons / spread, lambdaMicrons * spread]`.
        public var spread: Double
        public var componentCount: Int
        /// Read only for the bloom, where the within-group weights follow `lambda ** (2 - alpha)`
        /// so the assembled group decays as `r ** -alpha`.
        public var alpha: Double

        public init(lambdaMicrons: Double, spread: Double, componentCount: Int, alpha: Double = 3.0) {
            self.lambdaMicrons = lambdaMicrons
            self.spread = spread
            self.componentCount = componentCount
            self.alpha = alpha
        }
    }

    /// An entry of `_DIFFUSION_FILTER_SHAPES`. The three family weights sum to 1.
    public struct PSFShape: Sendable, Equatable {
        public var core: PSFGroup
        public var halo: PSFGroup
        public var bloom: PSFGroup
        public var coreWeight: Double
        public var haloWeight: Double
        public var bloomWeight: Double
        public var haloWarmthBase: Double
    }

    /// `_DIFFUSION_FILTER_SHAPES[family]`.
    public static func shape(for family: DiffusionFilterParams.Family) -> PSFShape {
        switch family {
        case .glimmerglass:
            return PSFShape(
                core: PSFGroup(lambdaMicrons: 10.0, spread: 1.5, componentCount: 2),
                halo: PSFGroup(lambdaMicrons: 50.0, spread: 2.0, componentCount: 3),
                bloom: PSFGroup(lambdaMicrons: 260.0, spread: 2.5, componentCount: 4, alpha: 3.2),
                coreWeight: 0.60, haloWeight: 0.30, bloomWeight: 0.10, haloWarmthBase: 0.0)
        case .blackProMist:
            return PSFShape(
                core: PSFGroup(lambdaMicrons: 16.0, spread: 1.5, componentCount: 2),
                halo: PSFGroup(lambdaMicrons: 95.0, spread: 2.0, componentCount: 3),
                bloom: PSFGroup(lambdaMicrons: 380.0, spread: 2.5, componentCount: 4, alpha: 3.5),
                coreWeight: 0.40, haloWeight: 0.47, bloomWeight: 0.13, haloWarmthBase: 0.65)
        case .proMist:
            return PSFShape(
                core: PSFGroup(lambdaMicrons: 14.0, spread: 1.5, componentCount: 2),
                halo: PSFGroup(lambdaMicrons: 150.0, spread: 2.0, componentCount: 3),
                bloom: PSFGroup(lambdaMicrons: 650.0, spread: 2.5, componentCount: 4, alpha: 2.9),
                coreWeight: 0.28, haloWeight: 0.42, bloomWeight: 0.30, haloWarmthBase: 0.40)
        case .cinebloom:
            return PSFShape(
                core: PSFGroup(lambdaMicrons: 20.0, spread: 1.5, componentCount: 2),
                halo: PSFGroup(lambdaMicrons: 200.0, spread: 2.0, componentCount: 3),
                bloom: PSFGroup(lambdaMicrons: 1000.0, spread: 2.5, componentCount: 4, alpha: 2.5),
                coreWeight: 0.22, haloWeight: 0.30, bloomWeight: 0.48, haloWarmthBase: 0.85)
        }
    }

    /// `_DIFFUSION_FAMILY_TOTAL_GAIN`: the family's deflection efficiency at a given commercial stop.
    ///
    /// Black Pro-Mist's "deep blacks" character lives here and in its low bloom weight. The model is
    /// energy conserving and absorbs nothing, whatever the GUI tooltip says.
    public static func totalGain(for family: DiffusionFilterParams.Family) -> Double {
        switch family {
        case .glimmerglass: return 0.65
        case .blackProMist: return 0.75
        case .proMist: return 1.05
        case .cinebloom: return 1.00
        }
    }

    /// `_DIFFUSION_STRENGTH_BREAKPOINTS`, the commercial filter stops.
    static let strengthBreakpoints = [0.125, 0.25, 0.5, 1.0, 2.0]
    /// `_DIFFUSION_STRENGTH_TOTAL_FRACTION`, the pre-gain deflected fraction at each stop.
    static let strengthTotalFraction = [0.10, 0.20, 0.35, 0.55, 0.75]

    /// `_strength_to_scatter`: strength and family to `p_s`, the deflected-photon fraction.
    ///
    /// Interpolated linearly in `log2(strength)`. `numpy.interp` clamps outside its breakpoints, so
    /// the fraction holds at 0.10 below strength 0.125 and at 0.75 above 2.0. Extrapolating instead
    /// would run away at both ends. The 0.99 ceiling is unreachable with the shipped gains, whose
    /// maximum product is 0.7875.
    public static func strengthToScatter(
        _ strength: Double, family: DiffusionFilterParams.Family
    ) -> Double {
        if strength <= 0 { return 0.0 }
        let logStrength = log2(Swift.max(strength, 1e-6))
        let base = Interpolation.npInterp(
            query: [logStrength], xp: strengthBreakpoints.map(log2), fp: strengthTotalFraction)[0]
        return Swift.min(Swift.max(base * totalGain(for: family), 0.0), 0.99)
    }

    /// `_expand_group`: a group block to sub-component lambdas and within-group weights summing to 1.
    static func expandGroup(
        _ group: PSFGroup, isBloom: Bool
    ) -> (lambdas: [Double], weights: [Double]) {
        let count = Swift.max(group.componentCount, 1)
        if count == 1 || group.spread <= 1.0 { return ([group.lambdaMicrons], [1.0]) }
        let lambdas = linspace(
            log(group.lambdaMicrons / group.spread), log(group.lambdaMicrons * group.spread),
            count: count
        ).map(exp)
        var weights =
            isBloom
            ? lambdas.map { pow($0, 2.0 - group.alpha) }
            : [Double](repeating: 1.0, count: count)
        let total = weights.reduce(0, +)
        for i in weights.indices { weights[i] /= total }
        return (lambdas, weights)
    }

    /// `_HALO_CHANNEL_WARMTH_AXIS`, biased toward yellow-green so a warm outer halo reads warm-yellow.
    static let haloWarmthAxis = [1.30, 0.15, -1.45]

    /// `_halo_channel_weights`: energy-conserving per-channel redistribution across the halo's
    /// sub-components. Returns three rows of `weights.count` entries, each row summing to
    /// `weights.sum()`.
    ///
    /// The clip to zero before the renormalise is not decoration. Cinebloom at its family base of
    /// 0.85 drives red's innermost weight to `1 + 0.85 * 1.30 * -1 = -0.105`, so red clips to 0, the
    /// row sums to 1.035, and the renormalise divides it back. Skipping the clip leaves a negative
    /// weight and a silently different halo colour.
    static func haloChannelWeights(_ weights: [Double], warmth: Double) -> [[Double]] {
        let count = weights.count
        if count < 2 { return [weights, weights, weights] }
        let clamped = Swift.min(Swift.max(warmth, -1.5), 1.5)

        // Inner to outer gradient, re-centred so it sums to zero against the weights.
        var gradient = linspace(-1.0, 1.0, count: count)
        let targetTotal = weights.reduce(0, +)
        var weighted = 0.0
        for i in 0..<count { weighted += gradient[i] * weights[i] }
        let average = weighted / targetTotal
        for i in 0..<count { gradient[i] -= average }

        var out: [[Double]] = []
        for channel in 0..<3 {
            var row = (0..<count).map { i in
                Swift.max(weights[i] * (1.0 + clamped * haloWarmthAxis[channel] * gradient[i]), 0.0)
            }
            let sum = row.reduce(0, +)
            if sum > 0 {
                for i in row.indices { row[i] *= targetTotal / sum }
            } else {
                row = weights
            }
            out.append(row)
        }
        return out
    }

    /// Per-group multipliers off a ``DiffusionFilterParams``.
    public struct PSFOverrides: Sendable, Equatable {
        public var coreIntensity: Double
        public var haloIntensity: Double
        public var bloomIntensity: Double
        public var coreSize: Double
        public var haloSize: Double
        public var bloomSize: Double

        public init(
            coreIntensity: Double = 1.0, haloIntensity: Double = 1.0, bloomIntensity: Double = 1.0,
            coreSize: Double = 1.0, haloSize: Double = 1.0, bloomSize: Double = 1.0
        ) {
            self.coreIntensity = coreIntensity
            self.haloIntensity = haloIntensity
            self.bloomIntensity = bloomIntensity
            self.coreSize = coreSize
            self.haloSize = haloSize
            self.bloomSize = bloomSize
        }

        var isIdentity: Bool {
            coreIntensity == 1.0 && haloIntensity == 1.0 && bloomIntensity == 1.0
                && coreSize == 1.0 && haloSize == 1.0 && bloomSize == 1.0
        }
    }

    /// `_overrides_from_params`. Nil when all six multipliers are exactly 1, by float equality.
    static func overrides(from params: DiffusionFilterParams) -> PSFOverrides? {
        let out = PSFOverrides(
            coreIntensity: params.coreIntensity, haloIntensity: params.haloIntensity,
            bloomIntensity: params.bloomIntensity, coreSize: params.coreSize,
            haloSize: params.haloSize, bloomSize: params.bloomSize)
        return out.isIdentity ? nil : out
    }

    /// `_resolve_family_cfg`.
    ///
    /// The intensities scale the three family weights, which are then renormalised so they still sum
    /// to 1, leaving the strength-to-`p_s` mapping alone. The sizes scale each group's `lambdaMicrons`
    /// only: `spread`, `componentCount` and `alpha` are never touched.
    ///
    /// Two clamps and one early return are load-bearing:
    ///
    /// - a negative intensity clamps to 0, and the other two renormalise around it;
    /// - a size of 0 becomes 1e-6, not 0;
    /// - **all three intensities at 0 reverts to the unmodified family**, discarding the size
    ///   overrides along with them.
    static func resolve(
        _ family: DiffusionFilterParams.Family, overrides: PSFOverrides?
    ) -> PSFShape {
        let base = shape(for: family)
        guard let overrides, !overrides.isIdentity else { return base }

        let coreWeight = base.coreWeight * Swift.max(overrides.coreIntensity, 0.0)
        let haloWeight = base.haloWeight * Swift.max(overrides.haloIntensity, 0.0)
        let bloomWeight = base.bloomWeight * Swift.max(overrides.bloomIntensity, 0.0)
        let total = coreWeight + haloWeight + bloomWeight
        if total <= 0 { return base }

        var out = base
        out.core.lambdaMicrons *= Swift.max(overrides.coreSize, 1e-6)
        out.halo.lambdaMicrons *= Swift.max(overrides.haloSize, 1e-6)
        out.bloom.lambdaMicrons *= Swift.max(overrides.bloomSize, 1e-6)
        out.coreWeight = coreWeight / total
        out.haloWeight = haloWeight / total
        out.bloomWeight = bloomWeight / total
        return out
    }

    /// `_bloom_max_lambda_um`: the overridden lambda times the untouched spread.
    static func bloomMaxLambdaMicrons(
        _ family: DiffusionFilterParams.Family, overrides: PSFOverrides?
    ) -> Double {
        let bloom = resolve(family, overrides: overrides).bloom
        return bloom.lambdaMicrons * bloom.spread
    }

    /// `_radial_components`.
    ///
    /// `radius` is in pixels when `pixelSizeMicrons` is the real pixel pitch, or in micrometres when
    /// `pixelSizeMicrons` is 1. `warmth` arrives already summed with the family base: this does not
    /// add the base itself, so adding it twice is the easy mistake.
    static func radialComponents(
        radius: [Double], family: DiffusionFilterParams.Family, spatialScale: Double,
        pixelSizeMicrons: Double, warmth: Double, overrides: PSFOverrides?
    ) -> (core: [Double], halo: [[Double]], bloom: [Double]) {
        let cfg = resolve(family, overrides: overrides)
        let scale = Swift.max(spatialScale, 1e-6)

        let (coreLambdas, coreWeights) = expandGroup(cfg.core, isBloom: false)
        let (haloLambdas, haloWeights) = expandGroup(cfg.halo, isBloom: false)
        let (bloomLambdas, bloomWeights) = expandGroup(cfg.bloom, isBloom: true)
        let haloPerChannel = haloChannelWeights(haloWeights, warmth: warmth)

        let toPixels = { (lambdas: [Double]) in lambdas.map { $0 * scale / pixelSizeMicrons } }
        let corePixels = toPixels(coreLambdas)
        let haloPixels = toPixels(haloLambdas)
        let bloomPixels = toPixels(bloomLambdas)

        // The 1e-6 floor is in pixel units, so an absurd spatialScale turns a component into a
        // 1.6e10-tall single-pixel spike that the later sum normalisation absorbs.
        func exponentialSum(_ lambdasPixels: [Double], _ weights: [Double]) -> [Double] {
            var total = [Double](repeating: 0, count: radius.count)
            for (weight, rawLambda) in zip(weights, lambdasPixels) {
                let lambda = Swift.max(rawLambda, 1e-6)
                let denominator = (2.0 * Double.pi) * (lambda * lambda)
                for i in radius.indices {
                    total[i] += weight * exp(-radius[i] / lambda) / denominator
                }
            }
            return total
        }

        let core = exponentialSum(corePixels, coreWeights).map { cfg.coreWeight * $0 }
        let bloom = exponentialSum(bloomPixels, bloomWeights).map { cfg.bloomWeight * $0 }
        let halo = (0..<3).map { channel in
            exponentialSum(haloPixels, haloPerChannel[channel]).map { cfg.haloWeight * $0 }
        }
        return (core, halo, bloom)
    }

    /// `diffusion_filter_radial_profile`: the analytic continuum profile in `1 / um**2`.
    ///
    /// No normalisation, unlike ``diffusionFilterPSF(kernelHeight:kernelWidth:family:spatialScale:pixelSizeMicrons:haloWarmth:overrides:)``,
    /// and no grid or truncation coupling. Nothing in the pipeline calls it; it exists as an
    /// analysis and test hook.
    public static func diffusionFilterRadialProfile(
        radiusMicrons: [Double], family: DiffusionFilterParams.Family = .blackProMist,
        spatialScale: Double = 1.0, haloWarmth: Double = 0.0, overrides: PSFOverrides? = nil
    ) -> (core: [Double], halo: [[Double]], bloom: [Double], totalPerChannel: [[Double]]) {
        let cfg = resolve(family, overrides: overrides)
        let parts = radialComponents(
            radius: radiusMicrons, family: family, spatialScale: spatialScale,
            pixelSizeMicrons: 1.0, warmth: cfg.haloWarmthBase + haloWarmth, overrides: overrides)
        let totalPerChannel = (0..<3).map { channel in
            radiusMicrons.indices.map { i in
                parts.halo[channel][i] + parts.core[i] + parts.bloom[i]
            }
        }
        return (parts.core, parts.halo, parts.bloom, totalPerChannel)
    }

    /// `diffusion_filter_psf`: the sampled per-channel PSF, each channel sum-normalised on the grid.
    ///
    /// The kernel is always odd-sided with the centre on a sample, so the PSF is exactly symmetric
    /// in both axes and convolution equals correlation. ``FFTConvolve2D`` relies on that and does not
    /// flip the kernel.
    ///
    /// Sum normalisation absorbs both the truncation loss and the `1 / (2 pi lambda^2)` prefactors,
    /// so only the relative radial shape survives. The truncation is not small: on a 29x29 grid at a
    /// 100 um pitch the pre-normalisation red sum is 4.0028.
    public static func diffusionFilterPSF(
        kernelHeight: Int, kernelWidth: Int, family: DiffusionFilterParams.Family,
        spatialScale: Double, pixelSizeMicrons: Double, haloWarmth: Double = 0.0,
        overrides: PSFOverrides? = nil
    ) -> ImageBuffer {
        precondition(kernelHeight > 0 && kernelWidth > 0, "PSF grid must be non-empty")
        let cfg = resolve(family, overrides: overrides)
        let centreY = kernelHeight / 2
        let centreX = kernelWidth / 2

        // Integer squares before the square root, matching the integer `ogrid` arithmetic.
        var radius = [Double](repeating: 0, count: kernelHeight * kernelWidth)
        for y in 0..<kernelHeight {
            let dy = y - centreY
            for x in 0..<kernelWidth {
                let dx = x - centreX
                radius[y * kernelWidth + x] = Double(dx * dx + dy * dy).squareRoot()
            }
        }

        let parts = radialComponents(
            radius: radius, family: family, spatialScale: spatialScale,
            pixelSizeMicrons: pixelSizeMicrons, warmth: cfg.haloWarmthBase + haloWarmth,
            overrides: overrides)

        var psf = ImageBuffer(height: kernelHeight, width: kernelWidth, channels: 3)
        for channel in 0..<3 {
            var plane = [Double](repeating: 0, count: radius.count)
            var sum = 0.0
            for i in plane.indices {
                let value = parts.core[i] + parts.halo[channel][i] + parts.bloom[i]
                plane[i] = value
                sum += value
            }
            psf.values.withUnsafeMutableBufferPointer { dst in
                for i in plane.indices { dst[i * 3 + channel] = plane[i] / sum }
            }
        }
        return psf
    }

    // MARK: - Diffusion filter

    /// Peak transient allocation the diffusion filter may make before it refuses the job.
    ///
    /// 256 MB. Measured on 3:2 frames at 35 mm and `spatialScale` 1, that covers a long edge up to
    /// 1616 px for glimmerglass, 1360 for black_pro_mist and 1088 for pro_mist and cinebloom. See
    /// ``applyDiffusionFilter(_:_:pixelSizeMicrons:memoryBudgetBytes:)`` for why the ceiling is this
    /// low and why no FFT arrangement raises it.
    public static let defaultMemoryBudgetBytes = 256 << 20

    /// Peak transient bytes ``applyDiffusionFilter(_:_:pixelSizeMicrons:memoryBudgetBytes:)`` would
    /// allocate: the two spectra, the per-channel PSF, and one mirror-padded plane.
    public static func diffusionFilterPeakBytes(
        imageHeight: Int, imageWidth: Int, radius: Int
    ) -> Int {
        let side = 2 * radius + 1
        let paddedHeight = imageHeight + 2 * radius
        let paddedWidth = imageWidth + 2 * radius
        let transforms = FFTConvolve2D.scratchBytes(
            imageHeight: paddedHeight, imageWidth: paddedWidth, kernelHeight: side, kernelWidth: side)
        let psf = side * side * 3 * MemoryLayout<Double>.size
        let padded = paddedHeight * paddedWidth * MemoryLayout<Double>.size
        return transforms + psf + padded
    }

    /// The PSF radius `apply_diffusion_filter_um` would pick.
    ///
    /// `ceil(max(8 * lambda_bloom_max_px, 5))`, then clamped to `max(min(H, W) / 2 - 1, 1)`. The 8x
    /// budget comes from the 2D radial CDF of a single exponential, `1 - (1 + r/l) exp(-r/l)`, which
    /// reaches 99.95% at `r = 8 l`. The clamp is not a formality: on a 6000x4000 frame it drags
    /// pro_mist from 2229 and cinebloom from 3429 down to 1999.
    public static func diffusionKernelRadius(
        _ params: DiffusionFilterParams, pixelSizeMicrons: Double, imageHeight: Int, imageWidth: Int
    ) -> Int {
        precondition(pixelSizeMicrons > 0, "pixel size must be positive")
        let bloomMaxPixels =
            bloomMaxLambdaMicrons(params.family, overrides: overrides(from: params))
            * params.spatialScale / pixelSizeMicrons
        let bound = Swift.max(Swift.min(imageHeight, imageWidth) / 2 - 1, 1)
        let wanted = Swift.max(8.0 * bloomMaxPixels, 5.0).rounded(.up)
        return wanted >= Double(bound) ? bound : Int(wanted)
    }

    /// `apply_diffusion_filter_um`: the energy-conserving convex combination
    /// `E_out = (1 - p_s) E_in + p_s (K_s * E_in)`.
    ///
    /// The boundary is `numpy.pad(mode: "reflect")`, whole-sample symmetric with the edge sample
    /// shared, which is ``BoundaryIndex/mirrorEdgeShared(_:count:)`` and *not* the fold the FIR blur
    /// uses. Reconstructing this operator with the other convention lands 4.7e-4 away on a random
    /// 48x60 frame and 6.6e-6 on a smoother one, so the error can sit inside the 1e-4 gate. The tests
    /// check the boundary directly as well as against a golden for that reason.
    ///
    /// - Throws: ``SpektraError/unsupportedSetting(_:value:)`` when the peak allocation would exceed
    ///   `memoryBudgetBytes`. The radius grows with resolution until the clamp catches it, and the
    ///   transform has to span the image plus four radii, so the working set grows faster than the
    ///   frame. Measured for black_pro_mist on 3:2 frames: 130 MB at a 1024 px long edge, 2.1 GB at
    ///   4000 px, 4.4 GB at 6000 px, where pro_mist and cinebloom both reach 6.7 GB. No FFT
    ///   arrangement fixes that. Overlap-save tiles cannot be smaller than the kernel, which at the
    ///   6000 px clamp is 3999x3999, so tiling raises the floor instead of lowering it. Callers that
    ///   hit this have to convolve at a lower working resolution.
    public static func applyDiffusionFilter(
        _ image: ImageBuffer, _ params: DiffusionFilterParams, pixelSizeMicrons: Double?,
        memoryBudgetBytes: Int = defaultMemoryBudgetBytes
    ) throws -> ImageBuffer {
        guard params.active else { return image }
        guard params.strength > 0, params.spatialScale > 0 else { return image }
        let scatterFraction = strengthToScatter(params.strength, family: params.family)
        guard scatterFraction > 0 else { return image }
        guard let pixelSizeMicrons else {
            preconditionFailure("an active diffusion filter needs a pixel size")
        }
        // The channel loop indexes a PSF that is always 3 wide, so a 1- or 4-channel image would
        // drop channels or read out of bounds.
        precondition(
            image.channels == 3, "the diffusion filter needs an RGB image, got \(image.channels)")

        let radius = diffusionKernelRadius(
            params, pixelSizeMicrons: pixelSizeMicrons, imageHeight: image.height,
            imageWidth: image.width)
        let kernelSide = 2 * radius + 1
        let paddedHeight = image.height + 2 * radius
        let paddedWidth = image.width + 2 * radius

        let peak = diffusionFilterPeakBytes(
            imageHeight: image.height, imageWidth: image.width, radius: radius)
        if peak > memoryBudgetBytes {
            throw SpektraError.unsupportedSetting(
                "diffusionFilter on \(image.width)x\(image.height) at a \(radius) px PSF radius",
                value: "peaks at \(peak >> 20) MB, budget is \(memoryBudgetBytes >> 20) MB"
            )
        }

        let psf = diffusionFilterPSF(
            kernelHeight: kernelSide, kernelWidth: kernelSide, family: params.family,
            spatialScale: params.spatialScale, pixelSizeMicrons: pixelSizeMicrons,
            haloWarmth: params.haloWarmth, overrides: overrides(from: params))

        var out = image
        for channel in 0..<3 {
            let blurred = FFTConvolve2D.convolveValid(
                image: mirrorPaddedPlane(image, channel: channel, radius: radius),
                height: paddedHeight, width: paddedWidth,
                kernel: GaussianFilter.plane(of: psf, channel: channel),
                kernelHeight: kernelSide, kernelWidth: kernelSide)
            out.values.withUnsafeMutableBufferPointer { dst in
                image.values.withUnsafeBufferPointer { src in
                    for p in 0..<blurred.count {
                        dst[p * 3 + channel] =
                            (1.0 - scatterFraction) * src[p * 3 + channel]
                            + scatterFraction * blurred[p]
                    }
                }
            }
        }
        return out
    }

    /// One channel of `numpy.pad(image, radius, mode: "reflect")`.
    ///
    /// Per channel because at a large radius the padded frame is the biggest single allocation in
    /// the operator: at a 1024 px long edge with cinebloom it is 65 MB for three channels.
    static func mirrorPaddedPlane(_ image: ImageBuffer, channel: Int, radius: Int) -> [Double] {
        let height = image.height + 2 * radius
        let width = image.width + 2 * radius
        var out = [Double](repeating: 0, count: height * width)
        let rows = (0..<height).map { BoundaryIndex.mirrorEdgeShared($0 - radius, count: image.height) }
        let columns = (0..<width).map {
            BoundaryIndex.mirrorEdgeShared($0 - radius, count: image.width)
        }
        let channels = image.channels
        image.values.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let source = rows[y] * image.width
                    let target = y * width
                    for x in 0..<width {
                        dst[target + x] = src[(source + columns[x]) * channels + channel]
                    }
                }
            }
        }
        return out
    }

    private static func triple(_ value: (Double, Double, Double)) -> [Double] {
        [value.0, value.1, value.2]
    }
}

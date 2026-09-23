import Foundation

/// The stochastic silver-halide particle model, the last step of development.
///
/// Ports `model/grain.py`. A pixel of film area holds a finite number of particles, a
/// density-dependent fraction of them develop, and the resulting density is quantised to whole
/// particles. The mean is preserved by construction and the variance follows the density.
///
/// Two topologies, selected by ``GrainParams/sublayersActive``. The layered one is the production
/// default: the channel density is split into three emulsion sublayers, each sublayer gets its own
/// particle population and its own particle size, and the three are summed.
///
/// What the reference does that this does not:
///
/// - The two-stage `Poisson(N / sat)` then `Binomial(K, p)` draw. Poisson thinning makes that
///   exactly `Poisson(N * p / sat)`, so one draw per pixel replaces two and no binomial sampler is
///   needed. Distributionally identical, verified by the closed-form moment gates.
/// - `use_fast_stats`. Upstream's Numba kernels replace both draws with normal approximations that
///   erase the skewness, and their thread-local RNG state makes the result depend on the thread
///   count. Not a behaviour to reproduce, so there is no knob for it.
/// - `method='gamma_beta'` and the `fixed_seed` argument. Both dead upstream, and `fixed_seed` is
///   inverted relative to its name.
public enum Grain {

    /// The clip `np.clip(density / density_max, 1e-6, 1 - 1e-6)` applies to the development
    /// probability. A density of -0.5, -0.01 and 0.0 all land on the floor and produce the same
    /// output plane.
    public static let probabilityFloor = 1e-6
    public static let probabilityCeiling = 1.0 - 1e-6

    /// The factor in `sat = 1 - p * u * (1 - 1e-6)`, which keeps `sat` above zero at `p = 1 - 1e-6,
    /// u = 1`. It bottoms out near 2e-6, which is what drives lambda to 1.2e8.
    public static let saturationScale = 1.0 - 1e-6

    /// The layered path's sublayer count is hardcoded upstream and is not `n_sub_layers`.
    public static let sublayerCount = 3

    /// Stream index the micro-structure clumping field uses, one past the three particle sublayers
    /// so it cannot collide with them.
    static let microStructureStream = sublayerCount

    // MARK: - Derived parameters

    /// Everything `apply_grain_to_density` computes before the first random draw.
    ///
    /// Pure arithmetic over the profile and the parameters, so it is gated exactly.
    public struct SingleLayerParameters: Sendable, Equatable {
        /// `nanmax(density_curves, axis=0)`, the usable range above base and fog.
        public let densityMaxCurves: [Double]
        public let densityMin: [Double]
        /// `density_max_curves + density_min`, the saturation point the probability divides by.
        public let densityMax: [Double]
        public let pixelArea: Double
        /// `particle_area_um2 * particle_scale`, per channel.
        public let particleArea: [Double]
        /// Already divided by ``subLayerCount``, as the reference divides it.
        public let particlesPerPixel: [Double]
        public let uniformity: [Double]
        public let subLayerCount: Int
        public let blurSigmaPixels: Double

        public init(params: GrainParams, pixelSizeMicrons: Double, densityMaxCurves: [Double]) {
            precondition(densityMaxCurves.count == 3, "density_max_curves must have 3 entries")
            let minimum = tupleToArray(params.densityMin)
            let scale = tupleToArray(params.particleScale)
            let area = (0..<3).map { params.particleAreaMicronsSquared * scale[$0] }
            let pixel = pixelSizeMicrons * pixelSizeMicrons
            let count = max(1, params.subLayerCount)

            self.densityMaxCurves = densityMaxCurves
            densityMin = minimum
            densityMax = (0..<3).map { densityMaxCurves[$0] + minimum[$0] }
            pixelArea = pixel
            particleArea = area
            // The reference only divides when the count exceeds 1, which is the same number.
            particlesPerPixel = (0..<3).map { pixel / area[$0] / Double(count) }
            uniformity = tupleToArray(params.uniformity)
            subLayerCount = count
            blurSigmaPixels = params.blur
        }

        /// `density_max / n_particles_per_pixel`, the density one developed particle contributes.
        public var odParticle: [Double] {
            (0..<3).map { densityMax[$0] / particlesPerPixel[$0] }
        }

        /// Mean, variance, skewness and excess kurtosis of the output at a flat input density.
        ///
        /// From `grain.md` section 4.2: the output is a sum of ``subLayerCount`` independent scaled
        /// Poissons, each seeing the full channel density with `N / subLayerCount` particles.
        public func closedFormMoments(density: Double, channel: Int) -> ParticleMoments {
            let population = ParticlePopulation(
                density: density + densityMin[channel],
                densityMax: densityMax[channel],
                particlesPerPixel: particlesPerPixel[channel],
                uniformity: uniformity[channel],
                // The sum over repeats is divided by the count, so each contribution shrinks.
                weight: 1.0 / Double(subLayerCount))
            return ParticleMoments(
                summing: Array(repeating: population, count: subLayerCount),
                offset: -densityMin[channel])
        }
    }

    /// Everything `apply_grain_to_density_layers` computes before the first random draw.
    ///
    /// Every `[sublayer][channel]` table is flattened to 9 entries, indexed `sublayer * 3 +
    /// channel`, which is also how `density_curves_layers` is laid out per exposure step.
    public struct LayeredParameters: Sendable, Equatable {
        /// `nanmax(density_curves_layers, axis=0)`, before `density_min_layers` is folded in.
        public let densityMaxLayersRaw: [Double]
        /// Sum over sublayers, per channel. Upstream's comment calls this `[sublayers, rgb]`; it is
        /// three values, one per channel.
        public let densityMaxTotal: [Double]
        /// Each sublayer's share of its channel's total. Columns sum to 1 exactly, which is what
        /// balances the final `-= density_min`.
        public let densityMaxFractions: [Double]
        public let densityMinLayers: [Double]
        /// `density_max_layers_raw + density_min_layers`. The reference rebinds the same name, and
        /// this is the value the saturation point uses.
        public let densityMaxLayers: [Double]
        public let densityMin: [Double]
        public let pixelSizeMicrons: Double
        public let pixelArea: Double
        public let particleAreaLayers: [Double]
        public let particlesPerPixel: [Double]
        public let odParticle: [Double]
        /// `blur_dye_clouds_um * sqrt(od_particle)`, in pixels.
        ///
        /// `blur_dye_clouds_um` is dimensionless despite its name, so the physical sigma is
        /// resolution invariant and the pixel sigma is not. At 8.75 um per pixel every entry is
        /// under 1/6 and the blur is an exact identity.
        public let dyeCloudSigmaPixels: [Double]
        public let uniformity: [Double]
        public let blurSigmaPixels: Double
        public let blurDyeClouds: Double
        public let microStructure: (Double, Double)
        public let microStructureBlurPixels: Double
        public let microStructureSigma: Double

        public init(
            params: GrainParams, pixelSizeMicrons: Double, densityMaxLayers rawMaxima: [Double]
        ) {
            precondition(rawMaxima.count == 9, "density_max_layers must have 9 entries")
            let minimum = tupleToArray(params.densityMin)
            let channelScale = tupleToArray(params.particleScale)
            let layerScale = tupleToArray(params.particleScaleLayers)

            var total = [Double](repeating: 0, count: 3)
            for sublayer in 0..<sublayerCount {
                for channel in 0..<3 { total[channel] += rawMaxima[sublayer * 3 + channel] }
            }
            var fractions = [Double](repeating: 0, count: 9)
            var minLayers = [Double](repeating: 0, count: 9)
            var maxLayers = [Double](repeating: 0, count: 9)
            var area = [Double](repeating: 0, count: 9)
            var particles = [Double](repeating: 0, count: 9)
            var od = [Double](repeating: 0, count: 9)
            var dyeSigma = [Double](repeating: 0, count: 9)
            let pixel = pixelSizeMicrons * pixelSizeMicrons

            for sublayer in 0..<sublayerCount {
                for channel in 0..<3 {
                    let i = sublayer * 3 + channel
                    fractions[i] = rawMaxima[i] / total[channel]
                    minLayers[i] = fractions[i] * minimum[channel]
                    maxLayers[i] = rawMaxima[i] + minLayers[i]
                    area[i] =
                        params.particleAreaMicronsSquared * channelScale[channel]
                        * layerScale[sublayer]
                    particles[i] = pixel * fractions[i] / area[i]
                    od[i] = maxLayers[i] / particles[i]
                    dyeSigma[i] = params.blurDyeCloudsMicrons * od[i].squareRoot()
                }
            }

            densityMaxLayersRaw = rawMaxima
            densityMaxTotal = total
            densityMaxFractions = fractions
            densityMinLayers = minLayers
            self.densityMaxLayers = maxLayers
            densityMin = minimum
            self.pixelSizeMicrons = pixelSizeMicrons
            pixelArea = pixel
            particleAreaLayers = area
            particlesPerPixel = particles
            odParticle = od
            dyeCloudSigmaPixels = dyeSigma
            uniformity = tupleToArray(params.uniformity)
            blurSigmaPixels = params.blur
            blurDyeClouds = params.blurDyeCloudsMicrons
            microStructure = params.microStructure
            microStructureBlurPixels = params.microStructure.0 / pixelSizeMicrons
            // The second entry is documented as nanometres, hence the factor of 1000.
            microStructureSigma = params.microStructure.1 * 0.001 / pixelSizeMicrons
        }

        public static func == (a: LayeredParameters, b: LayeredParameters) -> Bool {
            a.densityMaxLayersRaw == b.densityMaxLayersRaw && a.densityMin == b.densityMin
                && a.pixelArea == b.pixelArea && a.particleAreaLayers == b.particleAreaLayers
                && a.uniformity == b.uniformity && a.blurSigmaPixels == b.blurSigmaPixels
                && a.blurDyeClouds == b.blurDyeClouds && a.microStructure == b.microStructure
        }

        /// Mean, variance, skewness and excess kurtosis of the output at one pixel, given that
        /// pixel's three interpolated sublayer densities.
        ///
        /// The three sublayers are independent, so the central moments add. The mean tracks the
        /// interpolated sublayer sum, not the input density: interpolating three sublayer curves at
        /// the total-density abscissa is not exactly additive.
        public func closedFormMoments(sublayerDensities: [Double], channel: Int) -> ParticleMoments {
            precondition(sublayerDensities.count == sublayerCount, "need one density per sublayer")
            let populations = (0..<sublayerCount).map { sublayer -> ParticlePopulation in
                let i = sublayer * 3 + channel
                return ParticlePopulation(
                    density: sublayerDensities[sublayer] + densityMinLayers[i],
                    densityMax: densityMaxLayers[i],
                    particlesPerPixel: particlesPerPixel[i],
                    uniformity: uniformity[channel],
                    weight: 1.0)
            }
            return ParticleMoments(summing: populations, offset: -densityMin[channel])
        }
    }

    /// One particle population: `step * Poisson(lambda)`, with `step = weight * od_particle * sat`.
    ///
    /// The grain output of a `(channel, sublayer)` pair is exactly this, so a population's first
    /// four cumulants are `step^k * lambda`.
    public struct ParticlePopulation: Sendable, Equatable {
        /// The clipped development probability.
        public let probability: Double
        public let saturation: Double
        /// `n_particles_per_pixel * p / sat`, the rate of the one Poisson draw per pixel.
        public let lambda: Double
        /// The lattice spacing the output sits on, before any weighting.
        public let odTimesSaturation: Double
        /// `weight * odTimesSaturation`.
        public let step: Double

        init(
            density: Double, densityMax: Double, particlesPerPixel: Double, uniformity: Double,
            weight: Double
        ) {
            probability = probabilityOfDevelopment(density: density, densityMax: densityMax)
            saturation = Grain.saturation(probability: probability, uniformity: uniformity)
            lambda = particlesPerPixel * probability / saturation
            odTimesSaturation = densityMax / particlesPerPixel * saturation
            step = weight * odTimesSaturation
        }

        /// The `k`th cumulant.
        func cumulant(_ k: Int) -> Double {
            Foundation.pow(step, Double(k)) * lambda
        }
    }

    /// Central moments of a grain output distribution.
    ///
    /// The populations that make it up are independent, so their cumulants add. The second, third
    /// and fourth cumulants are the variance, the third central moment and the fourth minus
    /// `3 * variance^2`, which is what the standardised shape needs.
    public struct ParticleMoments: Sendable, Equatable {
        public let mean: Double
        public let variance: Double
        public let skewness: Double
        public let excessKurtosis: Double
        public let populations: [ParticlePopulation]

        public var standardDeviation: Double { variance.squareRoot() }

        init(summing populations: [ParticlePopulation], offset: Double) {
            var sums = [Double](repeating: 0, count: 5)
            for population in populations {
                for k in 1...4 { sums[k] += population.cumulant(k) }
            }
            mean = sums[1] + offset
            variance = sums[2]
            skewness = sums[2] > 0 ? sums[3] / (sums[2] * sums[2].squareRoot()) : 0
            excessKurtosis = sums[2] > 0 ? sums[4] / (sums[2] * sums[2]) : 0
            self.populations = populations
        }
    }

    // MARK: - Elementwise pieces

    /// `clip(density / density_max, 1e-6, 1 - 1e-6)`.
    ///
    /// NaN survives, as it does in NumPy: every comparison against it is false, so neither bound
    /// takes effect and the lambda downstream is NaN, which the Poisson sampler turns into 0.
    @inlinable
    public static func probabilityOfDevelopment(density: Double, densityMax: Double) -> Double {
        let ratio = density / densityMax
        if ratio.isNaN { return ratio }
        return Swift.min(Swift.max(ratio, probabilityFloor), probabilityCeiling)
    }

    /// `1 - p * u * (1 - 1e-6)`.
    ///
    /// Scales the per-particle density contribution and the particle count together, so the mean is
    /// untouched and the variance is multiplied by `sat`. At `u = 1` and full density the noise
    /// vanishes; `u < 1` leaves residual noise at Dmax.
    @inlinable
    public static func saturation(probability: Double, uniformity: Double) -> Double {
        1.0 - probability * uniformity * saturationScale
    }

    // MARK: - One particle population

    /// `layer_particle_model` for one `(channel, sublayer)` plane.
    ///
    /// - Parameters:
    ///   - density: a single-channel plane, with the fog floor already added.
    ///   - key: identifies the stream. The reference reseeds NumPy's global generator at the start
    ///     of every call with `[0, 1, 2][channel] + 10 * sublayer`, so the nine streams are
    ///     independent and the loop order does not matter. Here that schedule is carried by the
    ///     key's `channel` and `sublayer` fields instead, and the counter is the linear pixel index,
    ///     so the result is also independent of how the plane is split across threads.
    ///   - blurDyeClouds: `blur_dye_clouds_um`, dimensionless. Gated on the parameter, not on the
    ///     sigma it produces, which is what the reference does.
    public static func layerParticleModel(
        _ density: consuming ImageBuffer,
        densityMax: Double,
        particlesPerPixel: Double,
        uniformity: Double,
        key: PhiloxKey,
        blurDyeClouds: Double = 0.0,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        precondition(density.channels == 1, "the particle model works on single-channel planes")
        let odParticle = densityMax / particlesPerPixel

        var out = consume density
        out.values.withUnsafeMutableBufferPointer { buffer in
            guard let plane = buffer.baseAddress else { return }
            var source = Philox4x32(key: key)
            for i in 0..<buffer.count {
                let p = probabilityOfDevelopment(density: plane[i], densityMax: densityMax)
                let sat = saturation(probability: p, uniformity: uniformity)
                source.reset(counter: UInt64(i))
                let developed = Distributions.poisson(
                    lambda: particlesPerPixel * p / sat, &source)
                plane[i] = Double(developed) * odParticle * sat
            }
        }

        if blurDyeClouds > 0 {
            out = spatial.gaussian(out, sigma: blurDyeClouds * odParticle.squareRoot())
        }
        return out
    }

    // MARK: - Micro-structure

    /// `add_micro_structure`: a unit-mean lognormal clumping field, optionally blurred, multiplied
    /// into the grain.
    ///
    /// With default parameters neither gate opens at any realistic resolution. `sigma > 0.05` needs
    /// a pixel pitch under 0.6 um, which is a 35 mm frame at 58333 px wide, and `blur_px > 0.4`
    /// needs under 0.5 um. Ported because the thresholds depend on output resolution, not because
    /// production renders reach them.
    ///
    /// The field has shape `[H, W, 3]`, so each channel gets its own realisation. Mean preserving
    /// before the blur and after it.
    public static func addMicroStructure(
        _ image: consuming ImageBuffer,
        microStructure: (Double, Double),
        pixelSizeMicrons: Double,
        seed: UInt64,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        let blurPixels = microStructure.0 / pixelSizeMicrons
        let sigma = microStructure.1 * 0.001 / pixelSizeMicrons
        guard sigma > 0.05 else { return image }

        var clumping = ImageBuffer(
            height: image.height, width: image.width, channels: image.channels)
        for channel in 0..<image.channels {
            var source = Philox4x32(
                key: PhiloxKey(seed: seed, channel: channel, sublayer: microStructureStream))
            clumping.values.withUnsafeMutableBufferPointer { buffer in
                guard let field = buffer.baseAddress else { return }
                for pixel in 0..<image.pixelCount {
                    source.reset(counter: UInt64(pixel))
                    field[pixel * image.channels + channel] = Distributions.lognormalFromMeanStd(
                        mean: 1.0, std: sigma, &source)
                }
            }
        }
        if blurPixels > 0.4 {
            clumping = spatial.gaussian(clumping, sigma: blurPixels)
        }

        var out = consume image
        out.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for i in 0..<buffer.count { p[i] *= clumping.values[i] }
        }
        return out
    }

    // MARK: - The sublayer split

    /// `density_curves.interp_density_cmy_layers`, the contract the layered path depends on.
    ///
    /// - Returns: one buffer per RGB channel, each carrying that channel's three sublayer densities
    ///   in its three channel slots. Upstream's `[H, W, sublayer, channel]` array, transposed into
    ///   the layout the sampler wants.
    ///
    /// For positive stocks both the query and the axis are negated so the axis ascends; the
    /// sublayer values are not negated.
    ///
    /// Lives here rather than in ``DensityCurves`` because only the layered grain path splits a
    /// density into sublayers. The render path does not call this, since the nine planes together
    /// are three full frames; it takes them one at a time from
    /// ``sublayerPlane(_:channel:sublayer:densityCurves:densityCurvesLayers:positive:)``. This is
    /// the shape the reference returns and the shape the parity tests compare.
    public static func sublayerDensities(
        _ density: ImageBuffer,
        densityCurves: [Double],
        densityCurvesLayers: [Double],
        positive: Bool
    ) -> [ImageBuffer] {
        precondition(density.channels == 3, "density must have 3 channels")
        let steps = densityCurves.count / 3
        precondition(
            densityCurvesLayers.count == steps * 9,
            "density_curves_layers must be \(steps) x 3 x 3")

        return (0..<3).map { channel in
            var out = ImageBuffer(
                height: density.height, width: density.width, channels: sublayerCount)
            for sublayer in 0..<sublayerCount {
                let plane = sublayerPlane(
                    density,
                    channel: channel,
                    sublayer: sublayer,
                    densityCurves: densityCurves,
                    densityCurvesLayers: densityCurvesLayers,
                    positive: positive)
                writePlane(plane, into: &out, channel: sublayer)
            }
            return out
        }
    }

    /// One `(channel, sublayer)` plane of interpolated sublayer density.
    ///
    /// The nine planes are three full frames when materialised together, which is what
    /// ``sublayerDensities(_:densityCurves:densityCurvesLayers:positive:)`` hands back. The grain
    /// loop needs one at a time, so it interpolates them one at a time.
    ///
    /// Arithmetic copied from ``Interpolation/fastInterp(_:axis:values:)``'s shared-axis path, down
    /// to the reciprocal interval widths and the clamp to the endpoint values, so the two agree bit
    /// for bit.
    static func sublayerPlane(
        _ density: ImageBuffer,
        channel: Int,
        sublayer: Int,
        densityCurves: [Double],
        densityCurvesLayers: [Double],
        positive: Bool
    ) -> ImageBuffer {
        let steps = densityCurves.count / 3
        precondition(steps >= 2, "axis needs at least two samples")

        var axis = (0..<steps).map { densityCurves[$0 * 3 + channel] }
        if positive { for i in axis.indices { axis[i] = -axis[i] } }
        let curve = (0..<steps).map { densityCurvesLayers[$0 * 9 + sublayer * 3 + channel] }
        var invDx = [Double](repeating: 0, count: steps - 1)
        for i in 0..<(steps - 1) {
            let d = axis[i + 1] - axis[i]
            invDx[i] = d != 0 ? 1.0 / d : 0.0
        }

        var plane = ImageBuffer(height: density.height, width: density.width, channels: 1)
        density.values.withUnsafeBufferPointer { src in
            axis.withUnsafeBufferPointer { ax in
                curve.withUnsafeBufferPointer { ys in
                    invDx.withUnsafeBufferPointer { inv in
                        plane.values.withUnsafeMutableBufferPointer { dst in
                            let s = src.baseAddress!
                            let a = ax.baseAddress!
                            let y = ys.baseAddress!
                            let iv = inv.baseAddress!
                            let d = dst.baseAddress!
                            let first = a[0]
                            let last = a[steps - 1]

                            for pixel in 0..<density.pixelCount {
                                let value = s[pixel * 3 + channel]
                                let x = positive ? -value : value
                                if x.isNaN || x <= first {
                                    d[pixel] = y[0]
                                } else if x >= last {
                                    d[pixel] = y[steps - 1]
                                } else {
                                    let low =
                                        Interpolation.upperBound(x, a, stride: 1, count: steps) - 1
                                    let t = (x - a[low]) * iv[low]
                                    d[pixel] = y[low] + t * (y[low + 1] - y[low])
                                }
                            }
                        }
                    }
                }
            }
        }
        return plane
    }

    // MARK: - Topologies

    /// `apply_grain`.
    ///
    /// - Parameters:
    ///   - densityCurves: already fog-normalised by `develop`, `[exposure][cmy]` flattened.
    ///   - densityCurvesLayers: the raw profile array, `[exposure][sublayer][channel]` flattened.
    ///     Not normalised.
    ///   - seed: selects the grain realisation. ``GrainParams`` has no seed field upstream, where
    ///     the seeds are hardcoded, so it is passed separately.
    ///
    /// Returns the input unchanged when grain is off or bypassed, matching the reference, which
    /// returns the same object.
    ///
    /// Consumes `density`. A caller that still needs it gets a copy, and pays a frame for it.
    public static func apply(
        _ density: consuming ImageBuffer,
        pixelSizeMicrons: Double,
        params: GrainParams,
        densityCurves: [Double],
        densityCurvesLayers: [Double],
        positive: Bool,
        seed: UInt64 = 0,
        bypass: Bool = false,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        guard params.active, !bypass else { return density }

        if !params.sublayersActive {
            let derived = SingleLayerParameters(
                params: params,
                pixelSizeMicrons: pixelSizeMicrons,
                densityMaxCurves: nanMax(densityCurves, channels: 3))
            return applyToDensity(consume density, derived: derived, seed: seed, spatial: spatial)
        }

        let derived = LayeredParameters(
            params: params,
            pixelSizeMicrons: pixelSizeMicrons,
            densityMaxLayers: nanMax(densityCurvesLayers, channels: 9))
        // The density frame is released when this returns, before the closing blur allocates.
        let grain = accumulateLayers(
            consume density,
            derived: derived,
            densityCurves: densityCurves,
            densityCurvesLayers: densityCurvesLayers,
            positive: positive,
            seed: seed,
            spatial: spatial)
        return finishLayers(consume grain, derived: derived, seed: seed, spatial: spatial)
    }

    /// Sums the nine `(channel, sublayer)` particle populations, interpolating each sublayer plane
    /// as it is needed.
    static func accumulateLayers(
        _ density: consuming ImageBuffer,
        derived: LayeredParameters,
        densityCurves: [Double],
        densityCurvesLayers: [Double],
        positive: Bool,
        seed: UInt64,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        precondition(density.channels == 3, "density must have 3 channels")

        var out = ImageBuffer(height: density.height, width: density.width, channels: 3)
        for channel in 0..<3 {
            for sublayer in 0..<sublayerCount {
                let i = sublayer * 3 + channel
                var plane = sublayerPlane(
                    density,
                    channel: channel,
                    sublayer: sublayer,
                    densityCurves: densityCurves,
                    densityCurvesLayers: densityCurvesLayers,
                    positive: positive)
                addInPlace(&plane, derived.densityMinLayers[i])
                let grain = layerParticleModel(
                    consume plane,
                    densityMax: derived.densityMaxLayers[i],
                    particlesPerPixel: derived.particlesPerPixel[i],
                    uniformity: derived.uniformity[channel],
                    key: PhiloxKey(seed: seed, channel: channel, sublayer: sublayer),
                    blurDyeClouds: derived.blurDyeClouds,
                    spatial: spatial)
                accumulatePlane(grain, into: &out, channel: channel)
            }
        }
        return out
    }

    /// The clumping field, the fog subtraction and the closing blur, in that order.
    static func finishLayers(
        _ grain: consuming ImageBuffer,
        derived: LayeredParameters,
        seed: UInt64,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        var out = addMicroStructure(
            consume grain,
            microStructure: derived.microStructure,
            pixelSizeMicrons: derived.pixelSizeMicrons,
            seed: seed,
            spatial: spatial)

        out.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for i in stride(from: 0, to: buffer.count, by: 3) {
                for channel in 0..<3 { p[i + channel] -= derived.densityMin[channel] }
            }
        }

        if derived.blurSigmaPixels > 0 {
            blurInPlace(&out, sigma: derived.blurSigmaPixels, spatial: spatial)
        }
        return out
    }

    /// `apply_grain_to_density`: one particle population per channel, applied to the total channel
    /// density.
    ///
    /// The reference mutates its argument with `density_cmy += density_min`. This copies instead,
    /// and no behaviour depends on the mutation: in production the array is a temporary from the
    /// coupler stage, and the reference's own tests defend by passing a copy.
    ///
    /// `n_sub_layers` repeats and averages. Each repeat sees the full channel density with
    /// `N / n_sub_layers` particles, so the mean and variance do not move; only the skewness and
    /// the cost do.
    public static func applyToDensity(
        _ density: ImageBuffer,
        derived: SingleLayerParameters,
        seed: UInt64 = 0,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        precondition(density.channels == 3, "density must have 3 channels")

        var out = ImageBuffer(height: density.height, width: density.width, channels: 3)
        for channel in 0..<3 {
            var plane = extractPlane(density, channel: channel)
            addInPlace(&plane, derived.densityMin[channel])
            for repeatIndex in 0..<derived.subLayerCount {
                let grain = layerParticleModel(
                    plane,
                    densityMax: derived.densityMax[channel],
                    particlesPerPixel: derived.particlesPerPixel[channel],
                    uniformity: derived.uniformity[channel],
                    key: PhiloxKey(seed: seed, channel: channel, sublayer: repeatIndex),
                    spatial: spatial)
                accumulatePlane(grain, into: &out, channel: channel)
            }
        }

        let inverse = 1.0 / Double(derived.subLayerCount)
        out.values.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            for i in stride(from: 0, to: buffer.count, by: 3) {
                for channel in 0..<3 {
                    p[i + channel] = p[i + channel] * inverse - derived.densityMin[channel]
                }
            }
        }

        // The gate here is `> 0.4`, and the layered path's is `> 0`. Keep the asymmetry.
        if derived.blurSigmaPixels > 0.4 {
            out = spatial.gaussian(out, sigma: derived.blurSigmaPixels)
        }
        return out
    }

    /// `apply_grain_to_density_layers`, the production path.
    ///
    /// No division by a sublayer count: the split is carried by `density_max_fractions`, whose
    /// columns sum to 1, so `sum(density_min_layers)` over sublayers is `density_min` and the final
    /// subtraction balances.
    ///
    /// - Parameter layers: one buffer per RGB channel, three sublayers deep, as
    ///   ``sublayerDensities(_:densityCurves:densityCurvesLayers:positive:)`` returns. ``apply``
    ///   goes through ``accumulateLayers(_:derived:densityCurves:densityCurvesLayers:positive:seed:spatial:)``
    ///   instead, which never holds the whole split at once.
    public static func applyToDensityLayers(
        _ layers: [ImageBuffer],
        derived: LayeredParameters,
        seed: UInt64 = 0,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        precondition(layers.count == 3, "need one sublayer buffer per channel")
        let height = layers[0].height
        let width = layers[0].width

        var out = ImageBuffer(height: height, width: width, channels: 3)
        for channel in 0..<3 {
            precondition(layers[channel].channels == sublayerCount, "expected 3 sublayers")
            for sublayer in 0..<sublayerCount {
                let i = sublayer * 3 + channel
                var plane = extractPlane(layers[channel], channel: sublayer)
                addInPlace(&plane, derived.densityMinLayers[i])
                let grain = layerParticleModel(
                    consume plane,
                    densityMax: derived.densityMaxLayers[i],
                    particlesPerPixel: derived.particlesPerPixel[i],
                    uniformity: derived.uniformity[channel],
                    key: PhiloxKey(seed: seed, channel: channel, sublayer: sublayer),
                    blurDyeClouds: derived.blurDyeClouds,
                    spatial: spatial)
                accumulatePlane(grain, into: &out, channel: channel)
            }
        }
        return finishLayers(consume out, derived: derived, seed: seed, spatial: spatial)
    }

    // MARK: - Plane helpers

    /// One channel of a buffer as a single-channel plane, which is what the sampler and the 2D blur
    /// take.
    static func extractPlane(_ image: ImageBuffer, channel: Int) -> ImageBuffer {
        var plane = ImageBuffer(height: image.height, width: image.width, channels: 1)
        let stride = image.channels
        for pixel in 0..<image.pixelCount {
            plane.values[pixel] = image.values[pixel * stride + channel]
        }
        return plane
    }

    static func writePlane(_ plane: ImageBuffer, into image: inout ImageBuffer, channel: Int) {
        let stride = image.channels
        for pixel in 0..<image.pixelCount {
            image.values[pixel * stride + channel] = plane.values[pixel]
        }
    }

    /// Blurs each channel in place.
    ///
    /// ``SpatialFilter/gaussian(_:sigma:)`` dispatches per channel internally, so this is the same
    /// arithmetic with one plane live instead of a second full frame.
    static func blurInPlace(
        _ image: inout ImageBuffer, sigma: Double, spatial: some SpatialFilter
    ) {
        for channel in 0..<image.channels {
            let filtered = spatial.gaussian(extractPlane(image, channel: channel), sigma: sigma)
            writePlane(filtered, into: &image, channel: channel)
        }
    }

    static func accumulatePlane(_ plane: ImageBuffer, into image: inout ImageBuffer, channel: Int) {
        let stride = image.channels
        for pixel in 0..<image.pixelCount {
            image.values[pixel * stride + channel] += plane.values[pixel]
        }
    }

    static func addInPlace(_ plane: inout ImageBuffer, _ offset: Double) {
        for i in plane.values.indices { plane.values[i] += offset }
    }
}

private func tupleToArray(_ t: (Double, Double, Double)) -> [Double] { [t.0, t.1, t.2] }

import Foundation

/// A spatial blur the coupler and halation models need.
///
/// A protocol so the emulsion model can be built and gated without the diffusion module, and so a
/// Metal implementation can substitute later. ``NoSpatialFilter`` is the identity, the same result
/// `debug.lutMode` and `debug.deactivateSpatialEffects` select.
public protocol SpatialFilter: Sendable {
    /// Isotropic Gaussian blur, sigma in pixels.
    func gaussian(_ image: ImageBuffer, sigma: Double) -> ImageBuffer
    /// Energy-preserving exponential blur, decay constant in pixels.
    func exponential(_ image: ImageBuffer, decay: Double) -> ImageBuffer
}

/// Passes images through untouched.
public struct NoSpatialFilter: SpatialFilter {
    public init() {}
    public func gaussian(_ image: ImageBuffer, sigma: Double) -> ImageBuffer { image }
    public func exponential(_ image: ImageBuffer, decay: Double) -> ImageBuffer { image }
}

/// Development-inhibitor-releasing couplers.
///
/// Ports `model/couplers.py`. As density forms in one layer, inhibitor diffuses out and suppresses
/// density nearby, in the same layer and in the other two. This raises saturation and contrast.
/// When the inhibitor also spreads spatially, it raises local contrast and apparent sharpness.
public enum Couplers {

    /// `compute_dir_couplers_matrix`.
    ///
    /// Row is the donor layer that releases inhibitor, column is the receiver whose exposure drops.
    public static func inhibitionMatrix(_ p: DirCouplersParams) -> Matrix3 {
        let selfR = p.gammaSameLayerRGB.0 * p.inhibitionSameLayer
        let selfG = p.gammaSameLayerRGB.1 * p.inhibitionSameLayer
        let selfB = p.gammaSameLayerRGB.2 * p.inhibitionSameLayer
        let k = p.inhibitionInterlayer
        return Matrix3(
            selfR, p.gammaInterlayerRedToGreenBlue.0 * k, p.gammaInterlayerRedToGreenBlue.1 * k,
            p.gammaInterlayerGreenToRedBlue.0 * k, selfG, p.gammaInterlayerGreenToRedBlue.1 * k,
            p.gammaInterlayerBlueToRedGreen.0 * k, p.gammaInterlayerBlueToRedGreen.1 * k, selfB
        )
    }

    /// `compute_density_curves_before_dir_couplers`.
    ///
    /// The published curves already include the couplers' effect, so the model needs the curves as
    /// they would be without it. This inverts the relationship by shifting the exposure axis by the
    /// inhibitor each density level releases, then resampling.
    ///
    /// The shifted axis is not monotonic for positive stocks, so this uses
    /// ``Interpolation/npInterp(query:xp:fp:)``. See that function for what depends on it.
    ///
    /// - Parameters:
    ///   - curves: `[exposure][cmy]` flattened, already normalised to a zero floor.
    ///   - matrix: the inhibition matrix, already scaled by `amount`.
    public static func curvesBeforeCouplers(
        curves: [Double],
        logExposure: [Double],
        matrix: Matrix3,
        positive: Bool
    ) -> [Double] {
        let count = logExposure.count
        precondition(curves.count == count * 3, "curves must be \(count) x 3")

        // For positive stocks the interimage effect is assumed to act during silver development,
        // where silver density is d_max - d.
        var silver = curves
        if positive {
            var maxima = [Double](repeating: -Double.infinity, count: 3)
            for i in 0..<count {
                for c in 0..<3 {
                    let v = curves[i * 3 + c]
                    if !v.isNaN { maxima[c] = max(maxima[c], v) }
                }
            }
            for i in 0..<count {
                for c in 0..<3 { silver[i * 3 + c] = maxima[c] - curves[i * 3 + c] }
            }
        }

        // couplers_amount_curves = silver @ matrix, so the receiver column m accumulates every
        // donor layer k weighted by matrix[k, m].
        var shiftedAxis = [Double](repeating: 0, count: count * 3)
        for i in 0..<count {
            let r = silver[i * 3]
            let g = silver[i * 3 + 1]
            let b = silver[i * 3 + 2]
            for m in 0..<3 {
                let released = r * matrix[0, m] + g * matrix[1, m] + b * matrix[2, m]
                shiftedAxis[i * 3 + m] = logExposure[i] - released
            }
        }

        var out = [Double](repeating: 0, count: count * 3)
        for c in 0..<3 {
            let xp = (0..<count).map { shiftedAxis[$0 * 3 + c] }
            var fp = (0..<count).map { curves[$0 * 3 + c] }
            if positive { for i in fp.indices { fp[i] = -fp[i] } }
            var queried = Interpolation.npInterp(query: logExposure, xp: xp, fp: fp)
            if positive { for i in queried.indices { queried[i] = -queried[i] } }
            for i in 0..<count { out[i * 3 + c] = queried[i] }
        }
        return out
    }

    /// `compute_exposure_correction_dir_couplers`.
    ///
    /// Returns the log exposure the emulsion effectively saw, reduced by the inhibitor released
    /// around each pixel.
    ///
    /// Consumes `density` and returns its storage, written over. Nothing else may hold a reference,
    /// or the first write copies a whole frame.
    public static func correctedLogExposure(
        logRaw: ImageBuffer,
        density: consuming ImageBuffer,
        densityMax: [Double],
        matrix: Matrix3,
        diffusionSizePixels: Double,
        diffusionTailSizePixels: Double,
        diffusionTailWeight: Double,
        highExposureShift: Double = 0.0,
        positive: Bool,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        precondition(logRaw.channels == 3 && density.channels == 3)

        // Silver and the inhibitor it releases share one buffer. The matrix mixes the three channels
        // of a pixel, so all three silver values are read before any of them is overwritten.
        var correction = consume density
        correction.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in stride(from: 0, to: buf.count, by: 3) {
                var r = positive ? densityMax[0] - p[i] : p[i]
                var g = positive ? densityMax[1] - p[i + 1] : p[i + 1]
                var b = positive ? densityMax[2] - p[i + 2] : p[i + 2]
                r += highExposureShift * r * r
                g += highExposureShift * g * g
                b += highExposureShift * b * b
                for m in 0..<3 {
                    p[i + m] = r * matrix[0, m] + g * matrix[1, m] + b * matrix[2, m]
                }
            }
        }

        if diffusionSizePixels > 0 {
            diffuseInPlace(
                &correction,
                sigma: diffusionSizePixels,
                decay: diffusionTailSizePixels,
                tailWeight: diffusionTailWeight,
                spatial: spatial)
        }

        correction.combineInPlace(with: logRaw) { inhibitor, raw in raw - inhibitor }
        return correction
    }

    /// The Gaussian core and the exponential tail, blended and written back over `image`.
    ///
    /// Runs one channel at a time. Both whole-buffer filters dispatch per channel internally, so the
    /// arithmetic is unchanged, and the core, the tail and the mixture each hold one plane instead
    /// of a full frame.
    private static func diffuseInPlace(
        _ image: inout ImageBuffer,
        sigma: Double,
        decay: Double,
        tailWeight: Double,
        spatial: some SpatialFilter
    ) {
        for channel in 0..<image.channels {
            let plane = channelPlane(image, channel: channel)
            // The tail is a three-Gaussian mixture and allocates the most scratch, so it runs
            // before the core plane exists.
            let tail = spatial.exponential(plane, decay: decay)
            var mixed = spatial.gaussian(plane, sigma: sigma)
            mixed.combineInPlace(with: tail) { core, tail in
                (1 - tailWeight) * core + tailWeight * tail
            }
            write(plane: mixed, into: &image, channel: channel)
        }
    }

    private static func channelPlane(_ image: ImageBuffer, channel: Int) -> ImageBuffer {
        var plane = ImageBuffer(height: image.height, width: image.width, channels: 1)
        let stride = image.channels
        for pixel in 0..<image.pixelCount {
            plane.values[pixel] = image.values[pixel * stride + channel]
        }
        return plane
    }

    private static func write(plane: ImageBuffer, into image: inout ImageBuffer, channel: Int) {
        let stride = image.channels
        for pixel in 0..<image.pixelCount {
            image.values[pixel * stride + channel] = plane.values[pixel]
        }
    }

    /// `apply_density_correction_dir_couplers`.
    ///
    /// - Parameters:
    ///   - curves: normalised curves, `[exposure][cmy]` flattened.
    ///   - pixelSizeMicrons: `nil` when the pipeline was injected past preprocessing, as LUT bakes
    ///     are. The reference converts to pixel units only when `diffusion_size_um > 0`, so in this
    ///     case the non-spatial chemistry still runs.
    public static func applyDensityCorrection(
        density: consuming ImageBuffer,
        logRaw: ImageBuffer,
        pixelSizeMicrons: Double?,
        logExposure: [Double],
        curves: [Double],
        params: DirCouplersParams,
        positive: Bool,
        gammaFactor: Double = 1.0,
        spatial: some SpatialFilter
    ) -> ImageBuffer {
        guard params.active else { return density }

        var matrix = inhibitionMatrix(params)
        matrix = Matrix3(
            matrix.m00 * params.amount, matrix.m01 * params.amount, matrix.m02 * params.amount,
            matrix.m10 * params.amount, matrix.m11 * params.amount, matrix.m12 * params.amount,
            matrix.m20 * params.amount, matrix.m21 * params.amount, matrix.m22 * params.amount
        )

        let curvesBefore = curvesBeforeCouplers(
            curves: curves, logExposure: logExposure, matrix: matrix, positive: positive)

        var maxima = [Double](repeating: -Double.infinity, count: 3)
        for i in 0..<logExposure.count {
            for c in 0..<3 {
                let v = curves[i * 3 + c]
                if !v.isNaN { maxima[c] = max(maxima[c], v) }
            }
        }

        var sizePixels = 0.0
        var tailPixels = 0.0
        if params.diffusionSizeMicrons > 0, let pixelSizeMicrons {
            sizePixels = params.diffusionSizeMicrons / pixelSizeMicrons
            tailPixels = params.diffusionTailMicrons / pixelSizeMicrons
        }

        let corrected = correctedLogExposure(
            logRaw: logRaw,
            density: consume density,
            densityMax: maxima,
            matrix: matrix,
            diffusionSizePixels: sizePixels,
            diffusionTailSizePixels: tailPixels,
            diffusionTailWeight: params.diffusionTailWeight,
            positive: positive,
            spatial: spatial
        )

        return DensityCurves.densityFromLogExposure(
            logExposure: corrected, curves: curvesBefore, axis: logExposure,
            gammaFactor: gammaFactor)
    }
}

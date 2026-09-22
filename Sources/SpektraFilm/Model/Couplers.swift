import Foundation

/// A spatial blur the coupler and halation models need.
///
/// Declared as a protocol so the emulsion model can be built and gated before the diffusion module
/// lands, and so a Metal implementation can substitute later. ``NoSpatialFilter`` is the identity,
/// which is what `debug.lutMode` and `debug.deactivateSpatialEffects` select anyway.
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
/// density nearby, in the same layer and in the other two. That raises saturation and contrast, and
/// once the inhibitor is allowed to spread spatially, local contrast and apparent sharpness too.
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
    /// The shifted axis is not monotonic for positive stocks, which is why this uses
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
    public static func correctedLogExposure(
        logRaw: ImageBuffer,
        density: ImageBuffer,
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

        var silver = density
        silver.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in stride(from: 0, to: buf.count, by: 3) {
                for c in 0..<3 {
                    var d = positive ? densityMax[c] - p[i + c] : p[i + c]
                    d += highExposureShift * d * d
                    p[i + c] = d
                }
            }
        }

        var correction = silver
        correction.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in stride(from: 0, to: buf.count, by: 3) {
                let r = p[i]
                let g = p[i + 1]
                let b = p[i + 2]
                for m in 0..<3 {
                    p[i + m] = r * matrix[0, m] + g * matrix[1, m] + b * matrix[2, m]
                }
            }
        }

        if diffusionSizePixels > 0 {
            let core = spatial.gaussian(correction, sigma: diffusionSizePixels)
            let tail = spatial.exponential(correction, decay: diffusionTailSizePixels)
            var mixed = core
            mixed.values.withUnsafeMutableBufferPointer { buf in
                guard let p = buf.baseAddress else { return }
                for i in 0..<buf.count {
                    p[i] = (1 - diffusionTailWeight) * p[i] + diffusionTailWeight * tail.values[i]
                }
            }
            correction = mixed
        }

        var out = logRaw
        out.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count { p[i] -= correction.values[i] }
        }
        return out
    }

    /// `apply_density_correction_dir_couplers`.
    ///
    /// - Parameters:
    ///   - curves: normalised curves, `[exposure][cmy]` flattened.
    ///   - pixelSizeMicrons: `nil` when the pipeline was injected past preprocessing, which happens
    ///     for LUT bakes. The reference gates the conversion to pixel units on
    ///     `diffusion_size_um > 0` for exactly this case, so the non-spatial chemistry still runs.
    public static func applyDensityCorrection(
        density: ImageBuffer,
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
            density: density,
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

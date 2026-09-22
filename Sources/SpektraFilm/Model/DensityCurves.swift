import Foundation

/// The characteristic curves that turn log exposure into dye density.
///
/// Ports `model/density_curves.py` and the spectral helpers from `model/develop.py`.
public enum DensityCurves {

    /// `interpolate_exposure_to_density`.
    ///
    /// `gammaFactor` divides the exposure axis, which steepens or flattens the curve. The reference
    /// folds it into the axis and hands a per-channel axis to `fast_interp`, so a scalar gamma and
    /// three equal gammas take the same code path.
    public static func densityFromLogExposure(
        logExposure image: ImageBuffer,
        curves: [Double],
        axis logExposureAxis: [Double],
        gammaFactor: (Double, Double, Double)
    ) -> ImageBuffer {
        precondition(image.channels == 3, "log exposure buffer must have 3 channels")
        let count = logExposureAxis.count
        var perChannelAxis = [Double](repeating: 0, count: count * 3)
        let gamma = [gammaFactor.0, gammaFactor.1, gammaFactor.2]
        for i in 0..<count {
            for c in 0..<3 { perChannelAxis[i * 3 + c] = logExposureAxis[i] / gamma[c] }
        }
        return Interpolation.fastInterp(image, axis: perChannelAxis, values: curves)
    }

    /// `interpolate_exposure_to_density` with one gamma for all three channels.
    public static func densityFromLogExposure(
        logExposure image: ImageBuffer,
        curves: [Double],
        axis logExposureAxis: [Double],
        gammaFactor: Double = 1.0
    ) -> ImageBuffer {
        densityFromLogExposure(
            logExposure: image, curves: curves, axis: logExposureAxis,
            gammaFactor: (gammaFactor, gammaFactor, gammaFactor))
    }

    /// Subtracts each channel's minimum, ignoring NaN. `develop` does this before anything else, so
    /// the curves start at zero density and `density_min` from the grain model sets the real floor.
    public static func normalized(curves: [Double], minima: [Double]) -> [Double] {
        var out = curves
        for i in stride(from: 0, to: out.count, by: 3) {
            for c in 0..<3 { out[i + c] -= minima[c] }
        }
        return out
    }

    /// `compute_density_spectral`.
    ///
    /// Expands CMY density into a spectrum per pixel by weighting each dye's absorption curve, then
    /// adds the base density of the unexposed developed medium. Output has
    /// `ColourTables.wavelengthCount` channels, so a full frame cannot be converted in one
    /// allocation; call it through `ImageBuffer.mapPerPixel`.
    public static func spectralDensity(
        cmy: ImageBuffer,
        channelDensity: [Double],
        baseDensity: [Double]?
    ) -> ImageBuffer {
        precondition(cmy.channels == 3, "density buffer must have 3 channels")
        let wavelengths = ColourTables.wavelengthCount
        precondition(
            channelDensity.count == wavelengths * 3,
            "channel_density must be \(wavelengths) x 3")

        var out = ImageBuffer(height: cmy.height, width: cmy.width, channels: wavelengths)
        cmy.values.withUnsafeBufferPointer { src in
            channelDensity.withUnsafeBufferPointer { dye in
                out.values.withUnsafeMutableBufferPointer { dst in
                    let s = src.baseAddress!
                    let k = dye.baseAddress!
                    let d = dst.baseAddress!
                    for p in 0..<cmy.pixelCount {
                        let c = s[p * 3]
                        let m = s[p * 3 + 1]
                        let y = s[p * 3 + 2]
                        let base = p * wavelengths
                        for l in 0..<wavelengths {
                            d[base + l] =
                                c * k[l * 3] + m * k[l * 3 + 1] + y * k[l * 3 + 2]
                        }
                    }
                }
            }
        }
        if let baseDensity {
            precondition(baseDensity.count == wavelengths, "base_density must be \(wavelengths)")
            out.values.withUnsafeMutableBufferPointer { dst in
                guard let d = dst.baseAddress else { return }
                for p in 0..<cmy.pixelCount {
                    let base = p * wavelengths
                    for l in 0..<wavelengths { d[base + l] += baseDensity[l] }
                }
            }
        }
        return out
    }

    /// `utils/conversions.density_to_light`.
    ///
    /// Transmittance is `10^-density`, scaled by the incident light. NaN becomes 0, which is how
    /// wavelengths the datasheet does not cover end up contributing nothing.
    public static func densityToLight(
        _ density: ImageBuffer, illuminant: [Double]
    ) -> ImageBuffer {
        precondition(
            density.channels == illuminant.count,
            "illuminant has \(illuminant.count) samples, density has \(density.channels) channels")
        var out = density
        let channels = density.channels
        out.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for i in 0..<buf.count {
                let transmitted = Foundation.pow(10.0, -p[i]) * illuminant[i % channels]
                p[i] = transmitted.isNaN ? 0 : transmitted
            }
        }
        return out
    }

    /// Contracts a spectral buffer against an `[wavelength][3]` response, as the scanning and
    /// printing stages do with `contract("ijk, kl->ijl", light, response)`.
    public static func project(
        _ spectral: ImageBuffer, onto response: [Double], scale: Double = 1.0
    ) -> ImageBuffer {
        let wavelengths = spectral.channels
        precondition(
            response.count == wavelengths * 3, "response must be \(wavelengths) x 3")

        var out = ImageBuffer(height: spectral.height, width: spectral.width, channels: 3)
        spectral.values.withUnsafeBufferPointer { src in
            response.withUnsafeBufferPointer { resp in
                out.values.withUnsafeMutableBufferPointer { dst in
                    let s = src.baseAddress!
                    let r = resp.baseAddress!
                    let d = dst.baseAddress!
                    for p in 0..<spectral.pixelCount {
                        var acc = (0.0, 0.0, 0.0)
                        let base = p * wavelengths
                        for l in 0..<wavelengths {
                            let light = s[base + l]
                            acc.0 += light * r[l * 3]
                            acc.1 += light * r[l * 3 + 1]
                            acc.2 += light * r[l * 3 + 2]
                        }
                        d[p * 3] = acc.0 * scale
                        d[p * 3 + 1] = acc.1 * scale
                        d[p * 3 + 2] = acc.2 * scale
                    }
                }
            }
        }
        return out
    }
}

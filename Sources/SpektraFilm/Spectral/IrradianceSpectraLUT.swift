import Foundation

/// The shipped table of irradiance spectra, `[tc.x][tc.y][wavelength]` = 192 x 192 x 81.
///
/// `HANATOS2025_SPECTRA_LUT`. Grid coordinate for index `i` is `i / (gridSize - 1)`; axis 0 is
/// `tc.x`, axis 1 is `tc.y`, axis 2 is 380…780 nm at 5 nm. Cell `(0, j)` decodes to `xy = (1, 0)`
/// for every `j`, so the whole first row is degenerate in `xy` while holding different spectra: the
/// fit was done in `tc` space.
///
/// **Stays `Float16`.** The file is memory-mapped and every element is widened inside the
/// contraction, for 5.7 MiB resident instead of the 22.8 MiB a widened copy would cost. Every
/// binary16 is exactly representable in binary64, and the contraction accumulates in `Double`
/// either way, so this is bit-identical to the reference's `np.double(np.load(...))`.
public struct IrradianceSpectraLUT: Sendable {
    /// Bundle-relative location of the `.npy`.
    public static let resourceName = "irradiance_xy_tc"
    public static let resourceSubdirectory = "Resources/luts/spectral_upsampling"

    public let array: NumpyArray

    /// Samples per grid axis, 192. Both grid axes share it.
    public var gridSize: Int { array.shape[0] }
    /// Spectral samples per cell, 81.
    public var sampleCount: Int { array.shape[2] }

    private init(array: NumpyArray) throws {
        guard array.shape.count == 3, array.shape[0] == array.shape[1],
            array.shape[2] == SpectralShape.count
        else {
            throw SpektraError.malformedResource(
                "\(Self.resourceSubdirectory)/\(Self.resourceName).npy",
                reason:
                    "shape \(array.shape) is not (N, N, \(SpectralShape.count))")
        }
        self.array = array
    }

    private static let loaded: Result<IrradianceSpectraLUT, SpektraError> = {
        do {
            let array = try NumpyArrayReader.bundled(resourceName, subdirectory: resourceSubdirectory)
            return .success(try IrradianceSpectraLUT(array: array))
        } catch let error as SpektraError {
            return .failure(error)
        } catch {
            return .failure(
                .malformedResource(resourceName, reason: "\(error)"))
        }
    }()

    /// The mapped table, loaded once per process.
    public static func shared() throws -> IrradianceSpectraLUT { try loaded.get() }

    // MARK: - Access

    /// The 81-sample spectrum stored at a grid cell.
    public func spectrum(x i: Int, y j: Int) -> Spectrum {
        let base = (i * gridSize + j) * sampleCount
        return Spectrum(array.values(base..<(base + sampleCount)))
    }

    /// `(min, max, sum)` over the whole table, a cheap load-integrity check.
    public func statistics() -> (min: Double, max: Double, sum: Double) {
        var lowest = Double.infinity
        var highest = -Double.infinity
        var total = 0.0
        array.withPayload { raw in
            for index in 0..<array.count {
                let value = NumpyArray.double(
                    fromBinary16: raw.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self))
                if value < lowest { lowest = value }
                if value > highest { highest = value }
                total += value
            }
        }
        return (lowest, highest, total)
    }

    // MARK: - Contraction

    /// `contract('ijl,lm->ijm', spectra_lut, operand)`, optionally blurring along the wavelength
    /// axis first.
    ///
    /// The whole point of the subsystem: one 192 x 192 x 3 table per film, so nothing spectral
    /// happens per pixel. `(36864, 81) @ (81, 3)` is 9 MFLOP.
    ///
    /// Blurring per cell before the dot product is the same arithmetic as blurring the whole table
    /// and contracting after, and avoids materialising a second 22.8 MiB copy.
    public func contracted(with operand: SpectralMatrix, blur: HanatosSpectralBlur? = nil) -> ImageBuffer {
        let n = gridSize
        let k = sampleCount
        var out = ImageBuffer(height: n, width: n, channels: 3)

        array.withPayload { raw in
            var spectrum = [Double](repeating: 0, count: k)
            var blurred = [Double](repeating: 0, count: k)
            out.values.withUnsafeMutableBufferPointer { output in
                operand.values.withUnsafeBufferPointer { s in
                    for cell in 0..<(n * n) {
                        let base = cell * k
                        for l in 0..<k {
                            spectrum[l] = NumpyArray.double(
                                fromBinary16: raw.loadUnaligned(
                                    fromByteOffset: (base + l) * 2, as: UInt16.self))
                        }
                        if let blur {
                            blur.apply(spectrum, into: &blurred)
                            swap(&spectrum, &blurred)
                        }
                        var r = 0.0
                        var g = 0.0
                        var b = 0.0
                        for l in 0..<k {
                            let v = spectrum[l]
                            r += v * s[l * 3]
                            g += v * s[l * 3 + 1]
                            b += v * s[l * 3 + 2]
                        }
                        output[cell * 3] = r
                        output[cell * 3 + 1] = g
                        output[cell * 3 + 2] = b
                    }
                }
            }
        }
        return out
    }
}

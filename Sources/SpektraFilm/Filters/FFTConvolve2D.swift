import Accelerate
import Foundation

/// Linear 2D convolution through a pair of complex DFTs, standing in for
/// `scipy.signal.fftconvolve`.
///
/// Upstream calls `fftconvolve(padded, psf, mode='same')` and then crops the padding away. That
/// composite is exactly the *valid* region of the full linear convolution, which is what
/// ``convolveValid(image:height:width:kernel:kernelHeight:kernelWidth:)`` returns:
/// `same[i] == full[i + radius]`, and cropping `radius` off each side leaves
/// `full[t + kernelHeight - 1]`.
///
/// The transform length is free. `fftconvolve` computes the full linear convolution at a
/// `next_fast_len`-rounded size, so no output sample ever wraps, and the result does not depend on
/// the size chosen. This picks the smallest length `vDSP_DFT` accepts, which is why it can match
/// SciPy to roundoff while using different transform sizes.
public enum FFTConvolve2D {

    /// Smallest transform length `vDSP_DFT_zop_CreateSetupD` accepts that is at least `n`.
    ///
    /// vDSP restricts complex DFT lengths to `f * 2^k` with `f` in `{1, 3, 5, 15}` and `k >= 3`,
    /// so the minimum is 8. Rounding up is free: the convolution is linear either way.
    public static func supportedLength(atLeast n: Int) -> Int {
        precondition(n > 0, "transform length must be positive")
        var best = Int.max
        for factor in [1, 3, 5, 15] {
            var length = factor * 8
            while length < n { length *= 2 }
            best = Swift.min(best, length)
        }
        return best
    }

    /// Bytes of transform scratch a call would allocate, for the memory guard in
    /// ``Diffusion/applyDiffusionFilter(_:_:pixelSizeMicrons:scratchBudgetBytes:)``.
    ///
    /// Four `Double` planes at the padded transform size: real and imaginary for the image and for
    /// the kernel. Both spectra have to exist at once for the pointwise product.
    public static func scratchBytes(
        imageHeight: Int, imageWidth: Int, kernelHeight: Int, kernelWidth: Int
    ) -> Int {
        let rows = supportedLength(atLeast: imageHeight + kernelHeight - 1)
        let columns = supportedLength(atLeast: imageWidth + kernelWidth - 1)
        return 4 * rows * columns * MemoryLayout<Double>.size
    }

    /// The valid region of the full linear convolution of `image` with `kernel`.
    ///
    /// Result shape is `(height - kernelHeight + 1, width - kernelWidth + 1)`. The kernel is *not*
    /// flipped, so callers that want correlation must either flip it themselves or rely on the
    /// kernel being symmetric, which the diffusion-filter PSF is by construction.
    public static func convolveValid(
        image: [Double], height: Int, width: Int,
        kernel: [Double], kernelHeight: Int, kernelWidth: Int
    ) -> [Double] {
        precondition(image.count == height * width, "image is not \(height)x\(width)")
        precondition(
            kernel.count == kernelHeight * kernelWidth, "kernel is not \(kernelHeight)x\(kernelWidth)"
        )
        precondition(
            kernelHeight <= height && kernelWidth <= width,
            "kernel \(kernelHeight)x\(kernelWidth) exceeds image \(height)x\(width)"
        )

        let rows = supportedLength(atLeast: height + kernelHeight - 1)
        let columns = supportedLength(atLeast: width + kernelWidth - 1)
        let cells = rows * columns

        let imageReal = UnsafeMutablePointer<Double>.allocate(capacity: cells)
        let imageImaginary = UnsafeMutablePointer<Double>.allocate(capacity: cells)
        let kernelReal = UnsafeMutablePointer<Double>.allocate(capacity: cells)
        let kernelImaginary = UnsafeMutablePointer<Double>.allocate(capacity: cells)
        defer {
            imageReal.deallocate()
            imageImaginary.deallocate()
            kernelReal.deallocate()
            kernelImaginary.deallocate()
        }
        imageReal.initialize(repeating: 0, count: cells)
        imageImaginary.initialize(repeating: 0, count: cells)
        kernelReal.initialize(repeating: 0, count: cells)
        kernelImaginary.initialize(repeating: 0, count: cells)

        for i in 0..<height {
            for j in 0..<width { imageReal[i * columns + j] = image[i * width + j] }
        }
        for i in 0..<kernelHeight {
            for j in 0..<kernelWidth { kernelReal[i * columns + j] = kernel[i * kernelWidth + j] }
        }

        guard
            let rowForward = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(columns), .FORWARD),
            let columnForward = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(rows), .FORWARD),
            let rowInverse = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(columns), .INVERSE),
            let columnInverse = vDSP_DFT_zop_CreateSetupD(nil, vDSP_Length(rows), .INVERSE)
        else {
            preconditionFailure("vDSP rejected DFT length \(rows)x\(columns)")
        }
        defer {
            vDSP_DFT_DestroySetupD(rowForward)
            vDSP_DFT_DestroySetupD(columnForward)
            vDSP_DFT_DestroySetupD(rowInverse)
            vDSP_DFT_DestroySetupD(columnInverse)
        }

        transform(
            real: imageReal, imaginary: imageImaginary, rows: rows, columns: columns,
            rowSetup: rowForward, columnSetup: columnForward)
        transform(
            real: kernelReal, imaginary: kernelImaginary, rows: rows, columns: columns,
            rowSetup: rowForward, columnSetup: columnForward)

        for i in 0..<cells {
            let ar = imageReal[i]
            let ai = imageImaginary[i]
            let br = kernelReal[i]
            let bi = kernelImaginary[i]
            imageReal[i] = ar * br - ai * bi
            imageImaginary[i] = ar * bi + ai * br
        }

        transform(
            real: imageReal, imaginary: imageImaginary, rows: rows, columns: columns,
            rowSetup: rowInverse, columnSetup: columnInverse)

        // vDSP's forward and inverse transforms are both unscaled.
        let normalisation = 1.0 / Double(cells)
        let outHeight = height - kernelHeight + 1
        let outWidth = width - kernelWidth + 1
        var out = [Double](repeating: 0, count: outHeight * outWidth)
        for i in 0..<outHeight {
            let source = (i + kernelHeight - 1) * columns + kernelWidth - 1
            for j in 0..<outWidth { out[i * outWidth + j] = imageReal[source + j] * normalisation }
        }
        return out
    }

    /// Separable 2D DFT: every row, then every column.
    private static func transform(
        real: UnsafeMutablePointer<Double>, imaginary: UnsafeMutablePointer<Double>,
        rows: Int, columns: Int,
        rowSetup: vDSP_DFT_SetupD, columnSetup: vDSP_DFT_SetupD
    ) {
        let span = Swift.max(rows, columns)
        let gatherReal = UnsafeMutablePointer<Double>.allocate(capacity: span)
        let gatherImaginary = UnsafeMutablePointer<Double>.allocate(capacity: span)
        let resultReal = UnsafeMutablePointer<Double>.allocate(capacity: span)
        let resultImaginary = UnsafeMutablePointer<Double>.allocate(capacity: span)
        defer {
            gatherReal.deallocate()
            gatherImaginary.deallocate()
            resultReal.deallocate()
            resultImaginary.deallocate()
        }

        for row in 0..<rows {
            let base = row * columns
            vDSP_DFT_ExecuteD(
                rowSetup, real + base, imaginary + base, resultReal, resultImaginary)
            for j in 0..<columns {
                real[base + j] = resultReal[j]
                imaginary[base + j] = resultImaginary[j]
            }
        }
        for column in 0..<columns {
            for i in 0..<rows {
                gatherReal[i] = real[i * columns + column]
                gatherImaginary[i] = imaginary[i * columns + column]
            }
            vDSP_DFT_ExecuteD(
                columnSetup, gatherReal, gatherImaginary, resultReal, resultImaginary)
            for i in 0..<rows {
                real[i * columns + column] = resultReal[i]
                imaginary[i * columns + column] = resultImaginary[i]
            }
        }
    }
}

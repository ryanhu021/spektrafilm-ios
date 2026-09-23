#if canImport(Metal)
import Foundation
import Metal

/// ``OutputGamutCompressor`` on a ``GPUFrame``. The compressor's envelope is uploaded as built; the
/// GPU never rebuilds it.
enum MetalGamut {
    /// Mirrors the `GamutParams` struct in ``MetalKernels/gamut``.
    struct Parameters {
        var mode: UInt32 = 0
        var pixels: UInt32 = 0
        var hasLightnessKnee: UInt32 = 0
        var lightnessCount: UInt32 = 0
        var hueCount: UInt32 = 0
        var threshold: Float = 0
        var limit: Float = 0
        var power: Float = 0
        var lightnessThreshold: Float = 0
        var lightnessLimit: Float = 0
        var lightnessPower: Float = 0
        var lightnessWhite: Float = 0
        var lightnessFirst: Float = 0
        var lightnessLast: Float = 0
        var hueFirst: Float = 0
        var hueStep: Float = 0
        var F_L: Float = 0
        var N_bb: Float = 0
        var A_w: Float = 0
        var cz: Float = 0
        var inverseCz: Float = 0
        var chromaScale: Float = 0
        var chromaTerm: Float = 0
        var F_L4: Float = 0
    }

    /// ``OutputGamutCompressor/apply(to:)``, in place.
    static func apply(
        _ context: MetalContext, _ compressor: OutputGamutCompressor, to frame: GPUFrame
    ) throws {
        precondition(frame.channels == 3, "output gamut compression needs a 3-channel frame")
        var p = Parameters()
        let matrices: [Matrix3]
        switch compressor.kind {
        case .off:
            return
        case .acesRGC:
            p.mode = 0
            matrices = [.identity, .identity, .identity, .identity]
        case .perceptual(let space):
            p.mode = mode(space)
            matrices = foldedMatrices(compressor, space)
        }
        p.pixels = UInt32(frame.pixelCount)
        p.threshold = Float(compressor.knee.threshold)
        p.limit = Float(compressor.knee.limit)
        p.power = Float(compressor.knee.power)
        if let knee = compressor.lightnessCompression {
            p.hasLightnessKnee = 1
            p.lightnessThreshold = Float(knee.threshold)
            p.lightnessLimit = Float(knee.limit)
            p.lightnessPower = Float(knee.power)
        }
        p.lightnessWhite = Float(compressor.lightnessWhite)
        if let envelope = compressor.envelope {
            p.lightnessCount = UInt32(envelope.lightnessGrid.count)
            p.hueCount = UInt32(envelope.hueGrid.count)
            p.lightnessFirst = Float(envelope.lightnessGrid[0])
            p.lightnessLast = Float(envelope.lightnessGrid[envelope.lightnessGrid.count - 1])
            p.hueFirst = Float(envelope.hueGrid[0])
            p.hueStep = Float(envelope.hueGrid[1] - envelope.hueGrid[0])
        }
        if let vc = compressor.viewing {
            p.F_L = Float(vc.F_L)
            p.N_bb = Float(vc.N_bb)
            p.A_w = Float(vc.A_w)
            p.cz = Float(vc.c * vc.z)
            p.inverseCz = Float(1.0 / (vc.c * vc.z))
            p.chromaScale = Float((50000.0 / 13.0) * vc.N_c * vc.N_cb)
            p.chromaTerm = Float(vc.chromaExponentTerm)
            p.F_L4 = Float(spow(vc.F_L, 0.25))
        }

        var m: [Float] = matrices.flatMap { matrix in
            (0..<9).map { Float(matrix[$0 / 3, $0 % 3]) }
        }
        let table = try context.buffer(from: compressor.envelope?.values ?? [0])
        try context.dispatch("gamut_compress", count: frame.pixelCount) { e in
            e.setBuffer(frame.buffer, offset: 0, index: 0)
            e.setBytes(&m, length: m.count * 4, index: 1)
            e.setBuffer(table, offset: 0, index: 2)
            e.setBytes(&p, length: MemoryLayout<Parameters>.stride, index: 3)
        }
    }

    private static func mode(_ space: PerceptualSpace) -> UInt32 {
        switch space {
        case .oklch: return 1
        case .oklrab: return 2
        case .jzazbz: return 3
        case .cam16ucs: return 4
        }
    }

    /// RGB to the space's cone stage, cone to opponent, opponent to cone, and cone back to RGB.
    ///
    /// Everything linear between the output RGB and each non-linearity folds into the outer two:
    /// JzAzBz's ×100 and X'Y' shear, and CAM16's ×100 and `D_RGB` scaling. CAM16's opponent axes
    /// are not a matrix, so its middle two are unused.
    private static func foldedMatrices(
        _ compressor: OutputGamutCompressor, _ space: PerceptualSpace
    ) -> [Matrix3] {
        let toXYZ = compressor.rgbToXYZ
        let toRGB = compressor.xyzToRGB
        switch space {
        case .oklch, .oklrab:
            return [
                Oklab.xyzToLMS * toXYZ, Oklab.lmsPrimeToLab, Oklab.labToLMSPrime,
                toRGB * Oklab.lmsToXYZ,
            ]
        case .jzazbz:
            let b = ColourTables.jzazbz_b
            let g = ColourTables.jzazbz_g
            let scale = OutputGamutCompressor.jzazbzWhiteLuminance
            let shear = Matrix3(b, 0, -(b - 1), -(g - 1), g, 0, 0, 0, 1)
            let unshear = Matrix3(
                1 / b, 0, (b - 1) / b,
                (g - 1) / (g * b), 1 / g, (g - 1) * (b - 1) / (g * b),
                0, 0, 1)
            return [
                JzAzBz.xyzToLMS * shear * .diagonal(scale, scale, scale) * toXYZ,
                JzAzBz.lmsPrimeToIzAzBz, JzAzBz.izAzBzToLMSPrime,
                toRGB * .diagonal(1 / scale, 1 / scale, 1 / scale) * unshear * JzAzBz.lmsToXYZ,
            ]
        case .cam16ucs:
            let d = compressor.viewing!.dRGB
            return [
                .diagonal(d.0, d.1, d.2) * CAM16UCS.cat16 * .diagonal(100, 100, 100) * toXYZ,
                .identity, .identity,
                toRGB * .diagonal(0.01, 0.01, 0.01) * CAM16UCS.cat16Inverse
                    * .diagonal(1 / d.0, 1 / d.1, 1 / d.2),
            ]
        }
    }
}
#endif

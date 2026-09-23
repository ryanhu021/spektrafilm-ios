import Foundation

/// A point in the square ("quad") chromaticity space the Hanatos LUTs are indexed by.
///
/// `tc` in the reference. Both components are in `[0, 1]` after ``ChromaticityCoordinates/triToQuad``,
/// and a grid index `i` of a size-`N` LUT sits at `i / (N - 1)`.
public struct TCCoordinate: Sendable, Equatable {
    public var x: Double
    public var y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }
}

/// The warp between CIE 1931 `xy` and the LUT's square coordinates.
///
/// `spectral_upsampling._tri2quad` and `._quad2tri`. The warp spreads the visible locus over the
/// unit square far more evenly than raw `xy` does, so the shipped 192x192 irradiance table is
/// sampled in this space.
///
/// Used by two subsystems: the runtime `rgbToTCB` and the build-time input gamut compression bake.
public enum ChromaticityCoordinates {

    /// CIE `xy` to `tc`.
    ///
    /// Three details matter, and all three show up out of gamut:
    ///
    /// - `qy` divides by `fmax(1 - x, 1e-10)`, so a NaN `x` yields the `1e-10` guard, not NaN.
    /// - `qy` is computed from the **unclamped** `x`. Clamping first would move every out-of-locus
    ///   input.
    /// - `qx = (1 - x)^2` is symmetric about `x = 1`, so `x = 1.2` and `x = 0.8` both give 0.04.
    ///   Super-unit chromaticities alias onto valid cells without any error. Reproduced on purpose.
    ///
    /// NaN survives the clip, as it does in `np.clip`, and the LUT fetch catches it.
    @inlinable
    public static func triToQuad(x: Double, y: Double) -> TCCoordinate {
        let qy = y / npFmax(1.0 - x, 1e-10)
        let qx = (1.0 - x) * (1.0 - x)
        return TCCoordinate(x: clipUnit(qx), y: clipUnit(qy))
    }

    @inlinable
    public static func triToQuad(_ xy: Chromaticity) -> TCCoordinate {
        triToQuad(x: xy.x, y: xy.y)
    }

    /// `tc` back to CIE `xy`. No clamping, so it is only the inverse for `x` in `[0, 1)`.
    @inlinable
    public static func quadToTri(x: Double, y: Double) -> Chromaticity {
        let root = x.squareRoot()
        return Chromaticity(x: 1.0 - root, y: y * root)
    }

    @inlinable
    public static func quadToTri(_ tc: TCCoordinate) -> Chromaticity {
        quadToTri(x: tc.x, y: tc.y)
    }

    /// `np.clip(v, 0, 1)`, which passes NaN through because both of its comparisons are false.
    @inlinable
    public static func clipUnit(_ v: Double) -> Double {
        if v < 0 { return 0 }
        if v > 1 { return 1 }
        return v
    }
}

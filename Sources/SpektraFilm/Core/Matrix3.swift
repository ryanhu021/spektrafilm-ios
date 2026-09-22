import Foundation

/// A 3x3 matrix of `Double`, row-major.
///
/// Every matrix in the engine is 3x3: colourspace conversions, cone responses, the DIR-coupler
/// inhibition matrix. Multiplication order affects parity, and explicit operations make the order
/// easy to read off.
public struct Matrix3: Sendable, Equatable {
    public var m00, m01, m02: Double
    public var m10, m11, m12: Double
    public var m20, m21, m22: Double

    public init(
        _ m00: Double, _ m01: Double, _ m02: Double,
        _ m10: Double, _ m11: Double, _ m12: Double,
        _ m20: Double, _ m21: Double, _ m22: Double
    ) {
        self.m00 = m00
        self.m01 = m01
        self.m02 = m02
        self.m10 = m10
        self.m11 = m11
        self.m12 = m12
        self.m20 = m20
        self.m21 = m21
        self.m22 = m22
    }

    /// From nested rows, the shape ``ColourTables`` emits.
    public init(rows: [[Double]]) {
        precondition(rows.count == 3 && rows.allSatisfy { $0.count == 3 }, "expected 3x3")
        self.init(
            rows[0][0], rows[0][1], rows[0][2],
            rows[1][0], rows[1][1], rows[1][2],
            rows[2][0], rows[2][1], rows[2][2]
        )
    }

    public static let identity = Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)

    public static func diagonal(_ a: Double, _ b: Double, _ c: Double) -> Matrix3 {
        Matrix3(a, 0, 0, 0, b, 0, 0, 0, c)
    }

    @inlinable
    public subscript(row: Int, column: Int) -> Double {
        switch row * 3 + column {
        case 0: return m00
        case 1: return m01
        case 2: return m02
        case 3: return m10
        case 4: return m11
        case 5: return m12
        case 6: return m20
        case 7: return m21
        default: return m22
        }
    }

    /// Matrix product in written order: `lhs * rhs` applies `rhs` first.
    public static func * (lhs: Matrix3, rhs: Matrix3) -> Matrix3 {
        Matrix3(
            lhs.m00 * rhs.m00 + lhs.m01 * rhs.m10 + lhs.m02 * rhs.m20,
            lhs.m00 * rhs.m01 + lhs.m01 * rhs.m11 + lhs.m02 * rhs.m21,
            lhs.m00 * rhs.m02 + lhs.m01 * rhs.m12 + lhs.m02 * rhs.m22,
            lhs.m10 * rhs.m00 + lhs.m11 * rhs.m10 + lhs.m12 * rhs.m20,
            lhs.m10 * rhs.m01 + lhs.m11 * rhs.m11 + lhs.m12 * rhs.m21,
            lhs.m10 * rhs.m02 + lhs.m11 * rhs.m12 + lhs.m12 * rhs.m22,
            lhs.m20 * rhs.m00 + lhs.m21 * rhs.m10 + lhs.m22 * rhs.m20,
            lhs.m20 * rhs.m01 + lhs.m21 * rhs.m11 + lhs.m22 * rhs.m21,
            lhs.m20 * rhs.m02 + lhs.m21 * rhs.m12 + lhs.m22 * rhs.m22
        )
    }

    @inlinable
    public func apply(_ v: (Double, Double, Double)) -> (Double, Double, Double) {
        (
            m00 * v.0 + m01 * v.1 + m02 * v.2,
            m10 * v.0 + m11 * v.1 + m12 * v.2,
            m20 * v.0 + m21 * v.1 + m22 * v.2
        )
    }

    public var determinant: Double {
        m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
            + m02 * (m10 * m21 - m11 * m20)
    }

    /// Analytic inverse via the adjugate.
    ///
    /// The reference gets its inverses from LAPACK, which differs in the last couple of bits. That
    /// reaches about 1e-16 relative in the rendered pixel, twelve orders of magnitude under the
    /// 1e-4 gate, so LAPACK is not worth a dependency.
    public var inverse: Matrix3 {
        let det = determinant
        precondition(det != 0, "Matrix3 is singular")
        let invDet = 1.0 / det
        return Matrix3(
            (m11 * m22 - m12 * m21) * invDet,
            (m02 * m21 - m01 * m22) * invDet,
            (m01 * m12 - m02 * m11) * invDet,
            (m12 * m20 - m10 * m22) * invDet,
            (m00 * m22 - m02 * m20) * invDet,
            (m02 * m10 - m00 * m12) * invDet,
            (m10 * m21 - m11 * m20) * invDet,
            (m01 * m20 - m00 * m21) * invDet,
            (m00 * m11 - m01 * m10) * invDet
        )
    }

    /// Applies the matrix to every pixel of a 3-channel buffer, in place.
    ///
    /// A plain loop. `simd_double3x3` measured 2.3 ms against 3.8 ms over 4 MP, which is not worth
    /// the column-major conversion when the spectral contractions in the same pipeline do 27 times
    /// the arithmetic. Accelerate is worse still, 28.7 ms, since a K=3 gemm is all call overhead.
    public func apply(to buffer: inout ImageBuffer) {
        precondition(buffer.channels == 3, "Matrix3 applies to 3-channel buffers")
        let a = m00, b = m01, c = m02
        let d = m10, e = m11, f = m12
        let g = m20, h = m21, i = m22
        buffer.values.withUnsafeMutableBufferPointer { buf in
            guard let p = buf.baseAddress else { return }
            for k in stride(from: 0, to: buf.count, by: 3) {
                let r = p[k]
                let gg = p[k + 1]
                let bb = p[k + 2]
                p[k] = a * r + b * gg + c * bb
                p[k + 1] = d * r + e * gg + f * bb
                p[k + 2] = g * r + h * gg + i * bb
            }
        }
    }
}

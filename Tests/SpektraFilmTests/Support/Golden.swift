import Foundation
import Testing

@testable import SpektraFilm

/// Reads a parity fixture written by `Tools/parity/generate_goldens.py`.
///
/// `Tools/parity/spkg.py` documents the layout. A wrong magic, a truncated payload or an
/// unexpected shape all throw, because an empty array would make every comparison pass.
struct Golden {
    let shape: [Int]
    let values: [Double]

    var count: Int { values.count }

    init(_ name: String) throws {
        guard
            let url = Bundle.module.url(
                forResource: name, withExtension: "spkg", subdirectory: "Goldens")
        else {
            throw GoldenError.missing(name)
        }
        let blob = try Data(contentsOf: url)
        guard blob.count >= 12, blob.prefix(8) == Data("SPKG0001".utf8) else {
            throw GoldenError.malformed(name, "bad magic")
        }
        let rank = Int(blob.uint32(at: 8))
        guard rank > 0, rank <= 8, blob.count >= 12 + 4 * rank else {
            throw GoldenError.malformed(name, "implausible rank \(rank)")
        }
        var dims: [Int] = []
        for i in 0..<rank { dims.append(Int(blob.uint32(at: 12 + 4 * i))) }
        var offset = 12 + 4 * rank
        offset += (8 - offset % 8) % 8
        let expected = dims.reduce(1, *)
        guard blob.count == offset + expected * 8 else {
            throw GoldenError.malformed(
                name, "expected \(offset + expected * 8) bytes for shape \(dims), got \(blob.count)"
            )
        }
        shape = dims
        values = (0..<expected).map { blob.float64(at: offset + $0 * 8) }
    }

    /// The fixture as an ``ImageBuffer``, for rank-3 `[height][width][channel]` goldens.
    func imageBuffer() throws -> ImageBuffer {
        guard shape.count == 3 else {
            throw GoldenError.malformed("golden", "expected rank 3, got shape \(shape)")
        }
        return ImageBuffer(
            height: shape[0], width: shape[1], channels: shape[2], values: values)
    }

    enum GoldenError: Error, CustomStringConvertible {
        case missing(String)
        case malformed(String, String)

        var description: String {
            switch self {
            case .missing(let name):
                return "golden '\(name).spkg' is not in the test bundle; run `make goldens`"
            case .malformed(let name, let why):
                return "golden '\(name)' is malformed: \(why)"
            }
        }
    }
}

extension Data {
    fileprivate func uint32(at offset: Int) -> UInt32 {
        withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
    }

    fileprivate func float64(at offset: Int) -> Double {
        withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: Double.self) }
    }
}

// MARK: - Comparison

/// Difference statistics between a computed result and a golden.
struct ParityReport: CustomStringConvertible {
    let maxAbsolute: Double
    let rootMeanSquare: Double
    let worstIndex: Int
    let worstActual: Double
    let worstExpected: Double
    let nanMismatches: Int
    /// Values that are numerically equal but differ in the sign of zero.
    let signedZeroMismatches: Int
    let count: Int

    var description: String {
        var s = String(
            format: "max_abs=%.3e rms=%.3e over %d values", maxAbsolute, rootMeanSquare, count)
        if count > 0 {
            s += String(
                format: "; worst at [%d]: %.12g vs %.12g", worstIndex, worstActual, worstExpected)
        }
        if nanMismatches > 0 { s += "; \(nanMismatches) NaN mismatches" }
        if signedZeroMismatches > 0 {
            s += "; \(signedZeroMismatches) signed-zero mismatches"
        }
        return s
    }
}

/// Compares elementwise, treating NaN as equal to NaN.
///
/// Profiles encode missing data as JSON `null`, which the reference loads as NaN and carries
/// through the density curves, so stocks with partial datasheet coverage legitimately contain NaN.
/// Skipping NaN outright would hide a port that turned real numbers into NaN, which is what
/// ``ParityReport/nanMismatches`` counts.
func parity(_ actual: [Double], _ expected: [Double]) -> ParityReport {
    precondition(
        actual.count == expected.count,
        "parity compares equal-length arrays; got \(actual.count) vs \(expected.count)")
    var maxAbs = 0.0
    var sumSquares = 0.0
    var worstIndex = 0
    var nanMismatches = 0
    var signedZeroMismatches = 0
    var compared = 0
    for i in actual.indices {
        let a = actual[i]
        let e = expected[i]
        if a.isNaN || e.isNaN {
            if a.isNaN != e.isNaN { nanMismatches += 1 }
            continue
        }
        // abs(-0.0 - 0.0) is 0, so a signed-zero disagreement is invisible to the bounds below.
        // That is not academic: it hid an npFmax that broke the tie the opposite way from np.fmax,
        // in a fixture that contained the exact failing case.
        if a == 0 && e == 0 && a.sign != e.sign { signedZeroMismatches += 1 }
        let d = abs(a - e)
        if d > maxAbs {
            maxAbs = d
            worstIndex = i
        }
        sumSquares += d * d
        compared += 1
    }
    return ParityReport(
        maxAbsolute: maxAbs,
        rootMeanSquare: compared > 0 ? (sumSquares / Double(compared)).squareRoot() : 0,
        worstIndex: worstIndex,
        worstActual: compared > 0 ? actual[worstIndex] : .nan,
        worstExpected: compared > 0 ? expected[worstIndex] : .nan,
        nanMismatches: nanMismatches,
        signedZeroMismatches: signedZeroMismatches,
        count: compared
    )
}

/// The parity contract: absolute and RMS bounds, and no NaN appearing or disappearing.
///
/// Same tolerance the Android port settled on, so both ports promise the same thing.
func expectParity(
    _ actual: [Double],
    matches golden: String,
    maxAbsolute: Double = 1e-4,
    rootMeanSquare: Double = 1e-5,
    allowedSignedZeroMismatches: Int = 0,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let expected = try Golden(golden)
    let report = parity(actual, expected.values)
    // A fixture that is entirely NaN, or a port that turned everything into NaN, would otherwise
    // compare zero elements and pass.
    #expect(
        report.count > 0,
        "\(golden): no comparable values; \(report)",
        sourceLocation: sourceLocation)
    #expect(
        report.nanMismatches == 0,
        "\(golden): \(report)",
        sourceLocation: sourceLocation)
    #expect(
        report.signedZeroMismatches <= allowedSignedZeroMismatches,
        "\(golden): \(report)",
        sourceLocation: sourceLocation)
    #expect(
        report.maxAbsolute <= maxAbsolute,
        "\(golden): \(report) exceeds max_abs \(maxAbsolute)",
        sourceLocation: sourceLocation)
    #expect(
        report.rootMeanSquare <= rootMeanSquare,
        "\(golden): \(report) exceeds rms \(rootMeanSquare)",
        sourceLocation: sourceLocation)
}

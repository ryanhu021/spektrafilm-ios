import Foundation

/// The element types the reader accepts, spelled as NumPy's `descr` string.
///
/// Little-endian IEEE-754 only. Every array the engine loads is written by NumPy on a
/// little-endian host, and a big-endian or integer `descr` is far more likely to be the wrong file
/// than a file worth byte-swapping.
public enum NumpyDType: String, Sendable, CaseIterable {
    case float16 = "<f2"
    case float32 = "<f4"
    case float64 = "<f8"

    public var byteCount: Int {
        switch self {
        case .float16: return 2
        case .float32: return 4
        case .float64: return 8
        }
    }
}

/// A `.npy` array kept in its on-disk form, widened to `Double` on demand.
///
/// The payload is not copied and not converted at load. The spectra LUT is 192x192x81 `Float16`,
/// 5 971 968 bytes on disk and 23 887 872 bytes once widened, so a caller that only needs a few
/// spectra per pixel reads through ``subscript(_:)`` or ``withPayload(_:)`` and never pays for the
/// widened copy. ``values()`` is there for small arrays and for tests.
///
/// Float16 and Float32 widen to `Double` exactly, so values match NumPy's
/// `np.double(np.load(...))` bit for bit.
public struct NumpyArray: Sendable {
    /// Dimensions in C order. Empty for a 0-d array, which holds one element.
    public let shape: [Int]
    public let dtype: NumpyDType
    /// Identifies the array in error messages: a bundle-relative path or a file name.
    public let source: String
    /// Element count, `shape.reduce(1, *)`.
    public let count: Int
    /// Element strides in C order, in elements.
    public let strides: [Int]

    /// The whole file. Memory-mapped when the array came from ``NumpyArrayReader/mapped(at:)``.
    private let bytes: Data
    /// Where the payload starts, counted from the logical start of `bytes`.
    private let payloadOffset: Int

    init(shape: [Int], dtype: NumpyDType, source: String, bytes: Data, payloadOffset: Int) {
        self.shape = shape
        self.dtype = dtype
        self.source = source
        self.count = shape.reduce(1, *)
        self.bytes = bytes
        self.payloadOffset = payloadOffset
        var reversed: [Int] = []
        var running = 1
        for dimension in shape.reversed() {
            reversed.append(running)
            running *= dimension
        }
        self.strides = reversed.reversed()
    }

    /// Payload size in bytes, which is what the array costs resident while it stays unwidened.
    public var payloadByteCount: Int { count * dtype.byteCount }

    /// Flat element index of a C-order subscript.
    public func offset(_ indices: Int...) -> Int {
        precondition(
            indices.count == shape.count,
            "\(source): \(indices.count) indices for shape \(shape)")
        var flat = 0
        for (axis, index) in indices.enumerated() {
            precondition(
                index >= 0 && index < shape[axis],
                "\(source): index \(index) out of range on axis \(axis) of shape \(shape)")
            flat += index * strides[axis]
        }
        return flat
    }

    /// One element, widened.
    public subscript(_ flatIndex: Int) -> Double {
        precondition(
            flatIndex >= 0 && flatIndex < count,
            "\(source): index \(flatIndex) out of range for \(count) elements")
        return bytes.withUnsafeBytes { raw in
            Self.widen(raw, at: payloadOffset + flatIndex * dtype.byteCount, dtype: dtype)
        }
    }

    /// A contiguous run of elements, widened.
    public func values(_ range: Range<Int>) -> [Double] {
        precondition(
            range.lowerBound >= 0 && range.upperBound <= count,
            "\(source): range \(range) out of range for \(count) elements")
        let size = dtype.byteCount
        return bytes.withUnsafeBytes { raw in
            var out = [Double](repeating: 0, count: range.count)
            for i in 0..<range.count {
                out[i] = Self.widen(
                    raw, at: payloadOffset + (range.lowerBound + i) * size, dtype: dtype)
            }
            return out
        }
    }

    /// The whole array, widened. Costs `count * 8` bytes.
    public func values() -> [Double] { values(0..<count) }

    /// Hands the raw payload bytes to `body` without copying or widening them.
    ///
    /// The seam for stages that fold the LUT into a sum: widening element by element inside the
    /// contraction gives the same `Double` arithmetic as widening up front, at 2 bytes per element
    /// instead of 8.
    public func withPayload<R>(_ body: (UnsafeRawBufferPointer) throws -> R) rethrows -> R {
        try bytes.withUnsafeBytes { raw in
            try body(
                UnsafeRawBufferPointer(rebasing: raw[payloadOffset..<(payloadOffset + payloadByteCount)]))
        }
    }

    // MARK: - Widening

    private static func widen(
        _ raw: UnsafeRawBufferPointer, at byteOffset: Int, dtype: NumpyDType
    ) -> Double {
        switch dtype {
        case .float16:
            return double(fromBinary16: raw.loadUnaligned(fromByteOffset: byteOffset, as: UInt16.self))
        case .float32:
            return Double(
                Float(bitPattern: raw.loadUnaligned(fromByteOffset: byteOffset, as: UInt32.self)))
        case .float64:
            return Double(
                bitPattern: raw.loadUnaligned(fromByteOffset: byteOffset, as: UInt64.self))
        }
    }

    /// IEEE-754 binary16 to `Double`, by assembling the binary64 bit pattern.
    ///
    /// Hand-rolled because `Float16` is unavailable on x86_64 macOS, which the iOS simulator builds
    /// on an Intel host target. Every finite binary16 is exactly representable in binary64,
    /// including subnormals, so the finite paths are a relabelling of the exponent and a shift of
    /// the significand.
    ///
    /// Checked against `np.double(np.arange(65536, dtype=np.uint16).view(np.float16))` for all
    /// 65 536 bit patterns: identical, including sign of zero and NaN payloads. Also identical to
    /// `Double(Float16(bitPattern:))` on all 65 536, measured on arm64 where that type exists.
    public static func double(fromBinary16 bits: UInt16) -> Double {
        let sign = UInt64(bits & 0x8000) << 48
        let exponent = Int((bits >> 10) & 0x1F)
        let fraction = UInt64(bits & 0x03FF)
        if exponent == 0 {
            // No implicit leading 1: the value is fraction * 2^-24, and zero when fraction is 0.
            let magnitude = Double(fraction) * 0x1p-24
            return sign == 0 ? magnitude : -magnitude
        }
        if exponent == 0x1F {
            // Hardware conversion quiets a signalling NaN, so bit 51 goes on whenever the
            // significand is non-zero. Without it 1022 of the NaN patterns differ from NumPy.
            let payload = fraction == 0 ? UInt64(0) : (fraction << 42) | 0x0008_0000_0000_0000
            return Double(bitPattern: sign | (0x7FF << 52) | payload)
        }
        // 1008 = 1023 - 15 rebiases a binary16 exponent into binary64.
        return Double(bitPattern: sign | (UInt64(exponent + 1008) << 52) | (fraction << 42))
    }
}

/// Reads NumPy `.npy` files, format versions 1.0 and 2.0.
///
/// Format: <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>. The engine
/// needs it for `Resources/luts/spectral_upsampling/irradiance_xy_tc.npy`, the spectral-upsampling
/// LUT, and for any large parity fixture that is cheaper to ship as `.npy` than as text.
public enum NumpyArrayReader {

    static let magic = Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59])

    /// Memory-maps a `.npy` file. The pages stay unread until an element is asked for.
    public static func mapped(at url: URL) throws -> NumpyArray {
        try mapped(at: url, source: url.lastPathComponent)
    }

    /// Memory-maps a `.npy` file from the package bundle.
    ///
    /// `subdirectory` is relative to the bundle root, as in
    /// `bundled("irradiance_xy_tc", subdirectory: "Resources/luts/spectral_upsampling")`.
    public static func bundled(_ name: String, subdirectory: String) throws -> NumpyArray {
        let path = "\(subdirectory)/\(name).npy"
        guard
            let url = Bundle.module.url(
                forResource: name, withExtension: "npy", subdirectory: subdirectory)
        else {
            throw SpektraError.missingResource(path)
        }
        return try mapped(at: url, source: path)
    }

    private static func mapped(at url: URL, source: String) throws -> NumpyArray {
        let data: Data
        do {
            // .mappedIfSafe falls back to a plain read on a volume where mapping is unsafe, so
            // this is never worse than reading the file.
            data = try Data(contentsOf: url, options: [.mappedIfSafe])
        } catch {
            throw SpektraError.malformedResource(
                source, reason: "cannot be read: \(error.localizedDescription)")
        }
        return try parse(data, source: source)
    }

    /// Parses bytes already in memory.
    public static func parse(_ data: Data, source: String) throws -> NumpyArray {
        func fail(_ reason: String) -> SpektraError {
            SpektraError.malformedResource(source, reason: reason)
        }

        let base = data.startIndex
        guard data.count >= 10 else {
            throw fail("\(data.count) bytes is shorter than the 10-byte v1 preamble")
        }
        guard data[base..<(base + 6)] == magic else {
            throw fail("bad magic; a .npy file starts with \\x93NUMPY")
        }

        let major = data[base + 6]
        let minor = data[base + 7]
        let headerStart: Int
        let headerLength: Int
        switch (major, minor) {
        case (1, 0):
            headerStart = 10
            headerLength = Int(data.uint16(at: base + 8))
        case (2, 0):
            guard data.count >= 12 else { throw fail("truncated v2 preamble") }
            headerStart = 12
            headerLength = Int(data.uint32(at: base + 8))
        default:
            // v3 differs only by a UTF-8 header, which no writer here produces.
            throw fail("unsupported format version \(major).\(minor); expected 1.0 or 2.0")
        }

        let payloadOffset = headerStart + headerLength
        guard payloadOffset <= data.count else {
            throw fail(
                "header claims \(headerLength) bytes at offset \(headerStart), past the \(data.count)-byte file"
            )
        }
        let headerBytes = data[(base + headerStart)..<(base + payloadOffset)]
        guard let header = String(bytes: headerBytes, encoding: .ascii) else {
            throw fail("header is not ASCII")
        }

        guard let descr = token(forKey: "descr", in: header) else {
            throw fail("header has no 'descr': \(header.trimmingCharacters(in: .whitespaces))")
        }
        guard let dtype = NumpyDType(rawValue: descr) else {
            throw fail(
                "dtype '\(descr)' is not supported; expected one of "
                    + NumpyDType.allCases.map(\.rawValue).joined(separator: ", "))
        }

        guard let fortran = token(forKey: "fortran_order", in: header) else {
            throw fail("header has no 'fortran_order'")
        }
        switch fortran {
        case "False": break
        case "True":
            // Transposing silently would make every downstream index wrong in a way that still
            // produces plausible numbers.
            throw fail(
                "fortran_order is True; this reader is C-order only, re-save with numpy.ascontiguousarray"
            )
        default:
            throw fail("fortran_order is '\(fortran)', expected True or False")
        }

        guard let shapeToken = token(forKey: "shape", in: header),
            shapeToken.hasPrefix("("), shapeToken.hasSuffix(")")
        else {
            throw fail("header has no 'shape' tuple")
        }
        var shape: [Int] = []
        for field in shapeToken.dropFirst().dropLast().split(separator: ",") {
            let text = field.trimmingCharacters(in: .whitespaces)
            if text.isEmpty { continue }
            guard let dimension = Int(text), dimension >= 0 else {
                throw fail("shape \(shapeToken) has a non-numeric dimension '\(text)'")
            }
            shape.append(dimension)
        }

        var count = 1
        for dimension in shape {
            let (product, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw fail("shape \(shape) overflows Int") }
            count = product
        }
        // The element size has to be inside the same guard: shape (2000000000, 2000000000) gives a
        // count that fits an Int and a byte count that does not, and an unchecked multiply traps
        // the process instead of throwing.
        let (expected, sizeOverflow) = count.multipliedReportingOverflow(by: dtype.byteCount)
        guard !sizeOverflow else {
            throw fail("shape \(shape) of \(dtype.rawValue) needs more than Int.max bytes")
        }
        guard data.count - payloadOffset == expected else {
            throw fail(
                "shape \(shape) of \(dtype.rawValue) needs \(expected) payload bytes, file has \(data.count - payloadOffset)"
            )
        }

        return NumpyArray(
            shape: shape, dtype: dtype, source: source, bytes: data, payloadOffset: payloadOffset)
    }

    /// Pulls the value of `key` out of the header dict, as written text.
    ///
    /// A hand scanner, because the header is a Python literal: `'descr'` yields `<f2`, `'shape'`
    /// yields `(192, 192, 81)` parens included, `'fortran_order'` yields `False`. Key order,
    /// spacing and the trailing comma all vary between NumPy versions, so nothing here depends on
    /// them.
    static func token(forKey key: String, in header: String) -> String? {
        guard let keyRange = header.range(of: "'\(key)'") else { return nil }
        var index = keyRange.upperBound
        func skipSpaces() {
            while index < header.endIndex, header[index] == " " { index = header.index(after: index) }
        }
        skipSpaces()
        guard index < header.endIndex, header[index] == ":" else { return nil }
        index = header.index(after: index)
        skipSpaces()
        guard index < header.endIndex else { return nil }

        if header[index] == "'" || header[index] == "\"" {
            let quote = header[index]
            let start = header.index(after: index)
            guard let end = header[start...].firstIndex(of: quote) else { return nil }
            return String(header[start..<end])
        }
        if header[index] == "(" {
            guard let end = header[index...].firstIndex(of: ")") else { return nil }
            return String(header[index...end])
        }
        let start = index
        while index < header.endIndex, !", }".contains(header[index]) {
            index = header.index(after: index)
        }
        return String(header[start..<index])
    }
}

extension Data {
    fileprivate func uint16(at offset: Int) -> UInt16 {
        UInt16(self[offset]) | UInt16(self[offset + 1]) << 8
    }

    fileprivate func uint32(at offset: Int) -> UInt32 {
        UInt32(self[offset]) | UInt32(self[offset + 1]) << 8 | UInt32(self[offset + 2]) << 16
            | UInt32(self[offset + 3]) << 24
    }
}

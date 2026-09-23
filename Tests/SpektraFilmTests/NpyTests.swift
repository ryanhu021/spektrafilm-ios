import Darwin
import Foundation
import Testing

@testable import SpektraFilm

/// Checks the `.npy` reader against NumPy.
///
/// Two gates. The synthetic files under `Goldens/` cover the header variants: v1 and v2, all three
/// dtypes, empty, 0-d, 1-D, 3-D, and the three headers that must be refused. The real 192x192x81
/// float16 spectra LUT covers the strides and the widening at scale.
///
/// Tolerance is zero throughout. Every finite Float16 and Float32 widens to `Double` exactly, and
/// the NaN and infinity patterns are pinned by their own cases, so bit equality with
/// `np.double(np.load(...))` is the whole contract; rounding never enters.
@Suite("NumPy array reader")
struct NpyTests {

    static let lutSubdirectory = "Data/luts/spectral_upsampling"

    // MARK: - Helpers

    private static func caseFile(_ stem: String) throws -> NumpyArray {
        guard
            let url = Bundle.module.url(
                forResource: stem, withExtension: "npy", subdirectory: "Goldens")
        else {
            Issue.record("\(stem).npy is not in the test bundle; run `make goldens`")
            throw CocoaError(.fileNoSuchFile)
        }
        return try NumpyArrayReader.mapped(at: url)
    }

    private static func lut() throws -> NumpyArray {
        try NumpyArrayReader.bundled("irradiance_xy_tc", subdirectory: lutSubdirectory)
    }

    /// Bit equality against a golden, which `expectParity` cannot give: it treats NaN as a skip,
    /// `-0.0 == 0.0`, and `inf - inf` as NaN, so the widening's three special branches would all
    /// pass unchecked.
    private func expectBitIdentical(
        _ actual: [Double],
        matches golden: String,
        sourceLocation: SourceLocation = #_sourceLocation
    ) throws {
        let expected = try Golden(golden).values
        #expect(
            actual.count == expected.count,
            "\(golden): \(actual.count) values, golden has \(expected.count)",
            sourceLocation: sourceLocation)
        guard actual.count == expected.count else { return }
        var mismatches: [String] = []
        for i in actual.indices where actual[i].bitPattern != expected[i].bitPattern {
            mismatches.append(
                String(
                    format: "[%d] %016llx vs %016llx", i, actual[i].bitPattern,
                    expected[i].bitPattern))
        }
        #expect(
            mismatches.isEmpty,
            "\(golden): \(mismatches.count) bit mismatches, first \(mismatches.prefix(4).joined(separator: ", "))",
            sourceLocation: sourceLocation)
    }

    /// Builds a `.npy` blob around `dict`, padded and newline-terminated the way NumPy writes it.
    ///
    /// Lets a test reach header shapes no fixture carries: a v2 length above 65535, a declared
    /// shape with no payload behind it.
    private static func blob(
        dict: String, major: UInt8, padTo: Int = 0, payload: Data = Data()
    ) -> Data {
        let preamble = major == 1 ? 10 : 12
        var header = Array(dict.utf8)
        while header.count + 1 < padTo { header.append(0x20) }
        while (preamble + header.count + 1) % 64 != 0 { header.append(0x20) }
        header.append(0x0A)

        var out = NumpyArrayReader.magic
        out.append(contentsOf: [major, 0])
        if major == 1 {
            let length = UInt16(header.count)
            out.append(contentsOf: [UInt8(length & 0xFF), UInt8(length >> 8)])
        } else {
            let length = UInt32(header.count)
            out.append(contentsOf: (0..<4).map { UInt8((length >> (8 * $0)) & 0xFF) })
        }
        out.append(contentsOf: header)
        out.append(payload)
        return out
    }

    // MARK: - Header variants

    @Test(
        "every readable synthetic case decodes bit-exactly",
        arguments: [
            ("npy_case_v1_f8_2d", [3, 4], NumpyDType.float64),
            ("npy_case_v2_f8_2d", [3, 4], .float64),
            ("npy_case_v1_f4_1d", [7], .float32),
            ("npy_case_v1_f2_3d", [2, 3, 4], .float16),
            ("npy_case_v2_f2_3d", [2, 3, 4], .float16),
            ("npy_case_v1_f2_specials", [9], .float16),
            ("npy_case_v1_f4_specials", [19], .float32),
            ("npy_case_v1_f8_specials", [14], .float64),
            ("npy_case_v1_f2_sweep", [1089], .float16),
            ("npy_case_v1_f8_scalar", [], .float64),
        ])
    func readsCase(stem: String, shape: [Int], dtype: NumpyDType) throws {
        let array = try Self.caseFile(stem)
        #expect(array.shape == shape)
        #expect(array.dtype == dtype)
        #expect(array.count == shape.reduce(1, *))
        #expect(array.payloadByteCount == array.count * dtype.byteCount)
        try expectBitIdentical(array.values(), matches: "\(stem)_values")
    }

    /// The v2 preamble is 12 bytes with a uint32 length; v1 is 10 with a uint16. Same array, so
    /// mixing the two up shows as a shifted payload.
    @Test("v1 and v2 headers over the same array give the same values")
    func versionsAgree() throws {
        for (v1, v2) in [
            ("npy_case_v1_f8_2d", "npy_case_v2_f8_2d"), ("npy_case_v1_f2_3d", "npy_case_v2_f2_3d"),
        ] {
            let a = try Self.caseFile(v1)
            let b = try Self.caseFile(v2)
            #expect(a.shape == b.shape)
            #expect(a.values() == b.values())
        }
    }

    @Test("an empty array reads as zero elements and does not fail")
    func emptyArrays() throws {
        let flat = try Self.caseFile("npy_case_v1_f8_empty")
        #expect(flat.shape == [0])
        #expect(flat.count == 0)
        #expect(flat.payloadByteCount == 0)
        #expect(flat.values().isEmpty)

        let twoDimensional = try Self.caseFile("npy_case_v1_f8_empty_2d")
        #expect(twoDimensional.shape == [0, 3])
        #expect(twoDimensional.count == 0)
        #expect(twoDimensional.values().isEmpty)
    }

    /// NumPy writes `'shape': ()` for a 0-d array, which holds one element.
    @Test("a 0-d array holds one element")
    func scalarArray() throws {
        let array = try Self.caseFile("npy_case_v1_f8_scalar")
        #expect(array.shape.isEmpty)
        #expect(array.strides.isEmpty)
        #expect(array.count == 1)
        #expect(array[0] == 3.5)
    }

    @Test("strides are C order")
    func cOrderStrides() throws {
        let array = try Self.caseFile("npy_case_v1_f2_3d")
        #expect(array.strides == [12, 4, 1])
        #expect(array.offset(1, 2, 3) == 23)
        let values = array.values()
        #expect(array[array.offset(1, 2, 3)] == values[23])
        #expect(array[array.offset(0, 1, 0)] == values[4])
    }

    // MARK: - Rejections

    @Test("a column-major file is refused instead of transposed")
    func rejectsFortranOrder() throws {
        guard
            let url = Bundle.module.url(
                forResource: "npy_case_bad_fortran", withExtension: "npy", subdirectory: "Goldens")
        else {
            Issue.record("npy_case_bad_fortran.npy is not in the test bundle")
            return
        }
        // A 3x4 F-ordered payload has the same length as a C-ordered one, so nothing but the
        // header flag distinguishes them and a silent transpose would read plausible numbers.
        let error = #expect(throws: SpektraError.self) { try NumpyArrayReader.mapped(at: url) }
        #expect(String(describing: error).contains("fortran_order"))
    }

    @Test("unsupported dtypes name what is supported", arguments: ["bad_bigendian", "bad_int"])
    func rejectsDType(suffix: String) throws {
        guard
            let url = Bundle.module.url(
                forResource: "npy_case_\(suffix)", withExtension: "npy", subdirectory: "Goldens")
        else {
            Issue.record("npy_case_\(suffix).npy is not in the test bundle")
            return
        }
        let error = #expect(throws: SpektraError.self) { try NumpyArrayReader.mapped(at: url) }
        #expect(String(describing: error).contains("<f8"))
    }

    @Test("corrupt bytes are refused instead of being read as zeros")
    func rejectsCorruption() throws {
        guard
            let url = Bundle.module.url(
                forResource: "npy_case_v1_f8_2d", withExtension: "npy", subdirectory: "Goldens")
        else {
            Issue.record("npy_case_v1_f8_2d.npy is not in the test bundle")
            return
        }
        let good = try Data(contentsOf: url)

        var wrongMagic = good
        wrongMagic[1] = 0x4F
        #expect(throws: SpektraError.self) { try NumpyArrayReader.parse(wrongMagic, source: "t") }

        var wrongVersion = good
        wrongVersion[6] = 4
        #expect(throws: SpektraError.self) { try NumpyArrayReader.parse(wrongVersion, source: "t") }

        #expect(throws: SpektraError.self) {
            try NumpyArrayReader.parse(good.prefix(good.count - 8), source: "t")
        }
        #expect(throws: SpektraError.self) {
            try NumpyArrayReader.parse(good + Data([0, 0, 0, 0, 0, 0, 0, 0]), source: "t")
        }
        #expect(throws: SpektraError.self) { try NumpyArrayReader.parse(good.prefix(9), source: "t") }
        #expect(throws: SpektraError.self) { try NumpyArrayReader.parse(Data(), source: "t") }
    }

    @Test("a header length that runs past the file is refused")
    func rejectsOverlongHeader() throws {
        var blob = NumpyArrayReader.magic
        blob.append(contentsOf: [1, 0, 0xFF, 0xFF])
        #expect(throws: SpektraError.self) { try NumpyArrayReader.parse(blob, source: "t") }
    }

    /// 2 000 000 000 squared fits an `Int`; times 8 bytes it does not. An unchecked multiply here
    /// traps the process, which a caller cannot catch.
    @Test("a shape whose byte count overflows Int throws instead of trapping")
    func rejectsOverflowingByteCount() throws {
        let blob = Self.blob(
            dict: "{'descr': '<f8', 'fortran_order': False, 'shape': (2000000000, 2000000000), }",
            major: 1)
        let error = #expect(throws: SpektraError.self) {
            try NumpyArrayReader.parse(blob, source: "t")
        }
        #expect(String(describing: error).contains("Int.max"))

        // The per-dimension guard, for the case where the element count alone overflows.
        let wider = Self.blob(
            dict:
                "{'descr': '<f8', 'fortran_order': False, 'shape': (4000000000, 4000000000, 4000000000), }",
            major: 1)
        #expect(throws: SpektraError.self) { try NumpyArrayReader.parse(wider, source: "t") }
    }

    /// The whole reason format 2.0 exists is a header too long for a uint16. Every v2 fixture has a
    /// short header, so without this a reader that ignored the top two length bytes would pass.
    @Test("a v2 header longer than 65535 bytes uses the full uint32 length")
    func readsLongV2Header() throws {
        let values = [0.0, 1.0 / 3, -2.5, 1e300, -0.0, .pi]
        var payload = Data()
        for value in values {
            let bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: bits) { payload.append(contentsOf: $0) }
        }
        let blob = Self.blob(
            dict: "{'descr': '<f8', 'fortran_order': False, 'shape': (6,), }",
            major: 2, padTo: 70_000, payload: payload)
        #expect(blob.count > 65_535 + 12)

        let array = try NumpyArrayReader.parse(blob, source: "t")
        #expect(array.shape == [6])
        #expect(array.values().map(\.bitPattern) == values.map(\.bitPattern))
    }

    // MARK: - The shipped spectra LUT

    @Test("the bundled LUT has the shape and dtype the pipeline expects")
    func lutHeader() throws {
        let lut = try Self.lut()
        #expect(lut.shape == [192, 192, 81])
        #expect(lut.dtype == .float16)
        #expect(lut.strides == [192 * 81, 81, 1])
        #expect(lut.count == 2_985_984)
        // 5 971 968 payload bytes after a 128-byte header, per the spec's byte map.
        #expect(lut.payloadByteCount == 5_971_968)
    }

    /// `(0, 0)` and `(0, 191)` both decode to `xy = (1, 0)` but store different spectra, so a
    /// reader that collapsed axis 0 would still look plausible without them.
    @Test("LUT corners and centre match numpy.load")
    func lutCorners() throws {
        let lut = try Self.lut()
        var actual: [Double] = []
        for (i, j) in [(0, 0), (0, 191), (191, 0), (191, 191), (96, 96)] {
            let base = lut.offset(i, j, 0)
            actual += lut.values(base..<(base + 81))
        }
        try expectBitIdentical(actual, matches: "npy_lut_corners")
    }

    @Test("walking each LUT axis matches numpy.load")
    func lutAxisWalks() throws {
        let lut = try Self.lut()
        try expectBitIdentical(
            (0..<192).map { lut[lut.offset($0, 77, 12)] }, matches: "npy_lut_axis0_walk")
        try expectBitIdentical(
            (0..<192).map { lut[lut.offset(40, $0, 33)] }, matches: "npy_lut_axis1_walk")

        var block: [Double] = []
        for i in 5..<9 {
            for j in 10..<13 {
                let base = lut.offset(i, j, 0)
                block += lut.values(base..<(base + 81))
            }
        }
        try expectBitIdentical(block, matches: "npy_lut_block")
    }

    /// Reads every element the way a consumer should: through the raw payload, widening one at a
    /// time, with no 22.8 MiB intermediate. The sum runs left to right over the flat C-order
    /// array, which is the order the fixture sums in.
    @Test("LUT extrema and sum over all 2 985 984 elements match numpy")
    func lutExtrema() throws {
        let lut = try Self.lut()
        let stats = lut.withPayload { raw -> [Double] in
            var lowest = Double.infinity
            var highest = -Double.infinity
            var sum = 0.0
            for i in 0..<lut.count {
                let value = NumpyArray.double(
                    fromBinary16: raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self))
                lowest = min(lowest, value)
                highest = max(highest, value)
                sum += value
            }
            return [lowest, highest, sum]
        }
        try expectBitIdentical(stats, matches: "npy_lut_extrema")
    }

    @Test("withPayload hands back exactly the payload")
    func payloadExtent() throws {
        let array = try Self.caseFile("npy_case_v1_f2_3d")
        let widened = array.withPayload { raw -> [Double] in
            #expect(raw.count == array.payloadByteCount)
            return (0..<array.count).map {
                NumpyArray.double(
                    fromBinary16: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self))
            }
        }
        #expect(widened == array.values())
    }

    // MARK: - Memory

    /// The cost the spec flags: 5.97 MB of float16 on disk becomes 22.8 MiB of `Double` if the
    /// whole table is materialised. Mapping and widening per element avoids it, and this measures
    /// both so the claim is not just arithmetic.
    ///
    /// `phys_footprint` is what the iOS jetsam limit counts. Clean file-backed pages do not
    /// contribute, which is why mapping is nearly free here. The bound is loose because other
    /// tests allocate in the same process; the point is the order of magnitude.
    @Test("mapping the LUT costs far less than widening it")
    func footprint() throws {
        let baseline = physFootprint()
        let lut = try Self.lut()
        var checksum = 0.0
        for i in 0..<192 { checksum += lut[lut.offset(i, i, 40)] }
        let mapped = physFootprint() - baseline
        #expect(checksum > 0)

        let widenedBaseline = physFootprint()
        var values = lut.values()
        #expect(values.count == 2_985_984)
        let widened = physFootprint() - widenedBaseline
        values = []

        print(
            "npy footprint: mapped + 192 lookups \(mapped) B, values() \(widened) B, "
                + "payload on disk \(lut.payloadByteCount) B")
        // 2 985 984 doubles is 23 887 872 B. Large allocations come from fresh mmap'd regions, so
        // the growth cannot be hidden by reuse of another test's freed memory.
        #expect(widened >= 8 << 20, "widening the LUT grew the footprint by only \(widened) B")
    }

    private func physFootprint() -> Int {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Int(info.phys_footprint) : 0
    }
}

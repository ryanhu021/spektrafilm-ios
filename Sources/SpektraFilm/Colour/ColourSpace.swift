import Foundation

/// An RGB colourspace: conversion matrices, whitepoint and transfer function.
///
/// The seven registered spaces are the ones the reference GUI offers
/// (`spektrafilm_gui/options.py:RGBColorSpaces`). Matrices come from ``ColourTables``, dumped
/// straight from colour-science, because several spaces ship published matrices that a fresh
/// derivation from the primaries would not match digit for digit.
public struct ColourSpace: Sendable, Hashable, Identifiable {
    public let name: String
    public let whitepoint: Chromaticity
    public let matrixRGBToXYZ: Matrix3
    public let matrixXYZToRGB: Matrix3
    public let transfer: TransferFunction

    public var id: String { name }

    public static func == (lhs: ColourSpace, rhs: ColourSpace) -> Bool { lhs.name == rhs.name }
    public func hash(into hasher: inout Hasher) { hasher.combine(name) }
}

/// A CIE xy chromaticity pair.
public struct Chromaticity: Sendable, Hashable {
    public let x: Double
    public let y: Double

    public init(x: Double, y: Double) {
        self.x = x
        self.y = y
    }

    /// `colour.xy_to_XYZ`, i.e. `xyY_to_XYZ(xy_to_xyY(xy))` at Y = 1.
    public var XYZ: (Double, Double, Double) {
        let yy = 1.0 / y
        return (x * yy, 1.0, (1 - (x + y)) * yy)
    }
}

extension ColourSpace {
    /// Every registered colourspace, in the order the reference GUI lists them.
    public static let all: [ColourSpace] = ColourTables.colourspaces.map {
        ColourSpace(
            name: $0.name,
            whitepoint: Chromaticity(x: $0.whitepoint.x, y: $0.whitepoint.y),
            matrixRGBToXYZ: Matrix3(rows: $0.toXYZ),
            matrixXYZToRGB: Matrix3(rows: $0.fromXYZ),
            transfer: $0.transfer
        )
    }

    private static let byName: [String: ColourSpace] = Dictionary(
        uniqueKeysWithValues: all.map { ($0.name, $0) }
    )

    /// Looks a colourspace up by its canonical colour-science name, e.g. `"ProPhoto RGB"`.
    public static func named(_ name: String) throws -> ColourSpace {
        guard let cs = byName[name] else {
            throw SpektraError.unknownColourSpace(name, known: all.map(\.name))
        }
        return cs
    }

    public static let sRGB = try! named("sRGB")
    public static let displayP3 = try! named("Display P3")
    public static let proPhotoRGB = try! named("ProPhoto RGB")
    public static let bt2020 = try! named("ITU-R BT.2020")
}

/// Cone-response space used for von Kries chromatic adaptation.
public enum ChromaticAdaptationTransform: String, Sendable, CaseIterable {
    /// colour-science's default for `XYZ_to_RGB` / `RGB_to_RGB`, and therefore what the scanning
    /// stage uses.
    case cat02
    /// Named explicitly by the spectral-upsampling stage, where CAT02's cone primaries misbehave
    /// for blues and violets.
    case cat16
    case bradford
    case vonKries

    public var matrix: Matrix3 {
        switch self {
        case .cat02: return Matrix3(rows: ColourTables.cat02)
        case .cat16: return Matrix3(rows: ColourTables.cat16)
        case .bradford: return Matrix3(rows: ColourTables.bradford)
        case .vonKries: return Matrix3(rows: ColourTables.vonKries)
        }
    }
}

public enum Colour {

    /// `colour.adaptation.matrix_chromatic_adaptation_VonKries`.
    ///
    /// Upstream multiplies as `(inv(M) * D) * M`, and the grouping matters for parity.
    public static func chromaticAdaptationMatrix(
        from source: (Double, Double, Double),
        to destination: (Double, Double, Double),
        transform: ChromaticAdaptationTransform = .cat02
    ) -> Matrix3 {
        let m = transform.matrix
        let rgbW = m.apply(source)
        let rgbWR = m.apply(destination)
        let d = Matrix3.diagonal(rgbWR.0 / rgbW.0, rgbWR.1 / rgbW.1, rgbWR.2 / rgbW.2)
        return (m.inverse * d) * m
    }

    /// `colour.XYZ_to_RGB(XYZ, colourspace, illuminant:, apply_cctf_encoding: false)`.
    ///
    /// Given an `illuminant`, the adaptation matrix and the XYZ to RGB matrix go on as two separate
    /// products, the way upstream does it. The scanning stage passes the print's viewing-illuminant
    /// chromaticity through here.
    public static func XYZToRGB(
        _ buffer: inout ImageBuffer,
        colourspace: ColourSpace,
        illuminant: Chromaticity? = nil,
        transform: ChromaticAdaptationTransform = .cat02
    ) {
        precondition(buffer.channels == 3, "XYZ buffer must have 3 channels")
        if let illuminant {
            let cat = chromaticAdaptationMatrix(
                from: illuminant.XYZ, to: colourspace.whitepoint.XYZ, transform: transform)
            cat.apply(to: &buffer)
        }
        colourspace.matrixXYZToRGB.apply(to: &buffer)
    }

    /// `colour.matrix_RGB_to_RGB`.
    public static func matrixRGBToRGB(
        from input: ColourSpace,
        to output: ColourSpace,
        transform: ChromaticAdaptationTransform? = .cat02
    ) -> Matrix3 {
        var m = input.matrixRGBToXYZ
        if let transform {
            let cat = chromaticAdaptationMatrix(
                from: input.whitepoint.XYZ, to: output.whitepoint.XYZ, transform: transform)
            m = cat * input.matrixRGBToXYZ
        }
        return output.matrixXYZToRGB * m
    }

    /// `colour.RGB_to_RGB`.
    ///
    /// Upstream calls this with `input == output` just to run the output transfer function, in
    /// `ScanningStage._apply_cctf_encoding`. That path still multiplies by `fromXYZ * (CAT *
    /// toXYZ)`, which only comes close to identity, so the matrix step always runs here.
    public static func RGBToRGB(
        _ buffer: inout ImageBuffer,
        from input: ColourSpace,
        to output: ColourSpace,
        transform: ChromaticAdaptationTransform? = .cat02,
        applyDecoding: Bool = false,
        applyEncoding: Bool = false
    ) {
        precondition(buffer.channels == 3, "RGB buffer must have 3 channels")
        if applyDecoding { input.transfer.decode(&buffer) }
        matrixRGBToRGB(from: input, to: output, transform: transform).apply(to: &buffer)
        if applyEncoding { output.transfer.encode(&buffer) }
    }

    /// `colour.RGB_to_XYZ`, with the same two-product structure as ``XYZToRGB``.
    public static func RGBToXYZ(
        _ buffer: inout ImageBuffer,
        colourspace: ColourSpace,
        illuminant: Chromaticity? = nil,
        transform: ChromaticAdaptationTransform = .cat02
    ) {
        precondition(buffer.channels == 3, "RGB buffer must have 3 channels")
        colourspace.matrixRGBToXYZ.apply(to: &buffer)
        if let illuminant {
            let cat = chromaticAdaptationMatrix(
                from: colourspace.whitepoint.XYZ, to: illuminant.XYZ, transform: transform)
            cat.apply(to: &buffer)
        }
    }

    /// `colour.XYZ_to_xy`.
    public static func XYZToxy(_ xyz: (Double, Double, Double)) -> Chromaticity {
        let sum = xyz.0 + xyz.1 + xyz.2
        return Chromaticity(x: xyz.0 / sum, y: xyz.1 / sum)
    }
}

/// The engine's single spectral grid: `colour.SpectralShape(380, 780, 5)`, 81 samples.
///
/// Every spectral array in the reference is aligned to this grid before it reaches the engine, so
/// there is no resampling anywhere in the pipeline and no second grid to reconcile. The numbers come
/// from ``ColourTables``, which the extractor writes from colour-science itself.
///
/// Units are nanometres everywhere. The one place metres are needed is Planck's law, which is why
/// ``wavelengthsMetres`` exists; do not carry metres anywhere else.
public enum SpectralShape {
    public static let count = ColourTables.wavelengthCount
    public static let startNm = ColourTables.wavelengthStart
    public static let intervalNm = ColourTables.wavelengthInterval
    public static var endNm: Double { startNm + Double(count - 1) * intervalNm }

    /// 380, 385, … 780.
    public static let wavelengthsNm: [Double] = (0..<count).map(wavelengthNm)

    /// The same grid in metres, for Planck's law.
    public static let wavelengthsMetres: [Double] = wavelengthsNm.map { $0 * 1e-9 }

    @inlinable
    public static func wavelengthNm(_ index: Int) -> Double {
        startNm + Double(index) * intervalNm
    }

    @inlinable
    public static func wavelengthMetres(_ index: Int) -> Double { wavelengthNm(index) * 1e-9 }
}

// MARK: - Spectrum

/// One value per wavelength on ``SpectralShape``: an illuminant, a filter transmittance, a
/// band-pass window, a base density.
///
/// NumPy shape `(81,)`.
public struct Spectrum: Sendable, Equatable {
    public var values: [Double]

    public init(_ values: [Double]) {
        precondition(
            values.count == SpectralShape.count,
            "a spectrum needs \(SpectralShape.count) samples, got \(values.count)")
        self.values = values
    }

    public init(repeating value: Double) {
        self.values = [Double](repeating: value, count: SpectralShape.count)
    }

    public static let ones = Spectrum(repeating: 1)
    public static let zeros = Spectrum(repeating: 0)

    public var count: Int { values.count }

    @inlinable
    public subscript(index: Int) -> Double {
        get { values[index] }
        set { values[index] = newValue }
    }

    public var sum: Double { values.reduce(0, +) }
    public var mean: Double { sum / Double(values.count) }

    /// Elementwise product, the shape of every filter or illuminant multiply in the reference.
    public func multiplied(by other: Spectrum) -> Spectrum {
        Spectrum(zip(values, other.values).map(*))
    }

    public func scaled(by factor: Double) -> Spectrum {
        Spectrum(values.map { $0 * factor })
    }

    public static func * (lhs: Spectrum, rhs: Spectrum) -> Spectrum { lhs.multiplied(by: rhs) }
    public static func * (lhs: Spectrum, rhs: Double) -> Spectrum { lhs.scaled(by: rhs) }
}

// MARK: - SpectralMatrix

/// A three-channel spectral table: CMFs, cone fundamentals, log sensitivity, dye density, the
/// Mallett basis.
///
/// NumPy shape `(81, 3)`, flattened `[wavelength][channel]` row-major, so channel `c` at wavelength
/// index `i` is at `i * 3 + c`. Same order as ``ColourTables`` and as NumPy's C order, and a
/// transposed spectral table is the most likely way to produce a plausible-looking wrong render.
public struct SpectralMatrix: Sendable, Equatable {
    public static let channels = 3

    public var values: [Double]

    public init(_ values: [Double]) {
        precondition(
            values.count == SpectralShape.count * Self.channels,
            "a spectral matrix needs \(SpectralShape.count * Self.channels) values, got \(values.count)"
        )
        self.values = values
    }

    /// Builds the matrix from three per-channel spectra, i.e. `np.stack(..., axis: -1)`.
    public init(channels spectra: [Spectrum]) {
        precondition(spectra.count == Self.channels, "expected \(Self.channels) channel spectra")
        var flat = [Double](repeating: 0, count: SpectralShape.count * Self.channels)
        for i in 0..<SpectralShape.count {
            for c in 0..<Self.channels { flat[i * Self.channels + c] = spectra[c][i] }
        }
        self.values = flat
    }

    @inlinable
    public subscript(wavelength i: Int, channel c: Int) -> Double {
        get { values[i * Self.channels + c] }
        set { values[i * Self.channels + c] = newValue }
    }

    /// Column `c` as a spectrum.
    public func channel(_ c: Int) -> Spectrum {
        Spectrum((0..<SpectralShape.count).map { self[wavelength: $0, channel: c] })
    }

    /// `np.sum(m, axis: 0)`: one total per channel.
    public func columnSums() -> [Double] {
        var out = [Double](repeating: 0, count: Self.channels)
        for i in 0..<SpectralShape.count {
            for c in 0..<Self.channels { out[c] += self[wavelength: i, channel: c] }
        }
        return out
    }

    /// `contract("k,kc->c", spectrum, self)`: the spectral integration every stage ends with, such as
    /// light through the CMFs to XYZ or through the sensitivities to raw exposure.
    public func contracted(with spectrum: Spectrum) -> [Double] {
        var out = [Double](repeating: 0, count: Self.channels)
        for i in 0..<SpectralShape.count {
            let s = spectrum[i]
            for c in 0..<Self.channels { out[c] += s * self[wavelength: i, channel: c] }
        }
        return out
    }

    /// `contract("c,kc->k", channelWeights, self)`: the other direction, summing over the channel
    /// axis to get a spectrum. `develop.py` builds spectral density from CMY density this way.
    public func spectrum(weightedBy channelWeights: [Double]) -> Spectrum {
        precondition(channelWeights.count == Self.channels, "expected \(Self.channels) weights")
        var out = [Double](repeating: 0, count: SpectralShape.count)
        for i in 0..<SpectralShape.count {
            var acc = 0.0
            for c in 0..<Self.channels { acc += channelWeights[c] * self[wavelength: i, channel: c] }
            out[i] = acc
        }
        return Spectrum(out)
    }

    /// Scales every channel at wavelength `i` by `spectrum[i]`, i.e. `self * spectrum[:, None]`.
    /// The sensitivity-times-illuminant products in the spectral LUT bake are this shape.
    public func multipliedPerWavelength(by spectrum: Spectrum) -> SpectralMatrix {
        var out = self
        for i in 0..<SpectralShape.count {
            let s = spectrum[i]
            for c in 0..<Self.channels { out[wavelength: i, channel: c] *= s }
        }
        return out
    }

    /// Elementwise product of two `(81, 3)` tables.
    public func multiplied(by other: SpectralMatrix) -> SpectralMatrix {
        SpectralMatrix(zip(values, other.values).map(*))
    }
}

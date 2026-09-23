import Foundation
import Testing

@testable import SpektraFilm

/// Parity for Hanatos 2025 spectral upsampling: chromaticity warp, irradiance LUT, sensitivity
/// adaptation, the Mitchell 2D fetch and end-to-end raw.
///
/// Goldens come from `Tools/parity/fixtures/spectral.py`.
struct SpectralParityTests {

    // MARK: - Fixtures

    /// Every 8th grid index plus the last, matching `DECIMATION` in the fixture module.
    static let decimation: [Int] = Array(stride(from: 0, to: 192, by: 8)) + [191]

    /// The cells the spec pins, in fixture order.
    static let cells: [(Int, Int)] = [(0, 0), (0, 191), (191, 0), (191, 191), (96, 96), (85, 99)]

    static let illuminantLabels = ["D65", "D55", "D50", "T", "TH-KG3", "TH-KG3-L", "BB3400", "K75P"]

    static let colourSpaceNames = [
        "sRGB", "DCI-P3", "Display P3", "Adobe RGB (1998)", "ITU-R BT.2020", "ProPhoto RGB",
        "ACES2065-1",
    ]

    /// `nan_to_num(10 ** profile.log_sensitivity)`, the way `filming.py` builds it.
    static func sensitivity(_ stock: String) throws -> SpectralMatrix {
        let profile = try ProfileLibrary.load(stock)
        return SpectralMatrix(profile.data.logSensitivity.map { nanToNum(pow(10.0, $0)) })
    }

    static func adaptation(
        _ stock: String, window: Bool = true, surface: Bool = false, blur: Double = 0
    ) throws -> Hanatos2025SensitivityAdaptation {
        var adaptation = try ProfileLibrary.load(stock).hanatos2025Adaptation()
        adaptation.applyWindow = window
        adaptation.applySurface = surface
        adaptation.spectralGaussianBlur = blur
        return adaptation
    }

    /// The production `tc_lut` for `kodak_portra_400`, compression bypassed.
    static func portraTCLUT() throws -> ImageBuffer {
        try TCLUTBuilder.computeHanatos2025TCLUT(
            sensitivity: try sensitivity("kodak_portra_400"),
            adaptation: try adaptation("kodak_portra_400"))
    }

    /// Samples a 192 x 192 x C buffer on ``decimation`` along both grid axes.
    static func decimated(_ buffer: ImageBuffer) -> [Double] {
        var out: [Double] = []
        out.reserveCapacity(decimation.count * decimation.count * buffer.channels)
        for i in decimation {
            for j in decimation {
                for c in 0..<buffer.channels { out.append(buffer[i, j, c]) }
            }
        }
        return out
    }

    static func atCells(_ buffer: ImageBuffer) -> [Double] {
        cells.flatMap { i, j in (0..<buffer.channels).map { buffer[i, j, $0] } }
    }

    /// The 8-patch input the `su_tcb_*` and `su_raw_*` goldens were generated from.
    static func patchInput() throws -> ImageBuffer {
        try Golden("su_tcb_input").imageBuffer()
    }

    // MARK: - 1. The shipped irradiance LUT

    @Test func irradianceLUTShapeAndContents() throws {
        let lut = try IrradianceSpectraLUT.shared()
        #expect(lut.gridSize == 192)
        #expect(lut.sampleCount == 81)
        #expect(lut.array.dtype == .float16)

        try expectParity(
            Self.cells.flatMap { lut.spectrum(x: $0.0, y: $0.1).values },
            matches: "su_spectra_lut_cells")

        let stats = lut.statistics()
        // The sum is 3.2e6, so an absolute gate would be meaningless here.
        try expectParity(
            [stats.min, stats.max, stats.sum / 1e6], matches: "su_spectra_lut_stats_scaled")
    }

    @Test func irradianceLUTCornersDecodeToTheDegenerateChromaticity() throws {
        let lut = try IrradianceSpectraLUT.shared()
        let base = linspace(0, 1, count: lut.gridSize)
        // tc.x = 0 decodes to xy = (1, 0) for every tc.y, and the two cells hold different spectra
        // because the fit was done in tc space.
        for j in [0, 191] {
            let xy = ChromaticityCoordinates.quadToTri(x: base[0], y: base[j])
            #expect(xy.x == 1.0)
            #expect(xy.y == 0.0)
        }
        #expect(lut.spectrum(x: 0, y: 0).values != lut.spectrum(x: 0, y: 191).values)
    }

    // MARK: - 2. Chromaticity warp

    @Test func triToQuadAndBack() throws {
        let input = try Golden("su_tri2quad_input")
        var quad: [Double] = []
        var tri: [Double] = []
        for i in 0..<(input.count / 2) {
            let tc = ChromaticityCoordinates.triToQuad(x: input.values[i * 2], y: input.values[i * 2 + 1])
            quad += [tc.x, tc.y]
            let xy = ChromaticityCoordinates.quadToTri(tc)
            tri += [xy.x, xy.y]
        }
        try expectParity(quad, matches: "su_tri2quad")
        try expectParity(tri, matches: "su_quad2tri")
    }

    @Test func triToQuadFoldsSuperUnitChromaticity() {
        // qx = (1 - x)^2 is symmetric about x = 1, so 1.2 and 0.8 alias onto the same cell.
        #expect(
            ChromaticityCoordinates.triToQuad(x: 1.2, y: 0.5).x
                == ChromaticityCoordinates.triToQuad(x: 0.8, y: 0.5).x)
    }

    @Test func triToQuadUsesTheUnclampedXForY() {
        let tc = ChromaticityCoordinates.triToQuad(x: 0.999999, y: 0.5)
        #expect(tc.y == 1.0)
        #expect(abs(tc.x - 1.000000000057511e-12) < 1e-24)

        // The two only differ for x < 0, where 1 - x exceeds 1 and clipping x to 0 first would give
        // the larger ratio y instead of y / (1 - x). Oracle values from `_tri2quad`.
        let negative: [(Double, Double, Double)] = [
            (-2.0, 0.5, 0.16666666666666666),
            (-0.25, 0.9, 0.72),
            (-3.0, 0.05, 0.0125),
        ]
        for (x, y, expected) in negative {
            let quad = ChromaticityCoordinates.triToQuad(x: x, y: y)
            // (1 - x)^2 > 1 for every negative x, so the x coordinate saturates and only y carries
            // the distinction.
            #expect(quad.x == 1.0)
            #expect(abs(quad.y - expected) < 1e-15)
        }
    }

    @Test func negativeChromaticityXIsReachableFromOrdinaryInput() throws {
        // ProPhoto RGB (-0.2, 0.1, 1.0) is a saturated blue outside the input space, with a perfectly
        // finite b = 0.828. Its CIE x is -0.122, which is the branch above. Oracle: _rgb_to_tc_b.
        let rgb = ImageBuffer(height: 1, width: 1, channels: 3, values: [-0.2, 0.1, 1.0])
        let (tc, brightness) = SpectralUpsampling.rgbToTCB(
            rgb: rgb,
            colourSpace: try ColourSpace.named("ProPhoto RGB"),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"))
        #expect(abs(brightness[0] - 0.82823140170069209) < 1e-12)
        #expect(tc[0, 0, 0] == 1.0)
        #expect(abs(tc[0, 0, 1] - 0.01450835255853076) < 1e-12)
    }

    @Test func triToQuadDropsNaNInTheGuard() {
        // fmax(NaN, 1e-10) is 1e-10, so a NaN x yields a finite ratio but NaN qx.
        let tc = ChromaticityCoordinates.triToQuad(x: .nan, y: 0.5)
        #expect(tc.x.isNaN)
        #expect(tc.y == 1.0)
    }

    // MARK: - 3. Illuminant chromaticity

    @Test func illuminantChromaticityAndTC() throws {
        var xy: [Double] = []
        var tc: [Double] = []
        for label in Self.illuminantLabels {
            let chromaticity = try Illuminant(label: label).chromaticity
            xy += [chromaticity.x, chromaticity.y]
            let quad = ChromaticityCoordinates.triToQuad(chromaticity)
            tc += [quad.x, quad.y]
        }
        try expectParity(xy, matches: "su_illuminant_xy")
        try expectParity(tc, matches: "su_illuminant_tc")
    }

    // MARK: - 4. The composed CAT16 matrix

    @Test func composedRGBToXYZMatrices() throws {
        var values: [Double] = []
        for label in ["D55", "T"] {
            let illuminant = try Illuminant(label: label)
            for name in Self.colourSpaceNames {
                let m = SpectralUpsampling.composedRGBToXYZMatrix(
                    colourSpace: try ColourSpace.named(name), referenceIlluminant: illuminant)
                for row in 0..<3 {
                    for column in 0..<3 { values.append(m[row, column]) }
                }
            }
        }
        try expectParity(values, matches: "su_rgb_to_tc_b_matrices")
    }

    // MARK: - 5. rgb -> tc, b

    @Test(arguments: [("ProPhoto RGB", "prophoto"), ("ACES2065-1", "aces")])
    func rgbToTCB(space: String, tag: String) throws {
        let (tc, brightness) = SpectralUpsampling.rgbToTCB(
            rgb: try Self.patchInput(),
            colourSpace: try ColourSpace.named(space),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"))
        try expectParity(tc.values, matches: "su_tcb_\(tag)_tc")
        try expectParity(brightness, matches: "su_tcb_\(tag)_b")
    }

    @Test func brightnessKeepsItsSignForWideGamutInput() throws {
        // ACES2065-1 rgb = (-0.5, 1, -0.2) gives b < 0. Nothing here clamps it; filming.py floors
        // the raw later with log10(fmax(raw, 0) + 1e-10).
        let rgb = ImageBuffer(height: 1, width: 1, channels: 3, values: [-0.5, 1.0, -0.2])
        let (_, brightness) = SpectralUpsampling.rgbToTCB(
            rgb: rgb,
            colourSpace: try ColourSpace.named("ACES2065-1"),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"))
        // 1e-9, not tighter: ACES2065-1's composed CAT16 matrix differs from colour-science's in the
        // last bits because the adaptation inverse here is analytic and LAPACK's there.
        #expect(abs(brightness[0] - -0.087019446625840) < 1e-9)
    }

    // MARK: - 6. Band-pass window

    @Test func erf4Window() throws {
        let adaptation = try Self.adaptation("kodak_portra_400")
        let window = try SpectralBandpassWindow.erf4.evaluate(params: adaptation.windowParams)
        try expectParity(window.values, matches: "su_erf4_window")

        let sensitivity = try Self.sensitivity("kodak_portra_400")
        let normalisation = Hanatos2025SensitivityAdaptation.windowNormalisation(
            window: window, sensitivity: sensitivity,
            illuminant: adaptation.referenceIlluminant.spectrum)
        try expectParity(normalisation, matches: "su_erf4_normalization")
        try expectParity(
            try adaptation.normalisedWindow(sensitivity: sensitivity).values,
            matches: "su_erf4_window_normalised")
    }

    @Test func logiflex8Window() throws {
        let window = try SpectralBandpassWindow.logiflex8.evaluate(
            params: [415, 12, 667, 76, 430, 650, 1, 1])
        try expectParity(window.values, matches: "su_logiflex8_window")
    }

    @Test func windowModelsRejectTheWrongParameterCount() {
        #expect(throws: SpektraError.self) {
            try SpectralBandpassWindow.erf4.evaluate(params: [1, 2, 3])
        }
        #expect(throws: SpektraError.self) {
            try SpectralBandpassWindow.logiflex8.evaluate(params: [1, 2, 3, 4])
        }
    }

    // MARK: - 7. Log-exposure correction surface

    @Test func poly4Surface() throws {
        let adaptation = try Self.adaptation("kodak_portra_400")
        let surface = try LogExposureCorrectionSurface.poly4.evaluate(
            params: adaptation.surfaceParams,
            illuminantXY: adaptation.referenceIlluminant.chromaticity,
            gridSize: 192)
        try expectParity(Self.decimated(surface), matches: "su_poly4_surface")
        try expectParity(Self.atCells(surface), matches: "su_poly4_surface_cells")

        var extrema = [Double](repeating: .infinity, count: 3) + [Double](repeating: -.infinity, count: 3)
        for pixel in 0..<surface.pixelCount {
            for c in 0..<3 {
                let v = surface.values[pixel * 3 + c]
                extrema[c] = min(extrema[c], v)
                extrema[3 + c] = max(extrema[3 + c], v)
            }
        }
        try expectParity(extrema, matches: "su_poly4_surface_extrema")
    }

    @Test func poly4SurfaceIsZeroAtTheIlluminantChromaticity() throws {
        let adaptation = try Self.adaptation("kodak_portra_400")
        let surface = try LogExposureCorrectionSurface.poly4.evaluate(
            params: adaptation.surfaceParams,
            illuminantXY: adaptation.referenceIlluminant.chromaticity,
            gridSize: 192)
        // (85, 99) is the nearest grid cell to D55's tc. Dropping the polynomial's constant term
        // is what forces the correction to vanish there, which is what preserves white.
        for c in 0..<3 { #expect(abs(surface[85, 99, c]) < 3e-3) }
    }

    @Test func poly4WarpSurface() throws {
        let adaptation = try Self.adaptation("kodak_portra_400")
        // No shipped profile carries the warp strength, so append a synthetic alpha per channel.
        var params: [Double] = []
        for channel in 0..<3 {
            params += Array(adaptation.surfaceParams[(channel * 15)..<((channel + 1) * 15)]) + [0.5]
        }
        let surface = try LogExposureCorrectionSurface.poly4WarpXY.evaluate(
            params: params,
            illuminantXY: adaptation.referenceIlluminant.chromaticity,
            gridSize: 192)
        try expectParity(Self.decimated(surface), matches: "su_poly4_warp_surface")
    }

    @Test func surfaceModelsRejectTheWrongParameterCount() throws {
        let xy = Chromaticity(x: 1.0 / 3.0, y: 1.0 / 3.0)
        #expect(throws: SpektraError.self) {
            try LogExposureCorrectionSurface.poly4.evaluate(
                params: [Double](repeating: 0, count: 48), illuminantXY: xy, gridSize: 8)
        }
        #expect(throws: SpektraError.self) {
            try LogExposureCorrectionSurface.poly4WarpXY.evaluate(
                params: [Double](repeating: 0, count: 45), illuminantXY: xy, gridSize: 8)
        }
    }

    // MARK: - 8. Spectral blur

    @Test(arguments: [(1.0, "s1p0"), (2.0, "s2p0"), (4.0, "s4p0")])
    func spectralBlurOnTheLUTCorner(sigma: Double, tag: String) throws {
        let lut = try IrradianceSpectraLUT.shared()
        let blur = try #require(HanatosSpectralBlur(sigma: sigma))
        var out: [Double] = []
        var blurred = [Double](repeating: 0, count: 81)
        for i in 0..<4 {
            for j in 0..<4 {
                blur.apply(lut.spectrum(x: i, y: j).values, into: &blurred)
                out += blurred
            }
        }
        try expectParity(out, matches: "su_spectral_blur_\(tag)")
    }

    @Test func spectralBlurDeltaResponse() throws {
        var delta = [Double](repeating: 0, count: 81)
        delta[0] = 1
        var out: [Double] = []
        var blurred = [Double](repeating: 0, count: 81)
        for sigma in [1.0, 2.0, 4.0] {
            let blur = try #require(HanatosSpectralBlur(sigma: sigma))
            blur.apply(delta, into: &blurred)
            out += blurred
            // The reflect fold sends the truncated half back into the signal, so no energy is lost
            // at the edge.
            #expect(abs(blurred.reduce(0, +) - 1.0) < 1e-12)
        }
        try expectParity(out, matches: "su_spectral_blur_delta")
    }

    @Test func spectralBlurRadiusAndZeroSigma() throws {
        // radius = int(truncate * sigma + 0.5) with scipy's default truncate = 4.0.
        #expect(try #require(HanatosSpectralBlur(sigma: 2.0)).radius == 8)
        #expect(try #require(HanatosSpectralBlur(sigma: 1.0)).radius == 4)
        #expect(HanatosSpectralBlur(sigma: 0) == nil)
        #expect(HanatosSpectralBlur(sigma: -1) == nil)
    }

    // MARK: - 9. The per-film tc_lut

    @Test func tcLUTWindowOnly() throws {
        let lut = try Self.portraTCLUT()
        #expect(lut.height == 192)
        #expect(lut.width == 192)
        #expect(lut.channels == 3)
        try expectParity(lut.values, matches: "su_tc_lut_window_only")
        try expectParity(Self.atCells(lut), matches: "su_tc_lut_window_only_cells")
    }

    @Test func tcLUTVariants() throws {
        let sensitivity = try Self.sensitivity("kodak_portra_400")
        let cases: [(Hanatos2025SensitivityAdaptation, String)] = [
            (try Self.adaptation("kodak_portra_400", surface: true), "su_tc_lut_window_surface"),
            (try Self.adaptation("kodak_portra_400", blur: 4.0), "su_tc_lut_blur4"),
            (.noAdaptation, "su_tc_lut_no_adaptation"),
        ]
        for (adaptation, golden) in cases {
            let lut = try TCLUTBuilder.computeHanatos2025TCLUT(
                sensitivity: sensitivity, adaptation: adaptation)
            try expectParity(Self.decimated(lut), matches: golden)
        }
    }

    @Test func tcLUTSecondFilmAndIlluminant() throws {
        let stock = "kodak_vision3_500t"
        let adaptation = try Self.adaptation(stock)
        #expect(adaptation.referenceIlluminant == .incandescent)
        let lut = try TCLUTBuilder.computeHanatos2025TCLUT(
            sensitivity: try Self.sensitivity(stock), adaptation: adaptation)
        try expectParity(Self.decimated(lut), matches: "su_tc_lut_vision3_500t")
    }

    @Test func tcLUTThrowsWhenCompressionIsRequestedWithoutABake() throws {
        var spec = InputGamutCompressSpec()
        #expect(spec.active)
        let expected = SpektraError.unsupportedSetting(
            "io.input_gamut_compress.active", value: "true")
        #expect(throws: expected) {
            try TCLUTBuilder.computeHanatos2025TCLUT(
                sensitivity: try Self.sensitivity("kodak_portra_400"),
                adaptation: try Self.adaptation("kodak_portra_400"),
                gamutCompress: spec)
        }

        // An inactive spec is the identity, so it must not throw and must not change the LUT.
        spec.active = false
        let bypassed = try TCLUTBuilder.computeHanatos2025TCLUT(
            sensitivity: try Self.sensitivity("kodak_portra_400"),
            adaptation: try Self.adaptation("kodak_portra_400"),
            gamutCompress: spec)
        #expect(bypassed.values == (try Self.portraTCLUT()).values)
    }

    @Test func tcLUTCacheRebuildsOnEveryKeyComponent() throws {
        let stock = "kodak_portra_400"
        let sensitivity = try Self.sensitivity(stock)
        let adaptation = try Self.adaptation(stock)
        var cache = FilmingTCLUTCache()

        #expect(!cache.isCached(sensitivity: sensitivity, adaptation: adaptation))
        let first = try cache.lut(sensitivity: sensitivity, adaptation: adaptation)
        #expect(cache.isCached(sensitivity: sensitivity, adaptation: adaptation))
        #expect(try cache.lut(sensitivity: sensitivity, adaptation: adaptation).values == first.values)

        // The reference's test mutates spectral_gaussian_blur in place and re-sets the adaptation;
        // value semantics make that a different key.
        var mutated = adaptation
        mutated.spectralGaussianBlur = 4.0
        #expect(!cache.isCached(sensitivity: sensitivity, adaptation: mutated))
        _ = try cache.lut(sensitivity: sensitivity, adaptation: mutated)
        #expect(!cache.isCached(sensitivity: sensitivity, adaptation: adaptation))

        var otherSensitivity = sensitivity
        otherSensitivity[wavelength: 40, channel: 1] *= 1.01
        #expect(!cache.isCached(sensitivity: otherSensitivity, adaptation: mutated))

        var inactive = InputGamutCompressSpec()
        inactive.active = false
        _ = try cache.lut(sensitivity: sensitivity, adaptation: adaptation, gamutCompress: inactive)
        #expect(cache.isCached(sensitivity: sensitivity, adaptation: adaptation, gamutCompress: inactive))
        #expect(!cache.isCached(sensitivity: sensitivity, adaptation: adaptation, gamutCompress: nil))
    }

    // MARK: - 10. The Mitchell 2D fetch

    @Test func mitchellWeights() throws {
        try expectParity(
            linspace(-2.25, 2.25, count: 37).map(LUTInterpolation.mitchellWeight),
            matches: "su_mitchell_weights")
    }

    @Test func mitchellDoesNotInterpolate() {
        // B = C = 1/3 gives [1/18, 8/9, 1/18, 0] at frac 0. A grid-aligned fetch is a smoothing, so
        // substituting any interpolating kernel shifts every rendered pixel.
        let w = [
            LUTInterpolation.mitchellWeight(1), LUTInterpolation.mitchellWeight(0),
            LUTInterpolation.mitchellWeight(-1), LUTInterpolation.mitchellWeight(-2),
        ]
        #expect(abs(w[0] - 1.0 / 18.0) < 1e-15)
        #expect(abs(w[1] - 8.0 / 9.0) < 1e-15)
        #expect(abs(w[2] - 1.0 / 18.0) < 1e-15)
        #expect(w[3] == 0)
        #expect(abs(w.reduce(0, +) - 1.0) < 1e-15)
    }

    @Test func lut2DOnExactGridCoordinates() throws {
        let lut = try Self.portraTCLUT()
        let base = linspace(0, 1, count: 192)
        var coordinates = ImageBuffer(height: 192, width: 192, channels: 2)
        for i in 0..<192 {
            for j in 0..<192 {
                coordinates[i, j, 0] = base[i]
                coordinates[i, j, 1] = base[j]
            }
        }
        let fetched = LUTInterpolation.applyLUTCubic2D(lut: lut, coordinates: coordinates)
        try expectParity(fetched.values, matches: "su_lut2d_grid")

        // The same array against the *stored* LUT, which the fetch deliberately does not reproduce.
        // The oracle measures 0.21265295923 on this LUT (0.780 on the raw irradiance table), so this
        // pins the smoothing magnitude, not just the values.
        let smoothing = zip(fetched.values, lut.values).map { abs($0 - $1) }.max() ?? 0
        #expect(abs(smoothing - 0.21265295922986738) < 1e-9)
    }

    @Test func lut2DInterior() throws {
        let lut = try Self.portraTCLUT()
        let coordinates = try Golden("su_lut2d_random_input").imageBuffer()
        try expectParity(
            LUTInterpolation.applyLUTCubic2D(lut: lut, coordinates: coordinates).values,
            matches: "su_lut2d_random")
    }

    @Test func lut2DEdgesAndCorners() throws {
        let lut = try Self.portraTCLUT()
        let coordinates = try Golden("su_lut2d_edges_input").imageBuffer()
        try expectParity(
            LUTInterpolation.applyLUTCubic2D(lut: lut, coordinates: coordinates).values,
            matches: "su_lut2d_edges")
    }

    @Test func lut2DDegenerateFallsBackToBilinear() throws {
        let lut = ImageBuffer(height: 1, width: 1, channels: 3, values: [1, 2, 3])
        let coordinates = try Golden("su_lut2d_degenerate_input").imageBuffer()
        try expectParity(
            LUTInterpolation.applyLUTCubic2D(lut: lut, coordinates: coordinates).values,
            matches: "su_lut2d_degenerate")
    }

    @Test func coordinateBaseFractionEdgeCases() {
        #expect(LUTInterpolation.cubicCoordinateBaseFraction(-5, size: 192) == (0, 0))
        #expect(LUTInterpolation.cubicCoordinateBaseFraction(0, size: 192) == (0, 0))
        // The top edge lands in the last cell with fraction 1, so base + 1 stays in range.
        #expect(LUTInterpolation.cubicCoordinateBaseFraction(191, size: 192) == (190, 1.0))
        #expect(LUTInterpolation.cubicCoordinateBaseFraction(500, size: 192) == (190, 1.0))
        let (base, fraction) = LUTInterpolation.cubicCoordinateBaseFraction(3.25, size: 192)
        #expect(base == 3)
        #expect(fraction == 0.25)
    }

    @Test func coordinateBaseFractionGuardsNaN() throws {
        // Int(Double.nan) traps in Swift where Numba produced a garbage index. The fraction stays NaN
        // so every Mitchell weight is 0, which is what makes the fetch return exactly 0, as the
        // oracle does.
        let (base, fraction) = LUTInterpolation.cubicCoordinateBaseFraction(.nan, size: 192)
        #expect(base == 0)
        #expect(fraction.isNaN)

        let fetched = LUTInterpolation.applyLUTCubic2D(
            lut: try Self.portraTCLUT(),
            coordinates: ImageBuffer(height: 1, width: 2, channels: 2, values: [.nan, 0.5, 0.5, .nan]))
        // apply_lut_cubic_2d on the same LUT and coordinates, measured in the oracle.
        #expect(fetched.values == [0, 0, 0, 0, 0, 0])
    }

    @Test func safeIndexMirrorsWholeSample() {
        #expect(LUTInterpolation.safeIndex(-1, size: 192) == 1)
        #expect(LUTInterpolation.safeIndex(192, size: 192) == 190)
        #expect(LUTInterpolation.safeIndex(0, size: 192) == 0)
        #expect(LUTInterpolation.safeIndex(191, size: 192) == 191)
    }

    // MARK: - 11. End to end

    @Test func rgbToRawHanatos2025() throws {
        let rgb = try Self.patchInput()
        let raw = try SpectralUpsampling.rgbToRawHanatos2025(
            rgb: rgb,
            sensitivity: try Self.sensitivity("kodak_portra_400"),
            colourSpace: try ColourSpace.named("ProPhoto RGB"),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"),
            tcLUT: try Self.portraTCLUT())
        try expectParity(raw.values, matches: "su_raw_hanatos2025")

        // Midgray lands near 1 without any normalisation here: log_sensitivity is pre-balanced
        // offline, and the rest happens downstream in the colour reference service.
        for c in 0..<3 { #expect(abs(raw[0, 1, c] - 1.0) < 0.1) }
    }

    @Test func rgbToRawHanatos2025WithoutAPrebuiltLUT() throws {
        let raw = try SpectralUpsampling.rgbToRawHanatos2025(
            rgb: try Self.patchInput(),
            sensitivity: try Self.sensitivity("kodak_portra_400"),
            colourSpace: try ColourSpace.named("ProPhoto RGB"),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"))
        try expectParity(raw.values, matches: "su_raw_hanatos2025_fallback")
    }

    @Test func rgbToRawSurvivesNaNInput() throws {
        let rgb = ImageBuffer(
            height: 1, width: 2, channels: 3, values: [.nan, 0.5, 0.5, 0.184, 0.184, 0.184])
        let raw = try SpectralUpsampling.rgbToRawHanatos2025(
            rgb: rgb,
            sensitivity: try Self.sensitivity("kodak_portra_400"),
            colourSpace: try ColourSpace.named("ProPhoto RGB"),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"),
            tcLUT: try Self.portraTCLUT())
        for c in 0..<3 { #expect(raw[0, 0, c] == 0) }
        for c in 0..<3 { #expect(raw[0, 1, c].isFinite) }
    }

    @Test func mallett2019IsNotPorted() throws {
        #expect(
            throws: SpektraError.unsupportedSetting(
                "settings.rgb_to_raw_method", value: "mallett2019")
        ) {
            try SpectralUpsampling.rgbToRaw(
                method: .mallett2019,
                rgb: ImageBuffer(height: 1, width: 1, channels: 3, values: [0.5, 0.5, 0.5]),
                sensitivity: try Self.sensitivity("kodak_portra_400"),
                colourSpace: try ColourSpace.named("sRGB"),
                applyCCTFDecoding: false,
                referenceIlluminant: try Illuminant(label: "D65"))
        }
    }

    @Test func rgbToRawDispatchesHanatos2025() throws {
        let rgb = try Self.patchInput()
        let dispatched = try SpectralUpsampling.rgbToRaw(
            method: .hanatos2025,
            rgb: rgb,
            sensitivity: try Self.sensitivity("kodak_portra_400"),
            colourSpace: try ColourSpace.named("ProPhoto RGB"),
            applyCCTFDecoding: false,
            referenceIlluminant: try Illuminant(label: "D55"),
            tcLUT: try Self.portraTCLUT())
        try expectParity(dispatched.values, matches: "su_raw_hanatos2025")
    }
}

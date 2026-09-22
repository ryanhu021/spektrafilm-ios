import Foundation
import Testing

@testable import SpektraFilm

/// Checks every default in `RuntimePhotoParams` against the reference.
///
/// The defaults are the render. A single wrong one shifts every photo, and no other test in the
/// suite would catch it, so all 172 leaves are compared by name against a JSON dump of the Python
/// dataclass defaults. Keys are the Python paths, which also documents the mapping between the two
/// naming conventions.
@Suite("Parameter defaults")
struct ParamsDefaultsTests {

    /// The Swift defaults, flattened onto the reference's paths.
    static func swiftDefaults() -> [String: Value] {
        let camera = CameraParams()
        let enlarger = EnlargerParams()
        let scanner = ScannerParams()
        let film = FilmRenderingParams()
        let printRender = PrintRenderingParams()
        let io = IOParams()
        let debug = DebugParams()
        let settings = SettingsParams()
        let taps = TapsParams()

        var v: [String: Value] = [:]

        func diffusion(_ prefix: String, _ d: DiffusionFilterParams) {
            v["\(prefix).active"] = .bool(d.active)
            v["\(prefix).filter_family"] = .string(d.family.rawValue)
            v["\(prefix).strength"] = .number(d.strength)
            v["\(prefix).spatial_scale"] = .number(d.spatialScale)
            v["\(prefix).halo_warmth"] = .number(d.haloWarmth)
            v["\(prefix).core_intensity"] = .number(d.coreIntensity)
            v["\(prefix).core_size"] = .number(d.coreSize)
            v["\(prefix).halo_intensity"] = .number(d.haloIntensity)
            v["\(prefix).halo_size"] = .number(d.haloSize)
            v["\(prefix).bloom_intensity"] = .number(d.bloomIntensity)
            v["\(prefix).bloom_size"] = .number(d.bloomSize)
        }

        func glare(_ prefix: String, _ g: GlareParams) {
            v["\(prefix).active"] = .bool(g.active)
            v["\(prefix).percent"] = .number(g.percent)
            v["\(prefix).roughness"] = .number(g.roughness)
            v["\(prefix).blur"] = .number(g.blur)
        }

        // camera
        v["camera.exposure_compensation_ev"] = .number(camera.exposureCompensationEV)
        v["camera.auto_exposure"] = .bool(camera.autoExposure)
        v["camera.auto_exposure_method"] = .string(camera.autoExposureMethod.rawValue)
        v["camera.lens_blur_um"] = .number(camera.lensBlurMicrons)
        v["camera.film_format_mm"] = .number(camera.filmFormatMillimetres)
        v["camera.filter_uv[0]"] = .number(camera.filterUV.amplitude)
        v["camera.filter_uv[1]"] = .number(camera.filterUV.wavelength)
        v["camera.filter_uv[2]"] = .number(camera.filterUV.width)
        v["camera.filter_ir[0]"] = .number(camera.filterIR.amplitude)
        v["camera.filter_ir[1]"] = .number(camera.filterIR.wavelength)
        v["camera.filter_ir[2]"] = .number(camera.filterIR.width)
        diffusion("camera.diffusion_filter", camera.diffusionFilter)

        // enlarger
        v["enlarger.illuminant"] = .string(enlarger.illuminant)
        v["enlarger.print_exposure"] = .number(enlarger.printExposure)
        v["enlarger.print_exposure_compensation"] = .bool(enlarger.printExposureCompensation)
        v["enlarger.normalize_print_exposure"] = .bool(enlarger.normalizePrintExposure)
        v["enlarger.y_filter_shift"] = .number(enlarger.yFilterShift)
        v["enlarger.m_filter_shift"] = .number(enlarger.mFilterShift)
        v["enlarger.y_filter_neutral"] = .number(enlarger.yFilterNeutral)
        v["enlarger.m_filter_neutral"] = .number(enlarger.mFilterNeutral)
        v["enlarger.c_filter_neutral"] = .number(enlarger.cFilterNeutral)
        v["enlarger.lens_blur"] = .number(enlarger.lensBlur)
        v["enlarger.preflash_exposure"] = .number(enlarger.preflashExposure)
        v["enlarger.preflash_y_filter_shift"] = .number(enlarger.preflashYFilterShift)
        v["enlarger.preflash_m_filter_shift"] = .number(enlarger.preflashMFilterShift)
        diffusion("enlarger.diffusion_filter", enlarger.diffusionFilter)

        // scanner
        v["scanner.lens_blur"] = .number(scanner.lensBlur)
        v["scanner.white_correction"] = .bool(scanner.whiteCorrection)
        v["scanner.black_correction"] = .bool(scanner.blackCorrection)
        v["scanner.white_level"] = .number(scanner.whiteLevel)
        v["scanner.black_level"] = .number(scanner.blackLevel)
        v["scanner.unsharp_mask[0]"] = .number(scanner.unsharpMask.sigma)
        v["scanner.unsharp_mask[1]"] = .number(scanner.unsharpMask.amount)

        // film_render
        v["film_render.density_curve_gamma"] = .number(film.densityCurveGamma)
        let g = film.grain
        v["film_render.grain.active"] = .bool(g.active)
        v["film_render.grain.sublayers_active"] = .bool(g.sublayersActive)
        v["film_render.grain.particle_area_um2"] = .number(g.particleAreaMicronsSquared)
        v["film_render.grain.particle_scale[0]"] = .number(g.particleScale.0)
        v["film_render.grain.particle_scale[1]"] = .number(g.particleScale.1)
        v["film_render.grain.particle_scale[2]"] = .number(g.particleScale.2)
        v["film_render.grain.particle_scale_layers[0]"] = .number(g.particleScaleLayers.0)
        v["film_render.grain.particle_scale_layers[1]"] = .number(g.particleScaleLayers.1)
        v["film_render.grain.particle_scale_layers[2]"] = .number(g.particleScaleLayers.2)
        v["film_render.grain.density_min[0]"] = .number(g.densityMin.0)
        v["film_render.grain.density_min[1]"] = .number(g.densityMin.1)
        v["film_render.grain.density_min[2]"] = .number(g.densityMin.2)
        v["film_render.grain.uniformity[0]"] = .number(g.uniformity.0)
        v["film_render.grain.uniformity[1]"] = .number(g.uniformity.1)
        v["film_render.grain.uniformity[2]"] = .number(g.uniformity.2)
        v["film_render.grain.blur"] = .number(g.blur)
        v["film_render.grain.blur_dye_clouds_um"] = .number(g.blurDyeCloudsMicrons)
        v["film_render.grain.micro_structure[0]"] = .number(g.microStructure.0)
        v["film_render.grain.micro_structure[1]"] = .number(g.microStructure.1)
        v["film_render.grain.n_sub_layers"] = .number(Double(g.subLayerCount))

        let h = film.halation
        v["film_render.halation.active"] = .bool(h.active)
        v["film_render.halation.scatter_amount"] = .number(h.scatterAmount)
        v["film_render.halation.scatter_spatial_scale"] = .number(h.scatterSpatialScale)
        v["film_render.halation.halation_amount"] = .number(h.halationAmount)
        v["film_render.halation.halation_spatial_scale"] = .number(h.halationSpatialScale)
        v["film_render.halation.scatter_core_um[0]"] = .number(h.scatterCoreMicrons.0)
        v["film_render.halation.scatter_core_um[1]"] = .number(h.scatterCoreMicrons.1)
        v["film_render.halation.scatter_core_um[2]"] = .number(h.scatterCoreMicrons.2)
        v["film_render.halation.scatter_tail_um[0]"] = .number(h.scatterTailMicrons.0)
        v["film_render.halation.scatter_tail_um[1]"] = .number(h.scatterTailMicrons.1)
        v["film_render.halation.scatter_tail_um[2]"] = .number(h.scatterTailMicrons.2)
        v["film_render.halation.scatter_tail_weight[0]"] = .number(h.scatterTailWeight.0)
        v["film_render.halation.scatter_tail_weight[1]"] = .number(h.scatterTailWeight.1)
        v["film_render.halation.scatter_tail_weight[2]"] = .number(h.scatterTailWeight.2)
        v["film_render.halation.boost_ev"] = .number(h.boostEV)
        v["film_render.halation.boost_range"] = .number(h.boostRange)
        v["film_render.halation.protect_ev"] = .number(h.protectEV)
        v["film_render.halation.halation_strength[0]"] = .number(h.halationStrength.0)
        v["film_render.halation.halation_strength[1]"] = .number(h.halationStrength.1)
        v["film_render.halation.halation_strength[2]"] = .number(h.halationStrength.2)
        v["film_render.halation.halation_first_sigma_um[0]"] = .number(
            h.halationFirstSigmaMicrons.0)
        v["film_render.halation.halation_first_sigma_um[1]"] = .number(
            h.halationFirstSigmaMicrons.1)
        v["film_render.halation.halation_first_sigma_um[2]"] = .number(
            h.halationFirstSigmaMicrons.2)
        v["film_render.halation.halation_n_bounces"] = .number(Double(h.halationBounceCount))
        v["film_render.halation.halation_bounce_decay"] = .number(h.halationBounceDecay)
        v["film_render.halation.halation_renormalize"] = .bool(h.halationRenormalize)

        let d = film.dirCouplers
        v["film_render.dir_couplers.active"] = .bool(d.active)
        v["film_render.dir_couplers.amount"] = .number(d.amount)
        v["film_render.dir_couplers.inhibition_samelayer"] = .number(d.inhibitionSameLayer)
        v["film_render.dir_couplers.inhibition_interlayer"] = .number(d.inhibitionInterlayer)
        v["film_render.dir_couplers.gamma_samelayer_rgb[0]"] = .number(d.gammaSameLayerRGB.0)
        v["film_render.dir_couplers.gamma_samelayer_rgb[1]"] = .number(d.gammaSameLayerRGB.1)
        v["film_render.dir_couplers.gamma_samelayer_rgb[2]"] = .number(d.gammaSameLayerRGB.2)
        v["film_render.dir_couplers.gamma_interlayer_r_to_gb[0]"] = .number(
            d.gammaInterlayerRedToGreenBlue.0)
        v["film_render.dir_couplers.gamma_interlayer_r_to_gb[1]"] = .number(
            d.gammaInterlayerRedToGreenBlue.1)
        v["film_render.dir_couplers.gamma_interlayer_g_to_rb[0]"] = .number(
            d.gammaInterlayerGreenToRedBlue.0)
        v["film_render.dir_couplers.gamma_interlayer_g_to_rb[1]"] = .number(
            d.gammaInterlayerGreenToRedBlue.1)
        v["film_render.dir_couplers.gamma_interlayer_b_to_rg[0]"] = .number(
            d.gammaInterlayerBlueToRedGreen.0)
        v["film_render.dir_couplers.gamma_interlayer_b_to_rg[1]"] = .number(
            d.gammaInterlayerBlueToRedGreen.1)
        v["film_render.dir_couplers.diffusion_size_um"] = .number(d.diffusionSizeMicrons)
        v["film_render.dir_couplers.diffusion_tail_um"] = .number(d.diffusionTailMicrons)
        v["film_render.dir_couplers.diffusion_tail_weight"] = .number(d.diffusionTailWeight)
        glare("film_render.glare", film.glare)

        // print_render
        glare("print_render.glare", printRender.glare)
        let m = printRender.densityCurvesMorph
        v["print_render.density_curves_morph.active"] = .bool(m.active)
        v["print_render.density_curves_morph.gamma_factor"] = .number(m.gammaFactor)
        v["print_render.density_curves_morph.gamma_factor_fast"] = .number(m.gammaFactorFast)
        v["print_render.density_curves_morph.gamma_factor_slow"] = .number(m.gammaFactorSlow)
        v["print_render.density_curves_morph.gamma_factor_red"] = .number(m.gammaFactorRed)
        v["print_render.density_curves_morph.gamma_factor_green"] = .number(m.gammaFactorGreen)
        v["print_render.density_curves_morph.gamma_factor_blue"] = .number(m.gammaFactorBlue)
        v["print_render.density_curves_morph.developer_exhaustion"] = .number(
            m.developerExhaustion)

        // io
        v["io.input_color_space"] = .string(io.inputColourSpace)
        v["io.input_cctf_decoding"] = .bool(io.inputCCTFDecoding)
        v["io.output_color_space"] = .string(io.outputColourSpace)
        v["io.output_cctf_encoding"] = .bool(io.outputCCTFEncoding)
        v["io.input_gamut_compress.active"] = .bool(io.inputGamutCompress.active)
        v["io.input_gamut_compress.algorithm"] = .string(io.inputGamutCompress.algorithm.rawValue)
        v["io.input_gamut_compress.knee[0]"] = .number(io.inputGamutCompress.knee.threshold)
        v["io.input_gamut_compress.knee[1]"] = .number(io.inputGamutCompress.knee.limit)
        v["io.input_gamut_compress.knee[2]"] = .number(io.inputGamutCompress.knee.power)
        v["io.output_gamut_compress.algorithm"] = .string(io.outputGamutCompress.algorithm.rawValue)
        v["io.output_gamut_compress.knee[0]"] = .number(io.outputGamutCompress.knee.threshold)
        v["io.output_gamut_compress.knee[1]"] = .number(io.outputGamutCompress.knee.limit)
        v["io.output_gamut_compress.knee[2]"] = .number(io.outputGamutCompress.knee.power)
        if let lc = io.outputGamutCompress.lightnessCompression {
            v["io.output_gamut_compress.lightness_compression[0]"] = .number(lc.threshold)
            v["io.output_gamut_compress.lightness_compression[1]"] = .number(lc.limit)
            v["io.output_gamut_compress.lightness_compression[2]"] = .number(lc.power)
        }
        v["io.crop"] = .bool(io.crop)
        v["io.crop_center[0]"] = .number(io.cropCenter.x)
        v["io.crop_center[1]"] = .number(io.cropCenter.y)
        v["io.crop_size[0]"] = .number(io.cropSize.width)
        v["io.crop_size[1]"] = .number(io.cropSize.height)
        v["io.upscale_factor"] = .number(io.upscaleFactor)
        v["io.scan_film"] = .bool(io.scanFilm)

        // debug
        v["debug.deactivate_spatial_effects"] = .bool(debug.deactivateSpatialEffects)
        v["debug.deactivate_stochastic_effects"] = .bool(debug.deactivateStochasticEffects)
        v["debug.print_timings"] = .bool(debug.printTimings)
        v["debug.lut_mode"] = .bool(debug.lutMode)

        // settings
        v["settings.rgb_to_raw_method"] = .string(settings.rgbToRawMethod.rawValue)
        v["settings.apply_hanatos2025_adaptation_window"] = .bool(
            settings.applyHanatos2025AdaptationWindow)
        v["settings.apply_hanatos2025_adaptation_surface"] = .bool(
            settings.applyHanatos2025AdaptationSurface)
        v["settings.spectral_gaussian_blur"] = .number(settings.spectralGaussianBlur)
        v["settings.use_enlarger_lut"] = .bool(settings.useEnlargerLUT)
        v["settings.use_scanner_lut"] = .bool(settings.useScannerLUT)
        v["settings.lut_resolution"] = .number(Double(settings.lutResolution))
        v["settings.use_fast_stats"] = .bool(settings.useFastStats)
        v["settings.preview_max_size"] = .number(Double(settings.previewMaxSize))
        v["settings.preview_mode"] = .bool(settings.previewMode)
        v["settings.neutral_print_filters_from_database"] = .bool(
            settings.neutralPrintFiltersFromDatabase)

        // taps
        v["taps.inject"] = taps.inject.map { .string($0.rawValue) } ?? .null
        v["taps.collect"] = taps.collect.map { .string($0.rawValue) } ?? .null

        return v
    }

    @Test("every default matches the reference dataclass")
    func defaultsMatch() throws {
        let reference = try referenceDefaults()
        let swift = Self.swiftDefaults()

        let missing = Set(reference.keys).subtracting(swift.keys).sorted()
        let extra = Set(swift.keys).subtracting(reference.keys).sorted()
        #expect(missing.isEmpty, "not represented in Swift: \(missing.joined(separator: ", "))")
        #expect(extra.isEmpty, "no such reference default: \(extra.joined(separator: ", "))")

        for key in reference.keys.sorted() {
            guard let mine = swift[key] else { continue }
            #expect(mine == reference[key]!, "\(key): Swift \(mine), reference \(reference[key]!)")
        }
    }

    @Test("the reference dump covers the whole tree")
    func dumpIsComplete() throws {
        // Guards against the dump silently shrinking, which would make the test above vacuous.
        #expect(try referenceDefaults().count == 172)
    }

    // MARK: - Reading the dump

    enum Value: Equatable, CustomStringConvertible {
        case number(Double)
        case bool(Bool)
        case string(String)
        case null

        var description: String {
            switch self {
            case .number(let d): return "\(d)"
            case .bool(let b): return "\(b)"
            case .string(let s): return "\"\(s)\""
            case .null: return "null"
            }
        }
    }

    func referenceDefaults() throws -> [String: Value] {
        guard
            let url = Bundle.module.url(
                forResource: "params_defaults", withExtension: "json", subdirectory: "Goldens")
        else {
            throw Golden.GoldenError.missing("params_defaults.json")
        }
        let raw = try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
        guard let dict = raw as? [String: Any] else {
            throw Golden.GoldenError.malformed("params_defaults.json", "not an object")
        }
        return dict.mapValues { value in
            // Order matters. JSONSerialization returns NSNumber for both numbers and booleans, and
            // `as? Bool` succeeds for NSNumber(1.0), so the CFBoolean check has to come first or
            // every 0.0 and 1.0 default decodes as a boolean.
            if let object = value as AnyObject?, CFGetTypeID(object) == CFBooleanGetTypeID() {
                return .bool((value as? NSNumber)?.boolValue ?? false)
            }
            switch value {
            case let n as NSNumber: return .number(n.doubleValue)
            case let s as String: return .string(s)
            default: return .null
            }
        }
    }
}

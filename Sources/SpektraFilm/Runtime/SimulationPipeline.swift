import Foundation

/// Composes the stages into a tap topology and runs it.
///
/// Ports `runtime/pipeline.py`. The stage objects are built once and the topology declares how they
/// connect, so a caller can inject or collect at any named boundary without the stages knowing.
public final class SimulationPipeline {
    private let params: RuntimePhotoParams
    private let resizing: ResizingService
    private var enlarger: EnlargerService
    private let colourReference: ColorReferenceService
    private let filming: FilmingStage
    private let printing: PrintingStage
    private let scanning: ScanningStage
    private let topology: [Node]

    /// Per-node wall-clock times from the last run.
    public private(set) var timings: [String: TimeInterval] = [:]
    /// Total wall-clock time of the last run.
    public private(set) var elapsed: TimeInterval?

    public init(
        params rawParams: RuntimePhotoParams,
        resampler: any Resampler = UnavailableResampler()
    ) throws {
        try rawParams.validate()
        let params = Self.applyDebugOverrides(rawParams)
        self.params = params

        let spatial: any SpatialFilter =
            params.debug.deactivateSpatialEffects
            ? NoSpatialFilter() : FastSpatialFilter()

        resizing = ResizingService(
            io: params.io,
            filmFormatMillimetres: params.camera.filmFormatMillimetres,
            resampler: resampler)
        enlarger = EnlargerService(params.enlarger)
        colourReference = try ColorReferenceService(
            film: params.film, print: params.print, scanner: params.scanner, io: params.io)

        filming = try FilmingStage(
            film: params.film,
            filmRender: params.filmRender,
            camera: params.camera,
            io: params.io,
            settings: params.settings,
            resizing: resizing,
            enlarger: enlarger,
            colourReference: colourReference,
            spatial: spatial)

        // The midgray references depend on the film profile and the camera exposure, so the filming
        // stage computes them and the enlarger holds them for the printing stage.
        try filming.prepareMidgrayReferences()
        enlarger.densitySpectralMidgray = filming.densitySpectralMidgray
        enlarger.densitySpectralMidgrayCompensated = filming.densitySpectralMidgrayCompensated

        printing = try PrintingStage(
            film: params.film,
            filmRender: params.filmRender,
            print: params.print,
            printRender: params.printRender,
            enlargerParams: params.enlarger,
            settings: params.settings,
            enlarger: enlarger,
            resizing: resizing,
            colourReference: colourReference,
            spatial: spatial)

        scanning = try ScanningStage(
            film: params.film,
            filmRender: params.filmRender,
            print: params.print,
            printRender: params.printRender,
            scanner: params.scanner,
            io: params.io,
            settings: params.settings,
            colourReference: colourReference,
            spatial: spatial)

        topology = Self.buildTopology(
            io: params.io, resizing: resizing, filming: filming, printing: printing,
            scanning: scanning)
    }

    /// Runs the pipeline. Defaults to end to end, `rgbIn` to `rgbOut`.
    public func process(
        _ image: ImageBuffer, inject: Tap? = nil, collect: Tap? = nil
    ) throws -> ImageBuffer {
        let from = inject ?? params.taps.inject ?? .rgbIn
        let to = collect ?? params.taps.collect ?? .rgbOut

        timings.removeAll()
        let start = DispatchTime.now().uptimeNanoseconds
        defer { elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9 }

        return try runTopology(topology, inject: from, collect: to, image: image) {
            node, seconds in
            self.timings[node.label, default: 0] += seconds
        }
    }

    /// `debug.lut_mode` makes the pipeline a deterministic per-pixel transform, suitable for LUT
    /// sampling. Spatial effects, stochastic effects, auto-exposure and the scanner corrections all
    /// turn off regardless of their own settings.
    static func applyDebugOverrides(_ params: RuntimePhotoParams) -> RuntimePhotoParams {
        var p = params
        if p.debug.lutMode {
            p.debug.deactivateSpatialEffects = true
            p.debug.deactivateStochasticEffects = true
            p.camera.autoExposure = false
            p.scanner.whiteCorrection = false
            p.scanner.blackCorrection = false
            p.scanner.unsharpMask = (0, 0)
        }
        if p.debug.deactivateStochasticEffects {
            p.filmRender.grain.active = false
            p.filmRender.glare.active = false
            p.printRender.glare.active = false
        }
        if p.debug.deactivateSpatialEffects {
            p.camera.lensBlurMicrons = 0
            p.camera.diffusionFilter.active = false
            p.enlarger.lensBlur = 0
            p.enlarger.diffusionFilter.active = false
            p.scanner.lensBlur = 0
            p.filmRender.halation.active = false
        }
        return p
    }

    static func buildTopology(
        io: IOParams,
        resizing: ResizingService,
        filming: FilmingStage,
        printing: PrintingStage,
        scanning: ScanningStage
    ) -> [Node] {
        var nodes: [Node] = [
            Node(from: .rgbIn, to: .rgbPre, label: "preprocess") { image in
                let metered = try filming.autoExposure(image)
                return try resizing.cropAndRescale(metered)
            },
            Node(from: .rgbPre, to: .logExposureFilm, label: "filming.expose") { image in
                try filming.expose(image)
            },
            Node(from: .logExposureFilm, to: .cmyFilm, label: "filming.develop") { image in
                filming.develop(image)
            },
        ]

        if io.scanFilm {
            nodes.append(
                Node(from: .cmyFilm, to: .rgbOut, label: "scanning.scan_film") { image in
                    try scanning.scan(image)
                })
            return nodes
        }

        nodes.append(
            Node(from: .cmyFilm, to: .logExposurePrint, label: "printing.expose") { image in
                try printing.expose(image)
            })
        nodes.append(
            Node(from: .logExposurePrint, to: .cmyPrint, label: "printing.develop") { image in
                try printing.develop(image)
            })
        nodes.append(
            Node(from: .cmyPrint, to: .rgbOut, label: "scanning.scan_print") { image in
                try scanning.scan(image)
            })
        return nodes
    }

    /// The per-node timings, longest first.
    public func formattedTimings() -> String {
        var lines = timings.sorted { $0.value > $1.value }.map { label, seconds in
            String(format: "  %-22@ %7.1f ms", label as NSString, seconds * 1000)
        }
        if let elapsed {
            lines.append(String(format: "  %-22@ %7.1f ms", "total" as NSString, elapsed * 1000))
        }
        return lines.joined(separator: "\n")
    }
}

import CoreGraphics
import Foundation
import Observation
import Photos
import SpektraFilm
import SwiftUI

/// The editor's state, and the scheduler that decides which of the three render tiers to run.
///
/// The scheduling is the whole design problem. A render costs 83 ms at scrub size and 343 ms at
/// preview size, so neither can run per frame while a slider moves. Instead a moving control marks
/// the state dirty and a single task coalesces: it renders the latest parameters, and any edits that
/// arrived while it was working become one more render rather than a queue. Letting go of the control
/// asks for the larger size.
@Observable
@MainActor
final class EditorModel {

    // MARK: - Source

    /// The photo as imported, full size. Kept so export can re-render at full resolution.
    private(set) var source: CGImage?
    private(set) var sourceName: String?

    // MARK: - Parameters

    var params: RuntimePhotoParams? {
        didSet { if params != oldValue { requestRender(.settle) } }
    }

    /// The parameters as the pipeline actually runs them.
    ///
    /// `params` holds what the user edited, which is not what renders: digesting overrides the
    /// enlarger's neutral filter positions from the measured database and replaces the coupler gammas
    /// with fitted per-stock values. Readouts have to show these, or the dial numbers on screen are
    /// not the numbers in the print.
    var effective: RuntimePhotoParams? {
        guard let params else { return nil }
        return try? ParamsBuilder.digest(params)
    }

    /// Which pipeline boundary is on screen. The reason the app can show the virtual negative.
    var tap: Tap = .rgbOut {
        didSet { if tap != oldValue { requestRender(.settle) } }
    }

    // MARK: - Output

    private(set) var rendered: CGImage?
    private(set) var renderedQuality: RenderService.Quality?
    private(set) var lastRenderMilliseconds: Double?
    private(set) var stageTimings: [(String, Double)] = []
    private(set) var isRendering = false
    private(set) var failure: String?

    /// Advances on every edit. A render carrying an older generation is dropped.
    private var generation: UInt64 = 0
    private var pending: RenderService.Quality?
    private var renderTask: Task<Void, Never>?
    private let service = RenderService()

    // MARK: - Catalogue

    let filmStocks: [Profile]
    let printMedia: [Profile]

    init() {
        filmStocks = (try? ProfileLibrary.filmStocks) ?? []
        printMedia = (try? ProfileLibrary.printMedia) ?? []
        params = try? RuntimePhotoParams.make(
            film: "kodak_portra_400", print: "kodak_portra_endura")
        params?.io.inputColourSpace = ImageBridge.workingColourSpaceName
        // CoreGraphics already decoded to linear on import, so the engine must not decode again.
        params?.io.inputCCTFDecoding = false
    }

    // MARK: - Import

    func load(_ image: CGImage, named name: String?) {
        source = image
        sourceName = name
        failure = nil
        requestRender(.settle)
    }

    // MARK: - Editing

    /// Call while a control is moving. Coalesces into scrub-quality renders.
    func scrub(_ mutate: (inout RuntimePhotoParams) -> Void) {
        guard var current = params else { return }
        mutate(&current)
        params = current
        requestRender(.scrub)
    }

    /// Call when a control settles, or after a discrete choice.
    func settle() {
        requestRender(.settle)
    }

    /// Render at preview size with grain and the spatial effects on.
    func proof() {
        requestRender(.proof)
    }

    // MARK: - Scheduling

    private func requestRender(_ quality: RenderService.Quality) {
        guard source != nil, params != nil else { return }
        generation &+= 1
        // Keep the most demanding tier asked for since the last render started, so a scrub arriving
        // after a settle request does not downgrade it.
        pending = max(pending ?? quality, quality)
        guard renderTask == nil else { return }
        renderTask = Task { await self.drain() }
    }

    private func drain() async {
        defer { renderTask = nil }
        while let quality = pending {
            pending = nil
            guard let source, let params else { return }
            let mine = generation
            isRendering = true
            do {
                let result = try await service.render(
                    source: source,
                    params: params,
                    quality: quality,
                    tap: tap,
                    generation: mine,
                    currentGeneration: { [weak self] in await self?.generation ?? mine })
                if let result, result.generation == generation {
                    rendered = result.image
                    renderedQuality = result.quality
                    lastRenderMilliseconds = result.milliseconds
                    stageTimings = result.stages
                    failure = nil
                }
            } catch {
                failure = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
                print("[spektrafilm] render failed: \(failure ?? "")")
            }
            isRendering = false
        }
    }

    // MARK: - Export

    enum ExportState: Equatable {
        case idle
        case rendering
        case saving
        case saved(pixels: String, capped: Bool)
        case failed(String)
    }

    private(set) var exportState: ExportState = .idle

    /// Renders as large as this device allows and writes to the photo library.
    ///
    /// Roughly 2.3 s per megapixel. The size is capped by ``RenderBudget``: peak footprint is about
    /// 230 MB per megapixel, so a 12 MP frame would need 2.8 GB and be terminated. When the cap bites
    /// the result says so, because silently exporting something smaller than the source is the kind
    /// of thing a user discovers much later.
    func export() async {
        guard let source, let params else { return }
        exportState = .rendering
        do {
            let result = try await service.render(
                source: source,
                params: params,
                quality: .full,
                tap: .rgbOut,
                generation: generation,
                currentGeneration: { [weak self] in await self?.generation ?? 0 })
            guard let result else {
                exportState = .idle
                return
            }
            exportState = .saving
            try await PhotoLibrary.save(result.image)
            exportState = .saved(
                pixels: "\(result.pixelSize.width) x \(result.pixelSize.height)",
                capped: result.wasDownscaled)
        } catch {
            exportState = .failed(
                (error as? LocalizedError)?.errorDescription ?? String(describing: error))
        }
    }

    func dismissExport() { exportState = .idle }
}

/// Writes a rendered image to the photo library.
enum PhotoLibrary {
    enum SaveError: LocalizedError {
        case denied
        var errorDescription: String? {
            "Spektrafilm needs permission to add photos. Enable it in Settings."
        }
    }

    static func save(_ image: CGImage) async throws {
        let status = await PHPhotoLibrary.requestAuthorization(for: .addOnly)
        guard status == .authorized || status == .limited else { throw SaveError.denied }
        try await PHPhotoLibrary.shared().performChanges {
            PHAssetChangeRequest.creationRequestForAsset(from: UIImage(cgImage: image))
        }
    }
}

import CoreGraphics
import Foundation
import Observation
import Photos
import SpektraFilm
import SwiftUI

/// The editor's state, and the scheduler that picks which render tier to run.
///
/// A render costs 16 ms at scrub or settle size on an M4 Pro's GPU, and more on a phone, so neither
/// can be relied on to run every frame while a control moves. One task renders the latest parameters. Edits that arrive while it works collapse
/// into a single follow-up render. Releasing the control asks for the larger size.
///
/// Every finished render is shown, even if the parameters have moved on since it started. Discarding
/// superseded results would discard all of them during a continuous drag.
@Observable
@MainActor
final class EditorModel {

    // MARK: - Source

    /// The photo as imported, full size. Kept so export can re-render at full resolution.
    private(set) var source: CGImage?
    private(set) var sourceName: String?

    // MARK: - Parameters

    /// Written only through ``scrub(_:)``, which also picks the render tier.
    private(set) var params: RuntimePhotoParams?

    /// The parameters as the pipeline actually runs them.
    ///
    /// Digesting replaces the enlarger's neutral filter positions with measured values for the film,
    /// paper and lamp, and the coupler gammas with fitted per-stock values. Readouts show these so the
    /// numbers on screen are the numbers in the print.
    var effective: RuntimePhotoParams? {
        guard let params else { return nil }
        return try? ParamsBuilder.digest(params)
    }

    /// Which pipeline boundary to show: the print, or an intermediate such as the negative.
    var tap: Tap = .rgbOut {
        didSet { if tap != oldValue { requestRender(.settle) } }
    }

    // MARK: - Output

    private(set) var rendered: CGImage?
    private(set) var renderedQuality: RenderService.Quality?
    /// The tap `rendered` shows, which trails ``tap`` until the next render lands.
    private(set) var renderedTap: Tap = .rgbOut
    private(set) var lastRenderMilliseconds: Double?
    private(set) var stageTimings: [(String, Double)] = []
    private(set) var isRendering = false
    private(set) var failure: String?

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
            isRendering = true
            do {
                let result = try await service.render(
                    source: source, params: params, quality: quality, tap: tap)
                // A render of the previous photo, finishing after an import, would flash it back.
                if source === self.source {
                    rendered = result.image
                    renderedQuality = result.quality
                    renderedTap = result.tap
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
    /// ``RenderBudget`` caps the size to fit in memory, and the saved state
    /// reports when it did, so the user knows the export is smaller than the source.
    func export() async {
        guard let source, let params else { return }
        exportState = .rendering
        do {
            let result = try await service.render(
                source: source, params: params, quality: .full, tap: .rgbOut)
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

import SpektraFilm
import SwiftUI

/// The photo, fitted to the space above the editing strip.
///
/// Press and hold to see the original. When the view shows a stage other than the print, a capsule
/// names it.
struct Canvas: View {
    let model: EditorModel
    @State private var showingOriginal = false

    var body: some View {
        ZStack {
            if let failure = model.failure {
                ContentUnavailableView(
                    "Couldn't Render", systemImage: "exclamationmark.triangle",
                    description: Text(failure))
            } else if let image = showingOriginal ? model.source : model.rendered {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .scaledToFit()
                    .padding(.horizontal, 8)
                    .accessibilityLabel(showingOriginal ? "Original photo" : "Rendered print")
            } else {
                ProgressView()
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .onLongPressGesture(minimumDuration: 0.15, perform: {}) { pressing in
            showingOriginal = pressing
        }
        .overlay(alignment: .top) { badge }
        .overlay(alignment: .topTrailing) {
            if model.isRendering {
                ProgressView()
                    .controlSize(.small)
                    .padding(8)
                    .background(.ultraThinMaterial, in: Circle())
                    .padding(12)
            }
        }
    }

    @ViewBuilder
    private var badge: some View {
        let label =
            showingOriginal
            ? "Original" : model.renderedTap == .rgbOut ? nil : Stage.name(model.renderedTap)
        if let label {
            Text(label)
                .font(.footnote.weight(.semibold))
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(.ultraThinMaterial, in: Capsule())
                .padding(.top, 24)
        }
    }
}

/// The pipeline stages the view can show instead of the print.
enum Stage {
    static func name(_ tap: Tap) -> String {
        switch tap {
        case .rgbIn: return "Scene"
        case .rgbPre: return "Metered Scene"
        case .logExposureFilm: return "Film Exposure"
        case .cmyFilm: return "Negative"
        case .logExposurePrint: return "Paper Exposure"
        case .cmyPrint: return "Print Density"
        case .rgbOut: return "Print"
        }
    }

    /// The stages in pipeline order. Scanning the film skips the print.
    static func all(scanFilm: Bool) -> [Tap] {
        scanFilm
            ? [.rgbOut, .cmyFilm, .logExposureFilm]
            : [.rgbOut, .cmyPrint, .logExposurePrint, .cmyFilm, .logExposureFilm]
    }
}

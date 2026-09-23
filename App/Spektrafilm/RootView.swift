import PhotosUI
import SpektraFilm
import SwiftUI

/// One screen, laid out like the Photos editor: the photo on black, the editing strip under it.
struct RootView: View {
    @State private var model = EditorModel()
    @State private var pickedItem: PhotosPickerItem?
    @State private var showingDetails = false

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                if model.source == nil {
                    emptyState
                } else {
                    VStack(spacing: 0) {
                        Canvas(model: model)
                        EditingStrip(model: model)
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { toolbar }
            .toolbarBackground(.black, for: .navigationBar)
        }
        .tint(.yellow)
        .preferredColorScheme(.dark)
        .task {
            if SampleScene.isRequested, let image = SampleScene.make(width: 2400, height: 3200) {
                if let tap = SampleScene.requestedTap { model.tap = tap }
                model.load(image, named: "sample")
                // `-export` runs the export without a tap on the screen, for scripted testing of the
                // full tier and the memory cap.
                if ProcessInfo.processInfo.arguments.contains("-export") {
                    try? await Task.sleep(for: .seconds(3))
                    await model.export()
                }
            }
        }
        .task(id: pickedItem) { await loadPicked() }
        .sheet(isPresented: $showingDetails) { DetailsSheet(model: model) }
        .overlay { if model.exportState == .rendering || model.exportState == .saving { exporting } }
        .alert(
            exportAlertTitle, isPresented: exportAlertShown,
            actions: { Button("OK") { model.dismissExport() } },
            message: { Text(exportAlertMessage) })
    }

    // MARK: - Toolbar

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        if let params = model.params, model.source != nil {
            ToolbarItem(placement: .topBarLeading) {
                PhotosPicker(selection: $pickedItem, matching: .images, photoLibrary: .shared()) {
                    Label("Choose Photo", systemImage: "photo.on.rectangle")
                }
                .tint(.primary)
            }
            ToolbarItem(placement: .principal) {
                VStack(spacing: 0) {
                    Text(params.film.info.displayName)
                        .font(.subheadline.weight(.semibold))
                    Text(params.io.scanFilm ? "Scanned film" : params.print.info.displayName)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .lineLimit(1)
            }
            ToolbarItemGroup(placement: .topBarTrailing) {
                Menu {
                    Picker("View", selection: $model.tap) {
                        ForEach(Stage.all(scanFilm: params.io.scanFilm), id: \.self) { tap in
                            Text(Stage.name(tap)).tag(tap)
                        }
                    }
                } label: {
                    Label("View", systemImage: model.tap == .rgbOut ? "eye" : "eye.fill")
                }
                // Yellow only while showing a stage other than the print, so that state is seen.
                .tint(model.tap == .rgbOut ? .primary : .yellow)
                Button {
                    showingDetails = true
                } label: {
                    Label("Details", systemImage: "info.circle")
                }
                .tint(.primary)
                Button {
                    let editor = model
                    Task { await editor.export() }
                } label: {
                    Label("Export", systemImage: "square.and.arrow.up")
                }
                .tint(.primary)
                .disabled(model.exportState != .idle)
            }
        }
    }

    // MARK: - Empty state

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Photo", systemImage: "photo")
        } description: {
            Text("Choose a photo to expose onto film and print.")
        } actions: {
            PhotosPicker(selection: $pickedItem, matching: .images, photoLibrary: .shared()) {
                Text("Choose Photo")
            }
            .buttonStyle(.borderedProminent)
            .foregroundStyle(.black)
        }
    }

    // MARK: - Export

    private var exporting: some View {
        ZStack {
            Color.black.opacity(0.4).ignoresSafeArea()
            VStack(spacing: 12) {
                ProgressView()
                Text(model.exportState == .saving ? "Saving" : "Rendering at Full Size")
                    .font(.subheadline)
            }
            .padding(24)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
    }

    private var exportAlertShown: Binding<Bool> {
        Binding(
            get: {
                switch model.exportState {
                case .saved, .failed: return true
                default: return false
                }
            },
            set: { if !$0 { model.dismissExport() } })
    }

    private var exportAlertTitle: String {
        if case .failed = model.exportState { return "Couldn't Export" }
        return "Saved to Photos"
    }

    private var exportAlertMessage: String {
        switch model.exportState {
        case .saved(let pixels, let capped):
            return capped
                ? "\(pixels). Rendered smaller than the original to fit this device's memory."
                : pixels
        case .failed(let message):
            return message
        default:
            return ""
        }
    }

    private func loadPicked() async {
        guard let pickedItem else { return }
        guard let data = try? await pickedItem.loadTransferable(type: Data.self),
            let source = CGImageSourceCreateWithData(data as CFData, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { return }
        model.load(image, named: nil)
    }
}

#Preview {
    RootView()
}

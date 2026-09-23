import PhotosUI
import SpektraFilm
import SwiftUI

/// The darkroom.
///
/// One screen: the print above, the controls below, and the pipeline stages between them.
struct RootView: View {
    @State private var model = EditorModel()
    @State private var pickedItem: PhotosPickerItem?
    @State private var showDiagnostics = false

    var body: some View {
        ZStack {
            Safelight.ink.ignoresSafeArea()

            VStack(spacing: 0) {
                header

                PrintView(
                    image: model.rendered,
                    isRendering: model.isRendering,
                    tap: model.renderedTap,
                    quality: model.renderedQuality,
                    failure: model.failure
                )
                .frame(maxHeight: .infinity)

                caption

                if model.source != nil {
                    StageStrip(
                        tap: Binding(get: { model.tap }, set: { model.tap = $0 }),
                        scanFilm: model.params?.io.scanFilm ?? false)
                    Divider().overlay(Safelight.rule)
                    ControlDrawer(model: model)
                        .frame(height: 300)
                }
            }
        }
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
        .overlay { exportOverlay }
        .sheet(isPresented: $showDiagnostics) { DiagnosticsSheet(model: model) }
    }

    // MARK: - Header

    private var header: some View {
        // Hoisted: PhotosPicker's label builder is Sendable, so it cannot read main-actor state.
        let hasSource = model.source != nil
        return HStack(spacing: 10) {
            // The safelight itself.
            Circle()
                .fill(Safelight.amber)
                .frame(width: 6, height: 6)
                .shadow(color: Safelight.amber.opacity(0.8), radius: 6)

            Text("SPEKTRAFILM")
                .font(Safelight.label(11))
                .tracking(3.4)
                .foregroundStyle(Safelight.paper.opacity(0.9))

            Spacer(minLength: 0)

            if hasSource {
                Button {
                    showDiagnostics = true
                } label: {
                    Image(systemName: "waveform")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(Safelight.amberDim)
                }
                Button {
                    let editor = model
                    Task { await editor.export() }
                } label: {
                    Text("export").safelightLabel(true)
                }
                .disabled(model.exportState != .idle)
            }

            PhotosPicker(selection: $pickedItem, matching: .images, photoLibrary: .shared()) {
                Image(systemName: hasSource ? "arrow.triangle.2.circlepath" : "plus")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(Safelight.amber)
            }
        }
        .padding(.horizontal, Safelight.gutter)
        .padding(.vertical, 12)
        .background(Safelight.ink)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Safelight.rule).frame(height: Safelight.hairline)
        }
    }

    // MARK: - Caption

    @ViewBuilder
    private var caption: some View {
        if let params = model.effective, model.source != nil {
            VStack(spacing: 5) {
                Text("\(params.film.info.displayName)  \u{00B7}  \(params.print.info.displayName)")
                    .font(Safelight.display(15))
                    .foregroundStyle(Safelight.paper.opacity(0.88))
                    .multilineTextAlignment(.center)

                HStack(spacing: 14) {
                    readout("EV", String(format: "%+.2f", params.camera.exposureCompensationEV))
                    readout(
                        "Y",
                        String(format: "%.0f", params.enlarger.yFilterNeutral + params.enlarger.yFilterShift))
                    readout(
                        "M",
                        String(format: "%.0f", params.enlarger.mFilterNeutral + params.enlarger.mFilterShift))
                    if let ms = model.lastRenderMilliseconds {
                        readout("MS", String(format: "%.0f", ms))
                    }
                }
            }
            .padding(.top, 4)
            .padding(.bottom, 8)
        }
    }

    private func readout(_ key: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(key)
                .font(Safelight.label(9))
                .foregroundStyle(Safelight.amberDim)
            Text(value)
                .font(Safelight.readout(11))
                .foregroundStyle(Safelight.paper.opacity(0.8))
                .monospacedDigit()
        }
    }

    // MARK: - Export

    @ViewBuilder
    private var exportOverlay: some View {
        if model.exportState != .idle {
            ZStack {
                Safelight.ink.opacity(0.94).ignoresSafeArea()
                VStack(spacing: 16) {
                    switch model.exportState {
                    case .rendering:
                        DevelopingIndicator()
                        Text("Developing at full size")
                            .font(Safelight.display(16))
                            .foregroundStyle(Safelight.paper)
                        // The engine reports no progress, so there is no progress bar.
                        Text("a few seconds at full size")
                            .safelightLabel()
                    case .saving:
                        Text("Saving").font(Safelight.display(16)).foregroundStyle(Safelight.paper)
                    case .saved(let pixels, let capped):
                        Image(systemName: "checkmark")
                            .font(.system(size: 22, weight: .light))
                            .foregroundStyle(Safelight.amber)
                        Text("Saved to your library")
                            .font(Safelight.display(16))
                            .foregroundStyle(Safelight.paper)
                        Text(pixels).safelightLabel()
                        if capped {
                            Text("limited by this device's memory")
                                .font(Safelight.readout(10))
                                .foregroundStyle(Safelight.amberDim)
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 44)
                        }
                        Button {
                            model.dismissExport()
                        } label: {
                            Text("done").safelightLabel(true)
                        }
                    case .failed(let message):
                        Text(message)
                            .font(Safelight.readout(12))
                            .foregroundStyle(Safelight.paper.opacity(0.85))
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 40)
                        Button {
                            model.dismissExport()
                        } label: {
                            Text("dismiss").safelightLabel(true)
                        }
                    case .idle:
                        EmptyView()
                    }
                }
            }
            .transition(.opacity)
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

/// A slow amber pulse, the pace of a print coming up.
private struct DevelopingIndicator: View {
    @State private var phase = false

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Safelight.amber)
                    .frame(width: 5, height: 5)
                    .opacity(phase ? 1 : 0.22)
                    .animation(
                        .easeInOut(duration: 0.75)
                            .repeatForever(autoreverses: true)
                            .delay(Double(index) * 0.18),
                        value: phase)
            }
        }
        .onAppear { phase = true }
    }
}

/// The last render's time, split by stage.
private struct DiagnosticsSheet: View {
    let model: EditorModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Render").font(Safelight.display(20)).foregroundStyle(Safelight.paper)

            if let ms = model.lastRenderMilliseconds {
                HStack {
                    Text("total").safelightLabel()
                    Spacer()
                    Text(String(format: "%.0f ms", ms))
                        .font(Safelight.readout(12))
                        .foregroundStyle(Safelight.paper)
                }
            }

            ForEach(model.stageTimings, id: \.0) { stage, ms in
                HStack {
                    Text(stage.replacingOccurrences(of: ".", with: " \u{00B7} "))
                        .font(Safelight.readout(11))
                        .foregroundStyle(Safelight.paper.opacity(0.75))
                    Spacer()
                    Text(String(format: "%.0f ms", ms))
                        .font(Safelight.readout(11))
                        .foregroundStyle(Safelight.amber)
                        .monospacedDigit()
                }
            }

            Spacer()

            VStack(alignment: .leading, spacing: 4) {
                Text("film modeling powered by spektrafilm")
                // CC BY-SA 4.0 requires crediting the author and linking the source of the profiles.
                Text("film and paper profiles by Andrea Volpato, CC BY-SA 4.0")
                Link(
                    "github.com/andreavolpato/spektrafilm",
                    destination: URL(string: "https://github.com/andreavolpato/spektrafilm")!)
            }
            .font(Safelight.readout(9))
            .foregroundStyle(Safelight.amberDim)
            .tint(Safelight.amber)
        }
        .padding(Safelight.gutter)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Safelight.ink)
        .presentationDetents([.medium])
    }
}

#Preview {
    RootView()
}

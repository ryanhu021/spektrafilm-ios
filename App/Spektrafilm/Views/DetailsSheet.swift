import SpektraFilm
import SwiftUI

/// The settings that are set once rather than tuned, the last render's timings, and the credits.
struct DetailsSheet: View {
    let model: EditorModel
    @Environment(\.dismiss) private var dismiss

    /// Film formats by long edge, the length the engine spreads over the frame.
    private static let formats: [(name: String, millimetres: Double)] = [
        ("Half Frame", 24), ("35 mm", 35), ("645", 56), ("6×7", 70), ("6×9", 84), ("4×5", 127),
    ]

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Auto Exposure", isOn: binding(\.camera.autoExposure))
                    Picker("Film Format", selection: binding(\.camera.filmFormatMillimetres)) {
                        ForEach(Self.formats, id: \.millimetres) { format in
                            Text(format.name).tag(format.millimetres)
                        }
                    }
                } footer: {
                    Text("The format sets the grain, halation and diffusion scale.")
                }

                Section("Output") {
                    Picker("Colour Space", selection: binding(\.io.outputColourSpace)) {
                        ForEach(ColourSpace.all.map(\.name), id: \.self) { Text($0).tag($0) }
                    }
                }

                if let total = model.lastRenderMilliseconds {
                    Section("Last Render") {
                        LabeledContent("Total", value: milliseconds(total))
                        ForEach(model.stageTimings, id: \.0) { stage, ms in
                            LabeledContent(label(stage), value: milliseconds(ms))
                        }
                    }
                }

                Section {
                    Text("Film modeling powered by spektrafilm.")
                    // CC BY-SA 4.0 requires crediting the author and linking the source of the
                    // profiles.
                    Text("Film and paper profiles by Andrea Volpato, CC BY-SA 4.0.")
                    Link(
                        "github.com/andreavolpato/spektrafilm",
                        destination: URL(string: "https://github.com/andreavolpato/spektrafilm")!)
                } header: {
                    Text("About")
                }
                .font(.footnote)
            }
            .navigationTitle("Details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .presentationDetents([.medium, .large])
    }

    /// A parameter edited as a discrete choice: one render at the settled size.
    private func binding<Value>(
        _ keyPath: WritableKeyPath<RuntimePhotoParams, Value>
    )
        -> Binding<Value>
    where Value: Equatable {
        Binding(
            get: { model.params![keyPath: keyPath] },
            set: { new in
                model.scrub { $0[keyPath: keyPath] = new }
                model.settle()
            })
    }

    private func milliseconds(_ ms: Double) -> String { String(format: "%.0f ms", ms) }

    /// `filming.expose` as "Filming · Expose".
    private func label(_ stage: String) -> String {
        stage.split(separator: ".")
            .map { $0.replacingOccurrences(of: "_", with: " ").capitalized }
            .joined(separator: " · ")
    }
}

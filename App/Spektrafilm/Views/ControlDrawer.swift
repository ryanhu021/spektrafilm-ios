import SpektraFilm
import SwiftUI

/// The controls, grouped in pipeline order: camera, film, enlarger, paper.
///
/// Exposure is set at the camera, colour balance at the enlarger, and contrast by the paper.
struct ControlDrawer: View {
    @Bindable var model: EditorModel

    enum Bench: String, CaseIterable, Identifiable {
        case camera
        case film
        case enlarger
        case paper

        var id: String { rawValue }
        var title: String {
            switch self {
            case .camera: return "camera"
            case .film: return "film"
            case .enlarger: return "enlarger"
            case .paper: return "paper"
            }
        }
    }

    /// `-bench <name>` opens on another bench, for scripted screenshots.
    @State private var bench: Bench =
        SampleScene.argument(after: "-bench").flatMap(Bench.init(rawValue:)) ?? .film

    var body: some View {
        VStack(spacing: 0) {
            benchPicker
            Divider().overlay(Safelight.rule)
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    switch bench {
                    case .camera: cameraBench
                    case .film: filmBench
                    case .enlarger: enlargerBench
                    case .paper: paperBench
                    }
                }
                .padding(Safelight.gutter)
            }
            .scrollIndicators(.hidden)
        }
        .background(Safelight.ink)
        .safelightGrain(opacity: 0.025)
    }

    private var benchPicker: some View {
        HStack(spacing: 0) {
            ForEach(Bench.allCases) { item in
                Button {
                    bench = item
                } label: {
                    VStack(spacing: 6) {
                        Text(item.title)
                            .safelightLabel(bench == item)
                        Rectangle()
                            .fill(bench == item ? Safelight.amber : .clear)
                            .frame(height: 1.5)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 12)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .animation(.easeOut(duration: 0.15), value: bench)
    }

    // MARK: - Camera

    @ViewBuilder
    private var cameraBench: some View {
        if let params = model.params {
            Knob(
                label: "exposure",
                value: scrubBinding(\.camera.exposureCompensationEV),
                range: -4...4,
                unit: "EV",
                format: "%+.2f",
                onEnded: { model.settle() })

            Toggle(isOn: autoExposureBinding) {
                Text("auto exposure").safelightLabel(params.camera.autoExposure)
            }
            .tint(Safelight.amberDim)

            Knob(
                label: "film format",
                value: scrubBinding(\.camera.filmFormatMillimetres),
                range: 8...120,
                unit: "MM",
                format: "%.0f",
                help: "Sets the pixel pitch, so it scales grain, halation and coupler diffusion.",
                onEnded: { model.settle() })
        }
    }

    private var autoExposureBinding: Binding<Bool> {
        Binding(
            get: { model.params?.camera.autoExposure ?? false },
            set: { new in
                model.scrub { $0.camera.autoExposure = new }
                model.settle()
            })
    }

    // MARK: - Film

    @ViewBuilder
    private var filmBench: some View {
        StockList(
            profiles: model.filmStocks,
            selection: model.params?.film.info.stock,
            onSelect: { stock in
                guard let profile = try? ProfileLibrary.load(stock) else { return }
                model.scrub { $0.film = profile }
                model.settle()
            })

        if model.params != nil {
            SectionLabel("development")
            Knob(
                label: "curve gamma",
                value: scrubBinding(\.filmRender.densityCurveGamma),
                range: 0.5...2.0,
                unit: "\u{0393}",
                format: "%.2f",
                onEnded: { model.settle() })

            Knob(
                label: "coupler amount",
                value: scrubBinding(\.filmRender.dirCouplers.amount),
                range: 0...2,
                unit: "\u{00D7}",
                format: "%.2f",
                help: "Inhibitor released during development. Raises saturation and local contrast.",
                onEnded: { model.settle() })

            Knob(
                label: "halation",
                value: scrubBinding(\.filmRender.halation.halationAmount),
                range: 0...4,
                unit: "\u{00D7}",
                format: "%.2f",
                help: "Light reflected off the film base back into the emulsion.",
                onEnded: { model.settle() })
        }
    }

    // MARK: - Enlarger

    @ViewBuilder
    private var enlargerBench: some View {
        if let params = model.effective {
            SectionLabel("dichroic head")
            Text("Neutral is measured for this paper, lamp and film. Shifts are from there.")
                .font(Safelight.readout(10))
                .foregroundStyle(Safelight.amberDim)
                .padding(.bottom, 2)

            DichroicDial(
                label: "yellow", tint: Safelight.yellow,
                shift: dialBinding(\.enlarger.yFilterShift),
                range: -40...40, neutral: params.enlarger.yFilterNeutral,
                onEnded: { model.settle() })

            DichroicDial(
                label: "magenta", tint: Safelight.magenta,
                shift: dialBinding(\.enlarger.mFilterShift),
                range: -40...40, neutral: params.enlarger.mFilterNeutral,
                onEnded: { model.settle() })

            SectionLabel("exposure")
            Knob(
                label: "print exposure",
                value: scrubBinding(\.enlarger.printExposure),
                range: 0.25...4,
                unit: "\u{00D7}",
                format: "%.2f",
                onEnded: { model.settle() })

            Knob(
                label: "pre-flash",
                value: scrubBinding(\.enlarger.preflashExposure),
                range: 0...0.5,
                unit: "\u{00D7}",
                format: "%.3f",
                help: "Fogs the paper slightly before exposure, which holds highlights.",
                onEnded: { model.settle() })
        }
    }

    /// Writes through `scrub`, so a dragged control renders at 320 px. Its `onEnded` asks for 640.
    private func scrubBinding(
        _ keyPath: WritableKeyPath<RuntimePhotoParams, Double>
    ) -> Binding<Double> {
        Binding(
            get: { model.params?[keyPath: keyPath] ?? 0 },
            set: { new in model.scrub { $0[keyPath: keyPath] = new } })
    }

    /// Writes one print-curve morph control through `scrub`. The morph turns on with the first edit.
    /// At its default values it reproduces the unmorphed curves exactly, so turning it on changes
    /// nothing until a value moves.
    private func morphBinding(
        _ keyPath: WritableKeyPath<PrintCurvesMorphParams, Double>
    ) -> Binding<Double> {
        Binding(
            get: { model.params?.printRender.densityCurvesMorph[keyPath: keyPath] ?? 0 },
            set: { new in
                model.scrub {
                    $0.printRender.densityCurvesMorph.active = true
                    $0.printRender.densityCurvesMorph[keyPath: keyPath] = new
                }
            })
    }

    /// The same, for a dial.
    private func dialBinding(
        _ keyPath: WritableKeyPath<RuntimePhotoParams, Double>
    ) -> Binding<Double> {
        Binding(
            get: { model.params?[keyPath: keyPath] ?? 0 },
            set: { new in model.scrub { $0[keyPath: keyPath] = new } })
    }

    // MARK: - Paper

    @ViewBuilder
    private var paperBench: some View {
        StockList(
            profiles: model.printMedia,
            selection: model.params?.print.info.stock,
            onSelect: { stock in
                guard let profile = try? ProfileLibrary.load(stock) else { return }
                model.scrub { $0.print = profile }
                model.settle()
            })

        if model.params != nil {
            SectionLabel("development")
            Knob(
                label: "paper contrast",
                value: morphBinding(\.gammaFactor),
                range: 0.6...1.6,
                unit: "\u{03B3}",
                format: "%.2f",
                help: "Steepens or flattens the paper's curves, like a harder or softer grade.",
                onEnded: { model.settle() })

            Knob(
                label: "developer exhaustion",
                value: morphBinding(\.developerExhaustion),
                range: 0...1,
                unit: "",
                format: "%.2f",
                help: "A depleted developer. Deep shadows reach full black later; midgray holds.",
                onEnded: { model.settle() })

            SectionLabel("scan")
            Knob(
                label: "sharpening",
                value: scrubBinding(\.scanner.unsharpMask.amount),
                range: 0...2,
                unit: "\u{00D7}",
                format: "%.2f",
                onEnded: { model.settle() })

            Knob(
                label: "glare",
                value: scrubBinding(\.printRender.glare.percent),
                range: 0...0.15,
                unit: "%",
                format: "%.3f",
                help: "Stray light in the scanner, which lifts the deepest shadows.",
                onEnded: { model.settle() })

            SectionLabel("output")
            Picker("", selection: outputSpaceBinding) {
                ForEach(ColourSpace.all.map(\.name), id: \.self) { name in
                    Text(name).tag(name)
                }
            }
            .pickerStyle(.menu)
            .tint(Safelight.amber)
        }
    }

    private var outputSpaceBinding: Binding<String> {
        Binding(
            get: { model.params?.io.outputColourSpace ?? "sRGB" },
            set: { new in
                model.scrub { $0.io.outputColourSpace = new }
                model.settle()
            })
    }
}

// MARK: - Pieces

private struct SectionLabel: View {
    let text: String
    init(_ text: String) { self.text = text }

    var body: some View {
        HStack(spacing: 8) {
            Text(text).safelightLabel()
            Rectangle()
                .fill(Safelight.rule)
                .frame(height: Safelight.hairline)
        }
        .padding(.top, 6)
    }
}

/// A labelled slider with a monospaced readout, and optional one-line explanation.
private struct Knob: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let unit: String
    let format: String
    var help: String?
    let onEnded: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(label).safelightLabel()
                Spacer(minLength: 0)
                Text("\(String(format: format, value)) \(unit)")
                    .font(Safelight.readout(11))
                    .foregroundStyle(Safelight.paper.opacity(0.85))
                    .monospacedDigit()
            }
            Slider(
                value: $value,
                in: range,
                onEditingChanged: { editing in if !editing { onEnded() } }
            )
            .tint(Safelight.amber)
            if let help {
                Text(help)
                    .font(Safelight.readout(10))
                    .foregroundStyle(Safelight.amberDim)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// The stock catalogue.
private struct StockList: View {
    let profiles: [Profile]
    let selection: String?
    let onSelect: (String) -> Void

    var body: some View {
        VStack(spacing: 0) {
            ForEach(profiles, id: \.info.stock) { profile in
                let active = profile.info.stock == selection
                Button {
                    onSelect(profile.info.stock)
                } label: {
                    HStack(spacing: 12) {
                        Rectangle()
                            .fill(active ? Safelight.amber : .clear)
                            .frame(width: 2)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(profile.info.displayName)
                                .font(Safelight.display(15))
                                .foregroundStyle(active ? Safelight.paper : Safelight.paper.opacity(0.6))
                            Text(subtitle(profile))
                                .font(Safelight.readout(9))
                                .foregroundStyle(Safelight.amberDim)
                        }
                        Spacer(minLength: 0)
                    }
                    .padding(.vertical, 9)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                Divider().overlay(Safelight.rule.opacity(0.5))
            }
        }
    }

    private func subtitle(_ profile: Profile) -> String {
        var parts = [profile.isPositive ? "positive" : "negative"]
        if profile.info.use == .cine { parts.append("cine") }
        if profile.isPaper { parts.append("paper") }
        parts.append(profile.info.referenceIlluminant)
        return parts.joined(separator: " \u{00B7} ")
    }
}

import SpektraFilm
import SwiftUI

/// The strip under the photo: the chosen tool's controls, and the row of tools.
struct EditingStrip: View {
    let model: EditorModel
    @State private var tool: Tool = SampleScene.argument(after: "-tool").flatMap(Tool.init) ?? .film
    @State private var adjustment = Adjustment.all[0].id

    enum Tool: String, CaseIterable, Identifiable {
        case film
        case paper
        case adjust

        var id: String { rawValue }

        var title: String {
            switch self {
            case .film: return "Film"
            case .paper: return "Paper"
            case .adjust: return "Adjust"
            }
        }

        var symbol: String {
            switch self {
            case .film: return "film"
            case .paper: return "photo.artframe"
            case .adjust: return "dial.medium"
            }
        }
    }

    var body: some View {
        VStack(spacing: 14) {
            Group {
                switch tool {
                case .film:
                    StockPicker(
                        profiles: model.filmStocks, selection: model.params?.film.info.stock
                    ) { profile in
                        model.scrub { $0.film = profile }
                        model.settle()
                    }
                case .paper:
                    StockPicker(
                        profiles: model.printMedia, selection: model.params?.print.info.stock
                    ) { profile in
                        model.scrub { $0.print = profile }
                        model.settle()
                    }
                case .adjust:
                    AdjustTool(model: model, selection: $adjustment)
                }
            }
            .frame(height: 104)

            HStack {
                ForEach(Tool.allCases) { item in
                    Button {
                        tool = item
                    } label: {
                        VStack(spacing: 4) {
                            Image(systemName: item.symbol)
                                .font(.title3)
                            Text(item.title)
                                .font(.caption2)
                        }
                        .frame(maxWidth: .infinity)
                        .foregroundStyle(tool == item ? Color.accentColor : Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .accessibilityAddTraits(tool == item ? .isSelected : [])
                }
            }
        }
        .padding(.top, 8)
        .padding(.bottom, 4)
        .sensoryFeedback(.selection, trigger: tool)
    }
}

/// A horizontal list of stocks, the selected one filled.
private struct StockPicker: View {
    let profiles: [Profile]
    let selection: String?
    let onSelect: (Profile) -> Void

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(profiles, id: \.info.stock) { profile in
                        let selected = profile.info.stock == selection
                        Button {
                            onSelect(profile)
                        } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(profile.info.displayName)
                                    .font(.footnote.weight(selected ? .semibold : .regular))
                                    .lineLimit(1)
                                Text(kind(profile))
                                    .font(.caption2)
                                    .foregroundStyle(selected ? Color.black.opacity(0.6) : .secondary)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .foregroundStyle(selected ? Color.black : .primary)
                            .background(
                                selected ? AnyShapeStyle(Color.accentColor) : AnyShapeStyle(.quaternary),
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                        }
                        .buttonStyle(.plain)
                        .id(profile.info.stock)
                        .accessibilityAddTraits(selected ? .isSelected : [])
                    }
                }
                .padding(.horizontal, 16)
            }
            .onAppear { proxy.scrollTo(selection, anchor: .center) }
        }
    }

    private func kind(_ profile: Profile) -> String {
        if profile.info.use == .cine { return profile.isPositive ? "Cine slide" : "Cine" }
        if profile.isPaper { return "Paper" }
        return profile.isPositive ? "Slide" : "Negative"
    }
}

/// A row of adjustments, and one slider for the selected one.
private struct AdjustTool: View {
    let model: EditorModel
    @Binding var selection: String

    private var current: Adjustment {
        Adjustment.all.first { $0.id == selection } ?? Adjustment.all[0]
    }

    var body: some View {
        let params = model.params
        let adjustment = current
        let value = params.map(adjustment.read) ?? adjustment.defaultValue

        VStack(spacing: 10) {
            ScrollViewReader { proxy in
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 14) {
                        ForEach(Adjustment.all) { item in
                            AdjustmentButton(
                                adjustment: item,
                                selected: item.id == selection,
                                modified: params.map { abs(item.read($0) - item.defaultValue) > 1e-9 }
                                    ?? false
                            ) {
                                selection = item.id
                                withAnimation { proxy.scrollTo(item.id, anchor: .center) }
                            }
                            .id(item.id)
                        }
                    }
                    .padding(.horizontal, 16)
                }
            }

            HStack {
                Text(adjustment.title)
                    .font(.footnote.weight(.semibold))
                Spacer()
                if abs(value - adjustment.defaultValue) > 1e-9 {
                    Button("Reset") {
                        model.scrub { adjustment.write(&$0, adjustment.defaultValue) }
                        model.settle()
                    }
                    .font(.footnote)
                }
                Text(adjustment.format(value, model.effective))
                    .font(.footnote.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 20)

            Slider(
                value: Binding(
                    get: { value },
                    set: { new in
                        let stepped = adjustment.step.map { (new / $0).rounded() * $0 } ?? new
                        model.scrub { adjustment.write(&$0, stepped) }
                    }),
                in: adjustment.range,
                onEditingChanged: { editing in if !editing { model.settle() } }
            )
            .padding(.horizontal, 20)
            .accessibilityLabel(adjustment.title)
            .accessibilityValue(adjustment.format(value, model.effective))
        }
        .sensoryFeedback(.selection, trigger: selection)
    }
}

/// One adjustment in the row: its symbol in a circle, filled when selected, ringed when changed.
private struct AdjustmentButton: View {
    let adjustment: Adjustment
    let selected: Bool
    let modified: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: adjustment.symbol)
                .font(.system(size: 17))
                .frame(width: 40, height: 40)
                .foregroundStyle(selected ? Color.black : modified ? Color.accentColor : .primary)
                .background(Circle().fill(selected ? Color.accentColor : Color.clear))
                .overlay {
                    Circle().strokeBorder(
                        modified && !selected ? Color.accentColor : Color.secondary.opacity(0.4),
                        lineWidth: 1)
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(adjustment.title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

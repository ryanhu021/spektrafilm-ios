import SpektraFilm
import SwiftUI

/// The pipeline as a row of stops you can tap into.
///
/// The engine can stop at any named tap between stages, so the app can show the negative, the paper
/// exposure and the print. Stepping through them shows what the simulation does.
struct StageStrip: View {
    @Binding var tap: Tap
    let scanFilm: Bool

    private var stops: [Tap] {
        scanFilm
            ? [.rgbPre, .logExposureFilm, .cmyFilm, .rgbOut]
            : [.rgbPre, .logExposureFilm, .cmyFilm, .logExposurePrint, .cmyPrint, .rgbOut]
    }

    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(stops.enumerated()), id: \.element) { index, stop in
                if index > 0 { connector(isPast: stops.firstIndex(of: tap).map { index <= $0 } ?? false) }
                stopButton(stop)
            }
        }
        .padding(.horizontal, Safelight.gutter)
        .padding(.vertical, 10)
    }

    private func stopButton(_ stop: Tap) -> some View {
        let active = stop == tap
        return Button {
            tap = stop
        } label: {
            VStack(spacing: 5) {
                Circle()
                    .fill(active ? Safelight.amber : Safelight.rule)
                    .frame(width: active ? 7 : 4, height: active ? 7 : 4)
                    .overlay {
                        if active {
                            Circle()
                                .strokeBorder(Safelight.amber.opacity(0.28), lineWidth: 4)
                                .frame(width: 15, height: 15)
                        }
                    }
                Text(shortName(stop))
                    .safelightLabel(active)
                    .fixedSize()
            }
            .frame(minWidth: 44)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .animation(.easeOut(duration: 0.18), value: active)
    }

    private func connector(isPast: Bool) -> some View {
        Rectangle()
            .fill(isPast ? Safelight.amberDim : Safelight.rule)
            .frame(height: Safelight.hairline)
            .frame(maxWidth: .infinity)
            .offset(y: -7)
    }

    private func shortName(_ tap: Tap) -> String {
        switch tap {
        case .rgbIn: return "scene"
        case .rgbPre: return "meter"
        case .logExposureFilm: return "expose"
        case .cmyFilm: return "neg"
        case .logExposurePrint: return "paper"
        case .cmyPrint: return "dev"
        case .rgbOut: return "print"
        }
    }
}

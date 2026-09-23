import SwiftUI

/// One filter of the enlarger's colour head.
///
/// A dichroic head moves filters into the light path, calibrated in Kodak CC units where 100 units
/// is one density. The engine adjusts yellow and magenta and holds cyan fixed, so the app shows two
/// dials. Each is drawn as its filter sliding into the beam, in the filter's own colour.
struct DichroicDial: View {
    let label: String
    let tint: Color
    /// Steps away from the measured neutral position for this paper, lamp and film.
    @Binding var shift: Double
    let range: ClosedRange<Double>
    /// The neutral the shift is measured from, shown so the absolute dial value is legible.
    let neutral: Double
    let onEnded: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 8) {
                RoundedRectangle(cornerRadius: 1)
                    .fill(tint)
                    .frame(width: 14, height: 3)
                Text(label)
                    .safelightLabel(shift != 0)
                Spacer(minLength: 0)
                Text(readout)
                    .font(Safelight.readout(11))
                    .foregroundStyle(shift == 0 ? Safelight.amberDim : Safelight.amber)
                    .monospacedDigit()
            }

            FilterTrack(
                shift: $shift, range: range, tint: tint, onEnded: onEnded)
        }
    }

    private var readout: String {
        let absolute = neutral + shift
        let sign = shift > 0 ? "+" : (shift < 0 ? "\u{2212}" : " ")
        return String(format: "%5.1f  %@%4.1f", absolute, sign, abs(shift))
    }
}

/// A horizontal track where the filled portion is the filter in the beam.
private struct FilterTrack: View {
    @Binding var shift: Double
    let range: ClosedRange<Double>
    let tint: Color
    let onEnded: () -> Void

    var body: some View {
        GeometryReader { geometry in
            let width = geometry.size.width
            let fraction = (shift - range.lowerBound) / (range.upperBound - range.lowerBound)
            let centreFraction = (0 - range.lowerBound) / (range.upperBound - range.lowerBound)

            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Safelight.panel)
                    .frame(height: 22)
                    .overlay {
                        Capsule().strokeBorder(Safelight.rule, lineWidth: Safelight.hairline)
                    }

                // The neutral detent, the position the measured database put this filter at.
                Rectangle()
                    .fill(Safelight.amberDim)
                    .frame(width: Safelight.hairline, height: 22)
                    .offset(x: centreFraction * width)

                // The filter itself, swept in from whichever side of neutral it is on.
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [tint.opacity(0.18), tint.opacity(0.62)],
                            startPoint: .leading, endPoint: .trailing)
                    )
                    .frame(width: abs(fraction - centreFraction) * width, height: 22)
                    .offset(x: min(fraction, centreFraction) * width)

                Capsule()
                    .fill(Safelight.paper)
                    .frame(width: 3, height: 26)
                    .shadow(color: .black.opacity(0.5), radius: 3)
                    .offset(x: fraction * width - 1.5)
            }
            .frame(height: 26)
            .contentShape(Rectangle())
            .gesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { value in
                        let f = min(max(value.location.x / width, 0), 1)
                        shift =
                            (range.lowerBound + f * (range.upperBound - range.lowerBound))
                            .rounded(toNearest: 0.5)
                    }
                    .onEnded { _ in onEnded() }
            )
        }
        .frame(height: 26)
    }
}

extension Double {
    func rounded(toNearest step: Double) -> Double {
        (self / step).rounded() * step
    }
}

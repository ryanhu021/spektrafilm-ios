import SwiftUI

/// The visual language: a darkroom under safelight.
///
/// A colour darkroom is lit by a dim amber safelight, the band print paper is least sensitive to.
/// Everything is dark and warm except the paper under the enlarger lamp. In this app the print is
/// the only brightly lit surface, and the controls stay dim.
enum Safelight {

    // MARK: - Surfaces

    /// Room black, slightly warm.
    static let ink = Color(red: 0.043, green: 0.035, blue: 0.031)
    /// A raised panel, as on the side of an enlarger column.
    static let panel = Color(red: 0.082, green: 0.067, blue: 0.063)
    /// The easel the print sits on.
    static let easel = Color(red: 0.118, green: 0.098, blue: 0.090)
    /// Hairlines and dividers.
    static let rule = Color(red: 0.204, green: 0.169, blue: 0.153)

    // MARK: - Light

    /// Safelight amber. The only saturated colour in the room.
    static let amber = Color(red: 1.0, green: 0.541, blue: 0.239)
    /// Amber at rest, for inactive labels.
    static let amberDim = Color(red: 0.541, green: 0.290, blue: 0.137)
    /// Print white. Warm, because paper is warm.
    static let paper = Color(red: 0.949, green: 0.918, blue: 0.878)

    // MARK: - Dichroic head

    /// The three filters of a colour head, at roughly their own hues.
    static let cyan = Color(red: 0.247, green: 0.722, blue: 0.769)
    static let magenta = Color(red: 0.831, green: 0.318, blue: 0.608)
    static let yellow = Color(red: 0.910, green: 0.773, blue: 0.278)

    // MARK: - Type

    /// Stock names and headings. New York, which reads like the plate label on a datasheet.
    static func display(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .serif)
    }

    /// Instrument labels. Monospaced, uppercased, widely tracked, as on a dial plate.
    static func label(_ size: CGFloat = 10) -> Font {
        .system(size: size, weight: .medium, design: .monospaced)
    }

    /// Numeric readouts. Monospaced so digits do not shift as values change.
    static func readout(_ size: CGFloat = 12, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }

    // MARK: - Metrics

    static let gutter: CGFloat = 20
    static let hairline: CGFloat = 0.5
    static let corner: CGFloat = 3
}

extension View {
    /// An instrument-plate label: small, uppercase, tracked, dim.
    func safelightLabel(_ active: Bool = false) -> some View {
        self
            .font(Safelight.label())
            .tracking(1.6)
            .textCase(.uppercase)
            .foregroundStyle(active ? Safelight.amber : Safelight.amberDim)
    }

    /// The faint grain wash that keeps large dark areas from reading as flat fill.
    func safelightGrain(opacity: Double = 0.035) -> some View {
        overlay {
            GrainOverlay()
                .opacity(opacity)
                .blendMode(.screen)
                .allowsHitTesting(false)
        }
    }
}

/// A static dither field, drawn once into an image and tiled.
///
/// Decoration only. It has nothing to do with the engine's grain model.
private struct GrainOverlay: View {
    var body: some View {
        Canvas { context, size in
            var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
            let step: CGFloat = 2
            var y: CGFloat = 0
            while y < size.height {
                var x: CGFloat = 0
                while x < size.width {
                    seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    if seed >> 60 > 11 {
                        context.fill(
                            Path(CGRect(x: x, y: y, width: 1, height: 1)),
                            with: .color(.white.opacity(Double(seed >> 56 & 0xFF) / 255.0)))
                    }
                    x += step
                }
                y += step
            }
        }
        .drawingGroup()
    }
}

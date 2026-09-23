import SpektraFilm
import SwiftUI

/// The print on the easel.
///
/// A render takes 16 to 33 ms on an M4 Pro's GPU depending on the tier, and longer on a phone, so it
/// cannot be relied on to update every frame. A finished render appears slightly flat and soft and settles to full
/// contrast over about 280 ms, the way a print comes up in the developer tray.
struct PrintView: View {
    let image: CGImage?
    let isRendering: Bool
    let tap: Tap
    let quality: RenderService.Quality?
    let failure: String?

    @State private var settled = false
    @State private var shownGeneration = 0

    var body: some View {
        GeometryReader { geometry in
            ZStack {
                easel

                if let image {
                    Image(decorative: image, scale: 1, orientation: .up)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .background(Safelight.ink)
                        // The develop settle. Low contrast and slightly soft, resolving to full.
                        .saturation(settled ? 1 : 0.72)
                        .brightness(settled ? 0 : 0.045)
                        .blur(radius: settled ? 0 : 1.6)
                        .overlay {
                            Rectangle()
                                .strokeBorder(Safelight.rule, lineWidth: Safelight.hairline)
                        }
                        .shadow(color: .black.opacity(0.55), radius: 22, y: 10)
                        .padding(Safelight.gutter)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .onChange(of: image) { _, _ in
                            settled = false
                            withAnimation(.easeOut(duration: 0.28)) { settled = true }
                        }
                        .onAppear {
                            withAnimation(.easeOut(duration: 0.28)) { settled = true }
                        }
                } else if let failure {
                    failedEasel(failure)
                } else {
                    emptyEasel
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
        }
        .overlay(alignment: .topTrailing) { statusBadge }
    }

    private var easel: some View {
        Safelight.easel
            .overlay {
                // A faint vignette, as from a single overhead lamp.
                RadialGradient(
                    colors: [.clear, .black.opacity(0.45)],
                    center: .center, startRadius: 40, endRadius: 520)
            }
            .safelightGrain()
            .ignoresSafeArea()
    }

    private func failedEasel(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 22, weight: .light))
                .foregroundStyle(Safelight.amber)
            Text("The render stopped")
                .font(Safelight.display(17))
                .foregroundStyle(Safelight.paper.opacity(0.85))
            Text(message)
                .font(Safelight.readout(11))
                .foregroundStyle(Safelight.amberDim)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 34)
        }
    }

    private var emptyEasel: some View {
        VStack(spacing: 14) {
            Image(systemName: "tray.and.arrow.down")
                .font(.system(size: 26, weight: .light))
                .foregroundStyle(Safelight.amberDim)
            Text("Load a negative")
                .font(Safelight.display(17))
                .foregroundStyle(Safelight.paper.opacity(0.75))
            Text("Any photo becomes the scene light")
                .safelightLabel()
        }
    }

    /// Names the tap when it is not the print, so an intermediate is not mistaken for a bad render.
    @ViewBuilder
    private var statusBadge: some View {
        if tap != .rgbOut || isRendering {
            HStack(spacing: 7) {
                if isRendering {
                    Circle()
                        .fill(Safelight.amber)
                        .frame(width: 5, height: 5)
                        .opacity(isRendering ? 1 : 0)
                        .animation(
                            .easeInOut(duration: 0.65).repeatForever(autoreverses: true),
                            value: isRendering)
                }
                if tap != .rgbOut {
                    Text(Self.tapName(tap))
                        .safelightLabel(true)
                }
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(Safelight.ink.opacity(0.82), in: RoundedRectangle(cornerRadius: 2))
            .overlay {
                RoundedRectangle(cornerRadius: 2)
                    .strokeBorder(Safelight.rule, lineWidth: Safelight.hairline)
            }
            .padding(Safelight.gutter + 6)
        }
    }

    static func tapName(_ tap: Tap) -> String {
        switch tap {
        case .rgbIn: return "scene"
        case .rgbPre: return "metered"
        case .logExposureFilm: return "film exposure"
        case .cmyFilm: return "negative"
        case .logExposurePrint: return "paper exposure"
        case .cmyPrint: return "print density"
        case .rgbOut: return "print"
        }
    }
}

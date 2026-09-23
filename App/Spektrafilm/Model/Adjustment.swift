import Foundation
import SpektraFilm

/// One slider in the Adjust tool: a value in the parameters, its range, and how to show it.
struct Adjustment: Identifiable, Sendable {
    let id: String
    let title: String
    let symbol: String
    let range: ClosedRange<Double>
    let step: Double?
    let read: @Sendable (RuntimePhotoParams) -> Double
    let write: @Sendable (inout RuntimePhotoParams, Double) -> Void
    /// The value as the readout shows it. `effective` is the digested parameters, where the
    /// enlarger's measured neutral lives.
    let format: @Sendable (_ value: Double, _ effective: RuntimePhotoParams?) -> String

    /// The engine's defaults, which reset returns each adjustment to.
    static let defaults = try? RuntimePhotoParams.make(
        film: "kodak_portra_400", print: "kodak_portra_endura")

    var defaultValue: Double { Self.defaults.map(read) ?? range.lowerBound }

    static let all: [Adjustment] = [
        Adjustment(
            id: "exposure", title: "Exposure", symbol: "plusminus.circle", range: -4...4, step: nil,
            read: { $0.camera.exposureCompensationEV },
            write: { $0.camera.exposureCompensationEV = $1 },
            format: { value, _ in String(format: "%+.2f EV", value) }),
        Adjustment(
            id: "filmContrast", title: "Film Contrast", symbol: "circle.lefthalf.filled",
            range: 0.5...2, step: nil,
            read: { $0.filmRender.densityCurveGamma },
            write: { $0.filmRender.densityCurveGamma = $1 },
            format: { value, _ in String(format: "%.2f", value) }),
        Adjustment(
            id: "couplers", title: "Couplers", symbol: "drop.halffull", range: 0...2, step: nil,
            read: { $0.filmRender.dirCouplers.amount },
            write: { $0.filmRender.dirCouplers.amount = $1 },
            format: { value, _ in String(format: "%.2f\u{00D7}", value) }),
        Adjustment(
            id: "halation", title: "Halation", symbol: "sun.haze", range: 0...4, step: nil,
            read: { $0.filmRender.halation.halationAmount },
            write: { $0.filmRender.halation.halationAmount = $1 },
            format: { value, _ in String(format: "%.2f\u{00D7}", value) }),
        Adjustment(
            id: "printExposure", title: "Print Exposure", symbol: "lightbulb", range: 0.25...4,
            step: nil,
            read: { $0.enlarger.printExposure },
            write: { $0.enlarger.printExposure = $1 },
            format: { value, _ in String(format: "%.2f\u{00D7}", value) }),
        Adjustment(
            id: "yellow", title: "Yellow Filter", symbol: "y.circle", range: -40...40, step: 0.5,
            read: { $0.enlarger.yFilterShift },
            write: { $0.enlarger.yFilterShift = $1 },
            format: { value, effective in
                filter(value, neutral: effective?.enlarger.yFilterNeutral)
            }),
        Adjustment(
            id: "magenta", title: "Magenta Filter", symbol: "m.circle", range: -40...40, step: 0.5,
            read: { $0.enlarger.mFilterShift },
            write: { $0.enlarger.mFilterShift = $1 },
            format: { value, effective in
                filter(value, neutral: effective?.enlarger.mFilterNeutral)
            }),
        Adjustment(
            id: "preflash", title: "Pre-flash", symbol: "bolt", range: 0...0.5, step: nil,
            read: { $0.enlarger.preflashExposure },
            write: { $0.enlarger.preflashExposure = $1 },
            format: { value, _ in String(format: "%.3f", value) }),
        Adjustment(
            id: "paperContrast", title: "Paper Contrast", symbol: "circle.righthalf.filled",
            range: 0.6...1.6, step: nil,
            read: { $0.printRender.densityCurvesMorph.gammaFactor },
            write: { params, value in
                // At its defaults the morph reproduces the unmorphed curves exactly, so turning it
                // on changes nothing until a value moves.
                params.printRender.densityCurvesMorph.active = true
                params.printRender.densityCurvesMorph.gammaFactor = value
            },
            format: { value, _ in String(format: "%.2f", value) }),
        Adjustment(
            id: "exhaustion", title: "Developer Exhaustion", symbol: "flask", range: 0...1,
            step: nil,
            read: { $0.printRender.densityCurvesMorph.developerExhaustion },
            write: { params, value in
                params.printRender.densityCurvesMorph.active = true
                params.printRender.densityCurvesMorph.developerExhaustion = value
            },
            format: { value, _ in String(format: "%.2f", value) }),
        Adjustment(
            id: "sharpening", title: "Sharpening", symbol: "triangle", range: 0...2, step: nil,
            read: { $0.scanner.unsharpMask.amount },
            write: { $0.scanner.unsharpMask.amount = $1 },
            format: { value, _ in String(format: "%.2f", value) }),
        Adjustment(
            id: "glare", title: "Glare", symbol: "sun.max", range: 0...0.15, step: nil,
            read: { $0.printRender.glare.percent },
            write: { $0.printRender.glare.percent = $1 },
            format: { value, _ in String(format: "%.3f%%", value) }),
    ]

    /// A filter position: the absolute CC value, and its shift from the measured neutral.
    private static func filter(_ shift: Double, neutral: Double?) -> String {
        let shiftText = String(format: "%+.1f", shift)
        guard let neutral else { return shiftText }
        return String(format: "%.1f", neutral + shift) + " (\(shiftText))"
    }
}

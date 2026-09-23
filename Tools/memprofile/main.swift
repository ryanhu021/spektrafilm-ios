// Peak-memory profiler for the render pipeline.
//
// Reports the high-water footprint and wall time for one frame size. Peak footprint is about 125 MB
// per megapixel, which caps export size on iOS below a 12 MP frame (see
// Sources/SpektraFilm/Runtime/RenderBudget.swift).
//
// Build and run:
//   swift build -c release --product memprofile
//   .build/release/memprofile 2                  # megapixels, whole render
//   .build/release/memprofile 2 --tap cmy_film   # peak to reach one tap
//   .build/release/memprofile 2 --spectral       # the spectral upsampling call alone
//   .build/release/memprofile 2 --warm           # time a second render, after one-time setup
//   .build/release/memprofile 2 --float          # the float32 entry point, as the app uses it
//
// Environment toggles: SPK_PREVIEW=1 for the app's preview tiers, SPK_METAL=1 for the Metal backend,
// and SPK_NO_HALATION, SPK_NO_GRAIN, SPK_NO_COUPLERS, SPK_NO_GLARE to switch one operator off.
//
// One measurement per process, and the three modes are mutually exclusive. phys_footprint is a
// whole-process high-water mark, so anything measured after something larger reads the larger figure.
//
// Always measure in release. Debug keeps bounds checks on every ImageBuffer subscript, and the same
// render takes over an order of magnitude longer.

import Foundation
import SpektraFilm

func footprintBytes() -> Int {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return result == KERN_SUCCESS ? Int(info.phys_footprint) : -1
}

func megabytes(_ bytes: Int) -> String {
    String(format: "%.0f MB", Double(bytes) / 1_048_576)
}

/// Samples the footprint on a background thread, so a peak inside a stage is not missed.
final class PeakSampler {
    fileprivate let lock = NSLock()
    fileprivate var peak = 0
    fileprivate var running = true

    func start() {
        nonisolated(unsafe) let box = self
        Thread.detachNewThread {
            let me = box
            while true {
                me.lock.lock()
                let go = me.running
                me.lock.unlock()
                if !go { return }
                let now = footprintBytes()
                me.lock.lock()
                me.peak = max(me.peak, now)
                me.lock.unlock()
                usleep(2000)
            }
        }
    }

    func stop() -> Int {
        lock.lock()
        running = false
        let value = peak
        lock.unlock()
        return value
    }
}

let arguments = CommandLine.arguments
let megapixelTarget = Double(arguments.count > 1 ? arguments[1] : "2") ?? 2

// A 4:3 frame of the requested size.
let height = Int((megapixelTarget * 1e6 / (4.0 / 3.0)).squareRoot().rounded())
let width = Int(Double(height) * 4.0 / 3.0)

var params = try RuntimePhotoParams.make(
    film: "kodak_portra_400", print: "kodak_portra_endura")
// Off unless asked for, so a measurement is of the render itself. SPK_AUTOEXPOSURE=1 turns it on,
// as the app has it, which adds the metering preview's downsample to preprocessing.
params.camera.autoExposure = ProcessInfo.processInfo.environment["SPK_AUTOEXPOSURE"] != nil

// Toggles for bisecting which operator owns the footprint.
if ProcessInfo.processInfo.environment["SPK_NO_HALATION"] != nil {
    params.filmRender.halation.active = false
}
if ProcessInfo.processInfo.environment["SPK_NO_GRAIN"] != nil {
    params.filmRender.grain.active = false
}
if ProcessInfo.processInfo.environment["SPK_NO_COUPLERS"] != nil {
    params.filmRender.dirCouplers.active = false
}
if ProcessInfo.processInfo.environment["SPK_NO_GLARE"] != nil {
    params.printRender.glare.active = false
}
// The app's scrub and settle tiers.
if ProcessInfo.processInfo.environment["SPK_PREVIEW"] != nil {
    params.settings.previewMode = true
}

let constructionStarted = DispatchTime.now().uptimeNanoseconds
let backend: ComputeBackend =
    ProcessInfo.processInfo.environment["SPK_METAL"] != nil ? .metal : .cpu
let simulator = try Simulator(params, resampler: SkimageResampler(), backend: backend)
let constructionSeconds = Double(DispatchTime.now().uptimeNanoseconds - constructionStarted) / 1e9

let baseline = footprintBytes()
// The float32 entry point writes its own input, so it gets no float64 frame to hold.
let floatInput = arguments.contains("--float")
let inputHeight = floatInput ? 1 : height
let inputWidth = floatInput ? 1 : width
var values = [Double](repeating: 0, count: inputWidth * inputHeight * 3)
for i in values.indices { values[i] = Double((i * 7919) % 1000) / 1000.0 * 1.2 }
let image = ImageBuffer(height: inputHeight, width: inputWidth, channels: 3, values: values)
let withInput = footprintBytes()

let megapixels = Double(width * height) / 1e6
let spectralOnly = arguments.contains("--spectral")
let tapOnly = arguments.contains("--tap")

if !spectralOnly && !tapOnly {
    // `--warm` renders once untimed first, so one-time work (the gamut envelopes, which are cached
    // for the process) is excluded, as it is for every render after the app's first.
    func renderFloat() throws -> Double {
        // The float32 entry point, as the app calls it: the pixels are written straight into the
        // input, and only one output value is read back.
        try simulator.processFloat(
            height: height, width: width,
            fill: { pixels in
                for i in 0..<pixels.count {
                    pixels[i] = Float(Double((i * 7919) % 1000) / 1000 * 1.2)
                }
            },
            read: { pixels, _, _ in Double(pixels[0]) })
    }
    if arguments.contains("--warm") {
        if floatInput { _ = try renderFloat() } else { _ = try simulator.process(image) }
    }
    let sampler = PeakSampler()
    sampler.start()
    let started = DispatchTime.now().uptimeNanoseconds
    let output: ImageBuffer
    if floatInput {
        output = ImageBuffer(height: 1, width: 1, channels: 1, values: [try renderFloat()])
    } else {
        output = try simulator.process(image)
    }
    let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
    let peak = sampler.stop()

    print("frame          \(width) x \(height)  (\(String(format: "%.2f", megapixels)) MP)")
    print("baseline       \(megabytes(baseline))")
    print("input buffer   \(megabytes(withInput - baseline))")
    print("peak           \(megabytes(peak))")
    print("peak per MP    \(megabytes(Int(Double(peak) / megapixels)))")
    print("construction   \(String(format: "%.1f ms", constructionSeconds * 1000))")
    print("time           \(String(format: "%.3f s  (%.2f s/MP)", seconds, seconds / megapixels))")
    print("checksum       \(String(format: "%.9f", output.values[0]))")
    print("\nstages, longest first:")
    for (stage, stageSeconds) in simulator.timings.sorted(by: { $0.value > $1.value }) {
        let name = stage.padding(toLength: 28, withPad: " ", startingAt: 0)
        print("  \(name) \(String(format: "%6.3f s", stageSeconds))")
    }
}

// Isolates the spectral upsampling call.
//
// `--tap` must not reach here. The bindings below are top level, so they are globals that live for
// the rest of the process. `tc`, `brightness` and `sampled` hold 96 MB at 2 MP, and a tap measured
// afterwards reads that on top of its own frames.
if spectralOnly {
    print("frame          \(width) x \(height)  (\(String(format: "%.2f", megapixels)) MP)")
    print("input buffer   \(megabytes(withInput - baseline))")
    // The profiler is outside the module, so build the sensitivity the same way the stage does.
    let sensitivity = nanToNum(params.film.data.logSensitivity.map { pow(10.0, $0) })
    var adaptation = try params.film.hanatos2025Adaptation()
    adaptation.applyWindow = params.settings.applyHanatos2025AdaptationWindow
    adaptation.applySurface = params.settings.applyHanatos2025AdaptationSurface
    let lut = try TCLUTBuilder.computeHanatos2025TCLUT(
        sensitivity: SpectralMatrix(sensitivity), adaptation: adaptation,
        gamutCompress: params.io.inputGamutCompress, compressionBake: TCLUTCompressionBake())
    let converter = Hanatos2025RawConverter(
        colourSpace: try ColourSpace.named(params.io.inputColourSpace),
        applyCCTFDecoding: params.io.inputCCTFDecoding,
        referenceIlluminant: try Illuminant(label: params.film.info.referenceIlluminant),
        tcLUT: lut)

    // Cumulative, because `tc`, `brightness` and `sampled` are globals too. Only the first line
    // measures one call alone. Comparing the second or third across a change needs a separate
    // process.
    print("\nspectral upsampling, cumulative high-water:")
    var s = PeakSampler()
    s.start()
    let (tc, brightness) = converter.tcAndBrightness(rgb: image)
    var isolated = s.stop()
    print(
        "  tcAndBrightness    \(megabytes(isolated))  (\(megabytes(Int(Double(isolated) / megapixels)))/MP)")

    s = PeakSampler()
    s.start()
    let sampled = converter.sampler.sample(lut: lut, coordinates: tc)
    isolated = s.stop()
    print(
        "  sampler.sample     \(megabytes(isolated))  (\(megabytes(Int(Double(isolated) / megapixels)))/MP)")
    print(
        "  tc channels \(tc.channels), sampled channels \(sampled.channels), brightness \(brightness.count)")

    s = PeakSampler()
    s.start()
    _ = converter.raw(rgb: image)
    isolated = s.stop()
    print(
        "  raw (whole)        \(megabytes(isolated))  (\(megabytes(Int(Double(isolated) / megapixels)))/MP)")
}

// One tap per process. phys_footprint is a high-water mark, so measuring several in one process
// reports the largest so far for every one of them.
if let tapIndex = arguments.firstIndex(of: "--tap"), tapIndex + 1 < arguments.count {
    let name = arguments[tapIndex + 1]
    let taps: [String: Tap] = [
        "rgb_pre": .rgbPre, "log_e_film": .logExposureFilm, "cmy_film": .cmyFilm,
        "log_e_print": .logExposurePrint, "cmy_print": .cmyPrint, "rgb_out": .rgbOut,
    ]
    guard let tap = taps[name] else {
        print("unknown tap \(name); one of \(taps.keys.sorted().joined(separator: ", "))")
        exit(1)
    }
    let s = PeakSampler()
    s.start()
    let started = DispatchTime.now().uptimeNanoseconds
    _ = try simulator.process(image, inject: nil, collect: tap)
    let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
    let peak = s.stop()
    print(
        "\(name.padding(toLength: 12, withPad: " ", startingAt: 0)) "
            + "peak \(megabytes(peak))  (\(megabytes(Int(Double(peak) / megapixels)))/MP)  "
            + String(format: "%.2f s", seconds))
}

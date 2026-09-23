// Peak-memory profiler for the render pipeline.
//
// Reports the high-water footprint and wall time for one frame size. This is the instrument for the
// memory-reduction work: peak footprint is about 148 MB per megapixel, which still caps export size
// on iOS below a 12 MP frame (see Sources/SpektraFilm/Runtime/RenderBudget.swift).
//
// Build and run:
//   swift build -c release --product memprofile
//   .build/release/memprofile 2                  # megapixels, whole render
//   .build/release/memprofile 2 --tap cmy_film   # peak to reach one tap
//   .build/release/memprofile 2 --spectral       # the spectral upsampling call alone
//
// One measurement per process, and the three modes are mutually exclusive: phys_footprint is a
// whole-process high-water mark, so anything measured after something larger reads the larger figure.
//
// Always measure in release. Debug keeps bounds checks on every ImageBuffer subscript and the same
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
params.camera.autoExposure = false

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

let simulator = try Simulator(params, resampler: SkimageResampler())

let baseline = footprintBytes()
var values = [Double](repeating: 0, count: width * height * 3)
for i in values.indices { values[i] = Double((i * 7919) % 1000) / 1000.0 * 1.2 }
let image = ImageBuffer(height: height, width: width, channels: 3, values: values)
let withInput = footprintBytes()

let megapixels = Double(width * height) / 1e6
let spectralOnly = arguments.contains("--spectral")
let tapOnly = arguments.contains("--tap")

if !spectralOnly && !tapOnly {
    let sampler = PeakSampler()
    sampler.start()
    let started = DispatchTime.now().uptimeNanoseconds
    let output = try simulator.process(image)
    let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
    let peak = sampler.stop()

    print("frame          \(width) x \(height)  (\(String(format: "%.2f", megapixels)) MP)")
    print("baseline       \(megabytes(baseline))")
    print("input buffer   \(megabytes(withInput - baseline))")
    print("peak           \(megabytes(peak))")
    print("peak per MP    \(megabytes(Int(Double(peak) / megapixels)))")
    print("time           \(String(format: "%.2f s  (%.2f s/MP)", seconds, seconds / megapixels))")
    print("checksum       \(String(format: "%.9f", output.values[0]))")
}

// Isolates the spectral upsampling call, which the bisection above points at.
//
// `--tap` must not reach here. The bindings below are top level, so they are globals that live for
// the rest of the process: `tc`, `brightness` and `sampled` are 96 MB of them at 2 MP, and a tap
// measured afterwards reads that on top of its own frames.
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

    // Cumulative, because `tc`, `brightness` and `sampled` are globals too: only the first line is a
    // reading of one call. Comparing the second or third across a change needs a separate process.
    print("\nspectral upsampling, cumulative high-water:")
    var s = PeakSampler()
    s.start()
    let (tc, brightness) = converter.tcAndBrightness(rgb: image)
    var isolated = s.stop()
    print("  tcAndBrightness    \(megabytes(isolated))  (\(megabytes(Int(Double(isolated) / megapixels)))/MP)")

    s = PeakSampler()
    s.start()
    let sampled = converter.sampler.sample(lut: lut, coordinates: tc)
    isolated = s.stop()
    print("  sampler.sample     \(megabytes(isolated))  (\(megabytes(Int(Double(isolated) / megapixels)))/MP)")
    print("  tc channels \(tc.channels), sampled channels \(sampled.channels), brightness \(brightness.count)")

    s = PeakSampler()
    s.start()
    _ = converter.raw(rgb: image)
    isolated = s.stop()
    print("  raw (whole)        \(megabytes(isolated))  (\(megabytes(Int(Double(isolated) / megapixels)))/MP)")
}

// One tap per process. phys_footprint is a high-water mark, so measuring several in one process
// reports the largest so far for every one of them, which is how two earlier rounds of this
// measurement mislocated the cost.
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

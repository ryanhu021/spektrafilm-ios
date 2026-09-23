# spektrafilm for iOS

A Swift port of [spektrafilm](https://github.com/andreavolpato/spektrafilm), Andrea Volpato's
spectral simulation of analog photography. It takes a scene-linear image, exposes it onto a virtual
film emulsion, prints it through a virtual colour enlarger onto paper, and scans the result. All of
it happens in spectral space, driven by published datasheet measurements.

The engine matches the reference end to end, and the iOS app runs it. Grain, the spatial effects and
the full print chain are ported. [Status](#status) lists what is not.

## Why port it

The reference implementation is Python and NumPy, which does not run on iOS. This port reproduces
the reference's arithmetic and checks every stage against fixtures generated from it at one pinned
upstream commit, so a render here should match a render there.

There is also an [Android port](https://github.com/thetechgeekko/Spektrafilm-android), still in
development. It was useful prior art, but the Python reference is the numeric source of truth.

## Status

| Layer | State |
|---|---|
| Colour: transfer functions, colourspaces, illuminants, observers, filters | Done, parity-tested |
| Core: buffers, matrices, interpolation, RNG, distributions, resampling | Done, parity-tested |
| Profiles: loading and validation | Done |
| Model: density curves, DIR couplers, grain, diffusion, glare, gamut compression | Done, parity-tested |
| Pipeline: filming, printing, scanning, params digest, public API | Done, parity-tested |
| App: photo import, editor, stage inspection, export | Works |
| Parity harness: fixture format, generator, comparison | Done, 458 fixtures |

Not ported. Each throws `SpektraError.unsupportedSetting`, and none is reachable from the default
render:

| Feature | Why |
|---|---|
| Mallett-2019 upsampling | sRGB only, and clips the input. Hanatos-2025 is the default. |
| Resampling at order 3 | Only reachable from `io.upscaleFactor`. Needs the cubic spline prefilter. |
| 3D enlarger and scanner LUTs | The reference calls them an approximation of its own direct path. |

## Layout

```
Sources/SpektraFilm/     Engine. No third-party dependencies, builds for iOS and macOS.
  Core/                  Buffers, matrices, interpolation, errors.
  Colour/                Colourspaces, transfer functions, illuminants, observers.
  Profiles/              Profile types and the bundled-profile loader.
  Model/                 The film and paper physics: curves, couplers, grain, diffusion.
  Runtime/               Parameters, the pipeline stages, and the public Simulator.
  Metal/                 GPU versions of the costliest operators.
  Generated/             Tables extracted from colour-science. Do not edit by hand.
  Data/                  28 film and paper profiles, the spectral LUT, filter curves.
Tests/SpektraFilmTests/  Parity tests and their committed fixtures.
App/                     iOS app. project.yml is the source of truth; the .xcodeproj is generated.
Tools/parity/            Fixture generator, table extractor, oracle setup.
Tools/memprofile/        Peak-memory and timing profiler.
Tools/release/           SideStore source generator for the release workflow.
Tools/icon/              Renders the app icon.
```

## Build and test

The engine, on macOS:

```sh
swift build
swift test
```

The app needs [XcodeGen](https://github.com/yonaskolb/XcodeGen):

```sh
brew install xcodegen
make app       # generates App/SpektrafilmApp.xcodeproj and builds for the simulator
make run       # builds in release, installs and launches on a booted simulator
```

`make help` lists every target.

## Install on a phone

Each `v*` tag builds an unsigned `.ipa` and attaches it to a
[GitHub release](https://github.com/ryanhu021/spektrafilm-ios/releases). Install it with
[SideStore](https://sidestore.io), which signs it with your own Apple ID, so no paid developer
account is needed.

1. On the phone, open the latest release in Safari and download `Spektrafilm.ipa` to Files. The
   repository is private, so sign in to GitHub first.
2. In SideStore, tap **+** on the My Apps tab and choose the file.

A free Apple ID signs apps for 7 days; SideStore refreshes them in the background. If the repository
becomes public, add `https://github.com/ryanhu021/spektrafilm-ios/releases/latest/download/sidestore-source.json`
as a source in SideStore instead, and updates appear there.

To release, tag and push: `git tag v0.2.0 && git push origin v0.2.0`. The tag sets the version and
the workflow run number sets the build number.

## Parity

The tests compare against fixtures produced by the reference implementation at upstream commit
[`3bb2c2d`](https://github.com/andreavolpato/spektrafilm/commit/3bb2c2d2801ff68b92019cf1dbcbb133d60832bc).
The fixtures are committed, so CI runs without Python. Tolerance is `max_abs <= 1e-4` and
`rms <= 1e-5`, the same contract the Android port uses. Grain and glare draw random numbers and
cannot match bit for bit, so their tests check the statistics instead.

The fixtures cover each operator on synthetic inputs, every stage boundary across four film, paper
and output-space combinations, the scan-the-negative topology, and a real photograph through four
stocks. 18% grey through Portra 400 onto Portra Endura, in sRGB, renders
`[0.4607145, 0.46055124, 0.46038181]` in both implementations.

Regenerating fixtures needs the oracle, a local checkout of upstream at the pinned commit:

```sh
make oracle     # clones upstream at the pin, builds a Python 3.13 venv
make goldens    # regenerates the fixtures
make tables     # regenerates Sources/SpektraFilm/Generated
```

A changed fixture means the render changed. Review the difference before committing it.

## Performance

Measured in release on an M4 Pro (10 performance and 4 efficiency cores), Portra 400 onto Portra
Endura, after one warm-up render, on the float64 CPU backend. These are not phone measurements; a
phone has fewer cores.

| What | Time | Peak memory |
|---|---|---|
| Simulator construction | 1.5 ms, 9 ms the first time | |
| 320 px preview, grain off | 19 ms | |
| 640 px preview, grain off | 57 ms | |
| 640 px with grain and spatial effects | 94 ms | |
| 2 MP | 0.56 s | 250 MB |
| 6 MP | 2.3 s | 739 MB |
| 12 MP | 3.7 s | 1472 MB |

The per-pixel, per-row and per-column loops run across cores. Each output value is computed the
same way however the frame is split, so a render is bit-identical at any core count, and
`ParallelTests` checks that.

`Simulator(params, backend: .metal)` moves the spectral contraction, the costliest operator, to the
GPU in float32. Apple GPUs have no float64, but a render through it stays within 6.6e-7 of the
oracle, inside the same 1e-4 tolerance as the CPU. On an M4 Pro the contraction takes 15.5 ms at
12 MP against 259 ms on the CPU. Using Metal adds 106 MB to peak memory: 82 MB of one-time driver
setup and 24 MB of staging buffers. The app uses it, and falls back to the CPU without a GPU.

The app renders at 320 px while a control is being dragged, at 640 px when it is released, and at
the largest size the device allows on export.

Memory limits export size more than time does. iOS terminates a foreground app at roughly 1.4 GB,
so `RenderBudget` reads the process's actual allowance, caps the export size to fit, and the app
tells the user when it has downscaled. Peak footprint is 123 to 125 MB per megapixel: 6 MP fits
with room, and 12 MP is close to the limit.

The peak depends on how many full frames are alive at once. Spectral upsampling, the coupler
correction, grain and halation each work one channel plane at a time or reuse their input buffer,
which keeps that number low.

`Tools/memprofile` measures both memory and time. Take one measurement per process:
`phys_footprint` is a high-water mark, so a second measurement in the same run reports the larger
of the two.

## Licensing

GPL-3.0-only, inherited from upstream, since this is a derivative work. The bundled profiles are
CC BY-SA 4.0 and the spectral LUT carries upstream's own licence, which forbids resale.
[docs/LICENSING.md](docs/LICENSING.md) has the details, including the attribution the profiles
require.

Film modeling powered by `spektrafilm`.

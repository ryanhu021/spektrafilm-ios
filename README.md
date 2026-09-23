# spektrafilm for iOS

A Swift port of [spektrafilm](https://github.com/andreavolpato/spektrafilm), Andrea Volpato's
spectral simulation of analog photography. It takes a scene-linear image, exposes it onto a virtual
film emulsion, prints it through a virtual colour enlarger onto paper, and scans the result. All of
it happens in spectral space, driven by published datasheet measurements.

> **Status: it renders.** The engine matches the reference end to end, and the iOS app runs it.
> Grain, the spatial effects and the whole print chain are in. See [Status](#status) for what is
> deliberately left out.

## Why port it

The reference implementation is Python and NumPy, which will not run on iOS. This port reproduces
the reference's arithmetic and checks every stage against fixtures generated from it, pinned to one
upstream commit. The aim is a render that matches spektrafilm, not one that merely looks filmic.

There is also an [Android port](https://github.com/thetechgeekko/Spektrafilm-android) under active
development. It is useful prior art, but the Python reference is the numeric source of truth.

## Status

| Layer | State |
|---|---|
| Colour: transfer functions, colourspaces, illuminants, observers, filters | Done, parity-gated |
| Core: buffers, matrices, interpolation, RNG, distributions, resampling | Done, parity-gated |
| Profiles: loading and validation | Done |
| Model: density curves, DIR couplers, grain, diffusion, glare, gamut compression | Done, parity-gated |
| Pipeline: filming, printing, scanning, params digest, public API | Done, parity-gated |
| App: photo import, editor, tap inspection, export | Works |
| Parity harness: fixture format, generator, comparison | Done, 454 fixtures |

Deliberately not ported. Each throws `SpektraError.unsupportedSetting` rather than silently doing
something else, and none is reachable from the default render:

| Feature | Why |
|---|---|
| Mallett-2019 upsampling | sRGB only, and clips the input. Hanatos-2025 is the default. |
| Print-curve morph | Needs a Brent solve per control point. Defaults off. |
| Resampling at order 3 | Only reachable from `io.upscaleFactor`. Needs the cubic spline prefilter. |
| 3D enlarger and scanner LUTs | The reference calls them an approximation of its own direct path. |

## Layout

```
Sources/SpektraFilm/     Engine. No third-party dependencies, builds for iOS and macOS.
  Core/                  Buffers, matrices, interpolation, errors.
  Colour/                Colourspaces, transfer functions, illuminants, observers.
  Profiles/              Profile types and the bundled-profile loader.
  Generated/             Tables extracted from colour-science. Do not edit by hand.
  Data/                  28 film and paper profiles, the spectral LUT, filter curves.
Tests/SpektraFilmTests/  Parity tests and their committed fixtures.
App/                     iOS app. project.yml is the source of truth; the .xcodeproj is generated.
Tools/parity/            Fixture generator, table extractor, oracle setup.
```

## Build and test

The engine, on macOS:

```sh
swift build
swift test
```

The app, which needs [XcodeGen](https://github.com/yonaskolb/XcodeGen):

```sh
brew install xcodegen
make project   # generates App/SpektrafilmApp.xcodeproj
make app       # builds for the simulator
```

`make help` lists every target.

## Parity

Every numeric claim is checked against fixtures produced by the reference implementation at upstream
commit
[`3bb2c2d`](https://github.com/andreavolpato/spektrafilm/commit/3bb2c2d2801ff68b92019cf1dbcbb133d60832bc).
The fixtures are committed, so CI runs the gate without Python. Tolerance is `max_abs <= 1e-4` and
`rms <= 1e-5`, the same contract the Android port uses. Grain and glare draw from an RNG and cannot
match bit for bit, so they are gated on their statistics.

Regenerating fixtures needs the oracle, a local checkout of upstream at the pinned commit:

```sh
make oracle     # clones upstream at the pin, builds a Python 3.13 venv
make goldens    # regenerates the fixtures
make tables     # regenerates Sources/SpektraFilm/Generated
```

A changed fixture means the render changed. Review the delta before committing it.

The reference render of 18% grey through Portra 400 onto Portra Endura, in sRGB, is
`[0.4607145, 0.46055124, 0.46038181]`, and the port matches it. Every stage boundary is gated too,
across four film, paper and output-space combinations plus the scan-the-negative topology.

## Performance

Measured in release on an M-series Mac, single-threaded. The engine is float64 CPU with no GPU path,
so these are the numbers the app is designed around rather than a target to beat later.

| What | Time | Peak memory |
|---|---|---|
| Simulator construction, per film | 14 to 23 ms | |
| 320 px preview, grain off | 83 ms | |
| 640 px preview, grain off | 343 ms | |
| 640 px with grain and spatial effects | 623 ms | |
| 2 MP | 3.9 s | 250 MB |
| 6 MP | 12.4 s | 738 MB |
| 12 MP | 25.7 s | 1471 MB |

The app uses the first three as an interaction ladder: a control being dragged renders at 320 px, a
release renders at 640 px, and export renders as large as the device allows.

Memory is the binding constraint, not time. iOS terminates a foreground app at roughly 1.4 GB, so
`RenderBudget` reads the process's real allowance and caps the export size, and the app says when the
cap bit. Peak footprint started at 299 MB per megapixel, which made a 12 MP export 2.8 GB and
guaranteed a kill; it is now 125, so 6 MP fits with room and 12 MP sits on the line.

That came from removing concurrently-live frames, not from allocating less in total. Spectral
upsampling held five frames where two suffice, the coupler correction held eight, grain materialised
the whole sublayer split, and halation held its input, a copy and two blurs. Each now works a channel
plane at a time or consumes its input. What matters for peak is how many frames are alive at one
instant.

`Tools/memprofile` is the instrument. It takes one measurement per process, because `phys_footprint`
is a high-water mark and measuring several things in one run reports the largest so far for every one
of them. Three separate rounds of this measurement mislocated the cost that way before the tool was
fixed.

## Licensing

GPL-3.0-only, inherited from upstream, since this is a derivative work. The bundled profiles are
CC BY-SA 4.0 and the spectral LUT carries upstream's own licence, which forbids resale.
[docs/LICENSING.md](docs/LICENSING.md) has the detail, including the attribution the profiles
require.

Film modeling powered by `spektrafilm`.

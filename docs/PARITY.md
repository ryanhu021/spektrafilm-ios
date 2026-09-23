# Parity

The port reproduces the reference's arithmetic, and the tests check it against fixtures the
reference produced at upstream commit
[`3bb2c2d`](https://github.com/andreavolpato/spektrafilm/commit/3bb2c2d2801ff68b92019cf1dbcbb133d60832bc).
The fixtures are committed, so CI runs without Python.

Tolerance is `max_abs <= 1e-4` and `rms <= 1e-5`. Grain and glare draw random numbers and cannot
match bit for bit, so their tests check the statistics instead.

The fixtures cover each operator on synthetic inputs, every stage boundary across four film, paper
and output-space combinations, the scan-the-negative topology, and a real photograph through four
stocks. 18% grey through Portra 400 onto Portra Endura, in sRGB, renders
`[0.4607145, 0.46055124, 0.46038181]` in both implementations.

## Regenerating fixtures

This needs the oracle, a local checkout of upstream at the pinned commit:

```sh
make oracle     # clones upstream at the pin, builds a Python 3.13 venv
make goldens    # regenerates the fixtures
make tables     # regenerates Sources/SpektraFilm/Generated
```

A changed fixture means the render changed. Review the difference before committing it.

## Not ported

Each throws `SpektraError.unsupportedSetting`, and none is reachable from the default render.

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

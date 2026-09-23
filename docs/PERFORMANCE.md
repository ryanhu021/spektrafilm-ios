# Performance

Measured in release on an M4 Pro (10 performance and 4 efficiency cores), Portra 400 onto Portra
Endura, after one warm-up render. These are not phone measurements; a phone has a smaller GPU and
fewer cores.

| Render | CPU backend | Metal backend |
|---|---|---|
| 320 px preview, grain off | 19 ms | 16 ms |
| 640 px preview, grain off | 57 ms | 16 ms |
| 640 px, all effects | 94 ms | 33 ms |
| 2 MP | 0.56 s, 250 MB | 0.12 s, 227 MB |
| 6 MP | 2.3 s, 739 MB | 0.27 s, 480 MB |
| 12 MP | 3.7 s, 1472 MB | 0.47 s, 914 MB |

The Metal figures are for `Simulator.processFloat`, which the app uses, with float32 in and out.
Simulator construction takes 1.5 ms, or 9 ms the first time in a process.

## The two backends

The CPU backend is float64 and is what the parity fixtures check. Its per-pixel, per-row and
per-column loops run across cores, and each output value is computed the same way however the frame
is split, so a render is bit-identical at any core count. `ParallelTests` checks that.

`Simulator(params, backend: .metal)` runs everything after preprocessing on the GPU in float32:
filming, development with couplers and grain, printing and scanning. The frame is uploaded once and
downloaded once. Every tap still matches the oracle within the 1e-4 tolerance, and grain draws from
the same Philox stream per pixel as the CPU. Two cases stay on the CPU: the FFT diffusion filters,
and JzAzBz output gamut compression. JzAzBz's PQ curve amplifies the GPU's rounding past the
tolerance on some GPU generations.

## Memory

Memory limits export size more than time does. iOS terminates a foreground app at roughly 1.4 GB,
so `RenderBudget` reads the process's actual allowance and caps the export size to fit, and the app
tells the user when it has downscaled. The CPU backend peaks at 125 MB per megapixel. The Metal
float32 path peaks at about 96 MB plus 72 MB per megapixel, so the same allowance fits roughly
twice the frame.

The peak depends on how many full frames are alive at once. On the CPU, spectral upsampling, the
coupler correction, grain and halation work one channel plane at a time or reuse their input
buffer. On the GPU, halation and the coupler diffusion work one channel at a time through a few
reused planes, each stage frees its input once it has consumed it, and the float32 entry point never
makes a float64 copy of the frame.

## Measuring

`Tools/memprofile` measures memory and time:

```sh
swift build -c release --product memprofile
.build/release/memprofile 12 --warm                        # CPU, 12 MP
SPK_METAL=1 .build/release/memprofile 12 --float --warm    # Metal, as the app renders
```

Take one measurement per process: `phys_footprint` is a high-water mark, so a second measurement in
the same run reports the larger of the two. `SPK_PREVIEW=1` selects the app's preview tiers, and
`SPK_AUTOEXPOSURE=1` adds metering, which the app has on.

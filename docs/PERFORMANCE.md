# Performance

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

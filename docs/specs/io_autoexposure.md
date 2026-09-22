# Port spec — `io_autoexposure`

Subsystem: image import, crop/rescale/preview resampling, auto-exposure metering, RAW import,
image export and metadata.

Reference: `/tmp/spektra-ref/spektrafilm` @ `3bb2c2d`. Oracle venv:
`/tmp/spektra-ref/spektrafilm/.venv/bin/python`.

Files covered:

| Reference file | Status |
| --- | --- |
| `src/spektrafilm/utils/autoexposure.py` | port in full |
| `src/spektrafilm/utils/crop_resize.py` | port in full (one live function) |
| `src/spektrafilm/utils/preview.py` | port in full |
| `src/spektrafilm/runtime/services/resize.py` | port in full (caller, but owns the semantics) |
| `src/spektrafilm/utils/io.py` | port pixel I/O + metadata; ICC table is data |
| `src/spektrafilm/utils/raw_file_processor.py` | port the post-demosaic chain; demosaic is platform-native |
| `src/spektrafilm/utils/measure.py` | **do not port** — dead code, see §9 |

Every number in this spec was printed from the oracle venv. Every algorithm in §2 was
re-implemented from scratch in NumPy and checked against `skimage`/`scipy` on the same inputs; the
agreement figures quoted are from those runs.

---

## 0. Conventions

- Images are `[height, width, channel]`, row-major, channel-interleaved. `ImageBuffer` already
  matches this.
- Engine working type is `Double` (float64). The reference's `_preprocess` casts the input with
  `np.double(...)` before anything in this subsystem runs, so from the pipeline's point of view
  everything here is float64. The RAW loader and the file readers are the exception and produce
  float32/float16 in some paths — §7.6 and §8.1 list which.
- Units: `pixel_size_um` is micrometres of film per pixel. `film_format_mm` is millimetres.
  `crop_size` is a **fraction of the long edge**, not of each axis. Auto-exposure mask
  coordinates are normalised by the **long edge in pixels** and are dimensionless.
- Mid-grey is `0.184` everywhere in this engine (also used by `FilmingStage._compute_density_
  spectral_midgray_to_balance_print`). Do not substitute 0.18.

### 0.1 Where this subsystem sits

`SimulationPipeline._preprocess` (`runtime/pipeline.py:191`) is the whole of the engine-side entry:

```python
def _preprocess(self, image):
    image = np.double(np.array(image)[:, :, 0:3])   # drop alpha, force float64
    image = self._filming_stage.auto_exposure(image)  # meter + scale
    image = self._resize_service.crop_and_rescale(image)  # crop, optional upscale, set pixel_size_um
    return image
```

Two orderings matter and are easy to get backwards:

1. **Auto-exposure runs before the crop.** Metering always sees the full uncropped frame.
2. **`pixel_size_um` is computed from the pre-crop long edge.** A crop is a physical sub-area of
   the same piece of film, so it must not change the micrometres-per-pixel scale.

`pixel_size_um` stays `nil`/`None` until `crop_and_rescale` runs. Every downstream µm→pixel
conversion (diffusion, halation, grain, glare) reads it. If the Swift port lets a caller inject at
a tap past `preprocess`, it must either run `crop_and_rescale` first or set `pixelSizeUm`
explicitly, or the spatial stages get a nil.

---

## 1. `measure_autoexposure_ev` — auto-exposure metering

`utils/autoexposure.py`. Called only from `FilmingStage.auto_exposure`
(`runtime/stages/filming.py:40`):

```python
def auto_exposure(self, image):
    if self._camera.auto_exposure:
        small_preview = self._resize_service.small_preview(image)   # max 256 px long edge
        autoexposure_ev = measure_autoexposure_ev(
            small_preview,
            self._io.input_color_space,        # default "ProPhoto RGB"
            self._io.input_cctf_decoding,      # default False
            method=self._camera.auto_exposure_method,  # default "center_weighted"
        )
        return image * 2 ** autoexposure_ev    # applied to the FULL image
    return image
```

**Metering runs on the 256-px preview, the gain is applied to the full image.** The preview
downsample is therefore inside the parity envelope. Measured EV drift between metering the
preview and metering the full frame, on a 800×1200 gradient+noise image:

| method | ΔEV (preview − full) |
| --- | --- |
| `average` | +0.000013 |
| `partial` | −0.000226 |
| `matrix` | +0.000367 |
| `median` | −0.000825 |
| `multi_zone` | −0.000846 |
| `center_weighted` | −0.000986 |
| `highlight_weighted` | +0.002826 |

0.0028 EV is a 0.20 % multiplicative error on the whole image, ~2000× the 1e-4 gate. `small_preview`
must be bit-faithful (§4.3), not "close enough".

`params.debug.lut_mode` forces `camera.auto_exposure = False` (`params_builder.py:106`), because
metering is image-global and cannot be baked into a 3-D LUT.

### 1.1 Luminance

```python
def _luminance_y(image, color_space, apply_cctf_decoding):
    image_XYZ = colour.RGB_to_XYZ(image, color_space, apply_cctf_decoding=apply_cctf_decoding)
    return image_XYZ[:, :, 1]
```

`colour.RGB_to_XYZ` is called with `illuminant=None`, so `illuminant_XYZ == illuminant_RGB ==
colourspace.whitepoint` and the default `chromatic_adaptation_transform='CAT02'` builds a CAT
matrix from a whitepoint to itself. Verified: that matrix deviates from identity by at most
1.4e-16, and the resulting Y row is bit-identical to `matrix_RGB_to_XYZ[1]` for sRGB and
ACES2065-1 and differs by 8e-20 for ProPhoto RGB.

**Swift:** `Y = row1 · rgb_linear` using `ColourTables.colourspaces[…].toXYZ[1]`. No adaptation.

Y rows (already in `Generated/ColourTables.swift`; reproduced so a reviewer can check without
opening it):

| colourspace | `toXYZ[1]` = (R, G, B) weights |
| --- | --- |
| sRGB | 0.2126, 0.7152, 0.0722 |
| Display P3 | 0.2289745640697487, 0.6917385218365064, 0.07928691409374498 |
| Adobe RGB (1998) | 0.29734, 0.62736, 0.07529 |
| ITU-R BT.2020 | 0.2627002120112669, 0.6779980715188711, 0.05930171646986194 |
| DCI-P3 | 0.2094916779127305, 0.7215952541610436, 0.0689130679262258 |
| ProPhoto RGB | 0.288, 0.7119, 0.0001 |
| ACES2065-1 | 0.3439664498, 0.7281660966, **−0.0721325464** |

ACES2065-1's blue weight is negative — Y can go negative for saturated blue AP0 input. That is not
a bug to fix; see §1.5.

Colourspace names go through `colour.utilities.validate_method`, which lowercases and matches
case-insensitively. `"srgb"`, `"SRGB"` and `"sRGB"` all resolve. The Swift lookup should be
case-insensitive on the same seven names (`spektrafilm_gui/options.py:RGBColorSpaces`).

### 1.2 Transfer functions (`apply_cctf_decoding=True`)

`TransferFunction` is referenced by `Generated/ColourTables.swift` but not yet defined — it belongs
to the colour/spectral subsystem. The constants this subsystem needs, all dumped from
colour 0.4.7:

- **sRGB / Display P3** (`eotf_sRGB`): `V <= 0.040449936 ? V/12.92 : spow((V+0.055)/1.055, 2.4)`.
  The threshold is `eotf_inverse_sRGB(0.0031308)` evaluated numerically, **not** 0.04045.
  `spow(a,p) = sign(a) * |a|**p` (mirrors negatives).
- **ProPhoto RGB** (`cctf_decoding_ROMMRGB`): `E_t = 16**(1.8/(1-1.8)) = 0.001953125`, so the
  breakpoint is `16*E_t = 0.03125`. `X < 0.03125 ? X/16 : spow(X, 1.8)`.
- **ITU-R BT.2020** (`oetf_inverse_BT2020`, 10-bit): `alpha = 1.099`, `beta = 0.018`, threshold
  `oetf_BT2020(beta) = 0.08124794403514049`.
  `E < 0.08124794403514049 ? E/4.5 : spow((E + 0.099)/1.099, 1/0.45)`.
- **Adobe RGB (1998)**: `gamma_function(a, exponent=2.19921875)` with
  `negative_number_handling='Indeterminate'`, i.e. plain `a**2.19921875` → **NaN for a < 0**.
- **DCI-P3**: same, `exponent=2.6`, **NaN for a < 0**.
- **ACES2065-1**: identity.

Spot checks at `x = [-0.2, -0.01, 0]`:

| space | decoded |
| --- | --- |
| sRGB | −0.01547988, −0.00077399, 0 |
| ProPhoto RGB | −0.0125, −0.000625, 0 |
| ITU-R BT.2020 | −0.04444444, −0.00222222, 0 |
| Adobe RGB (1998) | NaN, NaN, 0 |
| DCI-P3 | NaN, NaN, 0 |

The default pipeline config is `input_color_space="ProPhoto RGB"`, `input_cctf_decoding=False`, so
the common path is a bare matrix multiply.

### 1.3 Normalised coordinates

```python
def _normalized_coords(image):
    norm_shape = image.shape[0:2] / np.max(image.shape[0:2])   # tuple / np.int64 -> float64 array
    x = (np.arange(image.shape[1]) / image.shape[1] - 0.5) * norm_shape[1]
    y = (np.arange(image.shape[0]) / image.shape[0] - 0.5) * norm_shape[0]
    return x, y            # x: (W,), y: (H,)
```

Algebraically `x[i] = (i − W/2) / L` and `y[j] = (j − H/2) / L` with `L = max(H, W)`. Two traps:

1. **The mask centre is off by half a pixel.** The code divides by `W` and subtracts `0.5`, so
   `x = 0` sits at index `W/2`, not at the geometric centre `(W−1)/2`. Keep it. Do not "fix" it.
2. **The range is asymmetric:** `x ∈ [−0.5·W/L, (0.5 − 1/W)·W/L]`. The long edge spans
   `[−0.5, 0.5 − 1/L]`, so the last sample is one pixel short of `+0.5`.

Verified for a 6×10 image: `x = [−0.5, −0.4, …, 0.4]`, `y = [−0.3, −0.2, …, 0.2]`.

Compute it in the same order as Python — `(Double(i)/Double(W) - 0.5) * (Double(W)/Double(L))` —
so the float rounding matches. The closed form differs by ~1e-17, irrelevant at 1e-4, but there is
no reason to introduce the difference.

The mask is broadcast as `x**2 + y[:, None]**2`, giving `[H, W]`.

### 1.4 The seven methods

All produce `exposure`, then

```python
exposure_compensation_ev = -np.log2(exposure)
if np.isinf(exposure_compensation_ev):
    exposure_compensation_ev = 0.0        # prints a warning
return exposure_compensation_ev
```

**`average`** — `exposure = mean(Y) / 0.184` over all H·W samples.

**`median`** — `exposure = median(Y) / 0.184`. NumPy's median averages the two central order
statistics for even H·W. Swift must do the same, not pick the lower.

**`center_weighted`** (default) — `sigma = 0.2`:

```
mask[j][i] = exp(-(x[i]^2 + y[j]^2) / (2 * 0.2^2))
mask      /= sum(mask)
exposure   = sum(Y * mask) / 0.184
```

Note the exact grouping: the negation applies to the whole numerator before the division by
`2*sigma^2 = 0.08`.

**`partial`** — hard disc, radius < 0.15 of the long edge (Canon "Partial"):

```
radius[j][i] = sqrt(x[i]^2 + y[j]^2)
mask         = radius < 0.15
if mask has no true pixels: mask = all true
exposure     = mean(Y[mask]) / 0.184
```

The disc is genuinely tiny: on a 3:2 frame it covers ~7 % of the height. On small previews it can
collapse to a handful of pixels — for a 4×6 image it selects exactly one pixel (the only one with
radius 0.0, present because both H and W are even).

**`matrix`** — 5×5 zones, raised-cosine radial weights:

```
cell_h = H // 5 ; cell_w = W // 5          # integer division; remainder rows/cols are DROPPED
for r in 0..4, c in 0..4:
    cell = Y[r*cell_h : (r+1)*cell_h, c*cell_w : (c+1)*cell_w]
    if cell is empty: skip this zone entirely (no mean, no weight)
    mean_rc = mean(cell)
    dy = (r - 2) / 2 ; dx = (c - 2) / 2                  # both in {-1, -0.5, 0, 0.5, 1}
    dist = sqrt(dx^2 + dy^2) / sqrt(2)
    w_rc = 0.5 * (1 + cos(pi * dist))
weights /= sum(weights)
exposure = dot(weights, means) / 0.184
```

The last `H % 5` rows and `W % 5` columns never contribute. Cells are only empty when `H < 5` or
`W < 5`, in which case **all 25** are empty, `weights` is empty, `weights.sum()` is 0 (NumPy warns),
`dot` of empty arrays is `0.0`, `−log2(0) = +inf`, the `isinf` guard fires and the function returns
`0.0 EV`. Reproduce that outcome; the intermediate NaN is not observable.

Raw weights (row-major, `r` = 0..4 top to bottom):

```
0.000000000000000  0.104374920684305  0.197150066460593  0.104374920684305  0.000000000000000
0.104374920684305  0.500000000000000  0.722007920163107  0.500000000000000  0.104374920684305
0.197150066460593  0.722007920163107  1.000000000000000  0.722007920163107  0.197150066460593
0.104374920684305  0.500000000000000  0.722007920163107  0.500000000000000  0.104374920684305
0.000000000000000  0.104374920684305  0.197150066460593  0.104374920684305  0.000000000000000
```

`sum = 7.511631311969239`. Normalised:

```
0.000000000000000  0.013895106981355  0.026245972182693  0.013895106981355  0.000000000000000
0.013895106981355  0.066563437319306  0.096118657875639  0.066563437319306  0.013895106981355
0.026245972182693  0.096118657875639  0.133126874638612  0.096118657875639  0.026245972182693
0.013895106981355  0.066563437319306  0.096118657875639  0.066563437319306  0.013895106981355
0.000000000000000  0.013895106981355  0.026245972182693  0.013895106981355  0.000000000000000
```

The four corners weigh exactly 0 (`dist = 1`, `cos(pi) = -1`), so they are metered and discarded.
Hard-coding the normalised table is safe; hard-coding the raw table plus a runtime sum is safer if
someone later makes the grid configurable.

**`multi_zone`** — three concentric rings on the same `radius` map:

| ring | bounds (half-open) | weight |
| --- | --- | --- |
| spot | `[0.00, 0.05)` | 0.50 |
| mid | `[0.05, 0.25)` | 0.30 |
| outer | `[0.25, 0.50)` | 0.20 |

```
weighted_sum = 0 ; weight_total = 0
for (lo, hi), w in rings:
    mask = (radius >= lo) & (radius < hi)
    if mask empty: continue                 # weight is NOT accumulated
    weighted_sum += w * mean(Y[mask])
    weight_total += w
exposure = (weight_total > 0 ? weighted_sum / weight_total : mean(Y)) / 0.184
```

Empty rings renormalise over the survivors. The spot ring requires `radius < 0.05`, which needs
both H and W even to even contain `radius == 0`; on odd-sized previews it is frequently empty.
Pixels with `radius >= 0.5` (the frame corners, up to 0.707 on a square image) are outside all
rings and never metered.

**`highlight_weighted`** — `weights = Y^2`:

```
total = sum(Y^2)
if total < 1e-12: weights = ones ; total = H*W
exposure = sum(Y * Y^2) / total / 0.184        # = sum(Y^3) / sum(Y^2) / 0.184
```

The `1e-12` threshold is absolute, not relative to the pixel count. A large, very dark frame can
sum above 1e-12 and still be numerically hopeless; the reference does not care and neither should
the port.

**anything else** — `exposure = 1.0`, so `EV = -0.0` (negative zero) and `2 ** -0.0 == 1.0`.
`camera.auto_exposure_method` is a free-form string, so a typo silently disables metering.

### 1.5 Degenerate inputs

| input | reference behaviour |
| --- | --- |
| all-zero image, any method | `exposure = 0` → `EV = +inf` → guard → `0.0` |
| `H < 5` or `W < 5`, `matrix` | `0.0` (see above) |
| **negative Y** (ACES blue, or any space after a negative-clamping cctf) | `−log2(negative) = NaN`. `np.isinf(NaN)` is **False**, so the guard does not fire and the function returns NaN. `image * 2**NaN` poisons the entire frame. |

The NaN path is a live defect in the reference. Reproduce it for parity, but the Swift port should
log it loudly, and the port should consider a `strictNaN` flag that traps in debug builds. Do not
silently substitute 0.0 — that would diverge from the oracle.

### 1.6 Reference values

Deterministic fixture: `H=4, W=6`, all three channels equal `v[j][i] = (j*6+i)/23`.

```
x = [-0.5, -0.333333333333, -0.166666666667, 0.0, 0.166666666667, 0.333333333333]
y = [-0.333333333333, -0.166666666667, 0.0, 0.166666666667]
```

center_weighted normalised mask:

```
0.001391982499385 0.007899820997434 0.022387589548912 0.031681375658032 0.022387589548912 0.007899820997434
0.003944789744683 0.022387589548912 0.063445002864416 0.089783000754996 0.063445002864416 0.022387589548912
0.005582394903221 0.031681375658032 0.089783000754996 0.127054722367942 0.089783000754996 0.031681375658032
0.003944789744683 0.022387589548912 0.063445002864416 0.089783000754996 0.063445002864416 0.022387589548912
```

EV results:

| method | ProPhoto RGB, decode=False | sRGB, decode=True |
| --- | --- | --- |
| `average` | −1.4422223286050742 | −0.79161567179506376 |
| `median` | −1.4422223286050742 | −0.22189608019724413 |
| `center_weighted` | −1.7085596472845359 | −1.1035688090539002 |
| `partial` | −1.82555096815658 | −1.057167838069716 |
| `matrix` | 0.0 (H<5) | 0.0 (H<5) |
| `multi_zone` | −1.7166165966034157 | −1.0360226249043731 |
| `highlight_weighted` | −2.0575584783697494 | −2.0102137988508129 |

Larger fixture, `H=97, W=151`, `np.random.default_rng(20240101).random((97,151,3))`:

| method | sRGB, decode=True | ProPhoto RGB, decode=False |
| --- | --- | --- |
| `average` | −0.75451641490033017 | −1.4414600444511969 |
| `median` | −0.46089524506216073 | −1.4404191267905064 |
| `center_weighted` | −0.75904103042006721 | −1.443065323164586 |
| `partial` | −0.75610394986700957 | −1.4381580828695224 |
| `matrix` | −0.76063627899561381 | −1.4433201459195688 |
| `multi_zone` | −0.7391197802621795 | −1.4190427902603511 |
| `highlight_weighted` | −1.6270221355736465 | −1.8540898636569068 |

Transposing to 151×97 reproduces every value to ≤3e-16 (the masks are transpose-symmetric), which
is a cheap check that the axis order is right.

---

## 2. The resampling core

Three call sites use `skimage.transform.resize`/`rescale`, at spline orders 0, 1 and 3. They all
reduce to the same pipeline, and it is worth implementing exactly once.

```
skimage.resize(image, out_shape, order=o, mode='reflect', clip=True,
               preserve_range=p, anti_aliasing=aa)
  = 1. optional per-axis Gaussian prefilter  (scipy.ndimage.gaussian_filter, mode='mirror')
    2. optional cubic B-spline prefilter      (order >= 2 only)
    3. per-axis grid-mode zoom               (scipy.ndimage.zoom, mode='mirror', grid_mode=True)
    4. global clip to [min(input), max(input)]
```

skimage's `mode='reflect'` is translated to scipy's `mode='mirror'`
(`skimage/_shared/utils.py:_to_ndimage_mode`). These names are swapped relative to each other; get
this wrong and every edge pixel is off.

A from-scratch NumPy reimplementation of steps 1–4 matches skimage on random images to:

| call site | max abs diff |
| --- | --- |
| `resize_for_preview` (order 1, AA) | 3.3e-16 |
| `small_preview` (order 0, AA) | 2.2e-16 |
| `rescale` order 3 upscale (no AA) | 1.4e-15 |
| `rescale` order 3 downscale (AA) | 1.3e-15 |

### 2.1 Mirror index folding

scipy `'mirror'` reflects about the edge *samples*, period `2(n−1)`:

```
fold(i, n):
    if n == 1: return 0
    p = 2*(n - 1)
    i = abs(i) % p
    return (i > n - 1) ? p - i : i
```

`… d c b | a b c d | c b a …`. Used identically by the Gaussian correlation, the spline prefilter
and the zoom sampling.

### 2.2 Gaussian anti-alias prefilter

`skimage.resize` computes, with `factors[k] = in_shape[k] / out_shape[k]`:

```
sigma[k] = max(0, (factors[k] - 1) / 2)
```

then calls `scipy.ndimage.gaussian_filter(image, sigma, mode='mirror', cval=0)` with the default
`truncate=4.0`.

- Axes with `sigma <= 1e-15` are **skipped entirely** (`scipy.ndimage._filters.gaussian_filter`
  filters `sigmas[ii] > 1e-15`). The channel axis always has `factor = 1` → `sigma = 0` → skipped,
  so channels never mix. Verified: `sigma=0` and `sigma=1e-20` give bit-identical output.
- Per-axis kernel: `radius = int(truncate*sigma + 0.5)`, `x = -radius … radius`,
  `k = exp(-0.5/sigma^2 * x^2)`, `k /= sum(k)`.
- Applied as a **correlation**: `out[i] = Σ_j k[j] * in[fold(i + j - radius, n)]`. The kernel is
  symmetric so correlate == convolve, but the index arithmetic above is the one to write.
- Axes are filtered sequentially, axis 0 then axis 1, each pass reading the previous pass's output.

Sample kernels:

```
sigma=0.5    radius=2   [2.638650827373541e-04, 1.064507719735915e-01, 7.865707258873422e-01,
                         1.064507719735915e-01, 2.638650827373541e-04]
sigma=1.0    radius=4   [1.338306246147417e-04, 4.431861620031266e-03, 5.399112742070441e-02,
                         2.419714456566007e-01, 3.989434693560978e-01, 2.419714456566007e-01,
                         5.399112742070441e-02, 4.431861620031266e-03, 1.338306246147417e-04]
sigma=1.3125 radius=5   [2.145237613498289e-04, 2.923875980720189e-03, 2.230154945813083e-02,
                         9.519270626630860e-02, 2.273865933496215e-01, 3.039615023677381e-01,
                         2.273865933496215e-01, 9.519270626630860e-02, 2.230154945813083e-02,
                         2.923875980720189e-03, 2.145237613498289e-04]
```

Whether AA runs at all: `resize_for_preview` passes `anti_aliasing=True` explicitly. The two
`ResizingService` calls leave it `None`, and skimage then computes

```
anti_aliasing = (dtype != bool)
                and not (dtype is integer and order == 0)
                and any(out_shape[k] < in_shape[k] for k in axes)
```

The `any` runs over **all three axes including channels** (equal → not less). Consequences:

- `small_preview` (order 0, float64 input, downscaling): **AA is on.** The nearest-neighbour
  preview is Gaussian-blurred first. Easy to miss.
- `crop_and_rescale` with `upscale_factor > 1`: AA off.
- `crop_and_rescale` with `upscale_factor < 1`: AA on.

### 2.3 Grid-mode zoom geometry

Per axis, with `n` input samples and `m` output samples:

```
scale = n / m                       # exact integer ratio, recomputed inside scipy.ndimage.zoom
c(o)  = (o + 0.5) * scale - 0.5     # o = 0 … m-1
```

Verified against `ndi.zoom(..., grid_mode=True)` for n=4 → m=8 and m=3.

Sampling per order (scipy `ni_interpolation.c`; `start = floor(c) - order/2` for odd orders,
`floor(c + 0.5) - order/2` for even):

- **order 0**: `out[o] = in[fold(floor(c + 0.5), n)]`. Half rounds **up** (toward +inf), i.e.
  2.5 → 3. Confirmed: `n=8 → m=4` gives coords `[0.5, 2.5, 4.5, 6.5]` and values `[1, 3, 5, 7]`.
- **order 1**: `i0 = floor(c)`, `f = c - i0`,
  `out[o] = (1-f)*in[fold(i0,n)] + f*in[fold(i0+1,n)]`.
- **order 3**: `start = floor(c) - 1`, four taps `k = start … start+3`,
  `out[o] = Σ_k coeff[fold(k,n)] * B3(c - k)` over the **prefiltered** coefficients (§2.4), with

```
B3(t), t = |t| :  t < 1 → 2/3 - t^2 + t^3/2
                  t < 2 → (2 - t)^3 / 6
                  else  → 0
```

Axes are zoomed sequentially in axis order 0, 1, 2. The channel axis has `m == n == 3`, so its
coordinates are exactly `0, 1, 2`.

### 2.4 Cubic B-spline prefilter (order 3 only)

`scipy.ndimage.zoom(..., order=3, prefilter=True)` runs `spline_filter` along **every** axis
(including channels) before interpolating. For `mode='mirror'` there is no pre-padding
(`_prepad_for_spline_filter` returns `npad = 0`).

The filter is *exactly* the inverse of the B-spline synthesis operator under mirror folding.
Verified: building the `n×n` matrix `A[j][fold(j+off,n)] += B3(off)` for `off ∈ {−1,0,1}` and
comparing with scipy's operator gives `‖A·P − I‖∞ ≤ 2.3e-16` and `‖P − A⁻¹‖∞ ≤ 4.5e-16` for
`n ∈ {3, 4, 5, 9, 33}`.

So implement it as a tridiagonal solve, not as an IIR recursion with hand-derived boundary
initialisation (I tried the textbook Unser initialisation; it is wrong for small `n` — 6.0e-2 error
at `n=3`, 1.9e-3 at `n=5`, only converging to 1e-16 by `n=33`).

Build `A` with the generic accumulation — it needs no special cases, and it produces the right
thing for `n = 1` and `n = 2` as well (checked against scipy: `‖A·P − I‖∞` is 0.0 at `n = 2` and
`A = [[1.0]]` at `n = 1`, matching scipy's `P = [[1.0]]`):

```
A = zeros(n, n)
for j in 0..n-1:
    for (off, w) in [(-1, 1/6), (0, 4/6), (1, 1/6)]:
        A[j][fold(j + off, n)] += w
```

For `n >= 3` that expands to (rows are output samples):

```
row 0      : diag 4/6, super 2/6            # fold(-1) == 1, so the two 1/6 taps coalesce
row 1..n-2 : sub 1/6, diag 4/6, super 1/6
row n-1    : sub 2/6, diag 4/6
```

and for `n = 2` to `[[4/6, 2/6], [2/6, 4/6]]` (both rows fold).

Diagonally dominant (4/6 vs 2/6), so plain Thomas elimination in `Double` is stable and O(n). Solve
along axis 0, then axis 1, then axis 2.

The channel-axis pass is not a no-op in isolation, but because the channel zoom evaluates at exactly
`0, 1, 2` the prefilter/reconstruct pair round-trips: verified identity to 4.4e-16 on a 3-sample
axis. So an implementation that prefilters only the spatial axes and treats channels independently
is numerically equivalent **provided** step 4 (the clip) is still global — see below.

### 2.5 The global clip — the one that actually bites

`_clip_warp_output(image, out, mode, cval, clip=True)`:

```
min_val = min(input over the ENTIRE array, all channels)
max_val = max(input over the ENTIRE array, all channels)
clip(out, min_val, max_val, out=out)          # in place
```

(The `preserve_cval` branch only triggers for `mode='constant'`, which none of our call sites use.
NaN in the input switches to `nanmin`/`nanmax`.)

Bounds are computed over all three axes jointly, **not per channel**. With order 3 this is
material: on a 41×67×3 uniform-random image upscaled 2×, the unclipped range is
`[−0.2355, +1.2079]` against an input range of `[0.000314, 0.999765]`, and clipping per channel
instead of globally changes pixels by up to **3.75e-4** — above the 1e-4 gate. (`clip=False` makes
the two agree to 1.3e-15, which is how the cause was isolated.)

For order 0 and order 1 the operators are convex combinations of input samples, so the clip is a
no-op. Keep it anyway; it is one `min`/`max` pass.

### 2.6 Output-shape rules — the three call sites use three different roundings

| call site | shape rule |
| --- | --- |
| `resize_for_preview` | `Int(Double(h) * scale)` — **truncation** |
| `ResizingService.small_preview` | `max(rint(scale * n), 1)` — **round-half-to-even**, per axis, channel axis forced to input |
| `ResizingService.crop_and_rescale` | same as `small_preview` (both go through `rescale`) |

`rescale` computes `np.maximum(np.round(scale * orig_shape), 1)` over all axes then restores
`output_shape[-1] = orig_shape[-1]`. `np.round` is banker's rounding. Verified:
`h=3, scale=0.5 → 1.5 → 2`; `h=1, scale=0.5 → 0.5 → 0 → clamped to 1`;
`h=5, scale=0.25 → 1.25 → 1`.

`resize_for_preview`'s truncation is not cosmetic. For `max_size = 256`, **1389 of 11743** long-edge
sizes in `257…12000` produce a long edge of 255 rather than 256, because
`L * (256/L)` lands just below the integer. Examples: `L = 322, 347, 374` → 255;
`max_size=640, L = 1077, 1162, 1212` → 639. And `int()` vs `round()` on the short edge differ for
roughly a third of size pairs. Swift must write `Int(Double(h) * scale)` with the identical
`Double` arithmetic; anything tidier changes the shape.

Also: `resize` is called with a 2-tuple while the image is 3-D, so
`_preprocess_resize_output_shape` appends `image.shape[-1]`. Channel count is preserved and
`factors[2] == 1`.

---

## 3. `crop_image`

`utils/crop_resize.py:4`. `resize_image` in the same file is commented out — ignore it.

```python
def crop_image(image, center=(0.5, 0.5), size=(0.1, 0.1)):
    center = np.flip(center)                                  # (x,y) -> (row,col)
    shape  = image.shape[0:2]                                 # (H, W)
    cn = np.round(shape * np.array(center))                    # (round(H*cy), round(W*cx))
    sz = np.round(np.double(np.max(shape)) * np.flip(np.array(size)))   # (round(L*sy), round(L*sx))
    x0 = np.round(cn - sz/2)
    sz = np.int64(sz); x0 = np.int64(x0)
    x0[x0 < 0] = 0
    if x0[0] + sz[0] > shape[0]: x0[0] = shape[0] - sz[0]
    if x0[1] + sz[1] > shape[1]: x0[1] = shape[1] - sz[1]
    return image[x0[0]:x0[0]+sz[0], x0[1]:x0[1]+sz[1], :]
```

Semantics:

- `center` is `(x, y)` normalised to `[0, 1]`; `size` is `(sx, sy)`.
- **Both** crop extents are fractions of the **long edge** `L = max(H, W)`. `size=(0.1, 0.1)` on a
  6000×4000 frame is 600×600 pixels — square, not 600×400.
- All three roundings are `np.round`, i.e. **round-half-to-even**. Use `rint`, not `round`.
- The lower clamp runs first, then the fit clamp; the fit clamp is **not** re-clamped, so it can go
  negative again.

Worked examples (H, W, center, size → crop, with internals):

```
4000 6000 (0.5,0.5) (0.1,0.1)  -> (600, 600)    cn=[2000,3000] sz=[600,600]   x0=[1700,2700]
4000 6000 (0.5,0.5) (0.5,0.3)  -> (1800, 3000)  cn=[2000,3000] sz=[1800,3000] x0=[1100,1500]
4000 6000 (0.0,0.0) (0.2,0.2)  -> (1200, 1200)  cn=[0,0]       sz=[1200,1200] x0=[0,0]
4000 6000 (1.0,1.0) (0.2,0.2)  -> (1200, 1200)  cn=[4000,6000] sz=[1200,1200] x0=[2800,4800]
 100  100 (0.5,0.5) (0.05,0.05)-> (5, 5)        cn=[50,50]     sz=[5,5]       x0=[48,48]   # round(47.5)=48
```

Broken for `size` larger than `min(H,W)/L`:

```
4000 6000 (0.5,0.5) (1.0,1.0) -> (2000, 6000)   sz=[6000,6000] x0=[-2000, 0]
4000 6000 (0.5,0.5) (1.2,1.2) -> (3200, 1200)   sz=[7200,7200] x0=[-3200,-1200]
```

`x0 = −2000` becomes a Python negative slice start, i.e. index `H + x0`, so the "full frame" crop
silently returns the bottom half. **Recommended divergence:** the Swift port should throw
`CropError.sizeExceedsFrame` when `round(L*s) > H` or `> W`, and goldens should only cover valid
inputs. Document the divergence in the port's notes so nobody "fixes" the Swift side back to the
Python behaviour later.

---

## 4. `ResizingService`

`runtime/services/resize.py`. Constructed per pipeline with `(io_params, camera.film_format_mm)`.
Holds mutable `pixel_size_um`, initialised to `None`.

### 4.1 `pixel_size_um`

```python
self.pixel_size_um = self.film_format_mm * 1000 / np.max(image.shape[0:2])
```

`film_format_mm` default 35.0, so a 6000-px long edge gives 5.8333 µm/px, and an 800×1200 frame
gives 29.166666666666668 µm/px. Computed **before** the crop, from the pre-crop long edge.
Divided by `upscale_factor` when upscaling.

Verified: 800×1200, `crop=True, crop_size=(0.25,0.25), upscale_factor=2.0` →
shape (600, 600, 3), `pixel_size_um = 14.583333333333334`.

### 4.2 `crop_and_rescale`

```python
self.pixel_size_um = film_format_mm * 1000 / max(H, W)
if io.crop:
    image = crop_image(image, center=io.crop_center, size=io.crop_size)
if io.upscale_factor != 1.0:
    self.pixel_size_um /= io.upscale_factor
    image = rescale(image, io.upscale_factor, channel_axis=2, order=3)
return image
```

`rescale` defaults: `mode='reflect'` (→ scipy `'mirror'`), `clip=True`, `preserve_range=False`,
`anti_aliasing=None`. `preserve_range=False` routes through `img_as_float`, which is the identity
for float input — and the pipeline always hands it float64 — so it has no effect here. If the port
ever feeds integer data in, `img_as_float` would divide by the dtype max; don't.

The `!= 1.0` test is an exact float comparison. `upscale_factor = 1.0 + 1e-18` would still take the
rescale branch and produce a same-shape but spline-filtered image. Keep the exact comparison.

### 4.3 `small_preview`

```python
def small_preview(self, image, max_size=256):
    if max(image.shape[0:2]) > max_size:
        scale_factor = max_size / max(image.shape[0:2])
        return rescale(image, scale_factor, channel_axis=2, order=0)
    return image
```

`max_size` defaults to 256 and `FilmingStage.auto_exposure` never overrides it. Order 0 with
anti-aliasing auto-enabled (§2.2) — Gaussian blur, then nearest-neighbour pick. Shape by
`max(rint(scale * n), 1)`. Verified against the from-scratch implementation to 2.2e-16 and against
the real call: 800×1200 → **(171, 256, 3)**, float64.

### 4.4 `resize_for_preview`

`utils/preview.py`. Separate function, separate rules. Called by `runtime/process.py:
simulate_preview` (with `params.settings.preview_max_size`, default 640) and by the GUI
controller.

```python
def resize_for_preview(image, max_size):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale_factor = max_size / max(h, w)
        return resize(image, (int(h*scale_factor), int(w*scale_factor)),
                      preserve_range=True, anti_aliasing=True, order=1).astype(image.dtype)
    return image
```

- Truncating shape rule (§2.6).
- `preserve_range=True`: float32 and float64 pass through unchanged; other dtypes are cast to
  `float`.
- `anti_aliasing=True` explicitly.
- `.astype(image.dtype)` at the end. For a float16 input (half EXR, §8.1) `resize` upcasts to
  float32 internally and this casts back to float16 — a real precision loss that a Swift
  all-`Double` port will not reproduce. Feed the port `Double` and it is moot; do not add a
  float16 path.
- When `max(h,w) <= max_size` the **same array object** is returned, no copy and no dtype change.

Verified against the from-scratch implementation: 41×67 → 24 gives shape (14, 24, 3), max diff
3.3e-16.

---

## 5. Auto-exposure + resize, assembled

Golden-worthy end-to-end for `_preprocess`:

```
input 800x1200x3 float64 (gradient x noise, seed 5), io.input_color_space="ProPhoto RGB",
io.input_cctf_decoding=False, camera.auto_exposure=True, method="center_weighted"

small_preview(image)                 -> (171, 256, 3) float64
measure_autoexposure_ev(preview,...) -> -0.82234543664152648
image * 2**ev                        -> (800, 1200, 3) float64
crop_and_rescale(...)                -> (800, 1200, 3), pixel_size_um = 29.166666666666668
```

---

## 6. RAW import — `load_and_process_raw_file`

`utils/raw_file_processor.py`. Signature:

```python
load_and_process_raw_file(raw_path, white_balance='as_shot', temperature=None, tint=None,
                          lens_correction=False, output_colorspace='ACES2065-1',
                          output_cctf_encoding=False, lens_info_out=None) -> np.ndarray
```

Order of operations:

1. `rawpy.imread(path).postprocess(**params)` → `uint16 [H, W, 3]`;
   `.astype(np.float32) / np.float32(65535.0)`.
2. If `lens_correction`: read EXIF, look up lensfun, apply. (§6.5)
3. If a temperature-derived white balance was requested: chromatic adaptation. (§6.3)
4. Tint. (§6.4)
5. If `output_colorspace != 'ACES2065-1'`: `colour.RGB_to_RGB`. (§6.6)

### 6.1 `rawpy.postprocess` parameters

Explicitly set (`_postprocess_params`):

```python
{'output_color': rawpy.ColorSpace.ACES,   # LibRaw output_color = 6, AP0 primaries
 'output_bps': 16,
 'no_auto_bright': True,
 'gamma': (1, 1)}
```

plus `use_camera_wb: True` **only** for `'as_shot'`. The tests pin that `user_wb` is never passed
and that `use_camera_wb` is absent for every non-`as_shot` mode
(`tests/test_raw_file_processor.py:104-105, 151, 221-222`).

Everything else is a LibRaw default and part of the contract, whether the author intended it or
not. From `rawpy.Params()` in the pinned venv (rawpy 0.26.1, LibRaw 0.22.0):

```
demosaic_algorithm=None (LibRaw default AHD)   half_size=False        four_color_rgb=False
dcb_iterations=0        dcb_enhance=False      fbdd_noise_reduction=Off
noise_thr=None          median_filter_passes=0 use_auto_wb=False      user_wb=None
user_flip=None (apply EXIF orientation)        user_black=None        user_cblack=None
user_sat=None           auto_bright_thr=None   adjust_maximum_thr=0.75
bright=1.0              highlight_mode=Clip    exp_shift=None
exp_preserve_highlights=0.0                    no_auto_scale=False
chromatic_aberration=None                      bad_pixels_path=None
```

The load-bearing ones for a port: `adjust_maximum_thr=0.75`, `no_auto_scale=False` (LibRaw applies
its `scale_colors()` white-level normalisation and the WB multipliers), `highlight_mode=Clip`, and
`user_flip=None` so the output is already rotated upright per EXIF.

Because `'daylight'` passes no WB flag at all, LibRaw falls back to the camera's
`daylight_whitebalance` (the per-camera daylight preset baked into LibRaw's tables), **not** to
unity multipliers. That is the base the temperature-derived modes adapt from.

### 6.2 White-balance modes

```python
_TUNGSTEN_TEMPERATURE = 2850.0
_DAYLIGHT_REFERENCE_TEMPERATURE = 6504.0
```

| `white_balance` | behaviour |
| --- | --- |
| `'as_shot'` | `use_camera_wb=True`. No post-adaptation, no tint. |
| `'daylight'` | LibRaw daylight base, nothing else. |
| `'tungsten'` | adapt from 2850 K to 6504 K; `tint` forced to 1.0 |
| `'custom'` | requires `temperature`; adapt from it to 6504 K; `tint` used as given. Raises `ValueError('A custom raw white balance requires a temperature value.')` if `temperature is None`. |
| `(T, tint)` tuple | same as `'custom'` with those values (the `else` branch unpacks the argument). |

The adaptation is skipped entirely when `np.allclose(reference_white_xyz, scene_white_xyz)`
(default `rtol=1e-5, atol=1e-8`). `temperature=6504.0` therefore reproduces `'daylight'` bit-for-bit,
which `tests/test_raw_smoke.py:71` asserts at `atol=1e-6`.

Note the direction: the **source** white is the requested temperature and the **target** is the
6504 K reference. Setting a warm temperature makes the image warmer, not cooler. The composed
matrix for 2850 K maps ACES neutral `(1,1,1)` to `(1.1307, 1.2949, 2.9972)` — a large blue lift.
That is the reference behaviour; it is not a camera-style white balance.

### 6.3 Temperature → adaptation matrix

```python
def _whitepoint_xyz_from_temperature(T):
    method = 'CIE Illuminant D Series' if T >= 4000.0 else 'Kang 2002'
    xy = colour.CCT_to_xy(np.float64(T), method=method)
    return np.asarray(colour.xy_to_XYZ(xy), dtype=np.float64)
```

`xy_to_XYZ(xy)` at `Y = 1` is `XYZ = (x/y, 1, (1 - x - y)/y)`.

**CIE Illuminant D Series** (`T >= 4000`):

```
T <= 7000:  x = -4.607e9/T^3 + 2.9678e6/T^2 + 0.09911e3/T + 0.244063
T >  7000:  x = -2.0064e9/T^3 + 1.9018e6/T^2 + 0.24748e3/T + 0.23704
y = -3.000*x^2 + 2.870*x - 0.275
```

**Kang 2002** (`T < 4000`):

```
T <= 4000:  x = -0.2661239e9/T^3 - 0.2343589e6/T^2 + 0.8776956e3/T + 0.179910
T >  4000:  x = -3.0258469e9/T^3 + 2.1070379e6/T^2 + 0.2226347e3/T + 0.24039
T <= 2222:            y = -1.1063814*x^3 - 1.34811020*x^2 + 2.18555832*x - 0.20219683
2222 < T <= 4000:     y = -0.9549476*x^3 - 1.37418593*x^2 + 2.09137015*x - 0.16748867
T > 4000:             y =  3.0817580*x^3 - 5.8733867*x^2 + 3.75112997*x - 0.37001483
```

(The `T > 4000` branches inside Kang are unreachable given the `>= 4000` dispatch, but keep the
piecewise structure verbatim so `T = 3999.9999` cannot land somewhere different.)

**There is a discontinuity at exactly 4000 K** where the dispatch switches methods:

```
T = 3999.9  ->  xy = (0.3805327242685768, 0.3767363321504229)
T = 4000.0  ->  xy = (0.38234362499999996, 0.3837662610155782)
```

That is a 0.007 jump in `y`. It is in the reference and must be reproduced. Any golden fixture
should straddle it.

Reference whitepoints:

```
T = 2850.0 (tungsten)   xy = (0.4475242509058118,  0.4076398122955084)
                        XYZ = (1.0978423534877653, 1.0, 0.35530370790595045)
T = 6504.0 (reference)  xy = (0.31271405688264753, 0.3291190991371872)
                        XYZ = (0.9501546938553648, 1.0, 1.0882590676722474)
T = 2000.0              XYZ = (1.2749754623088834, 1.0, 0.14478009129025976)
T = 2500.0              XYZ = (1.151657913725796,  1.0, 0.2654612682196285)
T = 3200.0              XYZ = (1.0603995725469662, 1.0, 0.44537821752140255)
T = 5000.0              XYZ = (0.9639632771097949, 1.0, 0.8241448070045999)
T = 5500.0              XYZ = (0.9565428301805666, 1.0, 0.9202635863189644)
T = 7500.0              XYZ = (0.9494202019722441, 1.0, 1.2249299693876758)
T = 10000.0             XYZ = (0.9549004098096291, 1.0, 1.4701418420789278)
T = 25000.0             XYZ = (0.9805894654203436, 1.0, 1.9440655690663686)
```

`_apply_white_balance_adaptation` then:

```python
src = src / src[1] ; tgt = tgt / tgt[1]                 # already 1.0, no-op in practice
xyz = colour.RGB_to_XYZ(rgb, ACES2065-1, chromatic_adaptation_transform=None,
                        apply_cctf_decoding=False)
xyz = colour.chromatic_adaptation(xyz, src, tgt, method='Von Kries')   # transform defaults to CAT02
return colour.XYZ_to_RGB(xyz, ACES2065-1, chromatic_adaptation_transform=None,
                         apply_cctf_encoding=False).astype(np.float32)
```

Von Kries with CAT02:

```
RGB_w  = M_cat · XYZ_w
RGB_wr = M_cat · XYZ_wr
M_CAT  = M_cat^-1 · diag(RGB_wr / RGB_w) · M_cat
```

`M_cat` (CAT02) is already in `ColourTables.cat02`:

```
 0.7328  0.4296 -0.1624
-0.7036  1.6975  0.0061
 0.0030  0.0136  0.9834
```

The whole chain collapses to one 3×3: `M = M_XYZ→ACES · M_CAT · M_ACES→XYZ`. Precompute per
temperature. Reference matrices (rows top to bottom):

```
T = 2850 K -> 6504 K
  8.17776821287017897e-01 -1.08104022060625540e-01  4.21001736708105534e-01
 -1.93354841590416132e-02  1.11116275292896471e+00  2.03030730008995824e-01
  1.65309273757869857e-02  1.93609781704005418e-02  2.96131911506597367e+00
T = 3200 K -> 6504 K
  8.45268310728457428e-01 -8.70369300249490413e-02  3.03552382526468822e-01
 -1.56526434234307982e-02  1.08062272471886889e+00  1.43606372649417002e-01
  1.18973140120950480e-02  1.37396559362265320e-02  2.38571474792138183e+00
T = 5500 K -> 6504 K
  9.68685649955430272e-01 -1.33261289091053895e-02  4.15521818029199949e-02
 -2.40841181280809042e-03  1.00460130071041842e+00  1.92254268706809686e-02
  1.62518152980918726e-03  1.84659200305400777e-03  1.17875982230706033e+00
T = 9000 K -> 6504 K
  1.05017662837406522e+00  1.84006883900578941e-02 -5.19534645875594572e-02
  3.33857939169037365e-03  1.00071562354650956e+00 -2.35055114784713699e-02
 -2.02780948278741095e-03 -2.26673731394056582e-03  7.88492235738278979e-01
```

ACES2065-1 matrices (also in `ColourTables`):

```
RGB -> XYZ                                   XYZ -> RGB
0.9525523959  0.0           9.36786e-05      1.0498110175  0.0          -9.74845e-05
0.3439664498  0.7281660966 -0.0721325464    -0.4959030231  1.3733130458  0.0982400361
0.0           0.0           1.0088251844     0.0           0.0           0.9912520182
```

The result is cast back to `float32` after the float64 adaptation.

### 6.4 Tint

```python
if tint is None or np.isclose(tint, 1.0): return rgb
return (rgb * np.array([1.0, tint, 1.0], dtype=np.float32)).astype(np.float32)
```

A plain multiply of the **ACES G channel** in float32. The docstring says "both green channels",
which is a leftover from camera-native G1/G2 multipliers; the code has one G. `np.isclose` default
tolerances (`rtol=1e-5, atol=1e-8`). `tint` is only ever set by the temperature-derived paths —
`'as_shot'` and `'daylight'` leave `tint_multiplier = None`.

### 6.5 Lens correction (`lens_correction=True`)

lensfunpy database lookup by EXIF make/model/lens, then `lensfunpy.Modifier` with
`ModifyFlags.ALL` (vignetting, TCA, distortion, geometry, scale), then per-channel
`scipy.ndimage.map_coordinates(..., order=1, mode="nearest")` over the subpixel distortion map.

`_find_lens_candidates` / `_select_lens_candidate` / `_lens_model_score` are a heuristic
best-match over the lensfun DB (exact normalised match, then compact-substring match, then shared
word count, then maker match, then focal-range fit, then aperture-range fit, then lensfun's own
`score`). Early-outs: empty `lens_model` → no DB query at all; no camera match → no correction; no
lens match → no correction.

**Recommendation: do not port.** There is no lensfun on iOS and no license-clean way to ship its
database, Apple's RAW pipeline already applies its own (non-disableable) maker-note lens
corrections, and the whole feature is opt-in and off by default. Keep `lensCorrection` out of the
Swift API surface for now and record it as deferred. If it comes back, it needs its own spec.

### 6.6 Output colourspace conversion

```python
if output_colorspace != 'ACES2065-1':
    rgb = colour.RGB_to_RGB(rgb, input_colourspace=ACES2065-1,
                            output_colourspace=colour.RGB_COLOURSPACES[output_colorspace],
                            apply_cctf_decoding=False, apply_cctf_encoding=output_cctf_encoding)
```

`chromatic_adaptation_transform` defaults to `'CAT02'`, so this **does** adapt from the ACES
whitepoint (0.32168, 0.33767) to the destination whitepoint. Exact string comparison against
`'ACES2065-1'`, so `'aces2065-1'` would take the conversion branch and hit an identity-ish CAT.

Composed `ACES2065-1 → X` matrices with CAT02:

```
-> ProPhoto RGB
  1.23938034178473000e+00 -1.63967822801400626e-01 -7.52333837983699266e-02
  3.61136186638134514e-03  1.08961364922170190e+00 -9.32657920819786601e-02
 -2.05967931567551873e-03 -2.25158834147137169e-03  1.00458557732885168e+00
-> sRGB
  2.52164942984330454e+00 -1.13688855422225910e+00 -3.84917593194445296e-01
 -2.75213551244026078e-01  1.36970515102632517e+00 -9.43924507765199206e-02
 -1.59250100904642922e-02 -1.47806368110799641e-01  1.16380581594243138e+00
-> Display P3
  2.02528732756913543e+00 -6.91962172308893786e-01 -3.33325155452011801e-01
 -1.82621506072665168e-01  1.28661600581793079e+00 -1.03994499686180839e-01
  8.58455250820089159e-03 -5.48162900550489066e-02  1.04623173759426935e+00
-> Adobe RGB (1998)
  1.72502401481613066e+00 -4.22887916254785723e-01 -3.02135661029031000e-01
 -2.75475166468154897e-01  1.36983077874929893e+00 -9.43442748254903624e-02
 -2.66674666837373418e-02 -8.53193513352806659e-02  1.11197970256945800e+00
-> ITU-R BT.2020
  1.49086870465700994e+00 -2.68712979082956105e-01 -2.22155725704625912e-01
 -7.92372106028326334e-02  1.17936858311110337e+00 -1.00131372460806445e-01
  2.77810076707935266e-03 -3.04336146315335489e-02  1.02765551391237042e+00
-> DCI-P3
  2.17531601279182008e+00 -8.29009858919383835e-01 -3.46306154081230555e-01
 -1.76608548687740230e-01  1.28106507366709232e+00 -1.04456524920948257e-01
  9.04622310120673304e-03 -6.72213424059809883e-02  1.05817511935224595e+00
```

**Dtype inconsistency, confirmed empirically:** the function returns `float32` when
`output_colorspace == 'ACES2065-1'` and `float64` otherwise, because `colour.RGB_to_RGB` upcasts
and never casts back. The docstring claims float32. The all-`Double` Swift port makes this moot,
but be aware when diffing against the oracle: the oracle's float32 branch has already rounded.

### 6.7 Golden values for the post-demosaic chain

Stubbing the demosaic with `uint16 [[[8192,16384,32768],[60000,1000,40000]]]` (shape 1×2×3), the
way `tests/test_raw_file_processor.py` does:

```
wb=as_shot   ocs=ACES2065-1   float32
  [[0.1250019,   0.2500038,    0.5000076  ],
   [0.9155413,   0.015259022,  0.61036086 ]]
wb=daylight  ocs=ACES2065-1   float32   (identical to as_shot with this stub)
wb=tungsten  ocs=ACES2065-1   float32
  [[0.28570133,  0.37689486,   1.4875889  ],
   [1.0040219,   0.12317483,   1.8229035  ]]
wb=custom T=3200 tint=0.9  ocs=ACES2065-1  float32
  [[0.23567909,  0.30600673,   1.1977978  ],
   [0.95782644,  0.08082928,   1.467249   ]]
wb=daylight  ocs=ProPhoto RGB   float64
  [[ 0.076315059581,  0.226225388371,  0.501480083527],
   [ 1.086282376955, -0.036992999706,  0.611239639254]]
wb=tungsten  ocs=sRGB           float64
  [[-0.280647979245,  0.297188793918,  1.671007335856],
   [ 1.690087519805, -0.279675557288,  2.087310628318]]
```

### 6.8 iOS: the native decoder, and where it must differ

There is no LibRaw and no rawpy. The native path is **`CIRAWFilter`** (iOS 15+,
`CoreImage`), which wraps Apple's RAW decoder, with `ImageIO`/`CGImageSource` for metadata.

To get something as close as possible to LibRaw's linear ACES output:

- `isGamutMappingEnabled = false`
- `boostAmount = 0`, `boostShadowAmount = 0` — Apple's default tone/shadow "boost" is the analogue
  of `no_auto_bright=False`; zeroing both is what `no_auto_bright=True` + `gamma=(1,1)` buys.
- `localToneMapAmount = 0`, `contrastAmount = 0`, `extendedDynamicRangeAmount = 0`
- `colorNoiseReductionAmount = 0`, `luminanceNoiseReductionAmount = 0`, `sharpnessAmount = 0`,
  `detailAmount = 0` — Apple applies NR and sharpening by default; LibRaw does not
  (`fbdd_noise_reduction=Off`, `median_filter_passes=0`).
- `exposure = 0`, `baselineExposure` left at the file default (there is no LibRaw equivalent of
  Apple's per-camera baseline; see the divergence list).
- White balance: `neutralTemperature` / `neutralTint` for a custom WB, or leave them at the
  filter's initial values for as-shot. `neutralChromaticity` takes an xy pair directly, which is
  the closest thing to the reference's whitepoint-driven model.
- Render with `CIContext.render(_:toBitmap:...)` / `createCGImage(...:colorSpace:)` into a
  **linear** working space. The repo already vendors
  `spektrafilm/data/icc/ellelstone/ACES-elle-V2-g10.icc` (1132 bytes, linear AP0), so
  `CGColorSpace(iccData:)` on that file gives a linear ACES2065-1 destination and the render lands
  directly in the reference's working space. Otherwise render to
  `CGColorSpace(name: .extendedLinearDisplayP3)` (or `.extendedLinearITUR_2020`) and apply the
  matrix to AP0 yourself — those spaces are the only guaranteed-available linear ones.

**Necessary divergences — do not try to gate these:**

| aspect | LibRaw (oracle) | Apple RAW |
| --- | --- | --- |
| demosaic | AHD | proprietary, undocumented, version-tagged (`decoderVersion`) |
| camera → XYZ matrix | Adobe DNG-derived tables compiled into LibRaw | Apple's own per-camera profiles |
| black/white level, highlight recovery | `adjust_maximum_thr=0.75`, `highlight_mode=Clip` | proprietary; `Clip` has no exact analogue |
| baseline exposure | none | per-camera `baselineExposure`, nonzero for most bodies |
| lens corrections | only via lensfun, opt-in | applied from maker notes, cannot be disabled |
| noise reduction | off | on by default, zeroable but the zero point is not LibRaw's |
| orientation | applied per EXIF (`user_flip=None`) | applied by default |

Expect several percent difference in the demosaiced linear RGB — far outside 1e-4. **The parity
gate must not include demosaic.** Structure it as:

1. The oracle writes committed 16-bit-linear-ACES fixtures (`.exr`/`.tif`) standing in for
   `raw.postprocess(...)` output.
2. Swift goldens cover steps 3–5 only (adaptation, tint, colourspace conversion) against those
   fixtures.
3. A separate, non-gating visual smoke test runs the real `CIRAWFilter` on a checked-in RAW and
   asserts only bounds/finiteness/shape, the way `tests/test_raw_smoke.py` does.

Put the decoder behind a protocol (`RawDecoding`) so the Android port (still in development; prior
art only, never a numeric source of truth) can supply its own. Android has no system decoder for
third-party RAW formats, so that side will likely have to vendor LibRaw through the NDK, which
means the Android backend *can* be bit-parity with the oracle while the iOS one cannot. The
protocol boundary is what lets both exist.

There is no RAW file in the reference repo (`find` for `.nef/.cr2/.arw/.dng/.raf` returns nothing),
so the reference tests all stub the reader. Follow the same pattern.

### 6.9 EXIF read (`_read_exif_metadata`)

Reads six tags via exiv2, returning `ExifData(make, model, lens_make, lens_model, focal_length,
f_number)`. Any exception → all-empty struct (`""` for strings, `0.0` for floats). Missing key →
same default. Strings are `str(value()).strip()`; floats are `toFloat()`.

| field | EXIF key |
| --- | --- |
| `make` | `Exif.Image.Make` |
| `model` | `Exif.Image.Model` |
| `lens_make` | `Exif.Photo.LensMake` |
| `lens_model` | `Exif.Photo.LensModel` |
| `focal_length` | `Exif.Photo.FocalLength` |
| `f_number` | `Exif.Photo.FNumber` |

Only consumed by the lens-correction path. On iOS these come from
`CGImageSourceCopyPropertiesAtIndex` → `kCGImagePropertyTIFFMake/Model` and
`kCGImagePropertyExifLensMake/LensModel/FocalLength/FNumber`. If §6.5 is deferred, this struct is
only needed for display.

---

## 7. Image I/O — `utils/io.py`

### 7.1 `load_image_oiio(filename)`

Picks a read type from the file's native format, reads, reshapes to
`(spec.height, spec.width, spec.nchannels)`, and normalises **only** the integer formats:

| file format | read as | returned dtype | scaling |
| --- | --- | --- | --- |
| uint8 | uint8 | float64 | `/ 255` |
| uint16 | uint16 | float64 | `/ 65535` |
| half | half | **float16** | none |
| float | float | **float32** | none |
| anything else | uint16 | float64 | `/ 65535` |

Verified by round-tripping an 5×7×3 image:

```
PNG      float64  range (0.000000, 0.996078)  err 3.9e-03
JPEG     float64  range (0.184314, 1.000000)  err 6.3e-01   (lossy, as expected)
TIFF  8  float64  err 3.9e-03
TIFF 16  float64  err 1.5e-05
TIFF 32  float32  err 3.0e-08
EXR half float16  err 2.4e-04
EXR 32   float32  err 3.0e-08
```

Callers slice `[..., :3]` (GUI) or `[:, :, 0:3]` (`_preprocess`), so alpha is dropped downstream,
not here.

On iOS: `CGImageSourceCreateWithURL` + `CGImageSourceCreateImageAtIndex`, or `vImage` for the
float formats. No OpenEXR in the system frameworks — EXR read/write needs either a vendored
decoder (against the no-third-party rule) or dropping EXR support. **Recommendation:** support
PNG/JPEG/TIFF via ImageIO on iOS and treat EXR as a desktop/macOS-only or later concern. Note
that ImageIO's 16-bit TIFF read gives you the integers; apply `/65535` yourself in `Double` to match.

### 7.2 `save_image_oiio(filename, image_data, bit_depth=16, *, color_space=None, cctf_encoding=True)`

Format is chosen by extension, `bit_depth` only matters for TIFF and EXR:

| ext | encoding | clip | scale | `bit_depth` |
| --- | --- | --- | --- | --- |
| `png` | uint8 | `[0,1]` | ×255 | ignored |
| `jpg`/`jpeg` | uint8 | `[0,1]` | ×255 | ignored |
| `tif`/`tiff` 8 | uint8 | `[0,1]` | ×255 | |
| `tif`/`tiff` 16 | uint16 | `[0,1]` | ×65535 | |
| `tif`/`tiff` 32 | float32 | no | no | |
| `exr` 16 | half | no | no | |
| `exr` 32 | float32 | no | no | |
| anything else | `ValueError` | | | |

Integer conversion is `(np.clip(x, 0, 1) * SCALE).astype(np.uint8/uint16)` — a **truncating** cast,
not a round. `0.5 * 255 = 127.5 → 127`. TIFF gets `spec.attribute("Compression", "zip")`
(deflate; lossless at every depth, and LZW is integer-only).

When `color_space` is set and the extension is not `exr`, the matching ICC profile bytes (§7.4) are
attached as `spec.attribute("ICCProfile", uint8[N], bytes)`, which OIIO routes to the JPEG APP2
marker, the PNG `iCCP` chunk or the TIFF `ICCProfile` tag. Missing profile → silently no embedding.

### 7.3 Metadata — `read_image_metadata` / `write_image_metadata`

`read_image_metadata(filename) -> ImageMetadata | None`: exiv2 open + `readMetadata()`, returning
a frozen `(exif, iptc, xmp)` triple. Any exception → `None`
(`tests/test_exif_metadata.py:34` pins this for a missing file).

`write_image_metadata(filename, source_metadata=None, *, saving_color_space=None,
saving_cctf_encoding=True)`:

1. **Early return for `.exr`** — EXR carries its own metadata, exiv2 is not used.
2. Read the just-written file with OIIO only to get `spec.width` / `spec.height`.
3. Open the destination with exiv2, `readMetadata()`.
4. If `source_metadata`: wholesale `setExifData` / `setIptcData` / `setXmpData` from the source.
5. Overwrite:
   - `Exif.Image.Orientation = 1` (pixels are already upright)
   - `Exif.Image.DateTime = now("%Y:%m:%d %H:%M:%S")`
   - `Exif.Image.Software = "spektrafilm"`
   - `Exif.Photo.PixelXDimension = spec.width`
   - `Exif.Photo.PixelYDimension = spec.height`
6. If `saving_color_space`:

| condition | `Exif.Photo.ColorSpace` | `Exif.Iop.InteroperabilityIndex` |
| --- | --- | --- |
| `sRGB` + encoded | `1` | `"R98"` |
| `Adobe RGB (1998)` + encoded | `65535` | `"R03"` |
| everything else | `65535` | not written |

   and `Xmp.photoshop.ICCProfile = name` or `f"{name} (linear)"` when not encoded.
7. `writeMetadata()`.

Pinned by `tests/test_exif_metadata.py:91-127`. The interop index is only *added*, never cleared,
so copying an `R98` source into a Display P3 output leaves a stale `R98` behind. Reproduce or
document; the Swift port should probably clear it in the `else` branch and note the divergence.

On iOS: `CGImageDestination` with a properties dictionary
(`kCGImagePropertyExifDictionary`, `kCGImagePropertyTIFFDictionary`, `kCGImagePropertyIPTCDictionary`,
`kCGImageDestinationMetadata` + `CGImageMetadata` for XMP). ImageIO will not faithfully round-trip
arbitrary MakerNote blobs the way exiv2's `setExifData` does — an accepted divergence, and one
worth surfacing in the UI if the app advertises "preserves your camera metadata".

### 7.4 ICC profile table

`_ICC_FILENAMES: dict[(color_space, cctf_encoded), path]`, resolved under
`spektrafilm/data/icc/`. Missing entry or unreadable file → `None` → no embedding.

| color space | encoded | linear |
| --- | --- | --- |
| sRGB | `ellelstone/sRGB-elle-V2-srgbtrc.icc` (9552 B) | `ellelstone/sRGB-elle-V2-g10.icc` (1384 B) |
| Adobe RGB (1998) | `ellelstone/ClayRGB-elle-V2-g22.icc` (1276 B) | `ellelstone/ClayRGB-elle-V2-g10.icc` (1276 B) |
| ProPhoto RGB | `ellelstone/LargeRGB-elle-V2-g18.icc` (1276 B) | `ellelstone/LargeRGB-elle-V2-g10.icc` (1276 B) |
| ITU-R BT.2020 | `ellelstone/Rec2020-elle-V2-rec709.icc` (9540 B) | `ellelstone/Rec2020-elle-V2-g10.icc` (1384 B) |
| ACES2065-1 | `ellelstone/ACES-elle-V2-g10.icc` (1132 B) | same file |
| Display P3 | `saucecontrol/DisplayP3-v2-micro.icc` (456 B) | — (falls through) |
| DCI-P3 | `saucecontrol/DCI-P3-v4.icc` (464 B) | — (falls through) |

11 distinct files, **29 016 bytes total**. The vendored `data/icc/` tree is 904 KB because it
carries the full upstream sets; ship only these 11. Licensing: ellelstone is CC BY-SA 3.0
(attribution required), saucecontrol is MIT. `docs/LICENSING.md` in the port repo needs both.

`ACES-elle-V2-g10.icc` doubles as the linear-AP0 `CGColorSpace` for the RAW render target (§6.8),
which is a good reason to bundle it even if ACES export is never exposed.

### 7.5 Neutral print filters and spectral filter loaders

`utils/io.py` also holds four functions that belong to other subsystems and are listed here only so
nobody ports them twice:

- `save_neutral_print_filters` / `read_neutral_print_filters` — JSON round-trip of
  `spektrafilm/data/filters/neutral_print_filters.json`. Already vendored at
  `Sources/SpektraFilm/Resources/filters/neutral_print_filters.json`. Owner: enlarger/print
  subsystem.
- `load_dichroic_filters(wavelengths, brand='thorlabs')` → `[wavelength, 3]` (c, m, y), CSV columns
  `(nm, percent)`, de-duplicated on the wavelength column via `np.unique(..., return_index=True)`,
  resampled with `scipy.interpolate.Akima1DInterpolator`, divided by 100.
- `load_filter(wavelengths, name='KG3', brand='schott', filter_type='heat_absorbing',
  percent_transmittance=False)` → `[wavelength]`, same de-dup + Akima, divided by 100 only when
  `percent_transmittance=True`.

Both resamplers are **Akima 1-D**, not cubic spline (the `CubicSpline` calls are commented out
directly above). Akima is not in Accelerate and must be written by hand. Owner: whichever
subsystem spec covers the enlarger filter stack; flagged here because the functions physically live
in `io.py`.

### 7.6 Dtype summary for the import boundary

```
load_image_oiio      -> float64 | float32 | float16 depending on file format
load_and_process_raw -> float32 (ACES output) | float64 (converted output)
_preprocess          -> np.double(...)  i.e. everything becomes float64 before the engine
```

The Swift port should normalise at the boundary: every importer returns `ImageBuffer` (`Double`),
and no engine code ever sees a narrower type.

---

## 8. Do not port: `utils/measure.py`

`measure_gamma`, `measure_slopes_at_exposure`, `measure_density_min`. A repo-wide grep (excluding
`.venv` and `.git`) finds **no callers anywhere** — not in `src/`, `tests/`, `scripts/` or
`examples/`. They are profile-authoring / research tools that survived in the package.

They would cost the most per line of anything in this subsystem:
`scipy.interpolate.interp1d(kind='cubic')`, `CubicSpline`, `scipy.optimize.least_squares` with
bounds (a trust-region reflective solver), and `scipy.special.erf`. Writing a bounded TRF
least-squares in Swift to fit a three-parameter erf toe, for code nothing calls, is not a good
trade.

If the iOS app ever grows a profile editor, revisit — and then spec it separately, because the
density-curve subsystem owns `log_exposure` / `density_curves` anyway.

---

## 9. Goldens

Generated by `Tools/parity/generate_goldens.py` into `Tests/SpektraFilmTests/Goldens/`
(existing repo convention, per `Tools/parity/upstream_pin.json`). All at max_abs ≤ 1e-4,
rms ≤ 1e-5. Nothing in this subsystem is stochastic, so no statistical gating is needed.

Shape-only assertions (exact integer equality, no tolerance) are as important as the pixel
comparisons here — §2.6 is the single most likely source of a silent off-by-one.

| fixture | oracle call | gates |
| --- | --- | --- |
| `autoexposure_ev_matrix.json` | `measure_autoexposure_ev` × 7 methods × {sRGB+decode, ProPhoto+no-decode} × {97×151, 151×97 transpose, 4×6, 8×8 zeros, 3×4} | every method, the transpose symmetry, the `H<5` and all-zero guards |
| `autoexposure_masks.npy` | the `center_weighted` mask and the `radius` map for 4×6, 97×151, 171×256 | the half-pixel centre offset and the asymmetric coordinate range |
| `autoexposure_negative_y.json` | `measure_autoexposure_ev(np.full((16,16,3), -0.05), 'ACES2065-1', False, m)` for 3 methods | that NaN propagates rather than being swallowed |
| `resample_gaussian_kernels.json` | `_gaussian_kernel1d(sigma, 0, int(4*sigma+0.5))` for sigma ∈ {0.5, 1.0, 1.3125, 2.34375, 4.7} | radius rule + normalisation |
| `resample_zoom_1d.npy` | `ndi.zoom(a, m/n, order=o, mode='mirror', grid_mode=True)` for a = arange(n), (n,m) ∈ {(4,8),(4,3),(8,4),(8,5),(5,11),(3,7),(37,74)}, o ∈ {0,1,3} | the coordinate formula, the order-0 half-up rounding, mirror folding, the order-3 4-tap stencil |
| `resample_spline_prefilter.npy` | `spline_filter1d(s, order=3, axis=0, mode='mirror')` for random `s` of length 3, 4, 5, 9, 33 | the tridiagonal solve, especially the short-axis boundary rows |
| `resize_for_preview.npy` | `resize_for_preview(img, ms)` for img ∈ {41×67×3, 97×151×3, 800×1200×3 seeds fixed}, ms ∈ {24, 256, 640} plus one no-op case where `max(h,w) <= ms` | order-1 + AA path, the truncating shape rule, the identity short-circuit |
| `resize_for_preview_shapes.json` | just the output shapes for every `(h, w, max_size)` in a 200-row table chosen to include the `max_size - 1` truncation cases (322/256, 347/256, 1077/640, …) | proves the port did not "clean up" the truncation |
| `small_preview.npy` | `ResizingService.small_preview(img)` for the same images, `max_size=256`, plus `h=1,w=512` and `h=3,w=512` | order-0 + AA path, `rint` half-to-even, the `max(…, 1)` clamp |
| `crop_and_rescale.npy` | `crop_and_rescale` over the matrix of `crop ∈ {False, True}` × `crop_size ∈ {(0.1,0.1),(0.5,0.3)}` × `upscale_factor ∈ {1.0, 0.61, 2.0}`, recording output **and** `pixel_size_um` | order-3 spline path both directions, the global clip, the pre-crop `pixel_size_um` rule |
| `crop_image.json` | `crop_image` internals (`cn`, `sz`, `x0`, output shape) for the §3 table, valid cases only | banker's rounding, long-edge-for-both-axes, the clamp order |
| `preprocess_end_to_end.npy` | `SimulationPipeline._preprocess` on an 800×1200×3 fixture with auto-exposure on, for 3 methods × 2 crop configs | that metering-before-crop and the 256-px preview are wired the right way round |
| `cct_to_whitepoint.json` | `_whitepoint_xyz_from_temperature(T)` for T ∈ {1700, 2000, 2222, 2500, 2850, 3200, 3999.9, 4000, 4000.1, 5000, 5500, 6504, 7000, 7000.1, 7500, 9000, 10000, 25000} | both CCT methods, the 4000 K discontinuity, the 7000 K branch inside CIE D, the 2222 K branch inside Kang |
| `wb_adaptation_matrices.json` | the composed ACES-RGB 3×3 for each T above, plus the `np.allclose` skip at exactly 6504.0 | CAT02 Von Kries composition and the skip threshold |
| `raw_postprocess_chain.json` | `load_and_process_raw_file` with a stubbed reader, over `white_balance ∈ {as_shot, daylight, tungsten, custom(3200, 0.9), custom(6504, 1.0), (4500, 1.1)}` × `output_colorspace ∈ {ACES2065-1, ProPhoto RGB, sRGB}` × `output_cctf_encoding ∈ {False, True}`, recording dtype too | §6.2–6.6 end to end, and the float32/float64 split |
| `aces_to_rgb_matrices.json` | `matrix_RGB_to_RGB(ACES2065-1, X, 'CAT02')` for all seven colourspaces | §6.6 |
| `cctf_decode.json` | each colourspace's `cctf_decoding` over a grid including negatives, 0, the breakpoints ±1 ulp, 0.18, 1.0, and >1 | the numerically-evaluated sRGB threshold, ROMM's 0.03125, BT.2020's 0.081247944, and the Adobe/DCI NaN-on-negative behaviour |
| `image_io_roundtrip.json` | `save_image_oiio` → `load_image_oiio` for every (ext, bit_depth) pair, recording dtype, shape and max error | §7.1/§7.2 encoding rules, especially the truncating integer cast |

`cctf_decode.json` overlaps the colour subsystem; coordinate ownership rather than duplicating the
fixture.

---

## 10. Risks

1. **The `rescale`/`resize` global clip (§2.5).** The single most likely parity failure. Per-channel
   clipping is the natural Swift implementation and it is wrong by 3.75e-4.
2. **Three different shape-rounding rules across three call sites (§2.6).** Off-by-one output
   shapes will fail loudly if goldens assert shapes and silently corrupt everything downstream if
   they do not.
3. **Anti-aliasing is auto-enabled for the order-0 `small_preview` (§2.2).** Skipping the Gaussian
   changes the metered EV by up to 3e-3, which is a 0.2 % gain error on the whole frame.
4. **The half-pixel offset in `_normalized_coords` (§1.3).** Looks like a bug, is load-bearing.
5. **RAW demosaic cannot be gated (§6.8).** If someone writes a golden that includes
   `CIRAWFilter`, CI will be permanently red or permanently loose. The fixture boundary has to sit
   at "16-bit linear ACES from the demosaicer".
6. **Negative luminance → NaN EV poisons the frame (§1.5).** Real reference defect. Reproducing it
   is correct for parity and awful for users; needs a product decision, not a quiet fix.
7. **No EXR in the system frameworks (§7.1).** Either drop EXR on iOS or accept a vendored decoder
   against the no-third-party rule.
8. **`scipy.ndimage`'s `'mirror'` is skimage's `'reflect'`.** Trivially easy to invert, and it only
   shows up on edge pixels, which is exactly where a coarse golden tolerance hides it.
9. **`upscale_factor != 1.0` is an exact float comparison (§4.2).** A GUI slider emitting
   `0.9999999999` takes the spline path.
10. **`pixel_size_um` is nil before `crop_and_rescale` (§0.1).** Any tap-injection API has to
    handle it.

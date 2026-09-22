# Licensing

This repository is a derivative work of [spektrafilm](https://github.com/andreavolpato/spektrafilm)
by Andrea Volpato. Three licences apply to three kinds of content.

## Code: GPL-3.0-only

All Swift code in `Sources/`, `Tests/` and `App/` is GPL-3.0-only, inherited from upstream. The
engine ports upstream's `src/spektrafilm` package: the algorithms, the pipeline topology and the
numeric behaviour are upstream's, re-expressed in Swift. That makes it a derivative work, so it
carries upstream's licence. Full text: [LICENSE](../LICENSE).

In practice, anything that links this engine must also be GPL-3.0-compatible and ship its source. If
that does not work for you, upstream is open to discussing alternatives, and that conversation is
with Andrea.

## Film and paper profiles: CC BY-SA 4.0

`Sources/SpektraFilm/Resources/profiles/*.json` are upstream's 28 measured film and print-paper
profiles, redistributed verbatim under CC BY-SA 4.0. Each file carries its own `metadata.license` and
`metadata.citation`, preserved unmodified. Redistribution must credit Andrea Volpato, link the
upstream project, and stay CC BY-SA 4.0.

The profiles come from Kodak and Fujifilm datasheets plus published reflectance datasets (Otsu,
Munsell, NIST human skin, forest colors, Japan colors). Each profile's `metadata.datasource` records
its provenance.

## Spectral LUT: spektrafilm LUT licence

`Sources/SpektraFilm/Resources/luts/spectral_upsampling/irradiance_xy_tc.npy` is upstream's
Hanatos-2025 spectral-upsampling table. Its licence allows commercial use and free sharing, and
prohibits resale. Full text: [SPEKTRAFILM_LICENSE.txt](../SPEKTRAFILM_LICENSE.txt).

[hanatos](https://github.com/hanatos) contributed the underlying coefficient fit to upstream.

## Filter measurements: vendor data

`Sources/SpektraFilm/Resources/filters/` holds digitised transmission curves for Schott
heat-absorbing glass, dichroic enlarger filters (Durst, Edmund Optics, Thorlabs) and one Canon lens.
The measurements belong to the respective vendors. Upstream digitised them, and they are
redistributed here on the same terms as the profiles. The `info.txt` files record each source.

## Attribution

Upstream asks that user-visible credit read *"film modeling powered by spektrafilm"*. The app shows
this in its About screen. Academic use should cite the upstream repository or its Zenodo DOI. See
[`Tools/parity/upstream_CITATION.cff`](../Tools/parity/upstream_CITATION.cff), a verbatim copy of
upstream's citation metadata.

## What is not derived from upstream

The SwiftUI application layer, the Accelerate-backed numeric primitives, the golden-fixture parity
harness and the CI configuration are original to this repository. They are GPL-3.0-only anyway,
because they link the engine.

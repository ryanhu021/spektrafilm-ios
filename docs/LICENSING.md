# Licensing

This repository is a derivative work of [spektrafilm](https://github.com/andreavolpato/spektrafilm)
by Andrea Volpato. Three separate licenses apply to three separate kinds of content.

## Code — GPL-3.0-only

All Swift code in `Sources/`, `Tests/`, and `App/` is licensed **GPL-3.0-only**, inherited from
upstream. The engine is a port of upstream's `src/spektrafilm` package: the algorithms, the
pipeline topology, and the numeric behaviour are upstream's, re-expressed in Swift. That makes it a
derivative work, so it carries upstream's license. Full text: [LICENSE](../LICENSE).

Practical consequence: anything that links this engine must also be GPL-3.0-compatible and ship its
source. If that does not work for you, upstream is [open to discussing
alternatives](https://github.com/andreavolpato/spektrafilm#readme) — that conversation is with
Andrea, not with this repository.

## Film and paper profiles — CC BY-SA 4.0

`Sources/SpektraFilm/Resources/profiles/*.json` are upstream's 28 measured film and print-paper
profiles, redistributed verbatim under **CC BY-SA 4.0**. Each file carries its own `metadata.license`
and `metadata.citation` fields; those are preserved unmodified. Redistribution must credit Andrea
Volpato, link the upstream project, and stay CC BY-SA 4.0.

The profiles are derived from Kodak and Fujifilm datasheets plus published reflectance datasets
(Otsu, Munsell, NIST human skin, forest colors, Japan colors); provenance is recorded in each
profile's `metadata.datasource`.

## Spectral LUT — spektrafilm LUT license

`Sources/SpektraFilm/Resources/luts/spectral_upsampling/irradiance_xy_tc.npy` is upstream's
Hanatos-2025 spectral-upsampling table, covered by the custom spektrafilm LUT license:
commercial use allowed, free sharing allowed, **resale prohibited**. Full text:
[SPEKTRAFILM_LICENSE.txt](../SPEKTRAFILM_LICENSE.txt).

The underlying coefficient fit was contributed to upstream by [hanatos](https://github.com/hanatos).

## Filter measurements — vendor data

`Sources/SpektraFilm/Resources/filters/` holds digitised transmission curves for Schott heat-absorbing
glass, dichroic enlarger filters (Durst / Edmund Optics / Thorlabs), and one Canon lens. The
measurements are the respective vendors'; the digitisation is upstream's, redistributed here on the
same terms as the profiles. `info.txt` files record each source.

## Attribution

Per upstream's request, user-visible credit reads *"film modeling powered by spektrafilm"*. The app
shows this in its About screen, and academic use should cite the upstream repository or its Zenodo
DOI — see [`Tools/parity/upstream_CITATION.cff`](../Tools/parity/upstream_CITATION.cff), a verbatim
copy of upstream's citation metadata.

## What is *not* derived from upstream

The SwiftUI application layer, the Accelerate-backed numeric primitives, the golden-fixture parity
harness, and the CI configuration are original to this repository. They are GPL-3.0-only anyway,
because they link the engine.

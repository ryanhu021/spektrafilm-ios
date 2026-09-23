# spektrafilm for iOS

A Swift port of [spektrafilm](https://github.com/andreavolpato/spektrafilm), Andrea Volpato's
spectral simulation of analog photography, and an iOS app that runs it. A photo is exposed onto a
simulated film stock, printed through a colour enlarger onto paper, and scanned, all in spectral
space from published datasheet measurements.

The engine matches the Python reference end to end: every stage is tested against fixtures the
reference generates at a pinned commit, to `max_abs <= 1e-4`.

## Install

Add this source in [SideStore](https://sidestore.io) and install Spektrafilm from it. No paid
developer account is needed.

```
https://github.com/ryanhu021/spektrafilm-ios/releases/latest/download/sidestore-source.json
```

## Build

```sh
swift test                  # engine and parity tests, on macOS
brew install xcodegen
make run                    # the app, on a booted simulator
```

`make help` lists every target.

## Docs

- [Parity](docs/PARITY.md): how the port is checked against the reference, and what is not ported.
- [Performance](docs/PERFORMANCE.md): timings, memory, the Metal backend, and how to measure.
- [Releasing](docs/RELEASING.md): cutting a release and installing it.
- [Licensing](docs/LICENSING.md): the three licences that apply, and the required attribution.

## Licence

GPL-3.0-only, inherited from upstream. The bundled profiles are CC BY-SA 4.0 and the spectral LUT
carries upstream's own licence. Film modeling powered by `spektrafilm`.

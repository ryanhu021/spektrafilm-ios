# Releasing

Pushing a `v*` tag builds an unsigned `.ipa` for device and attaches it to a
[GitHub release](https://github.com/ryanhu021/spektrafilm-ios/releases), with a SideStore source
describing it:

```sh
git tag v0.2.0 && git push origin v0.2.0
```

The tag sets the version and the workflow run number sets the build number. Release notes are
generated from the commits since the previous tag.

Nothing is signed. [SideStore](https://sidestore.io) signs the app on the phone with the user's own
Apple ID, so no paid developer account is needed. A free Apple ID signs apps for 7 days, and
SideStore refreshes them in the background.

## Installing

In SideStore, add this source and install Spektrafilm from it. New releases appear as updates.

```
https://github.com/ryanhu021/spektrafilm-ios/releases/latest/download/sidestore-source.json
```

Or download `Spektrafilm.ipa` from a release in Safari and open it with **+** on SideStore's My Apps
tab.

`Tools/release/sidestore_source.py` writes the source from the built app's Info.plist, so the
version, bundle identifier and privacy strings cannot disagree with the `.ipa`.

#!/usr/bin/env python3
"""Writes a SideStore (AltStore-format) source describing one release of the app.

Reads the bundle identifier, versions and privacy strings from the built app's Info.plist, so the
source cannot disagree with the .ipa it points at.

    Tools/release/sidestore_source.py --app path/to/SpektrafilmApp.app --ipa Spektrafilm.ipa \\
        --download-url URL --notes "..." --out sidestore-source.json

SideStore fetches sources without credentials, so the source only works while the repository is
public. A private repository's release can still be installed by downloading the .ipa and opening it
in SideStore.
"""

import argparse
import datetime
import json
import plistlib
from pathlib import Path

REPO = "https://github.com/ryanhu021/spektrafilm-ios"
RAW = "https://raw.githubusercontent.com/ryanhu021/spektrafilm-ios/main"
ICON = f"{RAW}/App/Spektrafilm/Assets.xcassets/AppIcon.appiconset/icon.png"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True, type=Path)
    parser.add_argument("--ipa", required=True, type=Path)
    parser.add_argument("--download-url", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with open(args.app / "Info.plist", "rb") as f:
        info = plistlib.load(f)

    privacy = {key: value for key, value in info.items() if key.endswith("UsageDescription")}
    source = {
        "name": "Spektrafilm",
        "identifier": f"{info['CFBundleIdentifier']}.source",
        "sourceURL": f"{REPO}/releases/latest/download/sidestore-source.json",
        "website": REPO,
        "iconURL": ICON,
        "apps": [
            {
                "name": info.get("CFBundleDisplayName", "Spektrafilm"),
                "bundleIdentifier": info["CFBundleIdentifier"],
                "developerName": "Ryan Hu",
                "subtitle": "Spectral film simulation",
                "localizedDescription": (
                    "Exposes a photo onto a simulated film stock, prints it through a colour "
                    "enlarger onto paper, and scans the result, all in spectral space. A port of "
                    "spektrafilm by Andrea Volpato. Film modeling powered by spektrafilm."
                ),
                "iconURL": ICON,
                "tintColor": "FF8A3D",
                "category": "photo-video",
                "versions": [
                    {
                        "version": info["CFBundleShortVersionString"],
                        "buildVersion": info["CFBundleVersion"],
                        "date": datetime.datetime.now(datetime.timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%SZ"),
                        "localizedDescription": args.notes,
                        "downloadURL": args.download_url,
                        "size": args.ipa.stat().st_size,
                        "minOSVersion": info.get("MinimumOSVersion", "17.0"),
                    }
                ],
                "appPermissions": {"entitlements": [], "privacy": privacy},
            }
        ],
        "news": [],
    }
    args.out.write_text(json.dumps(source, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

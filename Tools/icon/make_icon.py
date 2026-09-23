#!/usr/bin/env python3
"""Renders the app icon: a print coming up under the safelight, in the app's own palette.

Needs NumPy and Pillow. Writes App/Spektrafilm/Assets.xcassets/AppIcon.appiconset/icon.png.

    python3 Tools/icon/make_icon.py
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "App/Spektrafilm/Assets.xcassets/AppIcon.appiconset/icon.png"
S = 2048  # rendered at twice the size, then downsampled


def rgb(r, g, b):
    return np.array([r, g, b])


# Safelight.swift
INK, EASEL = rgb(.043, .035, .031), rgb(.118, .098, .090)
PAPER, AMBER = rgb(.949, .918, .878), rgb(1.0, .541, .239)
CYAN, MAGENTA, YELLOW = rgb(.247, .722, .769), rgb(.831, .318, .608), rgb(.910, .773, .278)


def main():
    y, x = np.mgrid[0:S, 0:S] / S

    # The room, dim and warm, lit from the safelight in the corner.
    d = np.hypot(x - 0.17, y - 0.15)
    img = INK + (EASEL - INK) * np.clip(1 - d, 0, 1)[..., None] ** 1.5
    img = img + AMBER * (np.exp(-(d / 0.2) ** 2) * 0.35)[..., None]
    core = np.clip(1 - d / 0.045, 0, 1) ** 0.5
    img = img * (1 - core[..., None]) + AMBER * core[..., None]

    # The print and its shadow.
    x0, x1, y0, y1 = 0.25, 0.79, 0.24, 0.84
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    mask = Image.fromarray((inside * 255).astype(np.uint8))
    shadow = np.asarray(mask.filter(ImageFilter.GaussianBlur(S * 0.025))) / 255.0
    shadow = np.roll(np.roll(shadow, int(S * 0.018), 0), int(S * 0.01), 1)
    img = img * (1 - 0.7 * shadow[..., None])
    img = np.where(inside[..., None], PAPER, img)

    # A sunset inside the border, still flat toward the bottom as it develops.
    m = 0.035
    ix0, ix1, iy0, iy1 = x0 + m, x1 - m, y0 + m, y1 - 0.11
    area = (x >= ix0) & (x <= ix1) & (y >= iy0) & (y <= iy1)
    u = np.clip((x - ix0) / (ix1 - ix0), 0, 1)
    v = np.clip((y - iy0) / (iy1 - iy0), 0, 1)
    sky = rgb(.98, .80, .55) * (1 - v[..., None]) ** 0.8 + rgb(.93, .45, .22) * v[..., None] ** 1.2
    sun = np.exp(-((u - 0.64) ** 2 + (v - 0.5) ** 2) / 0.006)
    sky = sky + rgb(1.0, .95, .80) * sun[..., None] * 0.9
    ground = rgb(.16, .11, .10) * (1 + 0.4 * (1 - v[..., None]))
    scene = np.where((v > 0.62 + 0.04 * np.sin(u * 5.0))[..., None], ground, sky)
    flat = np.clip((v - 0.35) / 0.65, 0, 1)[..., None] * 0.45
    scene = scene * (1 - flat) + PAPER * 0.72 * flat
    img = np.where(area[..., None], scene, img)

    # The three dichroic colours as patches in the bottom margin.
    for k, colour in enumerate([CYAN, MAGENTA, YELLOW]):
        px0 = ix0 + k * 0.058
        patch = (x >= px0) & (x < px0 + 0.05) & (y >= y1 - 0.075) & (y <= y1 - 0.04)
        img = np.where(patch[..., None], colour, img)

    img = np.clip(img, 0, 1)
    icon = Image.fromarray((img * 255 + 0.5).astype(np.uint8), "RGB")
    icon.resize((1024, 1024), Image.LANCZOS).save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

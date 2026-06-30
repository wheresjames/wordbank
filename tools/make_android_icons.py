#!/usr/bin/env python3
"""
Generate Android launcher icons from a source image.

Produces:
  - Legacy PNGs (ic_launcher, ic_launcher_round) at all mipmap densities
  - Adaptive foreground PNGs (ic_launcher_foreground) at all mipmap densities
  - mipmap-anydpi-v26/ic_launcher{,_round}.xml  (adaptive icon descriptors)
  - ic_launcher_background color entry in values/colors.xml

Usage:
    python3 tools/make_android_icons.py [source_image]

Default source: assets/images/otterico.png
"""

import sys
import re
from pathlib import Path
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT  = Path(__file__).parent.parent
SRC_IMAGE  = REPO_ROOT / "assets" / "images" / "otterico.png"
RES_DIR    = REPO_ROOT / "android" / "src" / "main" / "res"
COLORS_XML = RES_DIR / "values" / "colors.xml"

# ---------------------------------------------------------------------------
# Icon sizes
# ---------------------------------------------------------------------------

# Legacy launcher icon: 48dp baseline
LEGACY_PX = {
    "mdpi":    48,
    "hdpi":    72,
    "xhdpi":   96,
    "xxhdpi":  144,
    "xxxhdpi": 192,
}

# Adaptive icon canvas: 108dp baseline (safe zone is centre 72dp)
# The source image already has dark padding around the artwork, so filling
# the full 108dp canvas keeps the cube safely inside the 72dp safe zone.
ADAPTIVE_PX = {
    "mdpi":    108,
    "hdpi":    162,
    "xhdpi":   216,
    "xxhdpi":  324,
    "xxxhdpi": 432,
}

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_square(img: Image.Image) -> Image.Image:
    """Centre-crop to a square."""
    w, h = img.size
    side  = min(w, h)
    left  = (w - side) // 2
    top   = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def make_round(img: Image.Image) -> Image.Image:
    """Circular crop with transparent corners."""
    img  = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, img.size[0] - 1, img.size[1] - 1], fill=255)
    out  = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, mask=mask)
    return out


def resize(img: Image.Image, px: int) -> Image.Image:
    return img.resize((px, px), Image.LANCZOS)


def sample_bg(img: Image.Image) -> str:
    """Return the top-left corner pixel as an #RRGGBB hex string."""
    r, g, b, *_ = img.getpixel((0, 0))
    return f"#{r:02X}{g:02X}{b:02X}"

# ---------------------------------------------------------------------------
# XML / resource helpers
# ---------------------------------------------------------------------------

ADAPTIVE_ICON_XML = """\
<?xml version="1.0" encoding="utf-8"?>
<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">
    <background android:drawable="@color/ic_launcher_background"/>
    <foreground android:drawable="@mipmap/ic_launcher_foreground"/>
</adaptive-icon>
"""


def update_colors_xml(hex_color: str) -> None:
    """Insert or replace the ic_launcher_background entry in colors.xml."""
    text  = COLORS_XML.read_text()
    entry = f'    <color name="ic_launcher_background">{hex_color}</color>'
    if 'name="ic_launcher_background"' in text:
        text = re.sub(
            r'[ \t]*<color name="ic_launcher_background">.*?</color>',
            entry,
            text,
        )
    else:
        text = text.replace("</resources>", f"{entry}\n</resources>")
    COLORS_XML.write_text(text)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC_IMAGE
    if not source.exists():
        sys.exit(f"Error: source image not found: {source}")

    src = make_square(Image.open(source).convert("RGBA"))
    print(f"Source : {source}  ({src.size[0]}×{src.size[1]})")

    bg_color = sample_bg(src)
    print(f"BG color sampled from corner: {bg_color}\n")

    # Legacy icons ---------------------------------------------------------
    print("Legacy icons (48dp baseline):")
    for density, px in LEGACY_PX.items():
        folder = RES_DIR / f"mipmap-{density}"
        folder.mkdir(parents=True, exist_ok=True)
        resize(src, px).save(folder / "ic_launcher.png", "PNG")
        make_round(resize(src, px)).save(folder / "ic_launcher_round.png", "PNG")
        print(f"  mipmap-{density:<10}  {px}×{px}")

    # Adaptive foreground icons --------------------------------------------
    print("\nAdaptive foreground icons (108dp baseline):")
    for density, px in ADAPTIVE_PX.items():
        folder = RES_DIR / f"mipmap-{density}"
        folder.mkdir(parents=True, exist_ok=True)
        resize(src, px).save(folder / "ic_launcher_foreground.png", "PNG")
        print(f"  mipmap-{density:<10}  {px}×{px}")

    # Adaptive icon XML descriptors ----------------------------------------
    anydpi = RES_DIR / "mipmap-anydpi-v26"
    anydpi.mkdir(parents=True, exist_ok=True)
    (anydpi / "ic_launcher.xml").write_text(ADAPTIVE_ICON_XML)
    (anydpi / "ic_launcher_round.xml").write_text(ADAPTIVE_ICON_XML)
    print(f"\nAdaptive XML  → {anydpi.relative_to(REPO_ROOT)}/ic_launcher{{,_round}}.xml")

    # Background color in colors.xml ---------------------------------------
    update_colors_xml(bg_color)
    print(f"colors.xml    → ic_launcher_background = {bg_color}")

    print("\nDone.")


if __name__ == "__main__":
    main()

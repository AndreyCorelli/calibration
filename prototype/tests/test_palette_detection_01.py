"""
Unit-level test — palette detection only, test case 01.

Inputs
------
test_data/test_01/pattern.png      → Im  (digital calibration palette)
test_data/test_01/photo_sample.jpg → camera capture

Checks
------
- Palette colours are extracted from Im (correct bar count and distinct colours)
- Palette is located in the photo

Output
------
tests/output/debug_test_palette_detection_01.jpg
  Annotated camera image with:
  - Combined palette bounding box (black outer / white inner double border)
  - Horizontal dividers showing the expected bar regions
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PIL import Image

from calibration.palette_detector import (
    detect_palette_in_photo,
    extract_palette_colors,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TESTS_DIR   = os.path.dirname(__file__)
_DATA        = os.path.join(_TESTS_DIR, "test_data", "test_01")
PATTERN_PATH = os.path.join(_DATA, "pattern.png")
PHOTO_PATH   = os.path.join(_DATA, "photo_sample.jpg")
OUTPUT_PATH  = os.path.join(_TESTS_DIR, "output", "debug_test_palette_detection_01.jpg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _draw_double_border(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    thickness: int = 2,
) -> None:
    """Black outer rect + white inner rect inset by *thickness* px."""
    x, y, w, h = bbox
    H, W = img.shape[:2]
    pt1 = (max(0, x),         max(0, y))
    pt2 = (min(W - 1, x + w), min(H - 1, y + h))
    cv2.rectangle(img, pt1, pt2, (0, 0, 0), thickness)

    pt1_in = (min(W - 1, x + thickness),     min(H - 1, y + thickness))
    pt2_in = (max(0,      x + w - thickness), max(0,      y + h - thickness))
    if pt1_in[0] < pt2_in[0] and pt1_in[1] < pt2_in[1]:
        cv2.rectangle(img, pt1_in, pt2_in, (255, 255, 255), thickness)


def _draw_bar_dividers(
    img: np.ndarray,
    palette_bbox: tuple[int, int, int, int],
    n_bars: int,
) -> None:
    """Draw horizontal divider lines between the expected bar regions."""
    x, y, w, h = palette_bbox
    H_img, W_img = img.shape[:2]
    strip_h = h / n_bars
    for i in range(1, n_bars):
        y_div = int(y + i * strip_h)
        x1 = max(0, x)
        x2 = min(W_img - 1, x + w)
        cv2.line(img, (x1, y_div), (x2, y_div), (0,   0,   0), 2)
        cv2.line(img, (x1, y_div + 2), (x2, y_div + 2), (255, 255, 255), 1)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_palette_detection_01() -> None:
    assert os.path.isfile(PATTERN_PATH), f"Missing fixture: {PATTERN_PATH}"
    assert os.path.isfile(PHOTO_PATH),   f"Missing fixture: {PHOTO_PATH}"

    im_pil    = Image.open(PATTERN_PATH)
    photo_pil = Image.open(PHOTO_PATH)
    photo_bgr = _pil_to_bgr(photo_pil)

    # ── Extract palette colours from Im ─────────────────────────────────────
    palette_colors, bar_bounds = extract_palette_colors(im_pil)

    print(f"\n[Pattern] Detected {len(palette_colors)} bar(s) in pattern.png:")
    for i, (color, bounds) in enumerate(zip(palette_colors, bar_bounds)):
        r, g, b = color
        print(f"  Bar {i + 1}: R:{r}, G:{g}, B:{b}  (rows {bounds[0]}–{bounds[1]})")

    assert len(palette_colors) >= 2, (
        f"Too few bars detected ({len(palette_colors)}); "
        "ensure pattern.png contains at least 2 distinct solid-colour bars."
    )

    # Bars should have distinct colours (min pairwise ΔE in RGB ≥ 30)
    for i in range(len(palette_colors)):
        for j in range(i + 1, len(palette_colors)):
            ci = np.array(palette_colors[i], dtype=float)
            cj = np.array(palette_colors[j], dtype=float)
            dist = float(np.linalg.norm(ci - cj))
            assert dist >= 30, (
                f"Bars {i+1} and {j+1} are too similar (RGB distance {dist:.1f}); "
                "choose more distinctive palette colours."
            )

    # ── Detect palette in photo ──────────────────────────────────────────────
    palette_bbox, _ = detect_palette_in_photo(photo_bgr, palette_colors)

    if palette_bbox is None:
        pytest.fail(
            "Palette not detected in photo_sample.jpg.\n"
            "Possible causes: poor lighting, palette not visible, colours too similar to background."
        )

    x, y, w, h = palette_bbox
    print(f"\n[Detection] Palette bbox in photo: x:{x}, y:{y}, w:{w}, h:{h}")
    print(f"[Detection] Aspect ratio (h/w): {h/max(w,1):.2f}  (expected ~{len(palette_colors):.1f})")

    # ── Save annotated debug image ───────────────────────────────────────────
    debug = photo_bgr.copy()
    _draw_double_border(debug, palette_bbox)
    _draw_bar_dividers(debug, palette_bbox, len(palette_colors))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, debug)
    print(f"\n[Debug] Annotated image saved to: {OUTPUT_PATH}")

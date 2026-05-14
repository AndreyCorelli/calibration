"""
Integration test — calibration pipeline, test case 01.

Inputs
------
test_data/test_01/pattern.png      → Im  (digital calibration palette)
test_data/test_01/photo_sample.jpg → camera capture (printed palette + paint stroke)

The test walks the full pipeline in order:
  1. Extract digital palette colours from Im
  2. Locate the palette in the photo
  3. Sample photographed palette colours and estimate the CCM
  4. Detect the paint stroke
  5. Sample and colour-correct the paint stroke

All intermediate values are printed to stdout for manual inspection.
If a detection step fails the test stops and reports why — no automatic recovery.

A debug image is written to tests/output/debug_test_pipeline_01.jpg showing
detected bounding boxes as double-border outlines (black outer, white inner).
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
    sample_palette_in_photo,
)
from calibration.color_correction import apply_ccm, apply_inverse_ccm, estimate_ccm
from calibration.paint_detector import detect_paint_stroke, refine_stroke_region, sample_paint_color

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TESTS_DIR   = os.path.dirname(__file__)
_DATA        = os.path.join(_TESTS_DIR, "test_data", "test_01")
PATTERN_PATH = os.path.join(_DATA, "pattern.png")
PHOTO_PATH   = os.path.join(_DATA, "photo_sample.jpg")
OUTPUT_PATH  = os.path.join(_TESTS_DIR, "output", "debug_test_pipeline_01.jpg")


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
    """
    Draw a high-contrast double border around *bbox* in-place.

    Outer rectangle: black  (visible on light backgrounds)
    Inner rectangle: white  (visible on dark backgrounds), inset by *thickness* px
    """
    x, y, w, h = bbox
    H, W = img.shape[:2]

    # Outer black rectangle
    pt1 = (max(0, x),          max(0, y))
    pt2 = (min(W - 1, x + w),  min(H - 1, y + h))
    cv2.rectangle(img, pt1, pt2, (0, 0, 0), thickness)

    # Inner white rectangle, inset by thickness on every side
    pt1_in = (min(W - 1, x + thickness),          min(H - 1, y + thickness))
    pt2_in = (max(0,      x + w - thickness),      max(0,      y + h - thickness))
    if pt1_in[0] < pt2_in[0] and pt1_in[1] < pt2_in[1]:
        cv2.rectangle(img, pt1_in, pt2_in, (255, 255, 255), thickness)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_calibration_pipeline_01() -> None:
    # ── Sanity-check inputs exist ───────────────────────────────────────────
    assert os.path.isfile(PATTERN_PATH), f"Missing fixture: {PATTERN_PATH}"
    assert os.path.isfile(PHOTO_PATH),   f"Missing fixture: {PHOTO_PATH}"

    im_pil    = Image.open(PATTERN_PATH)
    photo_pil = Image.open(PHOTO_PATH)
    photo_bgr = _pil_to_bgr(photo_pil)

    # ── Step 1: Extract digital palette colours from Im ─────────────────────
    palette_colors, bar_bounds = extract_palette_colors(im_pil)

    print(f"\n[Step 1] Detected {len(palette_colors)} palette bar(s) in pattern.png:")
    for i, (color, bounds) in enumerate(zip(palette_colors, bar_bounds)):
        r, g, b = color
        print(f"  Bar {i + 1}: R:{r}, G:{g}, B:{b}  (rows {bounds[0]}–{bounds[1]})")

    if not palette_colors:
        pytest.fail("Step 1 failed — no colour bars detected in pattern.png")

    # ── Step 2: Locate palette in the photo ─────────────────────────────────
    palette_bbox, _ = detect_palette_in_photo(photo_bgr, palette_colors)

    if palette_bbox is None:
        pytest.fail(
            "Step 2 failed — palette not detected in photo_sample.jpg.\n"
            "Possible causes: poor lighting, palette not visible, colours too similar to background."
        )

    x, y, w, h = palette_bbox
    print(f"\n[Step 2] Palette location in photo: x:{x}, y:{y}, w:{w}, h:{h}")

    # ── Step 3: Sample photographed colours and estimate CCM ────────────────
    photo_colors = sample_palette_in_photo(photo_bgr, palette_bbox, len(palette_colors))

    print(f"\n[Step 3] Colour comparison (digital vs photographed):")
    for i, (dig, pho) in enumerate(zip(palette_colors, photo_colors)):
        dr, dg, db = dig
        pr, pg, pb = pho
        print(f"  Bar {i + 1}  digital: R:{dr}, G:{dg}, B:{db}  |  photo: R:{pr}, G:{pg}, B:{pb}")

    M = estimate_ccm(palette_colors, photo_colors)
    print(f"\n[Step 3] Colour correction matrix (3×4):\n{np.round(M, 4)}")

    # ── Step 4a: Coarse paint stroke detection ──────────────────────────────
    coarse_bbox, paint_err = detect_paint_stroke(photo_bgr, palette_bbox)

    if coarse_bbox is None:
        pytest.fail(
            f"Step 4 failed — paint stroke not detected in photo_sample.jpg.\n"
            f"Reason: {paint_err}"
        )

    bx, by, bw, bh = coarse_bbox
    print(f"\n[Step 4a] Coarse stroke bbox: x:{bx}, y:{by}, w:{bw}, h:{bh}")

    # ── Step 4b: Refine to actual stroke contour ─────────────────────────────
    stroke_bbox, centroid, stroke_mask = refine_stroke_region(
        photo_bgr, coarse_bbox, palette_bbox
    )

    if stroke_bbox is None or centroid is None:
        pytest.fail(
            "Step 4b failed — could not isolate the stroke from the background.\n"
            "Check that the paint stroke contrasts with the paper."
        )

    sx, sy, sw, sh = stroke_bbox
    cx, cy = centroid
    print(f"[Step 4b] Refined stroke bbox: x:{sx}, y:{sy}, w:{sw}, h:{sh}")
    print(f"[Step 4b] Stroke centroid:     x:{cx}, y:{cy}")

    # ── Step 5: Sample and colour-correct the paint stroke ──────────────────
    paint_photo_color   = sample_paint_color(photo_bgr, centroid, search_bbox=stroke_bbox)
    paint_digital_color = apply_ccm(paint_photo_color, M)
    paint_back_to_cam   = apply_inverse_ccm(paint_digital_color, M)

    pr, pg, pb = paint_photo_color
    dr, dg, db = paint_digital_color
    br, bg, bb = paint_back_to_cam

    print(f"\n[Step 5] Raw sample (9×9 seed + ΔE filter): R:{pr}, G:{pg}, B:{pb}")
    print(f"[Step 5] Corrected (digital equivalent):     R:{dr}, G:{dg}, B:{db}")
    print(f"[Step 5] Back-converted to camera space:     R:{br}, G:{bg}, B:{bb}")

    # ── Debug image ──────────────────────────────────────────────────────────
    debug = photo_bgr.copy()
    _draw_double_border(debug, palette_bbox)
    _draw_double_border(debug, coarse_bbox)
    _draw_double_border(debug, stroke_bbox, thickness=3)

    cv2.drawMarker(debug, centroid, (0, 0, 0),      cv2.MARKER_CROSS, 20, 3)
    cv2.drawMarker(debug, centroid, (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

    # Three 30×30 swatches stacked vertically at top-left
    # Top:    digital colour back-converted to camera space
    # Middle: corrected digital colour
    # Bottom: raw sampled colour (what camera captured)
    sw = 30  # swatch side length
    swatches = [
        (paint_back_to_cam, "cam-space"),
        (paint_digital_color, "digital"),
        (paint_photo_color,   "raw"),
    ]
    for i, (color_rgb, _) in enumerate(swatches):
        y0, y1 = i * sw, (i + 1) * sw
        bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        debug[y0:y1, 0:sw] = bgr

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, debug)
    print(f"\n[Debug] Annotated image saved to: {OUTPUT_PATH}")

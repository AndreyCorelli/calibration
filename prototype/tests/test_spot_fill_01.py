"""
Spot-fill test: paint all pixels in picture.png that are within ΔE ≤ tolerance
of RGB(133, 61, 55) with a solid colour, then save the result.

Outputs
-------
tests/output/detected_10.png  — matched pixels filled blue  (tolerance 10)
tests/output/detected_20.png  — matched pixels filled green (tolerance 20)
tests/output/detected_30.png  — matched pixels filled yellow (tolerance 30)
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from utils.color_utils import delta_e_image, rgb_image_to_lab, rgb_to_lab

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TESTS_DIR   = os.path.dirname(__file__)
_DATA        = os.path.join(_TESTS_DIR, "test_data", "test_01")
_OUTPUT_DIR  = os.path.join(_TESTS_DIR, "output")
PICTURE_PATH = os.path.join(_DATA, "picture.png")

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

TARGET_COLOR = (133, 61, 55)

_CASES = [
    (10, (0,   0,   255), "detected_10.png"),   # blue
    (20, (0,   255, 0),   "detected_20.png"),   # green
    (30, (255, 255, 0),   "detected_30.png"),   # yellow
]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tolerance,fill_rgb,filename",
    _CASES,
    ids=["tol10", "tol20", "tol30"],
)
def test_spot_fill(tolerance: int, fill_rgb: tuple, filename: str) -> None:
    assert os.path.isfile(PICTURE_PATH), f"Missing fixture: {PICTURE_PATH}"

    img_arr = np.array(Image.open(PICTURE_PATH).convert("RGB"), dtype=np.uint8)

    de_map = delta_e_image(rgb_image_to_lab(img_arr), rgb_to_lab(TARGET_COLOR))
    mask   = de_map <= tolerance

    result = img_arr.copy()
    result[mask] = fill_rgb

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(_OUTPUT_DIR, filename)
    Image.fromarray(result).save(output_path)

    matched_pct = 100.0 * mask.sum() / mask.size
    print(f"\n[tol={tolerance}] {matched_pct:.1f}% of pixels matched → {output_path}")

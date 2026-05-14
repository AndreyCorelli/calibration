"""
Colour matching: find pixels in the target image (Pt) that are visually similar
to the calibrated paint colour using CIE76 ΔE in Lab space.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from utils.color_utils import rgb_image_to_lab, rgb_to_lab, delta_e_image


def find_matching_regions(
    target_pil: Image.Image,
    paint_color_rgb: tuple[int, int, int],
    tolerance: float = 20.0,
) -> np.ndarray:
    """
    Build a boolean mask of pixels in *target_pil* whose ΔE to *paint_color_rgb*
    is within *tolerance*.

    Returns
    -------
    mask : bool array of shape (H, W), True where the colour matches
    """
    target_rgb = np.array(target_pil.convert("RGB"), dtype=np.uint8)
    target_lab = rgb_image_to_lab(target_rgb)
    paint_lab = rgb_to_lab(paint_color_rgb)
    de_map = delta_e_image(target_lab, paint_lab)
    return de_map <= tolerance

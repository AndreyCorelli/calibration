import numpy as np
from skimage import color as skcolor


def rgb_to_lab(rgb_uint8) -> np.ndarray:
    """Single RGB pixel (tuple/array of 3 ints 0-255) → CIE Lab array [L, a, b]."""
    rgb_f = np.asarray(rgb_uint8, dtype=np.float64).reshape(1, 1, 3) / 255.0
    return skcolor.rgb2lab(rgb_f)[0, 0]


def rgb_image_to_lab(rgb_array: np.ndarray) -> np.ndarray:
    """RGB image (H, W, 3) uint8 → Lab image (H, W, 3) float64."""
    return skcolor.rgb2lab(rgb_array.astype(np.float64) / 255.0)


def delta_e_image(lab_image: np.ndarray, lab_color: np.ndarray) -> np.ndarray:
    """CIE76 ΔE from each pixel in lab_image (H, W, 3) to a single Lab color. Returns (H, W)."""
    diff = lab_image - lab_color
    return np.sqrt(np.einsum("...i,...i->...", diff, diff))

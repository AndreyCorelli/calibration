"""
Colour correction matrix (CCM) estimation and application.

We estimate a 3×4 affine transform M such that:
    digital_rgb ≈ M @ [photo_rgb | 1]

This compensates (partially) for printer shift, camera white balance, and lighting.
"""
from __future__ import annotations

import numpy as np


def estimate_ccm(
    digital_colors_rgb: list[tuple[int, int, int]],
    photo_colors_rgb: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    Fit a 3×4 affine colour correction matrix via least squares.

    Parameters
    ----------
    digital_colors_rgb : N reference colours from Im  (0-255)
    photo_colors_rgb   : N photographed colours sampled from the printed palette

    Returns
    -------
    M : (3, 4) float64 array
    """
    if len(digital_colors_rgb) < 2:
        return np.hstack([np.eye(3), np.zeros((3, 1))])  # identity fallback

    digital = np.array(digital_colors_rgb, dtype=np.float64) / 255.0  # (N, 3)
    photo = np.array(photo_colors_rgb, dtype=np.float64) / 255.0      # (N, 3)

    N = len(digital)
    # Augment with a bias column: photo_ext shape is (N, 4)
    photo_ext = np.hstack([photo, np.ones((N, 1))])

    # Solve: photo_ext @ M.T = digital  →  M.T = pinv(photo_ext) @ digital
    M_T, _, _, _ = np.linalg.lstsq(photo_ext, digital, rcond=None)
    return M_T.T  # (3, 4)


def apply_ccm(
    color_rgb: tuple[int, int, int],
    M: np.ndarray,
) -> tuple[int, int, int]:
    """
    Apply CCM to a single RGB colour (0-255 ints).

    Returns corrected RGB tuple clamped to [0, 255].
    """
    c = np.array(color_rgb, dtype=np.float64) / 255.0
    corrected = M @ np.append(c, 1.0)
    corrected = np.clip(corrected, 0.0, 1.0)
    return tuple((corrected * 255).round().astype(np.uint8))  # type: ignore[return-value]


def apply_inverse_ccm(
    color_rgb: tuple[int, int, int],
    M: np.ndarray,
) -> tuple[int, int, int]:
    """
    Invert the CCM: convert a digital colour back to the expected camera colour.

    M maps  camera → digital  via  digital = M3 @ camera + bias
    Inversion:  camera = M3⁻¹ @ (digital - bias)

    Useful for sanity-checking: apply_inverse_ccm(apply_ccm(x, M), M) ≈ x
    """
    M3   = M[:, :3]                                      # (3, 3) linear part
    bias = M[:, 3]                                       # (3,)   offset
    d    = np.array(color_rgb, dtype=np.float64) / 255.0
    camera, *_ = np.linalg.lstsq(M3, d - bias, rcond=None)
    camera = np.clip(camera, 0.0, 1.0)
    return tuple((camera * 255).round().astype(np.uint8))  # type: ignore[return-value]

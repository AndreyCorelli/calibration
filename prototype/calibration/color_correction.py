"""
Colour correction: estimation and application.

Two approaches are provided:

1. apply_ccm (legacy) — 3×4 affine matrix via least squares.
   Works well when the paint colour is close to one of the calibration colours.
   Breaks down for mixed colours because the underdetermined minimum-norm solution
   produces extreme cross-channel coefficients.

2. apply_ccm_idw (preferred) — inverse-distance-weighted additive shift.
   For a query camera colour, each calibration pair (photo_i → digital_i) contributes
   a shift of (digital_i − photo_i), weighted by 1/distance in camera space.
   This interpolates corrections locally, avoiding the extrapolation instability
   of the global matrix for colours that lie between calibration primaries.
"""
from __future__ import annotations

import numpy as np


def estimate_ccm(
    digital_colors_rgb: list[tuple[int, int, int]],
    photo_colors_rgb: list[tuple[int, int, int]],
) -> np.ndarray:
    """
    Fit a 3×4 affine colour correction matrix via least squares.

    Returns
    -------
    M : (3, 4) float64 array  — kept for apply_inverse_ccm / legacy use
    """
    if len(digital_colors_rgb) < 2:
        return np.hstack([np.eye(3), np.zeros((3, 1))])

    digital = np.array(digital_colors_rgb, dtype=np.float64) / 255.0
    photo   = np.array(photo_colors_rgb,   dtype=np.float64) / 255.0
    N = len(digital)
    photo_ext = np.hstack([photo, np.ones((N, 1))])
    M_T, _, _, _ = np.linalg.lstsq(photo_ext, digital, rcond=None)
    return M_T.T  # (3, 4)


def apply_ccm_idw(
    color_rgb: tuple[int, int, int],
    photo_calibration: list[tuple[int, int, int]],
    digital_calibration: list[tuple[int, int, int]],
) -> tuple[int, int, int]:
    """
    Apply inverse-distance-weighted colour correction.

    For the query colour, blend the per-calibration-pair additive shifts
    using weights proportional to 1/distance in camera RGB space.

    Parameters
    ----------
    color_rgb          : camera-space colour to correct (0-255)
    photo_calibration  : N camera-space calibration colours (palette as photographed)
    digital_calibration: N digital-space reference colours  (palette from Im)

    Returns
    -------
    Corrected RGB tuple clamped to [0, 255].
    """
    c       = np.array(color_rgb,          dtype=np.float64) / 255.0
    photo_n = np.array(photo_calibration,  dtype=np.float64) / 255.0  # (N, 3)
    digit_n = np.array(digital_calibration, dtype=np.float64) / 255.0 # (N, 3)

    dists = np.linalg.norm(photo_n - c, axis=1)           # (N,)
    w     = 1.0 / (dists + 1e-9)
    w    /= w.sum()

    shift     = w @ (digit_n - photo_n)                   # weighted additive shift
    corrected = np.clip(c + shift, 0.0, 1.0)
    return tuple((corrected * 255).round().astype(np.uint8))  # type: ignore[return-value]


def apply_ccm(
    color_rgb: tuple[int, int, int],
    M: np.ndarray,
) -> tuple[int, int, int]:
    """Apply the 3×4 matrix CCM (legacy). Prefer apply_ccm_idw for mixed colours."""
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

"""
Paint stroke detection and refinement.

Two-stage pipeline
------------------
1. detect_paint_stroke  — coarse: finds a candidate rectangular region near the palette
                           using edge/contour analysis.
2. refine_stroke_region — fine: clips the coarse region away from the palette,
                           estimates the paper background colour from the region's corners,
                           builds a ΔE mask to separate stroke from background,
                           finds the largest non-background contour, and returns a tight
                           bounding box plus the stroke's centre of gravity.
3. sample_paint_color   — samples the median colour from a small patch around the centroid.
"""
from __future__ import annotations

import numpy as np
import cv2

from utils.color_utils import rgb_to_lab, rgb_image_to_lab, delta_e_image


# ---------------------------------------------------------------------------
# Stage 1 — coarse region detection (unchanged)
# ---------------------------------------------------------------------------

def detect_paint_stroke(
    photo_bgr: np.ndarray,
    palette_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """
    Detect the pencil-bounded paint stroke region in the photo.

    The search area is restricted to the right of the detected palette
    at the vertical extent of the palette ±50%.

    Returns
    -------
    bbox  : (x, y, w, h) coarse region in full-photo coordinates, or None
    error : human-readable error string, or None on success
    """
    H_full, W_full = photo_bgr.shape[:2]

    if palette_bbox is not None:
        px, py, pw, ph = palette_bbox
        sx = max(0, px + pw - pw // 8)
        sy = max(0, py - ph // 2)
        sw = W_full - sx
        sh = min(H_full - sy, int(ph * 2.2))
    else:
        sx, sy = 0, 0
        sw, sh = W_full, H_full

    search = photo_bgr[sy : sy + sh, sx : sx + sw]
    SR_H, SR_W = search.shape[:2]

    if SR_H < 10 or SR_W < 10:
        return None, "Search region too small"

    gray    = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    edges_d = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = SR_H * SR_W * 0.003
    max_area = SR_H * SR_W * 0.45
    candidates: list[tuple[float, int, int, int, int]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        if w_c < 5 or h_c < 5:
            continue
        aspect = max(w_c, h_c) / max(min(w_c, h_c), 1)
        if aspect > 5:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        convexity = area / hull_area if hull_area > 0 else 0.0
        candidates.append((area * convexity / aspect, x_c + sx, y_c + sy, w_c, h_c))

    if not candidates:
        return None, "No paint stroke region detected"

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, bx, by, bw, bh = candidates[0]
    return (bx, by, bw, bh), None


# ---------------------------------------------------------------------------
# Stage 2 — fine refinement
# ---------------------------------------------------------------------------

def _estimate_background(region_rgb: np.ndarray, patch_frac: float = 0.12) -> np.ndarray:
    """
    Estimate the paper background colour by sampling the four corners of *region_rgb*.
    Returns a float64 RGB triplet (the median of all corner pixels).
    """
    H, W = region_rgb.shape[:2]
    ps = max(4, int(min(H, W) * patch_frac))
    corners = [
        region_rgb[:ps,   :ps,   :],
        region_rgb[:ps,   W-ps:, :],
        region_rgb[H-ps:, :ps,   :],
        region_rgb[H-ps:, W-ps:, :],
    ]
    pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(pixels, axis=0)  # float64, 0-255 range


def refine_stroke_region(
    photo_bgr: np.ndarray,
    coarse_bbox: tuple[int, int, int, int],
    palette_bbox: tuple[int, int, int, int] | None = None,
    de_bg_threshold: float = 12.0,
) -> tuple[tuple[int, int, int, int] | None, tuple[int, int] | None, np.ndarray]:
    """
    Refine a coarse stroke region to the actual painted area.

    Steps
    -----
    1. Clip *coarse_bbox* so it does not overlap with *palette_bbox*.
    2. Estimate the paper background colour from the clipped region's corners.
    3. Build a ΔE mask: pixels whose Lab distance from the background exceeds
       *de_bg_threshold* are considered stroke pixels.
    4. Morphological clean-up to remove noise.
    5. Take the largest contour as the stroke.
    6. Return its tight bounding box and centre-of-gravity centroid.

    Returns
    -------
    stroke_bbox : tight (x, y, w, h) around the stroke in full-photo coords, or None
    centroid    : (cx, cy) centre of gravity of the stroke contour, or None
    stroke_mask : uint8 binary mask inside the clipped region (for debug display)
    """
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    H_full, W_full = photo_rgb.shape[:2]

    # 1. Clip away palette overlap (palette is always to the left of the stroke)
    bx, by, bw, bh = coarse_bbox
    if palette_bbox is not None:
        palette_right = palette_bbox[0] + palette_bbox[2]
        if bx < palette_right:
            shift = palette_right - bx
            bx += shift
            bw -= shift
        if bw <= 0:
            empty = np.zeros((1, 1), dtype=np.uint8)
            return None, None, empty

    # Clip to image bounds
    bx = max(0, bx);  by = max(0, by)
    bw = min(W_full - bx, bw);  bh = min(H_full - by, bh)
    if bw <= 10 or bh <= 10:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return None, None, empty

    region = photo_rgb[by : by + bh, bx : bx + bw, :]

    # 2. Background estimation from corners
    bg_rgb  = _estimate_background(region)
    bg_lab  = rgb_to_lab(bg_rgb.astype(np.uint8))

    # 3. ΔE mask — stroke pixels are far from the background colour
    region_lab = rgb_image_to_lab(region)
    de_map     = delta_e_image(region_lab, bg_lab)
    raw_mask   = (de_map > de_bg_threshold).astype(np.uint8) * 255

    # 4. Morphological clean-up
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stroke_mask = cv2.morphologyEx(raw_mask,    cv2.MORPH_CLOSE, kernel, iterations=2)
    stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # 5. Largest contour = the stroke
    contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, stroke_mask

    largest = max(contours, key=cv2.contourArea)

    # Tight bbox in full-photo coordinates
    rx, ry, rw, rh = cv2.boundingRect(largest)
    stroke_bbox = (bx + rx, by + ry, rw, rh)

    # 6. Centre of gravity from image moments
    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]) + bx
        cy = int(M["m01"] / M["m00"]) + by
    else:
        cx = bx + rx + rw // 2
        cy = by + ry + rh // 2

    return stroke_bbox, (cx, cy), stroke_mask


# ---------------------------------------------------------------------------
# Stage 3 — colour sampling
# ---------------------------------------------------------------------------

def sample_paint_color(
    photo_bgr: np.ndarray,
    centroid: tuple[int, int],
    search_bbox: tuple[int, int, int, int] | None = None,
    seed_half: int = 4,
    de_filter: float = 15.0,
) -> tuple[int, int, int]:
    """
    Sample the paint colour using a two-step seed-and-filter approach.

    Step 1 — Seed
        Take a (2*seed_half+1)² patch around *centroid* (default 9×9) and
        compute its median as the seed colour.

    Step 2 — Filter
        Within *search_bbox* (or a 5× enlarged patch if None), keep only
        pixels whose ΔE from the seed colour is ≤ *de_filter*.  Take the
        median of the surviving pixels.

    This excludes near-white paper, pencil-border marks, and shadows that
    would otherwise drag the estimate toward white.
    """
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    H, W = photo_rgb.shape[:2]
    cx, cy = centroid

    # ── Step 1: seed from 9×9 patch ────────────────────────────────────────
    x1s = max(0, cx - seed_half);  x2s = min(W, cx + seed_half + 1)
    y1s = max(0, cy - seed_half);  y2s = min(H, cy + seed_half + 1)
    seed_patch = photo_rgb[y1s:y2s, x1s:x2s, :]
    if seed_patch.size == 0:
        seed_patch = photo_rgb[cy:cy+1, cx:cx+1, :]
    seed_color = np.median(seed_patch.reshape(-1, 3), axis=0).astype(np.uint8)

    # ── Step 2: filter wider region by ΔE to seed ──────────────────────────
    if search_bbox is not None:
        bx, by, bw, bh = search_bbox
        bx = max(0, bx);  by = max(0, by)
        bw = min(W - bx, bw);  bh = min(H - by, bh)
    else:
        r = seed_half * 5
        bx = max(0, cx - r);  by = max(0, cy - r)
        bw = min(W - bx, 2 * r + 1);  bh = min(H - by, 2 * r + 1)

    region = photo_rgb[by:by+bh, bx:bx+bw, :]
    if region.size == 0:
        return tuple(seed_color)  # type: ignore[return-value]

    seed_lab   = rgb_to_lab(seed_color)
    region_lab = rgb_image_to_lab(region)
    de_map     = delta_e_image(region_lab, seed_lab)

    matching = region[de_map <= de_filter]
    if matching.size == 0:
        return tuple(seed_color)  # type: ignore[return-value]

    med = np.median(matching, axis=0).astype(np.uint8)
    return tuple(med)  # type: ignore[return-value]

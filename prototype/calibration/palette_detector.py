"""
Palette detection: extract digital colors from Im and locate the palette in a photo.
"""
from __future__ import annotations

from itertools import combinations as _combinations

import numpy as np
import cv2
from PIL import Image

from utils.color_utils import rgb_to_lab, rgb_image_to_lab, delta_e_image

# Resize large photos to this dimension before running the color-search pass
_DETECT_MAX_DIM = 900


def extract_palette_colors(
    palette_pil: Image.Image,
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int]]]:
    """
    Extract the color of every bar in the digital palette image (Im).

    Returns
    -------
    colors_rgb : list of (R, G, B) tuples, one per detected bar (top→bottom)
    bar_bounds : list of (top_row, bottom_row) tuples matching colors_rgb
    """
    arr = np.array(palette_pil.convert("RGB"), dtype=np.uint8)
    H, W = arr.shape[:2]

    # Sample the horizontal centre band to reduce edge/margin noise
    lx = max(0, W // 4)
    rx = min(W, 3 * W // 4)
    band = arr[:, lx:rx, :]  # (H, band_width, 3)

    # Per-row median colour in Lab space
    row_med = np.median(band.reshape(H, -1, 3), axis=1).astype(np.float64)  # (H, 3)
    row_lab = rgb_image_to_lab(row_med.reshape(H, 1, 3)).reshape(H, 3)

    # Magnitude of row-to-row Lab difference → peaks = bar boundaries
    diffs = np.linalg.norm(np.diff(row_lab, axis=0), axis=1)  # (H-1,)

    # Smooth with a small uniform kernel
    k = max(3, H // 60)
    diffs_s = np.convolve(diffs, np.ones(k) / k, mode="same")

    # Threshold: mean + 1 std, with a minimum height requirement per bar
    thresh = diffs_s.mean() + diffs_s.std()
    min_bar_h = max(4, H // 15)

    # Collect transitions, merging those that are very close together
    raw = np.where(diffs_s > thresh)[0]
    boundaries: list[int] = [0]
    if raw.size > 0:
        group = [int(raw[0])]
        for t in raw[1:]:
            if t - group[-1] < min_bar_h:
                group.append(int(t))
            else:
                boundaries.append(int(np.mean(group)) + 1)
                group = [int(t)]
        boundaries.append(int(np.mean(group)) + 1)
    boundaries.append(H)

    colors: list[tuple[int, int, int]] = []
    bar_bounds: list[tuple[int, int]] = []

    for i in range(len(boundaries) - 1):
        top, bot = boundaries[i], boundaries[i + 1]
        if bot - top < min_bar_h:
            continue
        margin = max(1, (bot - top) // 5)
        inner = band[top + margin : bot - margin, :, :]
        if inner.size == 0:
            inner = band[top:bot, :, :]
        med = tuple(np.median(inner.reshape(-1, 3), axis=0).astype(np.uint8))
        colors.append(med)  # type: ignore[arg-type]
        bar_bounds.append((top, bot))

    return colors, bar_bounds


def _largest_blob_bbox(
    mask: np.ndarray,
    min_area: float,
) -> tuple[int, int, int, int] | None:
    """Return the bounding box of the largest contour in *mask* above *min_area*, or None."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  k, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return None
    _, best = max(valid, key=lambda t: t[0])
    return cv2.boundingRect(best)


def _shape_valid(
    bboxes: list[tuple[int, int, int, int]],
    square_aspect_max: float = 3.0,
    x_center_tol_fraction: float = 1.2,
    y_gap_max_fraction: float = 0.6,
) -> bool:
    """
    Check that *bboxes* (one per palette bar) form a vertical stack of squares.

    Rules
    -----
    - Each bbox must be roughly square: max(w,h)/min(w,h) ≤ square_aspect_max
    - All x-centres must lie within x_center_tol_fraction × mean_width of each other
    - When sorted top-to-bottom, consecutive bboxes must not overlap and the
      gap between them must be ≤ y_gap_max_fraction × mean_height
    """
    if len(bboxes) < 2:
        return True

    # Squareness
    for x, y, w, h in bboxes:
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > square_aspect_max:
            return False

    sorted_boxes = sorted(bboxes, key=lambda b: b[1])  # top → bottom by y

    # Horizontal alignment: x-centres close to each other
    x_centers = [b[0] + b[2] / 2 for b in sorted_boxes]
    mean_w = float(np.mean([b[2] for b in sorted_boxes]))
    mean_xc = float(np.mean(x_centers))
    if any(abs(xc - mean_xc) > x_center_tol_fraction * mean_w for xc in x_centers):
        return False

    # Vertical stacking: gap must be within [-5px overlap, y_gap_max_fraction * mean_h]
    mean_h = float(np.mean([b[3] for b in sorted_boxes]))
    for i in range(len(sorted_boxes) - 1):
        _, y_a, _, h_a = sorted_boxes[i]
        _, y_b, _, _   = sorted_boxes[i + 1]
        gap = y_b - (y_a + h_a)
        if gap < -5:                              # significant overlap → wrong blobs
            return False
        if gap > y_gap_max_fraction * mean_h:     # gap too large
            return False

    return True


def _detect_palette_by_shape(
    photo_bgr: np.ndarray,
    n_bars: int,
    min_height_fraction: float = 0.35,
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int, int, int]]]:
    """
    Shape-based palette detection fallback.

    Finds N square-shaped blobs with clear borders (Canny edges) that are
    stacked vertically and together span at least min_height_fraction of the
    image height.  Color-agnostic — works for printed palettes under any
    lighting where the color-based approach fails.

    Returns
    -------
    bbox      : combined (x, y, w, h) in original coordinates, or None
    bar_bboxes: individual bar bboxes top→bottom (empty on failure)
    """
    H_orig, W_orig = photo_bgr.shape[:2]

    # Work at a capped resolution for speed
    detect_dim = 800
    scale = min(1.0, detect_dim / max(H_orig, W_orig))
    small = (
        cv2.resize(photo_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else photo_bgr
    )
    H, W = small.shape[:2]

    gray    = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 20, 80)

    # Close small gaps so square borders form closed loops
    k3    = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k3, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Per-bar size constraints
    min_side = H * 0.08   # ≥ 8 % of image height
    max_side = H * 0.65   # ≤ 65 % of image height (single bar)

    seen: set[tuple[int, int, int, int]] = set()
    candidates: list[tuple[int, int, int, int]] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_side ** 2 * 0.3:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if min(w, h) < min_side or max(w, h) > max_side:
            continue

        # Axis-aligned bounding-box squareness (tilted squares still look square)
        if max(w, h) / max(min(w, h), 1) > 2.5:
            continue

        # Deduplicate by snapping to a 10 px grid
        key = (round(x / 10) * 10, round(y / 10) * 10,
               round(w / 10) * 10, round(h / 10) * 10)
        if key in seen:
            continue
        seen.add(key)

        candidates.append((x, y, w, h))

    # Keep only the 30 largest blobs to bound combination count
    candidates.sort(key=lambda b: b[2] * b[3], reverse=True)
    candidates = candidates[:30]

    if len(candidates) < n_bars:
        return None, []

    min_combined_h                             = H * min_height_fraction
    best_combo: list[tuple[int, int, int, int]] | None = None
    best_score                                 = float("inf")

    for combo in _combinations(candidates, n_bars):
        bboxes = list(combo)

        if not _shape_valid(
            bboxes,
            square_aspect_max=2.5,
            x_center_tol_fraction=1.5,
            y_gap_max_fraction=1.2,
        ):
            continue

        ys_  = [b[1]        for b in bboxes]
        y2s_ = [b[1] + b[3] for b in bboxes]
        if max(y2s_) - min(ys_) < min_combined_h:
            continue

        # Prefer groups of uniformly-sized blobs
        areas = [b[2] * b[3] for b in bboxes]
        score = float(np.std(areas)) / (float(np.mean(areas)) + 1.0)

        if score < best_score:
            best_score = score
            best_combo = sorted(bboxes, key=lambda b: b[1])

    if best_combo is None:
        return None, []

    # Scale back to original resolution
    inv = 1.0 / scale
    best_combo = [
        (int(b[0] * inv), int(b[1] * inv), int(b[2] * inv), int(b[3] * inv))
        for b in best_combo
    ]

    xs_  = [b[0]        for b in best_combo]
    ys_  = [b[1]        for b in best_combo]
    x2s_ = [b[0] + b[2] for b in best_combo]
    y2s_ = [b[1] + b[3] for b in best_combo]
    bx, by = min(xs_), min(ys_)
    bw = min(W_orig - bx, max(x2s_) - bx)
    bh = min(H_orig - by, max(y2s_) - by)

    return (bx, by, bw, bh), best_combo


def detect_palette_in_photo(
    photo_bgr: np.ndarray,
    palette_colors_rgb: list[tuple[int, int, int]],
    de_threshold: float = 35.0,
    de_adaptive_margin: float = 25.0,
    min_blob_area_frac: float = 0.002,
) -> tuple[tuple[int, int, int, int] | None, np.ndarray]:
    """
    Locate the palette in a photographed image.

    Strategy
    --------
    For each palette colour independently, find the largest matching blob.
    The matching threshold is adaptive: max(de_threshold, min_ΔE_in_photo + de_adaptive_margin).
    This handles camera/screen colour shifts that push the actual pixel values
    outside the fixed threshold while still using the tighter threshold when
    the colours are close to their digital originals.

    Then verify that the N blobs form a vertical stack of roughly-square regions
    (matching the palette's known geometry of equal solid-colour bars).
    This rejects scattered UI pixels that happen to match a palette colour.

    Returns
    -------
    bbox        : (x, y, w, h) in original photo coordinates, or None
    debug_mask  : combined colour-match mask at original scale (for display)
    """
    H_orig, W_orig = photo_bgr.shape[:2]

    scale = min(1.0, _DETECT_MAX_DIM / max(H_orig, W_orig))
    small = (
        cv2.resize(photo_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale < 1.0 else photo_bgr
    )
    H, W = small.shape[:2]
    photo_lab = rgb_image_to_lab(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    min_blob_area = H * W * min_blob_area_frac
    debug_combined = np.zeros((H, W), dtype=np.uint8)
    per_color_bboxes: list[tuple[int, int, int, int]] = []
    color_ok = True

    for c_rgb in palette_colors_rgb:
        c_lab    = rgb_to_lab(c_rgb)
        de       = delta_e_image(photo_lab, c_lab)
        adaptive = max(de_threshold, float(de.min()) + de_adaptive_margin)
        mask     = (de < adaptive).astype(np.uint8) * 255
        debug_combined = cv2.bitwise_or(debug_combined, mask)

        bbox = _largest_blob_bbox(mask, min_blob_area)
        if bbox is None:
            color_ok = False
            break
        per_color_bboxes.append(bbox)

    debug_mask = (
        cv2.resize(debug_combined, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
        if scale < 1.0 else debug_combined
    )

    if color_ok and _shape_valid(per_color_bboxes):
        xs  = [b[0]         for b in per_color_bboxes]
        ys  = [b[1]         for b in per_color_bboxes]
        x2s = [b[0] + b[2]  for b in per_color_bboxes]
        y2s = [b[1] + b[3]  for b in per_color_bboxes]
        bx, by = min(xs), min(ys)
        bw, bh = max(x2s) - bx, max(y2s) - by

        inv = 1.0 / scale
        bx = max(0,           int(bx * inv))
        by = max(0,           int(by * inv))
        bw = min(W_orig - bx, int(bw * inv))
        bh = min(H_orig - by, int(bh * inv))
        return (bx, by, bw, bh), debug_mask

    # ── Shape-based fallback ────────────────────────────────────────────────
    # Used when colour-based detection fails (e.g. printed palettes under
    # ambient lighting where palette hues bleed into the background).
    shape_bbox, _ = _detect_palette_by_shape(photo_bgr, len(palette_colors_rgb))
    if shape_bbox is not None:
        return shape_bbox, debug_mask

    return None, debug_mask


def sample_palette_in_photo(
    photo_bgr: np.ndarray,
    palette_bbox: tuple[int, int, int, int],
    n_bars: int,
) -> list[tuple[int, int, int]]:
    """
    Sample the median colour of each palette bar in the photographed image.

    Divides the detected bounding box into n_bars equal horizontal strips
    and samples the centre 60% of each strip to reduce edge contamination.
    """
    photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
    x, y, w, h = palette_bbox
    region = photo_rgb[y : y + h, x : x + w, :]
    HR = region.shape[0]
    strip_h = max(1, HR // n_bars)

    sampled: list[tuple[int, int, int]] = []
    for i in range(n_bars):
        top = i * strip_h
        bot = (i + 1) * strip_h if i < n_bars - 1 else HR
        margin = max(1, (bot - top) // 5)
        inner = region[top + margin : bot - margin, :, :]
        if inner.size == 0:
            inner = region[top:bot, :, :]
        med = tuple(np.median(inner.reshape(-1, 3), axis=0).astype(np.uint8))
        sampled.append(med)  # type: ignore[arg-type]

    return sampled

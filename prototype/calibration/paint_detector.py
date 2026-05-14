"""
Paint stroke detection and refinement.

Two-stage pipeline
------------------
1. detect_paint_stroke  — coarse: finds a candidate rectangular region near the palette
                           using saturation + shape + ring-quality analysis.
2. refine_stroke_region — fine: clips the coarse region away from the palette,
                           estimates the paper background colour from the region's corners,
                           builds a ΔE mask to separate stroke from background,
                           filters out pencil-border and low-saturation contours,
                           and returns a tight bounding box plus the stroke mask.
3. sample_paint_color   — samples the median colour from the eroded stroke mask.
"""
from __future__ import annotations

import numpy as np
import cv2

from utils.color_utils import rgb_to_lab, rgb_image_to_lab, delta_e_image


# ---------------------------------------------------------------------------
# Stage 1 helpers
# ---------------------------------------------------------------------------

def _build_stroke_search_roi(
    image_shape: tuple[int, int],
    palette_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """ROI to the right of the palette, bounded horizontally."""
    H, W = image_shape[:2]
    px, py, pw, ph = palette_bbox

    sx = px + pw                                              # right at palette edge
    ex = px + pw + max(int(8.0 * pw), int(0.35 * W))         # wider search

    sx = max(0, min(W - 1, sx))
    ex = max(sx + 1, min(W, ex))
    sy = max(0, min(H - 1, py))
    ey = max(sy + 1, min(H, py + ph))

    return sx, sy, ex - sx, ey - sy


def _build_colored_blob_mask(search_bgr: np.ndarray) -> np.ndarray:
    """First-pass saturation/value mask for colored blobs."""
    hsv = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((S > 40) & (V > 60)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    return mask


def _contour_shape_metrics(cnt: np.ndarray) -> dict[str, float]:
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    bbox_area = float(max(w * h, 1))
    extent = area / bbox_area
    hull_area = float(max(cv2.contourArea(cv2.convexHull(cnt)), 1.0))
    solidity = area / hull_area
    aspect = max(w, h) / max(min(w, h), 1)
    return {
        "area": area, "x": float(x), "y": float(y),
        "w": float(w), "h": float(h),
        "extent": extent, "solidity": solidity, "aspect": aspect,
    }


def _touches_roi_border(
    bbox: tuple[int, int, int, int],
    roi_shape: tuple[int, int],
    margin: int = 2,
) -> bool:
    x, y, w, h = bbox
    roi_h, roi_w = roi_shape[:2]
    return (
        y <= margin                   # top
        or x + w >= roi_w - margin    # right
        or y + h >= roi_h - margin    # bottom
        # left edge intentionally not checked: stroke may be adjacent to palette
    )


def _lab_color_variance(search_bgr: np.ndarray, cnt: np.ndarray) -> float:
    """Mean per-channel Lab std-dev inside the contour. Lower = more uniform."""
    search_rgb = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2RGB)
    search_lab = rgb_image_to_lab(search_rgb)
    mask = np.zeros(search_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)
    pixels = search_lab[mask > 0]
    if len(pixels) < 10:
        return 999.0
    return float(np.mean(np.std(pixels, axis=0)))


def _paper_ring_score(
    search_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    expand_frac: float = 0.7,
) -> tuple[float, bool]:
    """
    Inspect the ring around the candidate bbox.
    Returns (score 0-3, is_paper_like).
    Higher score = brighter, less saturated, more uniform surround.
    """
    H, W = search_bgr.shape[:2]
    x, y, w, h = bbox
    pad = int(max(w, h) * expand_frac)

    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(W, x + w + pad);  y2 = min(H, y + h + pad)
    if x2 <= x1 or y2 <= y1:
        return 0.0, False

    ring = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    ring[:, :] = 255
    ix1, iy1 = x - x1, y - y1
    ring[iy1:iy1 + h, ix1:ix1 + w] = 0

    patch = search_bgr[y1:y2, x1:x2]
    hsv   = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1][ring > 0]
    V = hsv[:, :, 2][ring > 0]

    if len(S) < 20:
        return 0.0, False

    med_s = float(np.median(S))
    med_v = float(np.median(V))
    std_v = float(np.std(V))

    is_paper_like = med_v > 120.0 and med_s < 80.0 and std_v < 55.0

    score  = max(0.0, min(1.0, (med_v - 80.0) / 120.0))
    score += max(0.0, min(1.0, (100.0 - med_s) / 100.0))
    score += max(0.0, min(1.0, (80.0 - std_v) / 80.0))
    return score, is_paper_like


def _score_candidate(
    *,
    area: float,
    min_area: float,
    max_area: float,
    aspect: float,
    extent: float,
    solidity: float,
    lab_variance: float,
    paper_score: float,
    candidate_center_y: float,
    palette_center_y: float,
    palette_h: float,
) -> float:
    area_norm      = max(0.0, min(1.0, (area - min_area) / max(max_area - min_area, 1.0)))
    aspect_score   = max(0.0, min(1.0, (3.0 - aspect) / 2.0))
    extent_score   = max(0.0, min(1.0, extent))
    solidity_score = max(0.0, min(1.0, solidity))
    variance_score = max(0.0, min(1.0, (22.0 - lab_variance) / 22.0))
    vert_dist      = abs(candidate_center_y - palette_center_y) / max(palette_h, 1.0)
    vertical_score = max(0.0, 1.0 - vert_dist * 2.0)
    return (
        1.0 * area_norm
        + 1.0 * aspect_score
        + 0.7 * extent_score
        + 0.7 * solidity_score
        + 1.5 * variance_score
        + 2.0 * paper_score
        + 0.8 * vertical_score
    )


# ---------------------------------------------------------------------------
# Stage 1 — coarse region detection
# ---------------------------------------------------------------------------

def detect_paint_stroke(
    photo_bgr: np.ndarray,
    palette_bbox: tuple[int, int, int, int] | None = None,
    n_bars: int = 3,
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    """
    Detect a compact, color-consistent paint stroke on paper near the palette.

    Searches a bounded ROI to the right of the palette (within its vertical
    extent, no farther than 8× the palette width or 35% of image width).
    Each candidate contour is scored on shape compactness, internal colour
    uniformity, and the quality of the paper-like surround — not just area.

    Returns
    -------
    bbox  : (x, y, w, h) coarse region in full-photo coordinates, or None
    error : human-readable reason string, or None on success
    """
    if palette_bbox is None:
        return None, "Palette bbox is required for reliable stroke detection"

    H, W = photo_bgr.shape[:2]
    _, py, pw, ph = palette_bbox

    sx, sy, sw, sh = _build_stroke_search_roi((H, W), palette_bbox)
    search = photo_bgr[sy:sy + sh, sx:sx + sw]

    if search.shape[0] < 10 or search.shape[1] < 10:
        return None, "Stroke search region too small"

    mask = _build_colored_blob_mask(search)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No colored candidate contours found"

    roi_h, roi_w = search.shape[:2]
    min_area = roi_h * roi_w * 0.002
    max_area = (pw * ph) / max(n_bars, 1) * 0.85
    palette_center_y = py + ph / 2.0

    scored: list[tuple[float, tuple[int, int, int, int]]] = []

    for cnt in contours:
        m = _contour_shape_metrics(cnt)
        area = m["area"]
        x, y, w, h = int(m["x"]), int(m["y"]), int(m["w"]), int(m["h"])
        bbox = (x, y, w, h)

        if area < min_area or area > max_area:
            continue
        if _touches_roi_border(bbox, search.shape):
            continue
        if m["aspect"] > 3.0 or m["extent"] < 0.20 or m["solidity"] < 0.45:
            continue

        lab_var = _lab_color_variance(search, cnt)
        if lab_var > 30.0:
            continue

        paper_score, _ = _paper_ring_score(search, bbox)
        # paper_score used as a scoring weight only — hard rejection would
        # discard real strokes on non-paper backgrounds (e.g. screen captures)

        s = _score_candidate(
            area=area, min_area=min_area, max_area=max_area,
            aspect=m["aspect"], extent=m["extent"], solidity=m["solidity"],
            lab_variance=lab_var, paper_score=paper_score,
            candidate_center_y=sy + y + h / 2.0,
            palette_center_y=palette_center_y, palette_h=float(ph),
        )
        scored.append((s, (sx + x, sy + y, w, h)))

    if not scored:
        return None, "No valid paint stroke candidate found"

    scored.sort(key=lambda t: t[0], reverse=True)
    bx, by, bw, bh = scored[0][1]

    # Expand by 25% so refine_stroke_region has background pixels at corners,
    # clamped to the palette's vertical band
    mx, my = bw // 4, bh // 4
    bx = max(0, bx - mx)
    bw = min(W - bx, bw + 2 * mx)
    by = max(py,        by - my)
    bh = min(py + ph - by, bh + 2 * my)

    return (bx, by, bw, bh), None


# ---------------------------------------------------------------------------
# Stage 2 — fine refinement
# ---------------------------------------------------------------------------

def _estimate_background(
    photo_rgb: np.ndarray,
    clipped_bbox: tuple[int, int, int, int],
    strip_px: int = 30,
) -> np.ndarray:
    """
    Estimate the paper background colour from strips OUTSIDE the clipped stroke
    region (above, below, right).  Sampling from outside the coarse region avoids
    the failure where a large stroke fills its own corners and inverts the mask.
    Falls back to corner patches of the region itself if no external strip yields
    enough pixels.
    """
    H_full, W_full = photo_rgb.shape[:2]
    bx, by, bw, bh = clipped_bbox

    strips: list[np.ndarray] = []

    # Strip above the clipped region (same x-span)
    if by >= strip_px:
        s = photo_rgb[by - strip_px : by, bx : bx + bw]
        if s.size > 0:
            strips.append(s.reshape(-1, 3))

    # Strip below the clipped region
    if by + bh + strip_px <= H_full:
        s = photo_rgb[by + bh : by + bh + strip_px, bx : bx + bw]
        if s.size > 0:
            strips.append(s.reshape(-1, 3))

    # Strip to the right of the clipped region (NOT left — that's where the palette is)
    if bx + bw + strip_px <= W_full:
        s = photo_rgb[by : by + bh, bx + bw : bx + bw + strip_px]
        if s.size > 0:
            strips.append(s.reshape(-1, 3))

    if strips:
        pixels = np.concatenate(strips, axis=0)
        return np.median(pixels, axis=0)

    # Fallback: corner patches of the region itself
    region = photo_rgb[by : by + bh, bx : bx + bw]
    ps = max(4, int(min(bh, bw) * 0.12))
    corners = [
        region[:ps, :ps],
        region[:ps, bw - ps :],
        region[bh - ps :, :ps],
        region[bh - ps :, bw - ps :],
    ]
    pixels = np.concatenate([c.reshape(-1, 3) for c in corners], axis=0)
    return np.median(pixels, axis=0)


def refine_stroke_region(
    photo_bgr: np.ndarray,
    coarse_bbox: tuple[int, int, int, int],
    palette_bbox: tuple[int, int, int, int] | None = None,
    de_bg_threshold: float = 12.0,
) -> tuple[
    tuple[int, int, int, int] | None,
    tuple[int, int] | None,
    np.ndarray,
    tuple[int, int, int, int] | None,
]:
    """
    Refine a coarse stroke region to the actual painted area.

    Steps
    -----
    1. Clip *coarse_bbox* so it does not overlap with *palette_bbox*.
    2. Estimate the paper background colour from the clipped region's corners.
    3. Build a ΔE mask: pixels whose Lab distance from the background exceeds
       *de_bg_threshold* are considered stroke pixels.
    4. Morphological clean-up to remove noise.
    5. Filter contours: reject low-saturation (pencil/border) and hollow shapes.
    6. Return the largest surviving contour's tight bbox, centre-of-gravity
       centroid, binary mask, and the clipped region bbox.

    Returns
    -------
    stroke_bbox  : tight (x, y, w, h) in full-photo coords, or None
    centroid     : (cx, cy) centre of gravity, or None
    stroke_mask  : uint8 binary mask in *clipped_bbox* coordinate space
    clipped_bbox : (x, y, w, h) of the palette-clipped coarse region, or None
    """
    empty = np.zeros((1, 1), dtype=np.uint8)

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
            return None, None, empty, None

    bx = max(0, bx);  by = max(0, by)
    bw = min(W_full - bx, bw);  bh = min(H_full - by, bh)
    if bw <= 10 or bh <= 10:
        return None, None, empty, None

    clipped_bbox = (bx, by, bw, bh)
    region = photo_rgb[by : by + bh, bx : bx + bw, :]

    # 2. Background estimation from strips outside the clipped region
    bg_rgb = _estimate_background(photo_rgb, clipped_bbox)
    bg_lab = rgb_to_lab(bg_rgb.astype(np.uint8))

    # 3. ΔE mask — stroke pixels differ from background
    region_lab = rgb_image_to_lab(region)
    de_map     = delta_e_image(region_lab, bg_lab)
    raw_mask   = (de_map > de_bg_threshold).astype(np.uint8) * 255

    # 4. Morphological clean-up
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_mask  = cv2.morphologyEx(raw_mask,    cv2.MORPH_CLOSE, close_kernel, iterations=2)
    stroke_mask  = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN,  open_kernel,  iterations=1)

    # 5. Filter contours: reject pencil borders (low saturation) and hollow shapes
    contours, _ = cv2.findContours(stroke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, stroke_mask, clipped_bbox

    region_bgr = photo_bgr[by:by + bh, bx:bx + bw]
    hsv_region = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)

    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        cnt_mask = np.zeros(stroke_mask.shape, dtype=np.uint8)
        cv2.drawContours(cnt_mask, [cnt], -1, 255, thickness=-1)
        median_s = float(np.median(hsv_region[:, :, 1][cnt_mask > 0]))
        if median_s < 20:                   # pencil / gray border
            continue
        valid.append(cnt)

    if not valid:
        return None, None, stroke_mask, clipped_bbox

    largest = max(valid, key=cv2.contourArea)

    # 6. Tight bbox and centre of gravity
    rx, ry, rw, rh = cv2.boundingRect(largest)
    stroke_bbox = (bx + rx, by + ry, rw, rh)

    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]) + bx
        cy = int(M["m01"] / M["m00"]) + by
    else:
        cx = bx + rx + rw // 2
        cy = by + ry + rh // 2

    return stroke_bbox, (cx, cy), stroke_mask, clipped_bbox


# ---------------------------------------------------------------------------
# Stage 3 — colour sampling
# ---------------------------------------------------------------------------

def sample_paint_color(
    photo_bgr: np.ndarray,
    stroke_mask: np.ndarray,
    clipped_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int]:
    """
    Sample the paint colour from the eroded stroke mask.

    Erodes the mask by one step to avoid sampling fringe / edge pixels, then
    returns the median RGB of all surviving masked pixels.  Falls back to the
    full mask if erosion removes everything.
    """
    bx, by, bw, bh = clipped_bbox
    region_bgr = photo_bgr[by:by + bh, bx:bx + bw]
    region_rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)

    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    inner_mask = cv2.erode(stroke_mask, kernel, iterations=1)

    pixels = region_rgb[inner_mask > 0]
    if pixels.size == 0:
        pixels = region_rgb[stroke_mask > 0]
    if pixels.size == 0:
        # Last resort: centre pixel of the clipped region
        cy, cx = bh // 2, bw // 2
        pixels = region_rgb[cy:cy + 1, cx:cx + 1].reshape(-1, 3)

    med = np.median(pixels, axis=0).astype(np.uint8)
    return tuple(med)  # type: ignore[return-value]

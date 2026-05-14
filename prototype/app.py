"""
Paint-to-Image Colour Matcher — Streamlit prototype.

Workflow
--------
1. Upload target image (Pt) and digital calibration palette (Im) in the sidebar.
2. Capture a photo of the *printed* palette next to the paint stroke (via webcam).
3. The app detects the palette, estimates a colour correction matrix, detects the
   paint stroke, and searches Pt for matching regions.
4. Results are shown as a blink animation (original ↔ match mask).
"""
from __future__ import annotations

import base64
import glob
import sys
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Make project modules importable when Streamlit runs from an arbitrary cwd
sys.path.insert(0, os.path.dirname(__file__))

from calibration.palette_detector import (
    detect_palette_in_photo,
    extract_palette_colors,
    sample_palette_in_photo,
)
from calibration.color_correction import apply_ccm_idw, estimate_ccm
from calibration.paint_detector import detect_paint_stroke, refine_stroke_region, sample_paint_color
from calibration.color_matcher import find_matching_regions
from ui.highlighter import create_blink_gif

# ---------------------------------------------------------------------------
# Runtime persistence
# ---------------------------------------------------------------------------

_RUNTIME_DIR = os.path.join(os.path.dirname(__file__), "runtime")


def _save_runtime(uploaded_file, prefix: str) -> None:
    """Save *uploaded_file* to runtime/<prefix>.<ext>, removing any prior file for that prefix."""
    os.makedirs(_RUNTIME_DIR, exist_ok=True)
    for old in glob.glob(os.path.join(_RUNTIME_DIR, f"{prefix}.*")):
        os.remove(old)
    ext = os.path.splitext(uploaded_file.name)[1]
    dest = os.path.join(_RUNTIME_DIR, f"{prefix}{ext}")
    with open(dest, "wb") as fh:
        fh.write(uploaded_file.getvalue())


def _find_runtime(prefix: str) -> str | None:
    """Return path to the saved runtime file for *prefix*, or None."""
    matches = glob.glob(os.path.join(_RUNTIME_DIR, f"{prefix}.*"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _annotate(pil_img: Image.Image, bbox: tuple, color_bgr: tuple, label: str) -> Image.Image:
    bgr = _pil_to_bgr(pil_img)
    x, y, w, h = bbox
    cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, 3)
    cv2.putText(bgr, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _swatch(color_rgb: tuple, size: tuple = (80, 50)) -> Image.Image:
    return Image.new("RGB", size, color_rgb)


def _gif_html(gif_bytes: bytes, width: str = "100%") -> str:
    b64 = base64.b64encode(gif_bytes).decode()
    return (
        f'<img src="data:image/gif;base64,{b64}" '
        f'style="width:{width}; border-radius:6px;" />'
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Paint Colour Matcher", layout="wide")

st.title("Paint Colour Matcher")
st.caption(
    "Photograph your paint stroke next to the printed calibration palette — "
    "the app finds matching regions in your target image."
)

# ---------------------------------------------------------------------------
# Sidebar: image uploads
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Images")

    pt_file = st.file_uploader(
        "Target image (Pt)", type=["png", "jpg", "jpeg"],
        help="The image in which you want to find matching colour regions.",
    )
    if pt_file:
        _save_runtime(pt_file, "picture")
    else:
        _saved_pt = _find_runtime("picture")
        if _saved_pt:
            st.caption(f"Using saved: {os.path.basename(_saved_pt)}")

    im_file = st.file_uploader(
        "Calibration palette (Im)", type=["png"],
        help="The same PNG palette image you printed.",
    )
    if im_file:
        _save_runtime(im_file, "palette")
    else:
        _saved_im = _find_runtime("palette")
        if _saved_im:
            st.caption(f"Using saved: {os.path.basename(_saved_im)}")

    st.divider()
    st.header("Calibration photo")
    use_webcam = st.toggle("Use webcam", value=True)

    st.divider()
    st.header("About")
    st.markdown(
        "**Steps**\n"
        "1. Upload Pt and Im\n"
        "2. Photograph the printed palette + paint stroke\n"
        "3. Press **Run calibration**\n"
        "4. Adjust tolerance and inspect the blink animation"
    )

# ---------------------------------------------------------------------------
# Guard: need both images before proceeding
# ---------------------------------------------------------------------------

_pt_source  = pt_file  or _find_runtime("picture")
_im_source  = im_file  or _find_runtime("palette")

if not _pt_source or not _im_source:
    st.info("Upload the target image (Pt) and calibration palette (Im) in the sidebar to begin.")
    st.stop()

pt_img = Image.open(_pt_source)
im_img = Image.open(_im_source)

# ---------------------------------------------------------------------------
# Step 1: Preview uploaded images
# ---------------------------------------------------------------------------

st.subheader("Uploaded images")
st.image(pt_img, caption="Target image (Pt)", use_container_width=True)
st.image(im_img, caption="Calibration palette (Im)", width=120)

st.divider()

# ---------------------------------------------------------------------------
# Step 2: Calibration photo capture
# ---------------------------------------------------------------------------

st.subheader("Calibration photo")

if use_webcam:
    st.markdown(
        "Place the **printed palette** and your **paint stroke paper** in frame. "
        "Use diffuse lighting — avoid direct sunlight or harsh shadows."
    )
    photo_file = st.camera_input("Capture photo")
else:
    photo_file = st.file_uploader(
        "Upload calibration photo", type=["png", "jpg", "jpeg"], key="calib_photo"
    )

if not photo_file:
    st.stop()

photo_pil = Image.open(photo_file)
photo_bgr = _pil_to_bgr(photo_pil)

st.divider()

# ---------------------------------------------------------------------------
# Step 3: Run calibration
# ---------------------------------------------------------------------------

st.subheader("Calibration")

run_btn = st.button("Run calibration", type="primary")

if run_btn:
    # Clear previous results so the UI stays consistent
    for key in ("palette_colors", "ccm", "paint_color_digital", "palette_bbox", "paint_bbox"):
        st.session_state.pop(key, None)

    # 3a. Extract digital palette colours from Im
    with st.spinner("Extracting palette colours from Im…"):
        palette_colors, _ = extract_palette_colors(im_img)

    if not palette_colors:
        st.error("Could not extract any colour bars from the palette image. "
                 "Ensure Im contains clearly distinct solid-colour bars.")
        st.stop()

    st.session_state["palette_colors"] = palette_colors

    # Show extracted digital colours
    st.markdown(f"**Detected {len(palette_colors)} palette bar(s)**")
    st.image(
        [_swatch(c) for c in palette_colors],
        width=80,
        caption=[f"Bar {i + 1}" for i in range(len(palette_colors))],
    )

    # 3b. Detect palette in the photo
    with st.spinner("Locating palette in photo…"):
        palette_bbox, _dbg_mask = detect_palette_in_photo(photo_bgr, palette_colors)

    if palette_bbox is None:
        st.error(
            "Cannot recognise calibration palette in the photo.\n\n"
            "Tips: ensure the printed palette is fully visible with diffuse lighting "
            "and no strong glare."
        )
        st.stop()

    st.session_state["palette_bbox"] = palette_bbox

    # 3c. Sample photographed palette colours
    with st.spinner("Sampling photographed palette colours…"):
        photo_colors = sample_palette_in_photo(photo_bgr, palette_bbox, len(palette_colors))

    # Check that palette colours are sufficiently distinct (weak-palette guard)
    from utils.color_utils import rgb_to_lab
    labs = [rgb_to_lab(c) for c in palette_colors]
    min_de = float("inf")
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            de = float(np.linalg.norm(labs[i] - labs[j]))
            min_de = min(min_de, de)

    if min_de < 10:
        st.warning(
            "Palette colours are too similar (min ΔE ≈ {:.1f}). "
            "Colour correction may be inaccurate. "
            "Choose more distinctive colours.".format(min_de)
        )

    # 3d. Estimate CCM
    with st.spinner("Estimating colour correction matrix…"):
        M = estimate_ccm(palette_colors, photo_colors)

    st.session_state["ccm"] = M
    st.session_state["photo_colors"]    = photo_colors
    st.session_state["palette_colors_"] = palette_colors  # preserve for IDW at result time

    # 3e. Detect and refine paint stroke
    with st.spinner("Detecting paint stroke…"):
        coarse_bbox, paint_err = detect_paint_stroke(photo_bgr, palette_bbox, n_bars=len(palette_colors))

    centroid = None
    stroke_bbox = None
    stroke_mask_app = None
    clipped_bbox_app = None
    if coarse_bbox:
        with st.spinner("Refining stroke region…"):
            stroke_bbox, centroid, stroke_mask_app, clipped_bbox_app = refine_stroke_region(
                photo_bgr, coarse_bbox, palette_bbox
            )
        st.session_state["paint_bbox"] = stroke_bbox

    # Annotate the calibration photo
    annotated = _annotate(photo_pil, palette_bbox, (0, 200, 0), "Palette")
    if coarse_bbox:
        annotated = _annotate(annotated, coarse_bbox, (180, 180, 0), "Coarse")
    if stroke_bbox:
        annotated = _annotate(annotated, stroke_bbox, (0, 0, 220), "Stroke")

    st.image(annotated,
             caption="Detected regions — green: palette | yellow: coarse area | blue: refined stroke",
             use_container_width=True)

    if paint_err or coarse_bbox is None:
        st.warning(
            "Cannot recognise paint stroke automatically. "
            "Use the sliders below to mark it manually."
        )
    elif stroke_bbox is None:
        st.warning(
            "Coarse region found but stroke could not be isolated from the background. "
            "Try adjusting lighting or use the manual controls below."
        )

    # 3f. Sample & correct paint colour
    if stroke_mask_app is not None and clipped_bbox_app is not None:
        with st.spinner("Sampling and correcting paint colour…"):
            paint_photo = sample_paint_color(photo_bgr, stroke_mask_app, clipped_bbox_app)
            paint_digital = apply_ccm_idw(paint_photo, photo_colors, palette_colors)

        st.session_state["paint_color_digital"] = paint_digital
        st.session_state["paint_photo"]         = paint_photo

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Photographed paint colour**")
            st.image(_swatch(paint_photo, (120, 60)), width=120)
            st.caption(f"RGB {paint_photo}")
        with c2:
            st.markdown("**Estimated digital equivalent** (after CCM)")
            st.image(_swatch(paint_digital, (120, 60)), width=120)
            st.caption(f"RGB {paint_digital}")

# ---------------------------------------------------------------------------
# Manual paint-region selection (shown when auto-detection failed or as override)
# ---------------------------------------------------------------------------

if "ccm" in st.session_state and "paint_color_digital" not in st.session_state:
    st.markdown("### Manual paint region selection")
    st.markdown("Drag the sliders to frame the paint stroke in the calibration photo.")

    ph, pw = photo_bgr.shape[:2]
    mc1, mc2 = st.columns(2)
    with mc1:
        paint_x = st.slider("X (left edge)", 0, pw - 1, pw // 2, key="man_x")
        paint_w = st.slider("Width", 10, pw // 2, max(20, pw // 8), key="man_w")
    with mc2:
        paint_y = st.slider("Y (top edge)", 0, ph - 1, ph // 2, key="man_y")
        paint_h = st.slider("Height", 10, ph // 2, max(20, ph // 8), key="man_h")

    paint_bbox_man = (paint_x, paint_y, paint_w, paint_h)
    preview = _annotate(photo_pil, paint_bbox_man, (0, 0, 220), "Paint")
    st.image(preview, caption="Manual paint region", use_container_width=True)

    if st.button("Use this region"):
        M = st.session_state["ccm"]
        # Run refinement on the manually selected region too
        stroke_bbox_man, centroid_man, mask_man, clipped_man = refine_stroke_region(
            photo_bgr, paint_bbox_man, st.session_state.get("palette_bbox")
        )
        # Fall back: synthesise a full-region mask from the manual bbox
        if mask_man is None or clipped_man is None:
            cx = paint_x + paint_w // 2
            cy = paint_y + paint_h // 2
            clipped_man = (paint_x, paint_y, paint_w, paint_h)
            mask_man = np.ones((paint_h, paint_w), dtype=np.uint8) * 255
        paint_photo = sample_paint_color(photo_bgr, mask_man, clipped_man)
        _pc = st.session_state.get("palette_colors_", [])
        _ph = st.session_state.get("photo_colors",    [])
        paint_digital = apply_ccm_idw(paint_photo, _ph, _pc) if _pc and _ph else paint_photo
        st.session_state["paint_color_digital"] = paint_digital
        st.session_state["paint_photo"]         = paint_photo
        st.session_state["paint_bbox"] = stroke_bbox_man or paint_bbox_man
        st.rerun()

# ---------------------------------------------------------------------------
# Step 4: Results
# ---------------------------------------------------------------------------

if "paint_color_digital" not in st.session_state:
    st.stop()

st.divider()
st.subheader("Matching results")

paint_digital: tuple = st.session_state["paint_color_digital"]
paint_photo: tuple   = st.session_state.get("paint_photo", paint_digital)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Photographed stroke**")
    st.image(_swatch(paint_photo, (120, 60)), width=120)
    st.caption(f"RGB {paint_photo}")
with col_b:
    st.markdown("**Digital equivalent (matched against Pt)**")
    st.image(_swatch(paint_digital, (120, 60)), width=120)
    st.caption(f"RGB {paint_digital}")

# ΔE statistics: give the user a concrete anchor for the tolerance slider
from calibration.color_matcher import find_matching_regions as _fmr
from utils.color_utils import rgb_image_to_lab, rgb_to_lab, delta_e_image
_target_lab  = rgb_image_to_lab(np.array(pt_img.convert("RGB"), dtype=np.uint8))
_paint_lab   = rgb_to_lab(paint_digital)
_de_flat     = delta_e_image(_target_lab, _paint_lab).ravel()
_de_p5, _de_p25, _de_p50 = float(np.percentile(_de_flat, 5)), float(np.percentile(_de_flat, 25)), float(np.percentile(_de_flat, 50))
st.caption(
    f"ΔE in target — closest 5 % of pixels: ≤ {_de_p5:.1f} | "
    f"closest 25 %: ≤ {_de_p25:.1f} | median: {_de_p50:.1f}"
)

tolerance = st.slider(
    "Tolerance (ΔE)",
    min_value=1, max_value=60, value=min(20, max(1, int(_de_p25))),
    help="Lower = stricter match.  Higher = broader match.",
)

with st.spinner("Searching target image…"):
    mask = _de_flat.reshape(_target_lab.shape[:2]) <= tolerance

match_pct = 100.0 * mask.sum() / mask.size
st.metric("Matched area", f"{match_pct:.1f}%")

with st.spinner("Building blink animation…"):
    gif_bytes = create_blink_gif(pt_img, mask)

st.markdown(_gif_html(gif_bytes), unsafe_allow_html=True)
st.caption(
    "**Blink animation** — alternates between the original image and the same image "
    "with matched regions highlighted in green."
)

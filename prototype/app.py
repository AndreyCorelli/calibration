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
from calibration.color_correction import apply_ccm, estimate_ccm
from calibration.paint_detector import detect_paint_stroke, refine_stroke_region, sample_paint_color
from calibration.color_matcher import find_matching_regions
from ui.highlighter import create_blink_gif

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
    im_file = st.file_uploader(
        "Calibration palette (Im)", type=["png"],
        help="The same PNG palette image you printed.",
    )

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

if not pt_file or not im_file:
    st.info("Upload the target image (Pt) and calibration palette (Im) in the sidebar to begin.")
    st.stop()

pt_img = Image.open(pt_file)
im_img = Image.open(im_file)

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

    # 3e. Detect and refine paint stroke
    with st.spinner("Detecting paint stroke…"):
        coarse_bbox, paint_err = detect_paint_stroke(photo_bgr, palette_bbox, n_bars=len(palette_colors))

    centroid = None
    stroke_bbox = None
    if coarse_bbox:
        with st.spinner("Refining stroke region…"):
            stroke_bbox, centroid, _, _ = refine_stroke_region(
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
    if centroid:
        with st.spinner("Sampling and correcting paint colour…"):
            paint_photo = sample_paint_color(photo_bgr, centroid)
            paint_digital = apply_ccm(paint_photo, M)

        st.session_state["paint_color_digital"] = paint_digital

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
        stroke_bbox_man, centroid_man, _, _ = refine_stroke_region(
            photo_bgr, paint_bbox_man, st.session_state.get("palette_bbox")
        )
        # Fall back to centre of the manual bbox if refinement fails
        if centroid_man is None:
            centroid_man = (paint_x + paint_w // 2, paint_y + paint_h // 2)
        paint_photo = sample_paint_color(photo_bgr, centroid_man)
        paint_digital = apply_ccm(paint_photo, M)
        st.session_state["paint_color_digital"] = paint_digital
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

tolerance = st.slider(
    "Tolerance (ΔE)",
    min_value=1, max_value=60, value=20,
    help="Lower = stricter match.  Higher = broader match.",
)

with st.spinner("Searching target image…"):
    mask = find_matching_regions(pt_img, paint_digital, tolerance=float(tolerance))

match_pct = 100.0 * mask.sum() / mask.size
st.metric("Matched area", f"{match_pct:.1f}%")

with st.spinner("Building blink animation…"):
    gif_bytes = create_blink_gif(pt_img, mask)

st.markdown(_gif_html(gif_bytes), unsafe_allow_html=True)
st.caption(
    "**Blink animation** — alternates between the original image and the match mask "
    "(white = matched colour, black = no match)."
)

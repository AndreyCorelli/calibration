"""
Blink animation: generates an animated GIF that alternates between the original
target image and a black-and-white mask (white = colour match, black = no match).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

_MAX_GIF_DIM = 1200  # Resize the GIF if either dimension exceeds this


def create_blink_gif(
    target_pil: Image.Image,
    mask: np.ndarray,
    duration_ms: int = 700,
) -> bytes:
    """
    Parameters
    ----------
    target_pil  : original target image (Pt)
    mask        : boolean (H, W) array — True where colour matches
    duration_ms : display time per frame in milliseconds

    Returns
    -------
    Raw GIF bytes suitable for embedding in an <img> tag or writing to disk.
    """
    target_rgb = target_pil.convert("RGB")
    W, H = target_rgb.size

    # Downscale if too large (GIFs with 256-colour palettes look poor at high res anyway)
    if max(W, H) > _MAX_GIF_DIM:
        scale = _MAX_GIF_DIM / max(W, H)
        new_w, new_h = int(W * scale), int(H * scale)
        target_rgb = target_rgb.resize((new_w, new_h), Image.LANCZOS)
        mask = (
            np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (new_w, new_h), Image.NEAREST
                )
            )
            > 0
        )
        W, H = new_w, new_h

    # Build the mask frame (white = match, black = no match)
    mask_arr = np.zeros((H, W, 3), dtype=np.uint8)
    mask_arr[mask] = [255, 255, 255]
    mask_frame = Image.fromarray(mask_arr)

    buf = io.BytesIO()
    target_rgb.save(
        buf,
        format="GIF",
        save_all=True,
        append_images=[mask_frame],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    buf.seek(0)
    return buf.getvalue()

# Python Desktop Prototype: Paint-to-Image Color Matching App

## Goal

Build a fully functional desktop prototype for macOS, Windows, and Linux.

The app lets the user:

1. load a target image `Pt`
2. load a digital calibration palette image `Im`
3. load a photographed calibration scene containing:
   - printed `Im`
   - one paint stroke inside a hand-drawn rectangle
4. calibrate photographed colors against original digital palette colors
5. estimate the paint stroke’s digital-equivalent color
6. find matching regions in `Pt`
7. preview and export highlighted results

The prototype should be practical and usable, not merely a technical demo.

---

# Recommended Stack

Use:

```text
Python 3.11+
PySide6
OpenCV
NumPy
Pillow
scikit-image
```

Purpose:

```text
PySide6      desktop UI
OpenCV       image processing, contour detection, perspective correction
NumPy        array operations
Pillow       image loading/saving helpers
scikit-image Lab color conversion / ΔE if desired
```

Alternative:

```text
OpenCV can handle Lab conversion directly.
scikit-image is optional but useful.
```

---

# Core Concept

The same palette image `Im` exists in two forms:

```text
digital Im  -> uploaded into app
printed Im  -> photographed next to paint stroke
```

The app compares:

```text
original digital palette colors
vs
photographed printed palette colors
```

Then it estimates how the photographed paint stroke would look in the original digital palette space.

---

# User Workflow

## Step 1 — Load Target Image

User selects:

```text
Pt
```

This is the image where color matches will be searched.

---

## Step 2 — Load Palette Image

User selects:

```text
Im
```

The app validates that `Im` looks like a valid palette.

Expected palette:

```text
vertical strip
3 or more solid color bars
no padding
no gradients
no transparency
```

Example:

```text
+---------+
| Color A |
+---------+
| Color B |
+---------+
| Color C |
+---------+
```

---

## Step 3 — Load Calibration Photo

User selects a photo taken externally, for example using phone/camera.

The photo should contain:

```text
printed Im
small paper with one paint stroke
hand-drawn rectangle around the stroke
```

Example physical layout:

```text
[ Color A ]
[ Color B ]   [ pencil square with paint stroke ]
[ Color C ]
```

---

## Step 4 — Detect Palette

The app detects the printed palette in the calibration photo.

For prototype reliability, detection may be semi-automatic:

- app tries automatic detection first
- user can manually adjust/confirm palette rectangle if needed

Minimum acceptable prototype behavior:

```text
automatic detection when photo is clean
manual fallback when detection fails
```

---

## Step 5 — Detect Stroke Area

The app detects the hand-drawn rectangle near the printed palette.

For prototype reliability:

- app tries automatic rectangle detection first
- user can manually draw/adjust stroke rectangle if needed

Minimum acceptable prototype behavior:

```text
manual stroke box selection must exist
```

This prevents the prototype from being blocked by imperfect contour detection.

---

## Step 6 — Calibrate

The app samples colors from:

```text
digital Im
photographed printed Im
paint stroke area
```

It computes:

```text
photo palette colors -> digital palette colors
```

Then applies this correction to the paint stroke color.

---

## Step 7 — Match Target Image

The app searches `Pt` for pixels close to the estimated paint color.

Recommended:

```text
CIE Lab color space
ΔE-style distance
```

User controls:

```text
tolerance slider
```

---

## Step 8 — Preview Result

The app displays:

- original target image
- highlighted match preview
- optional mask preview
- estimated paint color swatch

---

## Step 9 — Export Result

The app can export:

- highlighted preview PNG
- binary mask PNG
- diagnostic report JSON

---

# UI Requirements

## Main Window Layout

Recommended layout:

```text
+------------------------------------------------------------+
| Toolbar                                                    |
| Load Pt | Load Im | Load Calibration Photo | Calibrate     |
+-------------------------+----------------------------------+
| Left panel              | Main preview                      |
|                         |                                  |
| Target image info       | zoomable image canvas             |
| Palette info            |                                  |
| Calibration status      |                                  |
| Stroke color swatch     |                                  |
| Tolerance slider        |                                  |
| Highlight mode          |                                  |
+-------------------------+----------------------------------+
| Status / warnings                                          |
+------------------------------------------------------------+
```

---

# Required UI Features

## File Loading

Buttons:

```text
Load Target Image
Load Palette Image
Load Calibration Photo
```

Supported formats:

```text
PNG
JPG
JPEG
```

---

## Preview Canvas

The canvas should support:

- zoom in
- zoom out
- pan
- fit to window
- show original
- show result
- show mask

Mouse behavior:

```text
mouse wheel      zoom
drag             pan
double click     fit to window
```

---

## Manual Selection Tools

The prototype should include manual fallback tools:

```text
Select Palette Rectangle
Select Stroke Rectangle
```

User should be able to draw rectangles on the calibration photo.

This is important for a fully functional prototype.

Automatic detection can be imperfect, but the user must still be able to proceed.

---

# Palette Image Validation

After loading `Im`, validate:

- image is RGB-compatible
- no alpha dependency
- has at least 3 color bars
- bars are sufficiently different
- bars are mostly solid

If invalid, show warning.

Example messages:

```text
Palette colors are too similar.
Palette appears noisy or gradient-like.
Use solid color blocks without padding.
```

---

# Palette Bar Extraction

For the prototype, assume vertical stacked bars.

Given `Im`:

```text
height = H
bar_count = N
bar_height = H / N
```

Sample each bar from its central region.

Avoid edges.

Example:

```text
sample x from 30% to 70% width
sample y from 20% to 80% of each bar
```

Use median color, not average.

---

# Calibration Photo Processing

## Palette Detection

Automatic detection should attempt:

1. find large rectangular regions
2. locate stacked color blocks
3. verify that internal sampled colors correspond approximately to uploaded `Im`
4. estimate perspective transform

If automatic detection fails, user selects palette rectangle manually.

---

## Manual Palette Rectangle

The user draws a rectangle around the printed palette.

For the prototype, manual rectangle can be axis-aligned.

Optional improvement:

```text
4-point perspective selection
```

Minimum:

```text
axis-aligned rectangle
```

If photo is taken reasonably straight, this is enough.

---

## Sampling Printed Palette

Once palette rectangle is known:

1. split it into `N` equal horizontal segments
2. sample central area of each segment
3. compute median photographed RGB color

Avoid edges and borders.

---

# Stroke Detection

## Automatic Stroke Box Detection

Search near the palette rectangle.

Recommended search area:

```text
to the right of the middle palette bar
```

Detect likely pencil rectangle:

- rectangular contour
- dark border
- light interior
- reasonable size

---

## Manual Stroke Rectangle

Required fallback.

User draws a rectangle around the paint stroke area.

Sampling should ignore the outer border.

Example:

```text
use inner 70% of selected rectangle
```

---

# Stroke Color Sampling

Inside the stroke rectangle:

1. ignore border area
2. ignore very light background pixels if possible
3. reject outliers
4. compute median paint color

A simple prototype method:

```text
convert area to Lab
estimate background as lightest/most common pixels
select pixels sufficiently different from background
take median of selected pixels
```

Fallback:

```text
take median of central region
```

---

# Color Calibration

## Input

```text
digital palette colors:
D1, D2, D3, ...

photographed palette colors:
P1, P2, P3, ...

photographed stroke color:
S_photo
```

Need estimate:

```text
S_digital
```

---

## Minimum Algorithm

Use per-channel affine correction.

For each RGB channel:

```text
digital_channel = a * photographed_channel + b
```

Fit `a` and `b` using palette samples.

This works with 3+ bars.

Clamp result to:

```text
0..255
```

---

## Better Algorithm

Fit a 3x4 affine color transform:

```text
[Rd]   [a11 a12 a13 b1] [Rp]
[Gd] = [a21 a22 a23 b2] [Gp]
[Bd]   [a31 a32 a33 b3] [Bp]
```

This allows channel mixing.

Needs at least 4 palette colors for a well-conditioned fit.

With only 3 colors, use simpler per-channel correction.

---

## Recommended Prototype Rule

If palette has:

```text
3 colors:
    use per-channel affine correction

4+ colors:
    use 3x4 affine correction
```

---

# Target Matching

## Convert to Lab

Convert:

```text
target image Pt
estimated stroke color
```

to Lab.

Compute distance for each pixel:

```text
distance = LabDistance(pixel, stroke)
```

For prototype:

```text
Euclidean Lab distance is acceptable
```

Later:

```text
CIEDE2000
```

---

## Tolerance

User setting:

```text
tolerance: 1..100
```

Initial default:

```text
20
```

Lower values:
- stricter
- fewer matches

Higher values:
- broader
- more matches

---

# Highlight Modes

## Original

Show original `Pt`.

---

## Mask

Show black/white mask.

```text
white = match
black = non-match
```

---

## Desaturate Non-Matches

Matching pixels remain original.

Non-matching pixels become grayscale.

---

## Tint Non-Matches

Matching pixels remain original.

Non-matching pixels are tinted or darkened.

---

## Blink Mode

Optional for prototype.

Matching pixels alternate between:

```text
black
white
```

Useful but not required for first fully functional version.

---

# Export

## Export Highlighted Image

Save current preview as:

```text
PNG
```

---

## Export Mask

Save binary mask as:

```text
PNG
```

---

## Export Diagnostics

Save JSON:

```json
{
  "target_image": "...",
  "palette_image": "...",
  "calibration_photo": "...",
  "palette_color_count": 3,
  "digital_palette_rgb": [],
  "photo_palette_rgb": [],
  "photo_stroke_rgb": [],
  "estimated_stroke_rgb": [],
  "tolerance": 20,
  "highlight_mode": "desaturate_non_matches"
}
```

---

# Diagnostics Panel

The app should show:

- loaded image sizes
- number of palette colors
- sampled digital palette colors
- sampled photo palette colors
- sampled photo stroke color
- corrected estimated stroke color
- current tolerance
- number/percentage of matched pixels

---

# Error Handling

## Missing Inputs

```text
Load target image first.
Load palette image first.
Load calibration photo first.
```

---

## Invalid Palette

```text
Palette is invalid or too noisy.
Use solid color bars without padding.
```

---

## Palette Detection Failed

```text
Could not detect printed palette automatically.
Please select the palette rectangle manually.
```

---

## Stroke Detection Failed

```text
Could not detect paint stroke automatically.
Please select the stroke rectangle manually.
```

---

## Calibration Failed

```text
Could not compute reliable color correction.
Try using more distinct palette colors.
```

---

# Project Structure

Recommended:

```text
paint_matcher/
  app/
    main.py
    ui/
      main_window.py
      image_canvas.py
      selection_tools.py
    core/
      image_io.py
      palette.py
      calibration.py
      stroke.py
      matching.py
      highlight.py
      diagnostics.py
    models/
      data.py
    resources/
      icons/
  tests/
    test_palette.py
    test_calibration.py
    test_matching.py
  pyproject.toml
  README.md
```

---

# Core Modules

## image_io.py

Responsibilities:

- load image as RGB NumPy array
- save image
- convert between PIL/QImage/NumPy

---

## palette.py

Responsibilities:

- validate palette image
- extract digital palette colors
- sample printed palette colors from selected rectangle
- check color distinctness

---

## calibration.py

Responsibilities:

- fit color correction transform
- apply transform to stroke color
- clamp output RGB values

---

## stroke.py

Responsibilities:

- detect stroke rectangle if possible
- sample representative stroke color
- reject background pixels

---

## matching.py

Responsibilities:

- convert RGB to Lab
- compute per-pixel color distance
- build match mask

---

## highlight.py

Responsibilities:

- generate preview images
- generate grayscale/tinted/mask modes

---

## diagnostics.py

Responsibilities:

- collect calibration metadata
- export JSON report

---

# Testing Requirements

## Unit Tests

Test:

- palette color extraction
- palette distinctness validation
- affine color correction
- Lab matching
- highlight mask generation

---

## Synthetic Tests

Generate artificial images:

- palette with known colors
- simulated photographed palette with known color shift
- simulated paint stroke
- target image with known matching regions

Verify:

```text
estimated stroke color is close to expected
matched regions are detected
```

---

# MVP Acceptance Criteria

The prototype is considered fully functional when:

1. user can load `Pt`
2. user can load `Im`
3. user can load calibration photo
4. app extracts digital palette colors
5. user can select palette rectangle manually
6. user can select stroke rectangle manually
7. app computes estimated paint color
8. app finds matching pixels in `Pt`
9. user can adjust tolerance
10. preview updates after tolerance change
11. user can switch highlight modes
12. user can export result PNG
13. user can export mask PNG
14. original `Pt` is never modified

Automatic detection is desirable but not required for MVP completeness.

Manual fallback is required.

---

# Implementation Priority

## Phase 1 — Functional Core

- load images
- extract palette colors
- manual palette rectangle
- manual stroke rectangle
- color correction
- Lab matching
- preview result

---

## Phase 2 — Usable Desktop UI

- zoom/pan canvas
- tolerance slider
- highlight mode selector
- diagnostics panel
- export buttons

---

## Phase 3 — Automatic Assistance

- auto-detect palette
- auto-detect stroke rectangle
- warnings for poor photo quality
- better sampling

---

## Phase 4 — Polishing

- better visual design
- persistent settings
- recent files
- export diagnostics
- packaging for macOS

---

# Notes

For the first version, prioritize reliability over automation.

A manual but accurate workflow is better than an automatic workflow that fails unpredictably.

The essential prototype is:

```text
load images
select palette rectangle
select stroke rectangle
calibrate
match
preview
export
```
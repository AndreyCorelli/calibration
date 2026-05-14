# Paint-to-Image Color Matching Android App

## Overview

An Android application for matching a real-world paint color to regions of a digital image.

The user:

1. uploads a target image `Pt`
2. uploads a calibration palette image `Im`
3. prints `Im`
4. places a paint stroke near the printed palette
5. photographs the setup using the app
6. receives a visualization of regions in `Pt` matching the paint color

The application performs approximate color calibration between:
- original digital palette colors
- printed palette photographed by the device camera

Then it estimates the original digital equivalent of the photographed paint stroke color and searches the target image for matching regions.

This application is intended for personal/private use and not commercial distribution.

---

# Goals

The app should:

- estimate the digital color equivalent of a real paint stroke
- compensate partially for:
  - camera white balance
  - printer color shifts
  - lighting conditions
- find similar colors inside a target image
- provide configurable highlighting/preview modes

The app is NOT expected to provide professional-grade color science accuracy.

---

# Terminology

## Pt

Target image.

The image in which matching color regions are searched.

Example:
- artwork
- reference image
- photo
- render

---

## Im

Calibration palette image.

A specially prepared image containing several large solid color bars.

The same exact digital image is:
- uploaded into the app
- printed physically

The printed version is photographed during calibration.

---

# Calibration Palette Requirements

## General

The palette image should:

- contain several large solid-color bars
- contain no gradients
- contain no noise
- contain no transparency
- contain no padding/margins between bars
- use PNG format
- preferably use sRGB

---

## Recommended Layout

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

Vertical arrangement is preferred.

---

## Recommended Resolution

Example:
- 200x600 PNG

Exact size is not important.

What matters:
- large solid regions
- easy recognition
- clear separation between colors

---

## Recommended Colors

The bars should use clearly distinctive colors.

Recommended:
- red
- green
- blue

or:
- black
- white
- red
- green
- blue

Avoid:
- similar colors
- dark-on-dark combinations
- low saturation colors

---

# Physical Setup

## Printed Palette

The user prints the uploaded palette image `Im`.

The printout may be reused multiple times.

---

## Paint Stroke

The user creates:
- one paint stroke
- inside a hand-drawn rectangle/square
- on a separate small piece of paper

The small paper is placed:
- near the printed palette
- preferably to the right of the middle bar
- not necessarily pixel-perfect

Example:

```text
[ Color A ]
[ Color B ]   [ pencil square with paint stroke ]
[ Color C ]
```

---

# Calibration Capture Workflow

## User Actions

The user presses:
- "Calibrate"

The app:
- opens the camera
- shows capture instructions

The user photographs:
- printed palette
- paint stroke paper

Both should fit inside the camera frame.

---

# Camera Capture Recommendations

The app should recommend:

- avoid shadows
- avoid glare
- use diffuse lighting
- keep camera approximately perpendicular
- avoid motion blur
- keep all elements visible

---

# Image Processing Pipeline

## Step 1 — Detect Palette

The app detects:
- palette position
- orientation
- scale
- perspective

The app identifies:
- all color bars

---

## Step 2 — Sample Palette Colors

For each detected bar:

- sample photographed color
- compare with original digital color from uploaded `Im`

---

## Step 3 — Estimate Color Correction

Using:
- photographed palette colors
- original palette colors

Estimate a color transform.

Purpose:
- partially compensate for:
  - printer shift
  - lighting shift
  - camera shift

Exact algorithm is implementation-defined.

---

## Step 4 — Detect Paint Stroke Region

The app searches:
- near the palette
- preferably near the middle bar

The app detects:
- pencil rectangle/square
- paint stroke inside

If multiple regions are found:
- choose largest plausible candidate

---

## Step 5 — Sample Paint Color

Extract representative paint color.

Recommended:
- median color
- robust against:
  - glare
  - shadows
  - paper visibility
  - brush texture

---

## Step 6 — Convert Paint Color

Apply inverse/derived color correction.

Estimate:
- original digital equivalent of photographed paint color

---

## Step 7 — Search Target Image

Search image `Pt` for:
- visually similar colors

Recommended:
- Lab color space
- ΔE color distance

Avoid:
- naive RGB Euclidean distance

---

# Matching Settings

## Tolerance

User-adjustable.

Controls:
- strictness of matching

Lower:
- fewer matches
- more precise

Higher:
- broader matches

---

# Highlighting Modes

## Mode 1 — Desaturation

- matching regions remain unchanged
- non-matching regions become grayscale

---

## Mode 2 — Tint Overlay

- non-matching regions tinted:
  - red
  - blue
  - gray
  - configurable

---

## Mode 3 — Blink

Matching regions alternate:
- black
- white

Purpose:
- visually emphasize matching areas

---

## Mode 4 — Mask Overlay

Semi-transparent overlay highlighting matches.

---

# Original Image Preservation

The original target image `Pt` must remain unchanged.

All highlighting should:
- operate on previews
- generate derived images only

---

# Error Handling

The app should detect and report failures.

Examples:

## Palette Not Found

```text
Cannot recognize calibration palette.
```

---

## Paint Stroke Not Found

```text
Cannot recognize paint stroke.
```

---

## Weak Palette

```text
Palette colors are too similar.
Choose more distinctive colors.
```

---

## Poor Lighting

```text
Lighting conditions are unsuitable for calibration.
```

---

# Technical Recommendations

## Color Space

Recommended:
- CIE Lab

Recommended similarity metric:
- ΔE

---

## Perspective Correction

Recommended:
- homography/perspective transform

---

## Robust Sampling

Recommended:
- median color
- outlier rejection

Avoid:
- raw average color

---

# Non-Goals

The app is NOT intended to:

- provide spectrophotometer-grade accuracy
- identify paint brands
- estimate pigment chemistry
- provide printer calibration
- perform professional color management

---

# Complexity Estimate

## Prototype

Moderate complexity.

Main components:
- image upload
- camera capture
- palette detection
- color correction
- paint detection
- color matching
- preview rendering

---

## Difficult Areas

Most difficult parts:
- reliable color correction
- paint detection robustness
- lighting variability
- printer inconsistency

---

# Future Ideas

Possible future improvements:

- multiple paint strokes
- reusable calibration cards
- QR/marker-assisted palette detection
- live preview mode
- histogram visualization
- automatic dominant-color extraction
- clustering similar regions
- exportable masks
- compare several candidate paint colors
- palette recommendations
- camera white balance locking
- RAW image capture

---
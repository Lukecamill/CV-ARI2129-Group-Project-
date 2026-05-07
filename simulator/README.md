# Watershed Segmentation Simulator

An interactive Streamlit simulator that walks step by step through a
**marker-controlled watershed** segmentation pipeline using OpenCV.
It is part of a Computer Vision Learning Pack and is designed so that
peers can change parameters, inspect every intermediate result, and
build intuition about how the algorithm works.

## What this simulator does

The simulator implements the classic OpenCV watershed recipe and
renders every intermediate stage:

1. Load an input image (uploaded or a synthetic fallback).
2. Convert to grayscale.
3. Apply a Gaussian blur to denoise.
4. Threshold the image (Otsu or manual, with optional inversion).
5. Apply morphological opening to clean the mask.
6. Dilate to obtain the *sure background* region.
7. Compute the distance transform of the cleaned mask.
8. Threshold the distance transform to obtain the *sure foreground*.
9. Subtract the two masks to obtain the *unknown* region.
10. Label markers with `cv2.connectedComponents`.
11. Apply `cv2.watershed`.
12. Overlay watershed boundaries on the original image.
13. Display a final labelled segmentation.

Each stage is shown alongside a short explanation of what it does and
why it matters, plus a panel of warnings about common failure modes.

## Project layout

```
simulator/
├── app.py              # Streamlit UI and parameter controls
├── watershed_utils.py  # Reusable pipeline functions and visualisation helpers
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

Requires Python 3.9+.

```bash
pip install -r simulator/requirements.txt
```

The dependency list uses `opencv-python-headless` so it works on
machines without a display server (CI, remote VMs, etc.). On a local
desktop you may swap it for `opencv-python` if you prefer.

## Running

From the repository root:

```bash
streamlit run simulator/app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`); open it
in a browser.

## Expected input images

The simulator works best with images that have:

- Reasonably uniform illumination.
- Objects that are clearly brighter (or darker) than the background.
- Round-ish objects that may touch or overlap (the typical watershed
  use case: cells, coins, beans, grains).

If you do not upload anything, the app generates a synthetic image of
overlapping circular blobs so the pipeline still runs.

## Adjustable parameters

| Parameter                          | What it controls                                                                 |
|------------------------------------|----------------------------------------------------------------------------------|
| **Blur kernel size**               | Gaussian filter size used to denoise. Larger = smoother but blurrier edges.      |
| **Threshold method**               | `otsu` picks the cut-off automatically; `manual` lets you set the value.          |
| **Manual threshold value**         | The cut-off (0-255) when `manual` is selected.                                    |
| **Invert threshold**               | Tick this when objects are *darker* than the background.                          |
| **Morphology kernel size**         | Size of the structuring element for the opening operation.                       |
| **Morphology iterations**          | How many times the opening is repeated.                                          |
| **Dilation iterations**            | Controls how large the "sure background" region is.                              |
| **Foreground threshold ratio**     | Fraction of `max(distance_transform)` used to keep only strong foreground peaks. |

## Why marker-controlled watershed?

A vanilla watershed treats every local minimum (or every local
maximum, depending on the gradient image) as a marker. Real images
have many noisy local extrema, so plain watershed produces a mess of
tiny regions - the textbook **over-segmentation** problem.

Marker-controlled watershed avoids this by giving the algorithm a
small, deliberate set of seeds: a handful of *sure foreground* points
inside each object plus a *sure background* outside. The flooding then
only happens in the *unknown* band between them, so each object ends
up with exactly one label.

## Troubleshooting

- **"Too many regions / objects split into pieces"** - increase the
  foreground threshold ratio, increase the blur kernel, or add more
  morphology iterations.
- **"Two touching objects merged into one"** - decrease the foreground
  threshold ratio so each object retains its own distance-transform
  peak, or use a stronger opening.
- **"Background is being labelled as object"** - try inverting the
  threshold, or switch from Otsu to manual.
- **"Streamlit not found"** - make sure you installed
  `requirements.txt` in the active environment.
- **"Distance transform is empty / sure foreground is black"** - the
  cleaned mask probably has no foreground left. Lower the morphology
  iterations, or check if you need to invert the threshold.
- **"OpenCV import errors on a server"** - use
  `opencv-python-headless` (already in `requirements.txt`).

---


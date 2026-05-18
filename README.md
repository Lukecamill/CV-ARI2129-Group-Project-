# ARI2129 — Watershed Segmentation Learning Pack

A Computer Vision (ARI2129) group project that explains **marker-controlled
watershed segmentation** as a small, self-contained learning pack. It pairs a
step-by-step code walkthrough with an interactive simulator, a presentation,
study notes, and a quiz, so a reader can both understand the theory and
experiment with the algorithm hands-on.

## Contents

| Path                          | What it is                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `Tutorial/watershed.ipynb`    | Code walkthrough that builds the watershed pipeline from scratch.           |
| `Tutorial/images/`            | Sample image (`water_coins.jpg`) used by the tutorial.                      |
| `simulator/`                  | Interactive Streamlit app for experimenting with the pipeline.              |
| `Comp_Vis_slides.pptx` / `.pdf` | Project presentation slides.                                              |
| `study notes cv.docx` / `.pdf`  | Study notes covering the underlying computer vision concepts.             |
| `quiz_with_rationale.docx` / `.pdf` | Quiz questions with explained answers; `quiz_link.txt` holds the form link. |
| `requirements.txt`            | Python dependencies for the tutorial and simulator.                         |

## The tutorial

`Tutorial/watershed.ipynb` walks through marker-controlled watershed one stage
at a time, with a markdown explanation before each step:

1. Load the raw image and convert BGR → RGB.
2. Convert to grayscale.
3. Threshold into a binary image.
4. Apply the distance transform.
5. Generate markers — sure foreground, sure background, and the unknown band.
6. Label markers with `cv2.connectedComponents`.
7. Compute a morphological gradient image as the flooding surface.
8. Build a priority queue ordered by gradient value.
9. Run a manual flooding implementation of the watershed algorithm.

The notebook does not just call `cv2.watershed` — it implements the flooding
process itself so the mechanics are visible.

## The simulator

`simulator/` is a Streamlit app that runs the classic OpenCV watershed recipe
and renders every intermediate result, with parameters you can change live.
See `simulator/README.md` for full details.

## Installation

Requires Python 3.9+.

```bash
pip install -r requirements.txt
```

The dependency list uses `opencv-python-headless` so it works on machines
without a display server.

## Running

Tutorial notebook:

```bash
jupyter notebook Tutorial/watershed.ipynb
```

Interactive simulator (from the repository root):

```bash
streamlit run simulator/app.py
```

## Group project

This repository is the deliverable for the ARI2129 Computer Vision group task.
Alongside the code it includes the presentation, study notes, and quiz listed
above. The AI usage journal is exported separately as `ai_journal.pdf`.

## Github REPO link
https://github.com/Lukecamill/CV-ARI2129-Group-Project-.git

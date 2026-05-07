"""
app.py
------
Streamlit front-end for the marker-controlled watershed segmentation
simulator. The heavy lifting lives in `watershed_utils`; this module is
intentionally focused on the UI: parameter controls, layout, captions
and warnings.

Run with:

    pip install -r requirements.txt
    streamlit run simulator/app.py

Preset model
------------
Each entry in `wu.DEMO_IMAGES` has its own tuned configuration in
`DEMO_PARAMETER_PRESETS`. The flow is:

1. User picks a demo from the "Demo image" dropdown.
2. The app detects that the demo has changed (current value differs
   from `st.session_state["last_demo_choice"]`) and copies the
   demo's recommended parameters into session state - exactly once.
3. The user can freely tweak any slider afterwards. Manual edits stay
   put across reruns; presets are NOT re-applied on every interaction.
4. The "Reset parameters for this demo" button copies the demo's
   recommended parameters back over whatever the user has changed.
5. Uploading a custom image is independent of the preset machinery -
   the demo dropdown still drives the parameters even if the image
   shown is an upload.

This avoids both the manual-pre-tuning chore and the rerun-loop
pitfalls of writing into session state for already-rendered widgets.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass
from typing import Any, Dict

import cv2
import numpy as np
import streamlit as st
from PIL import Image

import watershed_utils as wu


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Watershed Segmentation Simulator",
    page_icon=":droplet:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Widget keys
# ---------------------------------------------------------------------------
# Single source of truth for session-state key names. Both the preset
# dictionaries and the `PipelineParams` dataclass refer to these
# constants, so a typo can only fail in one place.

K_DEMO               = "demo_choice"
K_LAST_DEMO          = "last_demo_choice"
K_MODE_LABEL         = "mode_label"
K_GRADIENT_LABEL     = "gradient_method_label"
K_BLUR               = "blur_kernel"
K_THRESH_METHOD      = "threshold_method"
K_THRESH_VALUE       = "manual_threshold"
K_INVERT             = "invert_threshold"
K_MORPH_KERNEL       = "morph_kernel"
K_MORPH_ITER         = "morph_iterations"
K_DILATION_ITER      = "dilation_iterations"
K_FG_METHOD_LABEL    = "foreground_method_label"
K_FG_RATIO           = "foreground_ratio"
K_PEAK_NB            = "peak_neighbourhood"
K_PEAK_MIN_H         = "peak_min_height"
K_MIN_AREA           = "min_component_area"
K_ALPHA              = "overlay_alpha"
K_CONTOURS           = "show_contours"
K_LABEL_IDS          = "show_label_ids"


# ---------------------------------------------------------------------------
# Per-demo parameter presets
# ---------------------------------------------------------------------------
# Each preset is a complete snapshot of every tunable widget for one
# demo image. Keys MUST match `wu.DEMO_IMAGES` exactly. Note that the
# demo-name key (`K_DEMO`) is NOT included - the demo IS the selector,
# not a parameter to itself.

DEMO_PARAMETER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Overlapping blobs (default)": {
        # The user-supplied "balanced" parameters - cleanest separation
        # of the touching circular blobs in the synthetic scene.
        K_MODE_LABEL:      "Binary mask watershed",
        K_GRADIENT_LABEL:  "Sobel (L2 magnitude)",
        K_BLUR:            9,
        K_THRESH_METHOD:   "manual",
        K_THRESH_VALUE:    151,
        K_INVERT:          False,
        K_MORPH_KERNEL:    7,
        K_MORPH_ITER:      4,
        K_DILATION_ITER:   4,
        K_FG_METHOD_LABEL: "Local maxima (peaks)",
        K_FG_RATIO:        0.45,
        K_PEAK_NB:         11,
        K_PEAK_MIN_H:      0.15,
        K_MIN_AREA:        10,
        K_ALPHA:            0.0,
        K_CONTOURS:        True,
        K_LABEL_IDS:       True,
    },
    "Touching coins": {
        # Coins are darker than the light grey background -> invert.
        # Centres are ~75 px apart with radius 38, so neighbouring coins
        # touch along a thin neck. Notes on the retune:
        #   * MORPH_KERNEL=7, MORPH_ITER=3 erodes ~9 px - enough to break
        #     every touching neck (the previous 5/2 left one pair fused,
        #     yielding 14 components instead of 15).
        #   * After that erosion the per-coin distance peaks sit at ~29 px,
        #     so PEAK_MIN_H drops to 0.2 (it was 0.3 of the original ~34 px
        #     peak height, which is now too high relative to the smaller
        #     post-morphology peaks).
        #   * PEAK_NB=31 gives each peak a comfortable margin within the
        #     ~75 px centre spacing without merging neighbours.
        #   * MIN_AREA=5: peaks are dilated into ~5x5 discs (~19 px area)
        #     by compute_sure_foreground_peaks. The previous floor of 30 px
        #     filtered every disc away, leaving zero foreground markers.
        K_MODE_LABEL:      "Binary mask watershed",
        K_GRADIENT_LABEL:  "Sobel (L2 magnitude)",
        K_BLUR:            5,
        K_THRESH_METHOD:   "otsu",
        K_THRESH_VALUE:    127,
        K_INVERT:          True,
        K_MORPH_KERNEL:    7,
        K_MORPH_ITER:      3,
        K_DILATION_ITER:   3,
        K_FG_METHOD_LABEL: "Local maxima (peaks)",
        K_FG_RATIO:        0.5,
        K_PEAK_NB:         31,
        K_PEAK_MIN_H:      0.2,
        K_MIN_AREA:        5,
        K_ALPHA:            0.35,
        K_CONTOURS:        True,
        K_LABEL_IDS:       True,
    },
    "Cells / fluorescent blobs": {
        # Bright irregular ellipses placed randomly with a soft Gaussian
        # halo. This preset balances two common failure modes:
        #   * If the peak neighbourhood is too large, nearby cell centres
        #     suppress each other and touching cells merge together.
        #   * If the peak neighbourhood / minimum height / area filter are
        #     too loose, tiny noise peaks become markers and one cell can be
        #     split into several regions.
        # These values aim to keep one strong marker per visible cell while
        # filtering out very small noisy peaks.
        K_MODE_LABEL:      "Binary mask watershed",
        K_GRADIENT_LABEL:  "Sobel (L2 magnitude)",
        K_BLUR:            7,
        K_THRESH_METHOD:   "otsu",
        K_THRESH_VALUE:    127,
        K_INVERT:          False,
        K_MORPH_KERNEL:    3,
        K_MORPH_ITER:      2,
        K_DILATION_ITER:   3,
        K_FG_METHOD_LABEL: "Local maxima (peaks)",
        K_FG_RATIO:        0.4,
        K_PEAK_NB:         15,
        K_PEAK_MIN_H:      0.15,
        K_MIN_AREA:        10,
        K_ALPHA:           0.4,
        K_CONTOURS:        True,
        K_LABEL_IDS:       False,
    },
    "Cells / fluorescent blobs": {
        # Bright irregular ellipses placed randomly with a soft Gaussian
        # halo. This preset balances two common failure modes:
        #   * If the peak neighbourhood is too large, nearby cell centres
        #     suppress each other and touching cells merge together.
        #   * If the peak neighbourhood / minimum height / area filter are
        #     too loose, tiny noise peaks become markers and one cell can be
        #     split into several regions.
        # These values aim to keep one strong marker per visible cell while
        # filtering out very small noisy peaks.
        K_MODE_LABEL:      "Binary mask watershed",
        K_GRADIENT_LABEL:  "Sobel (L2 magnitude)",
        K_BLUR:            7,
        K_THRESH_METHOD:   "otsu",
        K_THRESH_VALUE:    127,
        K_INVERT:          False,
        K_MORPH_KERNEL:    3,
        K_MORPH_ITER:      2,
        K_DILATION_ITER:   3,
        K_FG_METHOD_LABEL: "Local maxima (peaks)",
        K_FG_RATIO:        0.4,
        K_PEAK_NB:         15,
        K_PEAK_MIN_H:      0.15,
        K_MIN_AREA:        10,
        K_ALPHA:           0.4,
        K_CONTOURS:        True,
        K_LABEL_IDS:       False,
    },
    "Pills / capsules": {
        # Elongated objects: distance transform has a ridge along the
        # major axis, so a wider peak window + higher min height stop
        # one capsule from being split into multiple seeds.
        K_MODE_LABEL:      "Binary mask watershed",
        K_GRADIENT_LABEL:  "Sobel (L2 magnitude)",
        K_BLUR:            5,
        K_THRESH_METHOD:   "otsu",
        K_THRESH_VALUE:    127,
        K_INVERT:          False,
        K_MORPH_KERNEL:    5,
        K_MORPH_ITER:      2,
        K_DILATION_ITER:   4,
        K_FG_METHOD_LABEL: "Local maxima (peaks)",
        K_FG_RATIO:        0.5,
        K_PEAK_NB:         31,
        K_PEAK_MIN_H:      0.4,
        K_MIN_AREA:        50,
        K_ALPHA:            0.4,
        K_CONTOURS:        True,
        K_LABEL_IDS:       True,
    },
    "Rice grains": {
        # Many small grains (16x6 px) - need a small peak window and
        # lighter morphology to avoid erasing tiny objects entirely.
        K_MODE_LABEL:      "Binary mask watershed",
        K_GRADIENT_LABEL:  "Sobel (L2 magnitude)",
        K_BLUR:            3,
        K_THRESH_METHOD:   "otsu",
        K_THRESH_VALUE:    127,
        K_INVERT:          False,
        K_MORPH_KERNEL:    3,
        K_MORPH_ITER:      1,
        K_DILATION_ITER:   2,
        K_FG_METHOD_LABEL: "Local maxima (peaks)",
        K_FG_RATIO:        0.4,
        K_PEAK_NB:         9,
        K_PEAK_MIN_H:      0.15,
        K_MIN_AREA:        5,
        K_ALPHA:            0.45,
        K_CONTOURS:        True,
        K_LABEL_IDS:       False,
    },
}

# Used both as the initial value and as the fallback if a future demo
# image is added without an entry in DEMO_PARAMETER_PRESETS.
DEFAULT_DEMO = next(iter(wu.DEMO_IMAGES.keys()))


# ---------------------------------------------------------------------------
# Session-state plumbing
# ---------------------------------------------------------------------------

def get_preset_for_demo(demo: str) -> Dict[str, Any]:
    """Return the recommended parameters for `demo`, falling back gracefully."""
    return DEMO_PARAMETER_PRESETS.get(
        demo, DEMO_PARAMETER_PRESETS[DEFAULT_DEMO]
    )


def apply_demo_preset(demo: str) -> None:
    """Copy the demo's recommended values into `st.session_state`.

    This only writes the *parameter* keys - never the demo selector
    itself, so calling it cannot overwrite the user's demo choice.
    """
    for key, value in get_preset_for_demo(demo).items():
        st.session_state[key] = value


def initialize_session_state() -> None:
    """Seed session state on the very first script run.

    Subsequent runs see the `_initialised` sentinel and skip this
    block, so the user's manual tweaks persist across reruns.
    """
    if st.session_state.get("_initialised"):
        return
    st.session_state[K_DEMO] = DEFAULT_DEMO
    apply_demo_preset(DEFAULT_DEMO)
    st.session_state[K_LAST_DEMO] = DEFAULT_DEMO
    st.session_state["_initialised"] = True


def sync_preset_with_demo() -> None:
    """Apply the demo's preset *only* when the demo selection changes.

    By comparing against `last_demo_choice` we apply the preset on
    transitions only - so changing the demo loads recommended values
    once, but a slider tweak (or any other interaction) does not.
    """
    current = st.session_state[K_DEMO]
    if current != st.session_state.get(K_LAST_DEMO):
        apply_demo_preset(current)
        st.session_state[K_LAST_DEMO] = current


def reset_to_demo_preset() -> None:
    """`on_click` callback for the reset button.

    Re-applies the current demo's recommended parameters. The callback
    runs before the next render, so subsequent widget creations will
    see the reset values.
    """
    apply_demo_preset(st.session_state[K_DEMO])


initialize_session_state()
sync_preset_with_demo()


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineParams:
    """Strongly-typed snapshot of every pipeline-relevant slider value.

    Centralises the label-to-internal-code mapping for radio/select
    widgets so the rest of the app can pass `**asdict(params)` directly
    to `wu.run_pipeline`.
    """

    blur_kernel: int
    threshold_method: str
    manual_threshold: int
    invert_threshold: bool
    morph_kernel: int
    morph_iterations: int
    dilation_iterations: int
    foreground_method: str       # "ratio" | "peaks"
    foreground_ratio: float
    peak_neighbourhood: int
    peak_min_height: float
    min_component_area: int
    mode: str                    # "binary" | "gradient"
    gradient_method: str         # "sobel" | "morphological"
    overlay_alpha: float
    show_contours: bool
    show_label_ids: bool

    @classmethod
    def from_session(cls) -> "PipelineParams":
        ss = st.session_state
        return cls(
            blur_kernel=int(ss[K_BLUR]),
            threshold_method=str(ss[K_THRESH_METHOD]),
            manual_threshold=int(ss[K_THRESH_VALUE]),
            invert_threshold=bool(ss[K_INVERT]),
            morph_kernel=int(ss[K_MORPH_KERNEL]),
            morph_iterations=int(ss[K_MORPH_ITER]),
            dilation_iterations=int(ss[K_DILATION_ITER]),
            foreground_method=(
                "peaks" if str(ss[K_FG_METHOD_LABEL]).startswith("Local") else "ratio"
            ),
            foreground_ratio=float(ss[K_FG_RATIO]),
            peak_neighbourhood=int(ss[K_PEAK_NB]),
            peak_min_height=float(ss[K_PEAK_MIN_H]),
            min_component_area=int(ss[K_MIN_AREA]),
            mode=("binary" if str(ss[K_MODE_LABEL]).startswith("Binary") else "gradient"),
            gradient_method=(
                "morphological"
                if str(ss[K_GRADIENT_LABEL]).startswith("Morphological")
                else "sobel"
            ),
            overlay_alpha=float(ss[K_ALPHA]),
            show_contours=bool(ss[K_CONTOURS]),
            show_label_ids=bool(ss[K_LABEL_IDS]),
        )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Watershed Segmentation Simulator")
st.markdown(
    """
    This simulator walks through a **marker-controlled watershed**
    segmentation pipeline step by step. Each built-in demo image ships
    with its own tuned parameters; pick a demo in the sidebar and the
    recommended settings load automatically. Tweak any slider afterwards
    to explore - your edits stay put until you switch demos or click
    **Reset parameters for this demo**.

    Two segmentation surfaces are supported:

    - **Binary mask watershed** floods the original intensity image,
      starting from markers we derive from the distance transform.
    - **Gradient magnitude watershed** floods the *edge strength*
      image - the textbook topographic interpretation, more robust on
      objects with internal texture.
    """
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_from_upload(uploaded_file) -> np.ndarray:
    """Decode an uploaded file (PNG/JPG/etc.) into an OpenCV BGR array."""
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Demo image")
    st.selectbox(
        "Demo image (drives recommended parameters)",
        options=list(wu.DEMO_IMAGES.keys()),
        key=K_DEMO,
        help=(
            "Each demo has its own tuned configuration. Selecting a "
            "different demo loads its recommended parameters once; "
            "your manual tweaks afterwards are preserved until you "
            "change demo or click the reset button below."
        ),
    )
    st.button(
        "Reset parameters for this demo",
        on_click=reset_to_demo_preset,
        help="Restore the recommended parameters for the currently-selected demo.",
        use_container_width=True,
    )
    uploaded = st.file_uploader(
        "...or upload your own (display only)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help=(
            "Uploaded images REPLACE the displayed image but do NOT "
            "change parameters. The currently-selected demo still "
            "drives the recommended preset."
        ),
    )

    st.header("2. Segmentation mode")
    st.radio(
        "Watershed surface",
        options=["Binary mask watershed", "Gradient magnitude watershed"],
        key=K_MODE_LABEL,
        help=(
            "Binary mode floods the original intensity image. "
            "Gradient mode floods the edge-strength image - usually "
            "better when objects have internal texture or shading."
        ),
    )
    is_gradient_mode = st.session_state[K_MODE_LABEL].startswith("Gradient")
    st.selectbox(
        "Gradient operator",
        options=["Sobel (L2 magnitude)", "Morphological gradient"],
        key=K_GRADIENT_LABEL,
        help="How the gradient magnitude image is estimated.",
        disabled=not is_gradient_mode,
    )

    st.header("3. Pre-processing")
    st.slider(
        "Blur kernel size",
        min_value=1, max_value=25, step=2,
        key=K_BLUR,
        help="Gaussian kernel size. Larger values smooth more but blur weak edges.",
    )

    st.header("4. Thresholding")
    st.radio(
        "Method", options=["otsu", "manual"], horizontal=True,
        key=K_THRESH_METHOD,
        help="Otsu picks the cut-off automatically; manual lets you pick it.",
    )
    is_manual_threshold = st.session_state[K_THRESH_METHOD] == "manual"
    st.slider(
        "Manual threshold value",
        min_value=0, max_value=255, step=1,
        key=K_THRESH_VALUE,
        help="Only used when 'manual' is selected.",
        disabled=not is_manual_threshold,
    )
    st.checkbox(
        "Invert threshold (dark objects on light background)",
        key=K_INVERT,
        help="Tick this when objects are darker than the background.",
    )

    st.header("5. Morphology")
    st.slider(
        "Morphology kernel size",
        min_value=1, max_value=15, step=2,
        key=K_MORPH_KERNEL,
        help="Size of the structuring element used for opening.",
    )
    st.slider(
        "Morphology iterations",
        min_value=1, max_value=10, step=1,
        key=K_MORPH_ITER,
        help="How many times the opening is repeated.",
    )

    st.header("6. Sure background")
    st.slider(
        "Dilation iterations (sure background)",
        min_value=1, max_value=10, step=1,
        key=K_DILATION_ITER,
        help="Higher = larger 'sure background' region.",
    )

    st.header("7. Sure foreground (markers)")
    st.radio(
        "Seed extraction",
        options=["Distance ratio (global)", "Local maxima (peaks)"],
        key=K_FG_METHOD_LABEL,
        help=(
            "Distance-ratio thresholds the distance transform globally - "
            "fast but biased against small objects. Local maxima look "
            "for peaks in a sliding window - much better at separating "
            "touching objects of unequal size."
        ),
    )
    is_peaks_method = st.session_state[K_FG_METHOD_LABEL].startswith("Local")
    st.slider(
        "Foreground threshold ratio",
        min_value=0.1, max_value=0.9, step=0.05,
        key=K_FG_RATIO,
        help="Fraction of max(distance) kept as foreground (ratio mode).",
        disabled=is_peaks_method,
    )
    st.slider(
        "Peak neighbourhood (px)",
        min_value=3, max_value=51, step=2,
        key=K_PEAK_NB,
        help=(
            "Window size for the local-maxima search. Roughly the "
            "minimum spacing between distinct objects you expect."
        ),
        disabled=not is_peaks_method,
    )
    st.slider(
        "Peak minimum height (fraction of max)",
        min_value=0.0, max_value=0.9, step=0.05,
        key=K_PEAK_MIN_H,
        help="Suppress peaks shorter than this fraction of the global max distance.",
        disabled=not is_peaks_method,
    )

    st.header("8. Noise filtering")
    st.slider(
        "Minimum marker area (px)",
        min_value=0, max_value=300, step=5,
        key=K_MIN_AREA,
        help=(
            "Drop foreground components smaller than this before markers "
            "are built. Tiny markers cause classic over-segmentation."
        ),
    )

    st.header("9. Visualisation")
    st.slider(
        "Label overlay transparency",
        min_value=0.0, max_value=1.0, step=0.05,
        key=K_ALPHA,
        help="0 = original image only. 1 = pure label colours.",
    )
    st.checkbox(
        "Show region contours",
        key=K_CONTOURS,
        help="Trace each detected region with its own coloured outline.",
    )
    st.checkbox(
        "Show label IDs",
        key=K_LABEL_IDS,
        help="Annotate each region with its watershed label number.",
    )

    st.markdown("---")
    st.caption(
        "Tip: tweak one slider at a time and watch how the markers and "
        "final boundaries change. Switch demos to load a different "
        "tuned configuration."
    )


# ---------------------------------------------------------------------------
# Resolve parameters and input image
# ---------------------------------------------------------------------------

params = PipelineParams.from_session()
demo_choice = st.session_state[K_DEMO]

if uploaded is not None:
    image_bgr = load_image_from_upload(uploaded)
    image_source_note = (
        f"Using uploaded image. Parameters are still tuned for the "
        f"**{demo_choice}** demo - tweak the sliders if your image "
        "needs different settings."
    )
else:
    image_bgr = wu.DEMO_IMAGES[demo_choice]()
    image_source_note = (
        f"Showing the **{demo_choice}** demo with its recommended "
        "parameters. Switch demos in the sidebar, or upload an image, "
        "to override the display."
    )

st.info(image_source_note)


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

result = wu.run_pipeline(image_bgr=image_bgr, **asdict(params))


# ---------------------------------------------------------------------------
# Top-level statistics panel
# ---------------------------------------------------------------------------

stats_cols = st.columns(4)
stats_cols[0].metric(
    "Connected components in opened mask",
    result.num_opened_components,
    help="Distinct white blobs in the cleaned binary mask. If two "
         "touching objects share a blob, the count stays at 1.",
)
stats_cols[1].metric(
    "Markers fed to watershed",
    result.num_markers,
    help="Background label (1) + one label per sure-foreground component.",
)
stats_cols[2].metric(
    "Detected objects (post-watershed)",
    result.num_objects,
    help="Distinct labels >= 2 in the watershed output.",
)
stats_cols[3].metric(
    "Watershed ridge pixels",
    result.ridge_pixel_count,
    help="Pixels labelled -1 by cv2.watershed - the segmentation boundaries.",
)


# ---------------------------------------------------------------------------
# Helper for laying out a labelled image with a caption
# ---------------------------------------------------------------------------

def show_image(column, image: np.ndarray, title: str, description: str) -> None:
    """Render one stage: heading, image, short explanation."""
    column.markdown(f"**{title}**")
    column.image(image, channels="RGB", use_container_width=True)
    column.caption(description)


# ---------------------------------------------------------------------------
# Stage 1-2: Original and grayscale
# ---------------------------------------------------------------------------

st.subheader("Stages 1 - 2: Load and convert to grayscale")
col1, col2 = st.columns(2)
show_image(
    col1,
    wu.bgr_to_rgb(result.original_bgr),
    "Original image",
    "The raw input. Watershed works on intensity, so colour information "
    "is collapsed in the next step.",
)
show_image(
    col2,
    result.gray,
    "Grayscale",
    "Single-channel intensity image. Bright pixels usually correspond to "
    "objects (or background, depending on your scene).",
)


# ---------------------------------------------------------------------------
# Stage 3-4: Blur and threshold
# ---------------------------------------------------------------------------

st.subheader("Stages 3 - 4: Blur and threshold")
col1, col2 = st.columns(2)
show_image(
    col1,
    result.blurred,
    "Blurred (denoised)",
    "Gaussian blur removes high-frequency noise that would otherwise "
    "create spurious markers. Too much blur, however, erases real edges.",
)
show_image(
    col2,
    result.threshold_mask,
    "Threshold mask",
    "Binarises the image into foreground (white) and background (black). "
    "Otsu picks the cut-off automatically by maximising inter-class variance.",
)


# ---------------------------------------------------------------------------
# Stage 5-6: Opening and sure background
# ---------------------------------------------------------------------------

st.subheader("Stages 5 - 6: Morphological opening and sure background")
col1, col2 = st.columns(2)
show_image(
    col1,
    result.opened,
    "Opening result",
    "Erosion followed by dilation. Wipes out small noise specks while "
    "preserving the overall object shapes.",
)
show_image(
    col2,
    result.sure_background,
    "Sure background",
    "We dilate the cleaned mask: anything outside this region is "
    "definitely background, which is one of the seeds for watershed.",
)


# ---------------------------------------------------------------------------
# Stage 7-8: Distance transform and sure foreground
# ---------------------------------------------------------------------------

st.subheader("Stages 7 - 8: Distance transform and sure foreground")
col1, col2 = st.columns(2)
distance_visual = wu.normalise_for_display(result.distance_transform)
distance_coloured = cv2.applyColorMap(distance_visual, cv2.COLORMAP_VIRIDIS)
show_image(
    col1,
    cv2.cvtColor(distance_coloured, cv2.COLOR_BGR2RGB),
    "Distance transform",
    "Every foreground pixel is replaced by its distance to the nearest "
    "background pixel: D(p) = min over background q of ||p - q||. "
    "The peaks sit at object centres, so we use them as seeds when the "
    "binary mask has fused touching objects.",
)
fg_caption = (
    "Threshold of the distance transform. Higher 'foreground ratio' = "
    "fewer, more conservative seeds, which helps separate touching objects."
    if params.foreground_method == "ratio"
    else
    "Local maxima of the distance transform - one peak per object, "
    "even when the objects have very different sizes. Each peak is "
    "dilated into a small disc so connected components can latch onto it."
)
show_image(
    col2,
    result.sure_foreground,
    "Sure foreground (after noise filter)",
    fg_caption + (
        f" Tiny components below {params.min_component_area} px were dropped."
        if params.min_component_area > 0
        else ""
    ),
)


# ---------------------------------------------------------------------------
# Stage 9-10: Unknown region and markers
# ---------------------------------------------------------------------------

st.subheader("Stages 9 - 10: Unknown region and marker labelling")
col1, col2 = st.columns(2)
show_image(
    col1,
    result.unknown,
    "Unknown region",
    "Sure background minus sure foreground. This is the band of pixels "
    "that watershed will assign to either an object or the background.",
)
markers_preview = wu.colourise_markers(result.markers)
show_image(
    col2,
    cv2.cvtColor(markers_preview, cv2.COLOR_BGR2RGB),
    "Markers before watershed",
    "Connected components of the sure foreground, plus background label. "
    "Each colour is one starting seed. Watershed will flood from these "
    "seeds outward, label by label, until two flood fronts meet on a ridge.",
)


# ---------------------------------------------------------------------------
# Stage 11: Gradient surface (always shown so binary vs gradient is visible)
# ---------------------------------------------------------------------------

st.subheader("Stage 11: Gradient magnitude (the alternative flooding surface)")
gradient_coloured = cv2.applyColorMap(result.gradient, cv2.COLORMAP_INFERNO)
flood_caption = (
    "**Currently flooding the gradient image** - low values are basins, "
    "high values are ridges, so the watershed ridges land precisely on "
    "object edges."
    if params.mode == "gradient"
    else
    "Currently using **binary mode** - watershed is flooding the original "
    "image, not this gradient. Switch the segmentation mode in the sidebar "
    "to flood the gradient surface instead. This image is shown for comparison."
)
show_image(
    st,
    cv2.cvtColor(gradient_coloured, cv2.COLOR_BGR2RGB),
    f"Gradient magnitude ({params.gradient_method})",
    flood_caption,
)


# ---------------------------------------------------------------------------
# Stage 12-13: Watershed result
# ---------------------------------------------------------------------------

st.subheader(f"Stages 12 - 13: Run watershed ({params.mode} mode) and visualise")
col1, col2 = st.columns(2)
show_image(
    col1,
    wu.bgr_to_rgb(result.boundaries_overlay),
    "Watershed boundaries",
    "Red lines are pixels where `cv2.watershed` placed a ridge between "
    "neighbouring labels. These are the segmentation boundaries.",
)
show_image(
    col2,
    wu.bgr_to_rgb(result.labelled_overlay),
    "Final labelled segmentation",
    f"Each region tinted with a unique colour (alpha={params.overlay_alpha:.2f}). "
    f"Detected object count: **{result.num_objects}**. "
    "Toggle contours / label IDs in the sidebar for extra annotation.",
)


# ---------------------------------------------------------------------------
# Mathematical / conceptual companion
# ---------------------------------------------------------------------------

with st.expander("Mathematical intuition (why this whole pipeline works)", expanded=False):
    st.markdown(
        r"""
        **The topographic metaphor.**
        Treat a grayscale image as a 3-D landscape: pixel coordinates are
        the (x, y) and pixel intensity is the elevation. Bright pixels
        are mountain tops; dark pixels are valleys. Now imagine puncturing
        every basin and slowly raising the water level. Water rises in
        each basin separately until two pools are about to merge - at
        which point we build a dam. Those dams, taken together, are the
        **watershed lines**.

        **Why naive watershed over-segments.**
        Real images contain many small local minima from noise and
        texture. Each minimum becomes a separate basin and therefore a
        separate region. The result is a mosaic of hundreds of useless
        regions - the textbook over-segmentation problem.

        **Why we need markers.**
        Marker-controlled watershed swaps "every minimum is a seed" for
        "only my chosen labels are seeds". The flood only starts at
        explicit markers, so the number of regions is bounded by the
        number of markers. Choosing markers well is the entire game.

        **Distance transform: D(p) = min_{q in background} ||p - q||.**
        Each foreground pixel is labelled with its Euclidean distance to
        the nearest background pixel. For one convex blob, D peaks near
        the centroid. For two **touching** blobs, D has a saddle between
        them and one local maximum inside each. Thresholding D (or
        finding local maxima of D) therefore gives one seed per object
        even when the binary mask fused them - this is the trick that
        rescues watershed for touching circles.

        **Gradient magnitude as the topographic surface.**
        Instead of flooding the raw intensity image, we can flood the
        gradient magnitude `|grad I|`. Now flat regions inside an object
        are valleys (low gradient) and the object's boundary is a ridge
        (high gradient). The watershed dams therefore land exactly on
        the boundaries, which is what segmentation actually wants. This
        is closer to the original Beucher and Meyer (1993) formulation.
        """
    )


# ---------------------------------------------------------------------------
# Failure modes / limitations
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Common failure modes and limitations")

with st.expander("Click to expand: failure modes, limitations, and alternatives", expanded=False):
    st.warning(
        """
        **Over-segmentation** - too many markers split one object into several
        pieces. Usually caused by a low foreground-ratio, insufficient blur,
        a tiny peak neighbourhood, or noise leaking through to the markers.
        Remedies: increase the foreground ratio, increase the peak
        neighbourhood, raise the minimum-area filter, or blur more.
        """
    )
    st.warning(
        """
        **Under-segmentation** - two touching objects share one marker and
        end up labelled as one region. Remedies: switch to "Local maxima"
        for the foreground extraction, lower the foreground ratio, increase
        morphology iterations, or feed watershed the gradient image instead
        of the binary surface.
        """
    )
    st.warning(
        """
        **Noisy images** - speckle noise produces fake local extrema and
        spurious markers. Tune the blur kernel and the morphological
        opening to remove them, but do not over-smooth real edges.
        """
    )
    st.warning(
        """
        **Weak or fuzzy boundaries** - watershed leans on intensity gradients;
        if your edges are very faint, the ridge can drift across the
        unknown region. Consider edge enhancement (Canny gradient, CLAHE,
        unsharp mask) before the pipeline, or use the gradient mode.
        """
    )
    st.warning(
        """
        **Textured backgrounds** - background texture can outscore real
        object edges in the gradient image, sending the flood front in
        the wrong direction. Pre-segment with colour or texture cues, or
        suppress the background with a top-hat / bottom-hat transform.
        """
    )
    st.warning(
        """
        **Uneven illumination** - a single global threshold may capture only
        the bright half of the image. Pre-process with histogram
        equalisation, CLAHE, or background subtraction before thresholding.
        Adaptive thresholding (`cv2.adaptiveThreshold`) is another option.
        """
    )
    st.warning(
        """
        **Poor thresholding** - if the threshold is too high, objects break
        up; too low, objects merge into the background. Otsu fails when
        the histogram is not bimodal; manual thresholds need to be retuned
        per scene. Try multiple methods before blaming the watershed.
        """
    )
    st.warning(
        """
        **Incorrect markers** - if a marker straddles two objects or sits
        outside its object entirely, no amount of tuning downstream can
        recover. Inspect the "Markers before watershed" stage carefully -
        if it looks wrong, no later stage will be right.
        """
    )

    st.markdown(
        """
        ### When watershed is **not** a good choice

        - Objects with **fuzzy or absent boundaries** (e.g. a tumour
          fading into surrounding tissue): watershed needs a ridge to
          flood up to.
        - **Heavily textured** scenes: too many local extrema even after
          blurring; the gradient image is dominated by texture instead
          of object boundaries.
        - **Semantic segmentation** (cat vs. dog, road vs. sidewalk):
          watershed knows nothing about object identity. It only finds
          *regions*, not *classes*.
        - **Cluttered overlapping objects** with no consistent geometry
          (broken glass, crowd scenes): markers cannot be reliably
          derived from the distance transform.

        ### Alternatives worth knowing

        - **k-means / GMM clustering** on (R, G, B) or (R, G, B, x, y).
          Fast, parameter-light, but ignores spatial coherence.
        - **Region growing** from manually chosen seeds. Conceptually
          similar to watershed but with hand-crafted homogeneity rules.
        - **Graph cuts (min-cut / max-flow)**. Optimises a global
          energy balancing region appearance and boundary smoothness;
          much more robust to weak boundaries.
        - **Active contours / level sets**. Evolve a curve to minimise
          an energy functional; great for smooth object outlines, but
          sensitive to initialisation.
        - **Mean-shift segmentation**. Mode-seeking in the joint
          colour-spatial space; produces smooth piecewise-constant
          regions without any seeds.
        - **Deep learning (U-Net, Mask R-CNN, SAM)**. State of the art
          for medical, satellite, and natural images when labelled data
          is available; handles texture, occlusion, and class
          information that classical watershed ignores.

        Watershed shines when objects are **roundish**, **well separated
        in shape**, and **brighter (or darker) than the background** -
        coins, cells, grains, beans, pills. Outside that regime, the
        alternatives above usually win.
        """
    )

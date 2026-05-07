"""
watershed_utils.py
------------------
Reusable building blocks for a marker-controlled watershed segmentation
pipeline. Each function performs one well-defined step so the Streamlit
front-end can render the intermediate result of every stage.

Pipeline (binary-mask mode, classic OpenCV recipe):

    grayscale -> blur -> threshold -> morphological opening
              -> sure background (dilation)
              -> distance transform -> sure foreground (threshold OR peaks)
              -> remove tiny components (noise filter)
              -> unknown = bg - fg
              -> connected-component markers
              -> cv2.watershed on the original image
              -> draw boundaries + labelled overlay

Pipeline (gradient-magnitude mode):

    Same marker derivation, but `cv2.watershed` is fed the **gradient
    magnitude** image (3-channel) instead of the original. Watershed
    treats low values as basins and high values as ridges, so feeding
    it a gradient image makes the algorithm explicitly "climb the
    edges" - exactly the topographic interpretation the textbook uses.

Why the gradient-magnitude variant matters:
    Plain watershed on the *original* intensity image works when objects
    are flat-bright and the background is flat-dark. As soon as an object
    has internal texture (e.g. a coin's relief, a cell's nucleus,
    illumination gradients) the original image has many false basins
    inside the object. The gradient magnitude image is near-zero in flat
    regions and only spikes at edges, so the flood fronts meet exactly
    on the object boundaries. That is why most watershed papers use
    `|grad I|` rather than `I` itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class WatershedResult:
    """Holds every intermediate image and statistic produced by the pipeline.

    Storing them together keeps the Streamlit layer simple: one call into
    `run_pipeline` returns everything the UI needs to display.
    """

    original_bgr: np.ndarray        # input image in BGR (OpenCV native)
    gray: np.ndarray                # single-channel grayscale
    blurred: np.ndarray             # denoised grayscale
    threshold_mask: np.ndarray      # binary foreground/background mask
    opened: np.ndarray              # mask after morphological opening
    sure_background: np.ndarray     # certain background region
    distance_transform: np.ndarray  # raw distance transform (float32)
    sure_foreground: np.ndarray     # certain foreground region (post-filter)
    sure_foreground_raw: np.ndarray # foreground before tiny-component filter
    unknown: np.ndarray             # ambiguous region (bg - fg)
    markers: np.ndarray             # int32 marker image fed to watershed
    markers_after: np.ndarray       # marker image returned by cv2.watershed
    boundaries_overlay: np.ndarray  # original image with red boundary lines
    labelled_overlay: np.ndarray    # original blended with random label colours
    gradient: np.ndarray            # gradient magnitude image (always computed)
    flood_surface_bgr: np.ndarray   # the 3-channel image actually flooded
    num_objects: int                # number of detected foreground regions
    num_markers: int                # markers fed to watershed (objects + bg)
    num_opened_components: int      # connected components in the opened mask
    ridge_pixel_count: int          # number of -1 pixels (watershed ridges)
    mode: str = "binary"            # "binary" or "gradient"


# ---------------------------------------------------------------------------
# Synthetic demo images
# ---------------------------------------------------------------------------
#
# We ship several built-in scenes so the simulator is never empty and so
# students can compare watershed behaviour across canonical cases. Every
# generator returns a BGR uint8 array shaped (H, W, 3) so it slots into
# the same pipeline as a user upload.

def _add_noise(image: np.ndarray, sigma: float = 8.0, seed: int = 0) -> np.ndarray:
    """Add light Gaussian noise so denoising stages have something to do."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_synthetic_blobs(size: int = 400) -> np.ndarray:
    """Several overlapping bright circular blobs on a dark background.

    The textbook watershed scene: plain thresholding fuses the touching
    blobs into one component, so good segmentation requires the
    distance-transform / marker machinery.
    """
    image = np.full((size, size, 3), 30, dtype=np.uint8)
    blob_specs = [
        ((140, 160), 60, (235, 235, 235)),
        ((200, 170), 55, (220, 220, 220)),
        ((170, 230), 50, (245, 245, 245)),
        ((270, 250), 65, (230, 230, 230)),
        ((110, 280), 45, (240, 240, 240)),
        ((300, 130), 40, (225, 225, 225)),
    ]
    for centre, radius, colour in blob_specs:
        cv2.circle(image, centre, radius, colour, thickness=-1, lineType=cv2.LINE_AA)
    return _add_noise(image, sigma=8.0, seed=1)


def make_synthetic_coins(size: int = 420) -> np.ndarray:
    """Touching circles of similar size - the canonical "coins" example.

    Equal-radius coins make this a stress test for foreground extraction:
    the distance-transform peaks are all roughly the same height, so a
    global ratio cutoff has to be tuned carefully.
    """
    image = np.full((size, size, 3), 200, dtype=np.uint8)  # light grey background
    radius = 38
    centres = [
        (90, 100), (165, 110), (240, 100), (315, 115),
        (110, 180), (190, 195), (270, 185), (350, 195),
        (130, 270), (210, 280), (290, 270), (365, 285),
        (155, 360), (235, 365), (315, 355),
    ]
    rng = np.random.default_rng(7)
    for cx, cy in centres:
        shade = int(rng.integers(60, 130))
        cv2.circle(image, (cx, cy), radius, (shade, shade, shade), -1, cv2.LINE_AA)
        # Slight inner ring so coins have a tiny bit of texture (good for
        # demonstrating why gradient watershed > intensity watershed here).
        cv2.circle(image, (cx, cy), radius - 8, (shade + 30, shade + 30, shade + 30), 2, cv2.LINE_AA)
    return _add_noise(image, sigma=4.0, seed=2)


def make_synthetic_cells(size: int = 420) -> np.ndarray:
    """Irregular bright "cells" of varying size and orientation."""
    image = np.full((size, size, 3), 25, dtype=np.uint8)
    rng = np.random.default_rng(11)
    for _ in range(14):
        cx = int(rng.integers(60, size - 60))
        cy = int(rng.integers(60, size - 60))
        ax = int(rng.integers(28, 55))
        ay = int(rng.integers(28, 55))
        angle = int(rng.integers(0, 180))
        shade = int(rng.integers(200, 250))
        cv2.ellipse(image, (cx, cy), (ax, ay), angle, 0, 360,
                    (shade, shade, shade), -1, cv2.LINE_AA)
    # Soft halo so neighbours bleed into each other - mimics fluorescent cells.
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return _add_noise(image, sigma=10.0, seed=3)


def make_synthetic_pills(size: int = 420) -> np.ndarray:
    """Capsule-shaped pills laid out at angles - elongated touching objects."""
    image = np.full((size, size, 3), 40, dtype=np.uint8)
    rng = np.random.default_rng(17)
    for _ in range(12):
        cx = int(rng.integers(70, size - 70))
        cy = int(rng.integers(70, size - 70))
        length = int(rng.integers(70, 110))
        thick = int(rng.integers(22, 32))
        angle = float(rng.integers(0, 180))
        shade = int(rng.integers(210, 250))
        # Build the capsule by drawing a thick antialiased line.
        rad = np.deg2rad(angle)
        dx = int((length / 2) * np.cos(rad))
        dy = int((length / 2) * np.sin(rad))
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        cv2.line(image, p1, p2, (shade, shade, shade),
                 thickness=thick, lineType=cv2.LINE_AA)
    return _add_noise(image, sigma=6.0, seed=4)


def make_synthetic_rice(size: int = 420) -> np.ndarray:
    """Many small rice-grain ellipses scattered at random angles.

    Stress test for noise filtering: a cluster of small objects should
    survive, but tiny stray dots from noise should not become markers.
    """
    image = np.full((size, size, 3), 35, dtype=np.uint8)
    rng = np.random.default_rng(23)
    for _ in range(45):
        cx = int(rng.integers(20, size - 20))
        cy = int(rng.integers(20, size - 20))
        angle = int(rng.integers(0, 180))
        shade = int(rng.integers(210, 245))
        cv2.ellipse(image, (cx, cy), (16, 6), angle, 0, 360,
                    (shade, shade, shade), -1, cv2.LINE_AA)
    return _add_noise(image, sigma=8.0, seed=5)


# Registry consumed by the Streamlit demo dropdown. Keeping the labels
# here rather than in `app.py` means the UI stays declarative.
DEMO_IMAGES: Dict[str, Callable[[], np.ndarray]] = {
    "Overlapping blobs (default)": make_synthetic_blobs,
    "Touching coins": make_synthetic_coins,
    "Cells / fluorescent blobs": make_synthetic_cells,
    "Pills / capsules": make_synthetic_pills,
    "Rice grains": make_synthetic_rice,
}


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------

def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert an image to single-channel grayscale.

    Watershed-style thresholding generally operates on intensity, and
    most morphology kernels assume a single channel.
    """
    if image_bgr.ndim == 2:
        return image_bgr.copy()
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_blur(gray: np.ndarray, kernel_size: int) -> np.ndarray:
    """Smooth the image with a Gaussian blur.

    Blurring suppresses pixel-level noise that would otherwise create
    spurious local minima/maxima and trigger over-segmentation later.
    The kernel size is forced to be odd because Gaussian filters require it.
    """
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size == 1:
        return gray.copy()
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def threshold_image(
    gray: np.ndarray,
    method: str = "otsu",
    manual_value: int = 127,
    invert: bool = False,
) -> np.ndarray:
    """Binarise the image into foreground / background."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "otsu":
        flag |= cv2.THRESH_OTSU
        _, mask = cv2.threshold(gray, 0, 255, flag)
    else:
        _, mask = cv2.threshold(gray, int(manual_value), 255, flag)
    return mask


def morphological_opening(
    mask: np.ndarray,
    kernel_size: int,
    iterations: int,
) -> np.ndarray:
    """Open the mask: erode then dilate to wipe small specks."""
    kernel_size = max(1, int(kernel_size))
    iterations = max(1, int(iterations))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)


def compute_sure_background(opened_mask: np.ndarray, dilation_iterations: int) -> np.ndarray:
    """Dilate the cleaned mask to obtain a region we are *sure* is background."""
    dilation_iterations = max(1, int(dilation_iterations))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(opened_mask, kernel, iterations=dilation_iterations)


def compute_distance_transform(opened_mask: np.ndarray) -> np.ndarray:
    """Distance from every foreground pixel to the nearest background pixel.

    Mathematical intuition:
        D(p) = min over background pixels q of ||p - q||_2

    For a single convex object, D peaks roughly at the centroid. For two
    *touching* objects there is a saddle between them, with one local
    maximum sitting near the centre of each. That is why the distance
    transform - not the binary mask - is the right surface to seed
    watershed from when objects merge under thresholding.

    We use DIST_L2 with a 5x5 mask: it is an accurate Euclidean
    distance approximation while remaining cheap.
    """
    return cv2.distanceTransform(opened_mask, cv2.DIST_L2, 5)


def compute_sure_foreground(
    distance: np.ndarray,
    foreground_ratio: float,
) -> np.ndarray:
    """Threshold the distance transform globally to keep only strong peaks.

    `foreground_ratio` is a fraction of the **global** maximum distance.
    This is fast and intuitive but biases against small objects: if the
    biggest object has distance 50 and the smallest has distance 8, a
    ratio of 0.5 keeps only pixels with distance >= 25, erasing the
    small object entirely. The "peaks" method below avoids this.
    """
    foreground_ratio = float(np.clip(foreground_ratio, 0.0, 1.0))
    if distance.size == 0 or distance.max() == 0:
        return np.zeros_like(distance, dtype=np.uint8)
    cut_off = foreground_ratio * distance.max()
    _, sure_fg = cv2.threshold(distance, cut_off, 255, cv2.THRESH_BINARY)
    return sure_fg.astype(np.uint8)


def compute_sure_foreground_peaks(
    distance: np.ndarray,
    neighbourhood: int = 11,
    min_height_ratio: float = 0.2,
    dilation_size: int = 5,
) -> np.ndarray:
    """Local-maxima ("peak") seeds in the distance transform.

    Why this beats a global ratio for touching objects:

      * A peak is defined relative to its NEIGHBOURHOOD, not the whole
        image. Two touching objects of very different size each get a
        peak, even though one peak is much shorter than the other.
      * The minimum-height filter only suppresses peaks that are
        clearly noise (a fraction of the global max); legitimate small
        objects survive.
      * Dilating each peak into a small disc gives the connected-
        components stage something to grab onto, so the resulting
        markers are robust to one-pixel jitter.

    A pixel is a local maximum when its value equals the value of the
    pixel-wise dilation by a square structuring element of side
    `neighbourhood`. Implemented with `cv2.dilate` to stay OpenCV-only
    (no scipy dependency).
    """
    if distance.size == 0 or distance.max() == 0:
        return np.zeros(distance.shape, dtype=np.uint8)

    neighbourhood = max(3, int(neighbourhood) | 1)  # force odd, >= 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (neighbourhood, neighbourhood))
    dilated = cv2.dilate(distance, kernel)

    # `eq` flags pixels where the distance equals the local max - that's
    # the textbook definition of a local maximum (modulo plateaus).
    is_peak = (distance == dilated)
    is_peak &= distance > min_height_ratio * float(distance.max())

    # Convert the sparse Boolean peak map into solid disc seeds. Without
    # this step, single-pixel peaks would be too thin to survive the
    # later subtraction (sure_bg - sure_fg).
    peaks = is_peak.astype(np.uint8) * 255
    if dilation_size >= 3:
        seed_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )
        peaks = cv2.dilate(peaks, seed_kernel)
    return peaks


def remove_small_components(
    mask: np.ndarray,
    min_area: int,
) -> Tuple[np.ndarray, int]:
    """Drop connected components whose area is below `min_area`.

    Why this matters: tiny seeds caused by noise become individual
    markers and produce textbook over-segmentation. Filtering them out
    *before* watershed is far cheaper - and more interpretable - than
    trying to merge spurious regions afterwards.

    Returns the cleaned mask and the count of *removed* components.
    """
    if min_area <= 1 or mask.size == 0:
        return mask.copy(), 0

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    removed = 0
    for i in range(1, num):  # skip background label 0
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
        else:
            removed += 1
    return out, removed


def compute_unknown_region(
    sure_background: np.ndarray,
    sure_foreground: np.ndarray,
) -> np.ndarray:
    """The "unknown" band sits between confident foreground and background."""
    return cv2.subtract(sure_background, sure_foreground)


def build_markers(
    sure_foreground: np.ndarray,
    unknown: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Convert the foreground mask into integer marker labels.

    Marker convention used by `cv2.watershed`:
        0  -> "I don't know" (the watershed will fill these in)
        1  -> background
        >1 -> distinct object IDs

    We add 1 to the connected-component output so background becomes 1,
    then explicitly set the unknown band back to 0.
    """
    num_components, components = cv2.connectedComponents(sure_foreground)
    markers = components + 1
    markers[unknown == 255] = 0
    num_objects = max(0, num_components - 1)
    return markers.astype(np.int32), num_objects


def compute_gradient_magnitude(
    gray: np.ndarray,
    method: str = "sobel",
) -> np.ndarray:
    """Topographic surface for gradient-based watershed.

    Two common ways to estimate `|grad I|`:

      * "sobel": L2 norm of the Sobel x/y derivatives. Gives smooth,
        well-localised edge magnitudes; the standard textbook choice.
      * "morphological": dilation - erosion. Cheap, robust, and matches
        what some morphology textbooks call the "Beucher gradient";
        slightly thicker edges than Sobel.

    The output is normalised to uint8 so it can be fed directly to
    `cv2.watershed`, which insists on an 8-bit 3-channel input.
    """
    if method == "morphological":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def run_watershed(
    image_bgr: np.ndarray,
    markers: np.ndarray,
) -> np.ndarray:
    """Apply OpenCV's watershed on a copy of the markers."""
    markers_copy = markers.copy()
    cv2.watershed(image_bgr, markers_copy)
    return markers_copy


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def draw_boundaries(image_bgr: np.ndarray, markers_after: np.ndarray) -> np.ndarray:
    """Overlay watershed ridges (where markers == -1) in red on the image."""
    overlay = image_bgr.copy()
    overlay[markers_after == -1] = (0, 0, 255)
    return overlay


def colour_labels(
    image_bgr: np.ndarray,
    markers_after: np.ndarray,
    alpha: float = 0.45,
    show_contours: bool = False,
    show_label_ids: bool = False,
) -> np.ndarray:
    """Blend a random-colour label map with the original.

    `alpha` controls how much of the label colour bleeds through
    (0 = pure original, 1 = pure label colours). `show_contours`
    traces a thin white outline around each region; `show_label_ids`
    annotates every region with its numeric watershed label at its
    centroid - useful for cross-referencing with the marker preview.
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    h, w = markers_after.shape
    label_image = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(markers_after)
    rng = np.random.default_rng(seed=42)
    palette = {}
    for label in unique_labels:
        if label <= 1:
            palette[label] = (0, 0, 0)
        else:
            palette[label] = tuple(int(c) for c in rng.integers(50, 255, size=3))

    for label, colour in palette.items():
        label_image[markers_after == label] = colour

    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, label_image, alpha, 0)
    blended[markers_after == -1] = (255, 255, 255)

    if show_contours:
        # Outline each object label with its own colour so adjacent
        # regions are clearly distinguishable even if their fill colours
        # happen to be similar.
        for label in unique_labels:
            if label <= 1:
                continue
            region = (markers_after == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(blended, contours, -1, palette[label], 2)

    if show_label_ids:
        for label in unique_labels:
            if label <= 1:
                continue
            ys, xs = np.where(markers_after == label)
            if xs.size == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            text = str(int(label))
            # Draw black halo first, then white text - readable on any colour.
            cv2.putText(blended, text, (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(blended, text, (cx - 5, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return blended


def normalise_for_display(image: np.ndarray) -> np.ndarray:
    """Stretch a float/large-range image into uint8 for display."""
    if image.dtype == np.uint8:
        return image
    if image.size == 0:
        return image.astype(np.uint8)
    image = image.astype(np.float32)
    minimum, maximum = float(image.min()), float(image.max())
    if maximum - minimum < 1e-9:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - minimum) / (maximum - minimum)
    return (scaled * 255).astype(np.uint8)


def colourise_markers(markers: np.ndarray) -> np.ndarray:
    """Render a marker image as a colourful preview."""
    return cv2.applyColorMap(normalise_for_display(markers), cv2.COLORMAP_JET)


def count_ridge_pixels(markers_after: np.ndarray) -> int:
    """How many pixels did watershed assign to ridge boundaries (-1)?"""
    return int(np.count_nonzero(markers_after == -1))


def count_distinct_objects(markers_after: np.ndarray) -> int:
    """Distinct watershed labels excluding ridge (-1) and background (1)."""
    labels = np.unique(markers_after)
    return int(np.sum(labels >= 2))


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    image_bgr: np.ndarray,
    blur_kernel: int,
    threshold_method: str,
    manual_threshold: int,
    invert_threshold: bool,
    morph_kernel: int,
    morph_iterations: int,
    dilation_iterations: int,
    foreground_ratio: float,
    *,
    foreground_method: str = "ratio",
    peak_neighbourhood: int = 11,
    peak_min_height: float = 0.2,
    min_component_area: int = 0,
    mode: str = "binary",
    gradient_method: str = "sobel",
    overlay_alpha: float = 0.45,
    show_contours: bool = False,
    show_label_ids: bool = False,
) -> WatershedResult:
    """Run every step of the marker-controlled watershed pipeline.

    `mode`:
        - "binary"   -> classic recipe, watershed flooding on the original
                        BGR image (treats intensity as elevation).
        - "gradient" -> watershed flooding on the gradient magnitude image
                        (treats edge strength as elevation; better when
                        objects have internal texture).

    `foreground_method`:
        - "ratio" -> global threshold on the distance transform.
        - "peaks" -> local-maxima detection in a sliding neighbourhood.
                     Use this when objects of different size touch.

    `min_component_area` removes tiny noise components from the sure
    foreground before markers are built. Set to 0 to disable.
    """

    gray = to_grayscale(image_bgr)
    blurred = apply_blur(gray, blur_kernel)

    threshold_mask = threshold_image(
        blurred,
        method=threshold_method,
        manual_value=manual_threshold,
        invert=invert_threshold,
    )

    opened = morphological_opening(threshold_mask, morph_kernel, morph_iterations)
    sure_bg = compute_sure_background(opened, dilation_iterations)
    distance = compute_distance_transform(opened)

    if foreground_method == "peaks":
        sure_fg_raw = compute_sure_foreground_peaks(
            distance,
            neighbourhood=peak_neighbourhood,
            min_height_ratio=peak_min_height,
        )
    else:
        sure_fg_raw = compute_sure_foreground(distance, foreground_ratio)

    sure_fg, _removed = remove_small_components(sure_fg_raw, int(min_component_area))

    unknown = compute_unknown_region(sure_bg, sure_fg)
    markers, _ = build_markers(sure_fg, unknown)

    # Always compute the gradient image so the UI can show it for both
    # modes (illustrates the conceptual difference) - it is cheap.
    gradient = compute_gradient_magnitude(blurred, method=gradient_method)

    # Build the 3-channel surface that watershed will actually flood.
    if mode == "gradient":
        flood_surface = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
    else:
        flood_surface = image_bgr

    markers_after = run_watershed(flood_surface, markers)
    boundaries = draw_boundaries(image_bgr, markers_after)
    labelled = colour_labels(
        image_bgr,
        markers_after,
        alpha=overlay_alpha,
        show_contours=show_contours,
        show_label_ids=show_label_ids,
    )

    num_opened_components = max(
        0, cv2.connectedComponents(opened)[0] - 1
    )
    num_objects = count_distinct_objects(markers_after)
    num_markers = int(markers.max())  # background label is the highest before watershed... actually no
    # `markers` uses 1 for background and 2..N for objects. The largest
    # value equals (objects + 1), so the count of distinct *positive*
    # labels is exactly `markers.max()`.

    return WatershedResult(
        original_bgr=image_bgr,
        gray=gray,
        blurred=blurred,
        threshold_mask=threshold_mask,
        opened=opened,
        sure_background=sure_bg,
        distance_transform=distance,
        sure_foreground=sure_fg,
        sure_foreground_raw=sure_fg_raw,
        unknown=unknown,
        markers=markers,
        markers_after=markers_after,
        boundaries_overlay=boundaries,
        labelled_overlay=labelled,
        gradient=gradient,
        flood_surface_bgr=flood_surface,
        num_objects=num_objects,
        num_markers=num_markers,
        num_opened_components=num_opened_components,
        ridge_pixel_count=count_ridge_pixels(markers_after),
        mode=mode,
    )


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for display in Streamlit/Matplotlib."""
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

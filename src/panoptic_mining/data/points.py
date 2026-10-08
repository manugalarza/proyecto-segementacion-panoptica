"""Box -> interior points -> pseudo-mask conversion.

Point-supervised panoptic segmentation (Li et al., "Fully Convolutional
Networks for Panoptic Segmentation", i.e. Panoptic FCN) trains instance
kernels from sparse point annotations rather than dense masks. Our source
annotations (from the shared FLIR dataset, see ``Etiquetas.zip``) are
bounding boxes, not points or masks, so this module implements the explicit
transformation the methodology calls for:

    bounding box --> interior points --> pseudo-mask

This is deliberately NOT assumed away: earlier drafts of the methodology
treated boxes as if they already carried mask-level supervision, which an
external review flagged as incorrect. The functions here make each step
concrete and testable independently of the real dataset (they operate on
any image + box, so unit tests use synthetic images).

Pipeline
--------
1. ``sample_interior_points`` — deterministically sample candidate
   foreground points inside a box, biased toward the box center (mirrors
   the intuition in Panoptic FCN that centrality correlates with object
   membership for convex "thing" categories).
2. ``box_to_pseudo_mask`` — tries two heuristics in order, falling back to
   the full box only if both fail to cover a reasonable fraction of it:
     a. OpenCV's GrabCut, seeded with the box as a region prior and the
        sampled points as sure-foreground. GrabCut fits a global color
        model (a Gaussian mixture) over the whole box/image, which works
        well when the instance's color clearly differs from its
        surroundings.
     b. Flood-fill region growing from the same seed points, with a local
        color-distance tolerance, constrained to stay inside the box.
        Because this only compares each candidate pixel to its
        immediate neighbors (not a global model), it can still find a
        coherent connected region for texturally ambiguous classes like
        SDZI — bare, disturbed earth that a global color model tends to
        lump in with the surrounding terrain — where GrabCut sometimes
        collapses to almost nothing (observed on real data: 5 px
        surviving out of a ~17,000 px box).
   Both are heuristics, not ground truth — see docs/methodology.md, point
   1: evaluation always uses the independent dense ground-truth subset,
   never a pseudo-mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class PseudoMaskResult:
    mask: np.ndarray  # uint8, shape (H, W), values in {0, 1}
    points: np.ndarray  # shape (N, 2), (x, y) points used as FG seeds
    converged: bool
    method: str  # "grabcut" | "flood_fill" | "full_box_fallback"


def sample_interior_points(
    box: Box,
    n_points: int = 5,
    center_bias: float = 0.6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample candidate foreground points inside ``box``.

    One point is always the exact box center. The remaining ``n_points - 1``
    points are drawn from a truncated Gaussian centered on the box, so they
    stay inside the box but skew toward the middle — this avoids seeding
    GrabCut with points that land near the box edge, which for a non-convex
    or partially-occluded object is more likely to fall outside the true
    instance.

    Parameters
    ----------
    box:
        ``(x_min, y_min, x_max, y_max)`` in pixel coordinates.
    n_points:
        Total number of points to return (including the center point).
        Must be >= 1.
    center_bias:
        Standard deviation of the truncated Gaussian, as a fraction of the
        half-width/half-height of the box. Lower values cluster points
        more tightly around the center.
    rng:
        Optional numpy Generator for reproducibility. Defaults to a fresh
        default_rng() (i.e. non-deterministic across calls unless seeded
        by the caller).

    Returns
    -------
    np.ndarray of shape (n_points, 2), dtype float64, columns (x, y).
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")

    x_min, y_min, x_max, y_max = box
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Degenerate box: {box}")

    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    hw, hh = (x_max - x_min) / 2.0, (y_max - y_min) / 2.0

    points = [(cx, cy)]
    if n_points > 1:
        rng = rng or np.random.default_rng()
        remaining = n_points - 1
        # Rejection sampling keeps this exact rather than clamping, which
        # would pile extra mass on the box edges.
        accepted: list[tuple[float, float]] = []
        while len(accepted) < remaining:
            batch = remaining - len(accepted)
            xs = rng.normal(cx, center_bias * hw, size=batch)
            ys = rng.normal(cy, center_bias * hh, size=batch)
            for x, y in zip(xs, ys):
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    accepted.append((float(x), float(y)))
        points.extend(accepted)

    return np.asarray(points, dtype=np.float64)


def _flood_fill_region_growing(
    image: np.ndarray,
    points: np.ndarray,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    tolerance: int,
) -> np.ndarray:
    """Grow a foreground region from ``points`` via flood-fill, constrained
    to the ``[x_min, x_max) x [y_min, y_max)`` box. Uses OpenCV's
    FLOODFILL_MASK_ONLY so the image itself is untouched; the box's
    exterior is pre-marked as an obstacle in the fill mask so growth can
    never escape the box (same containment requirement as GrabCut's
    GC_BGD region — see box_to_pseudo_mask docstring history)."""
    h, w = image.shape[:2]
    # cv2.floodFill requires a mask 2 px larger than the image on each side.
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    fill_mask[: y_min + 1, :] = 1
    fill_mask[y_max + 1 :, :] = 1
    fill_mask[:, : x_min + 1] = 1
    fill_mask[:, x_max + 1 :] = 1

    flags = cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if not (x_min <= xi < x_max and y_min <= yi < y_max):
            continue
        # loDiff/upDiff bound how far a neighbor's color may drift from the
        # SEED pixel at each step; FIXED_RANGE compares against the
        # original seed color rather than the running neighbor, which
        # keeps the region from "drifting" color gradually across the
        # whole box on a subtle gradient.
        cv2.floodFill(
            image.copy(),
            fill_mask,
            seedPoint=(xi, yi),
            newVal=(255, 255, 255),
            loDiff=(tolerance, tolerance, tolerance),
            upDiff=(tolerance, tolerance, tolerance),
            flags=flags,
        )

    grown = (fill_mask[1:-1, 1:-1] == 255).astype(np.uint8)
    return grown


def box_to_pseudo_mask(
    image: np.ndarray,
    box: Box,
    points: np.ndarray | None = None,
    iterations: int = 5,
    box_margin: int = 2,
    min_foreground_fraction: float = 0.15,
    flood_fill_tolerance: int = 12,
) -> PseudoMaskResult:
    """Derive a pixel-level pseudo-mask for one instance from its box.

    Uses the box as GrabCut's rectangular prior (everything outside is
    "certain background", everything inside starts as "probable
    foreground") and marks ``points`` as "certain foreground" seeds, which
    is the same seeding logic as interactive point-prompted segmentation.
    This is a heuristic, not ground truth — it exists so training can use
    a mask-shaped target for images where only box annotations exist; the
    independent dense ground-truth subset (see docs/methodology.md) is
    what evaluation is scored against, never this.

    Parameters
    ----------
    image:
        BGR or grayscale image as a numpy array, shape (H, W, 3) or (H, W).
    box:
        ``(x_min, y_min, x_max, y_max)`` in pixel coordinates, clipped to
        the image bounds internally.
    points:
        Optional (N, 2) array of (x, y) foreground seed points, typically
        from :func:`sample_interior_points`. If ``None``, points are
        sampled with default parameters.
    iterations:
        GrabCut iteration count.
    box_margin:
        Pixels of slack added around the box before running GrabCut, so
        the object's true boundary (which rarely aligns exactly with the
        annotated box) isn't clipped.
    min_foreground_fraction:
        Minimum fraction of the (margin-padded) box area the GrabCut
        result must cover to be trusted. Below this, we fall back to the
        full box as the mask and report ``converged=False``. This matters
        for texturally ambiguous classes like SDZI: on real aerial frames,
        GrabCut can reclassify almost the entire box as background because
        disturbed earth doesn't stand out in color from the surrounding
        terrain, leaving only the handful of seed pixels as "foreground" —
        technically a valid GrabCut result, but useless as a pseudo-mask
        (observed on real data: 5 px survived out of a ~17,000 px box).
        When GrabCut falls short of this, flood-fill region growing is
        tried next; only if THAT also falls short does the function give
        up and return the full box.
    flood_fill_tolerance:
        Per-channel color distance (0-255) a neighboring pixel may differ
        from a seed point and still be grown into the region. Lower is
        stricter (smaller, more conservative regions); higher risks
        leaking into genuinely different terrain.

    Returns
    -------
    PseudoMaskResult with a binary mask the same height/width as ``image``.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape[:2]

    x_min, y_min, x_max, y_max = box
    x_min = max(0, int(np.floor(x_min)) - box_margin)
    y_min = max(0, int(np.floor(y_min)) - box_margin)
    x_max = min(w, int(np.ceil(x_max)) + box_margin)
    y_max = min(h, int(np.ceil(y_max)) + box_margin)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(f"Box outside image bounds after clipping: {box} vs image {w}x{h}")

    if points is None:
        points = sample_interior_points(box)

    # Everything outside the (margin-padded) box is marked as CERTAIN
    # background (GC_BGD), not merely "probable" (GC_PR_BGD). Using
    # GC_PR_BGD here was an earlier bug: GrabCut's color model was then
    # free to reclassify pixels outside the box as foreground whenever
    # their color resembled the seeded points, letting the mask bleed far
    # beyond the annotated box (observed on real data: a mask nearly 5x
    # larger than the box area). Since the box is itself the annotation
    # for "this instance is in here", the mask must not extend past it.
    grabcut_mask = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[y_min:y_max, x_min:x_max] = cv2.GC_PR_FGD
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            grabcut_mask[yi, xi] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    converged = True
    try:
        cv2.grabCut(
            image,
            grabcut_mask,
            None,
            bgd_model,
            fgd_model,
            iterCount=iterations,
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        # Degenerate cases (e.g. a uniform-color crop) can make GrabCut's
        # GMM fitting fail; fall back to the box itself as the mask rather
        # than raising, since a coarse pseudo-mask is still usable for
        # training and the caller can inspect `converged`.
        converged = False

    binary_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0
    ).astype(np.uint8)

    box_area = (x_max - x_min) * (y_max - y_min)
    min_pixels = min_foreground_fraction * box_area
    method = "grabcut"

    if not converged or binary_mask.sum() < min_pixels:
        # GrabCut's global color model didn't find a large-enough region —
        # try flood-fill's local, neighbor-by-neighbor growing instead
        # before giving up entirely.
        flood_mask = _flood_fill_region_growing(
            image, points, x_min, y_min, x_max, y_max, flood_fill_tolerance
        )
        if flood_mask.sum() >= min_pixels:
            binary_mask = flood_mask
            method = "flood_fill"
            converged = True
        else:
            binary_mask = np.zeros((h, w), dtype=np.uint8)
            binary_mask[y_min:y_max, x_min:x_max] = 1
            method = "full_box_fallback"
            converged = False

    return PseudoMaskResult(mask=binary_mask, points=points, converged=converged, method=method)

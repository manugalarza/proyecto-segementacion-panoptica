"""Pure numpy helpers for turning box annotations into per-pixel training
targets: a class-index map for the stuff branch, and Gaussian heatmaps for
the thing branch (CenterNet-style: a peak at each box center, radius scaled
to box size). Kept dependency-light (numpy only) and torch-free so they're
testable without the ``vision`` extra and reusable from both
``data/torch_dataset.py`` (this project's own YOLO-format data) and
``data/transfer_datasets.py`` (LandCover.ai, which needs the heatmap helper
but not the box rasterizer, since it already ships pixel masks).

This is a deliberately simple choice, not a claim that it's the best one:
rasterizing a box's *entire* interior as its class for the stuff target
is coarser than the pseudo-masks in ``data/points.py`` (which use
GrabCut/flood-fill to find the actual disturbed-earth region within a box).
Running pseudo-mask generation per-sample at training time was ruled out
here as too slow to be a per-epoch data loader step; using it to
pre-compute an offline mask cache is a reasonable follow-up once training
is otherwise working end to end, not implemented here.
"""

from __future__ import annotations

import numpy as np


def rasterize_boxes_to_class_map(
    boxes_pixels: list[tuple[int, tuple[float, float, float, float]]],
    height: int,
    width: int,
    class_name_to_index: dict[str, int],
    background_index: int | None = None,
) -> np.ndarray:
    """Build an (H, W) int32 class-index map by painting each box's full
    interior with its class index. Later boxes in the list win on overlap
    (drawn in order) — this is the same "last box wins" convention as most
    simple box-to-mask rasterizers and is fine for a coarse training
    target, though it means overlapping stuff regions of different classes
    are not disambiguated here.

    ``background_index`` controls what unannotated pixels get:

    - ``None`` (default): -1, meant to be used with
      ``nn.CrossEntropyLoss(ignore_index=-1)`` so those pixels contribute
      no gradient at all — the original behavior of this function.
    - an explicit class index: unannotated pixels are labeled as that
      class instead of ignored. **Real finding (2026-09-19/20, see
      docs/decisions.md)**: with only 2 stuff classes (river, SDZI) and no
      background label, the first full overnight fine-tuning run produced
      a stuff branch that never learned any spatial structure — most
      pixels in most frames fall outside any river/SDZI box and were
      never given a negative learning signal (`ignore_index=-1` means no
      gradient either way for them), so at inference the model just
      output a global class bias (river predicted for ~90% of pixels
      everywhere, unrelated to where water actually was, confirmed
      visually with `scripts/visualize_predictions.py`). Passing an
      explicit "none"/background class index here is the fix: those
      pixels now get real negative supervision instead of being skipped.

    ``boxes_pixels`` entries are ``(class_name, (x_min, y_min, x_max, y_max))``
    in pixel coordinates (see ``BoxAnnotation.to_pixels`` /
    ``FrameRecord.pixel_boxes`` in ``data/manifest.py``). Boxes whose class
    isn't in ``class_name_to_index`` are skipped (lets a caller build a
    stuff-only or thing-only map from a mixed annotation list).
    """
    fill_value = -1 if background_index is None else background_index
    class_map = np.full((height, width), fill_value=fill_value, dtype=np.int32)
    for class_name, (x_min, y_min, x_max, y_max) in boxes_pixels:
        if class_name not in class_name_to_index:
            continue
        x0, y0 = max(0, int(round(x_min))), max(0, int(round(y_min)))
        x1, y1 = min(width, int(round(x_max))), min(height, int(round(y_max)))
        if x1 <= x0 or y1 <= y0:
            continue
        class_map[y0:y1, x0:x1] = class_name_to_index[class_name]
    return class_map


def gaussian_heatmap_targets(
    boxes_pixels: list[tuple[int, tuple[float, float, float, float]]],
    height: int,
    width: int,
    class_name_to_index: dict[str, int],
    min_sigma: float = 1.0,
) -> np.ndarray:
    """Build a (num_thing_classes, H, W) float32 heatmap target with a 2D
    Gaussian peak (value 1.0 at the center) at each box's center, sized
    relative to the box (CenterNet-style: bigger box -> wider, flatter
    peak). Multiple boxes of the same class take the elementwise max, so
    nearby instances don't cancel each other out.
    """
    num_classes = len(class_name_to_index)
    heatmap = np.zeros((num_classes, height, width), dtype=np.float32)
    if num_classes == 0:
        return heatmap

    ys, xs = np.mgrid[0:height, 0:width]
    for class_name, (x_min, y_min, x_max, y_max) in boxes_pixels:
        if class_name not in class_name_to_index:
            continue
        class_index = class_name_to_index[class_name]
        cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        box_w, box_h = max(x_max - x_min, 1.0), max(y_max - y_min, 1.0)
        sigma = max(min_sigma, 0.15 * min(box_w, box_h))
        peak = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
        heatmap[class_index] = np.maximum(heatmap[class_index], peak.astype(np.float32))
    return heatmap


def resize_class_map_nearest(class_map: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Nearest-neighbor resize for an integer class map (must not
    interpolate class indices). Pure numpy so this module stays
    torch/cv2-independent; ``cv2.resize(..., interpolation=INTER_NEAREST)``
    is the faster option and is used instead in ``torch_dataset.py`` where
    cv2 is already a hard dependency — this is kept for the pieces (tests,
    ``transfer_datasets.py``) that only need numpy.
    """
    in_height, in_width = class_map.shape
    row_idx = (np.arange(out_height) * in_height / out_height).astype(np.int64).clip(0, in_height - 1)
    col_idx = (np.arange(out_width) * in_width / out_width).astype(np.int64).clip(0, in_width - 1)
    return class_map[row_idx[:, None], col_idx[None, :]]


def generate_thing_pseudo_masks(
    image: np.ndarray,
    boxes_pixels: list[tuple[str, tuple[float, float, float, float]]],
    height: int,
    width: int,
    class_name_to_index: dict[str, int],
) -> np.ndarray:
    """
    Generate pseudo-masks for thing (instance) classes.
    
    Returns (num_instances, height, width) binary mask array.
    Each channel is one instance, background is 0.
    
    Note: This is a placeholder that returns simple box rasterizations.
    For production, integrate box_to_pseudo_mask from data/points.py
    to use GrabCut-based segmentation.
    """
    # Identify thing boxes (non-stuff classes)
    thing_boxes = [
        (class_name, box) for class_name, box in boxes_pixels 
        if class_name in class_name_to_index
    ]
    
    if not thing_boxes:
        return np.zeros((0, height, width), dtype=np.uint8)
    
    masks = []
    for class_name, (x_min, y_min, x_max, y_max) in thing_boxes:
        mask = np.zeros((height, width), dtype=np.uint8)
        x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
        x_max, y_max = int(min(width, x_max)), int(min(height, y_max))
        mask[y_min:y_max, x_min:x_max] = 1
        masks.append(mask)
    
    return np.array(masks, dtype=np.uint8)

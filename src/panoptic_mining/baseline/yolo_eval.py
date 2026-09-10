"""Re-evaluation of the "Eyes in the Sky" YOLOv11 baseline on the shared
test split, so its recall (0.425 for SDZI) is directly comparable to this
project's panoptic-mask-derived recall.

Status: PLANNED — not yet implemented.

Blocked on downloading:
  - BASELINE_YOLO_WEIGHTS (yolov11_best100.pt from the shared OneDrive)
  - The same test split Eyes in the Sky evaluated on (see README "Data
    consumption" and docs/decisions.md for why this must be the original
    split, not Jorge's leakage-safe split, when the goal is reproducing
    their reported number rather than measuring on a stricter split)

Planned flow:
  1. Run yolov11_best100.pt inference on the test split images.
  2. Score box-level recall directly (should reproduce ~0.425 for SDZI,
     validating that we're using the same weights/split they used).
  3. Separately, run our panoptic model's predicted masks through
     evaluation.metrics.mask_based_recall_as_boxes against the SAME
     ground-truth boxes, to get the comparable number for our approach.
     A raw mask-IoU recall is NOT comparable to their box-IoU recall and
     must not be reported as if it were (see docs/decisions.md).
"""

raise NotImplementedError(
    "Baseline re-evaluation is not implemented yet — see module docstring for blockers. "
    "evaluation.metrics.mask_based_recall_as_boxes is already implemented and unit-tested, "
    "ready to be called once the baseline weights and test split are available."
)

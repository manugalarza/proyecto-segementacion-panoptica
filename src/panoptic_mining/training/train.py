"""Training loop entry point.

Status: PLANNED — not yet implemented. Depends on models/panoptic_fcn.py
and models/context_fusion.py, and on the data pipeline in data/ having a
real manifest to read from (currently only the box->points->pseudo-mask
transform in data/points.py is implemented, since it doesn't need the
dataset itself to be written and tested).

Planned responsibilities, once unblocked:
  - Load DATA_ROOT (see .env.example) and, once available, Jorge's
    leakage-safe split manifest (LEAKAGE_MANIFEST).
  - Build pseudo-masks on the fly from box annotations via
    data.points.box_to_pseudo_mask for training images that lack dense
    ground truth.
  - Run the 3-stage experimental design from docs/methodology.md:
      1. context-ablation on validation (base vs. context-fusion)
      2. 2x2 matrix: {raw, clean} x {original split, leakage-safe split}
      3. final test evaluation, only for the configuration selected in
         stages 1-2 (the test set is not touched before that point)
"""

raise NotImplementedError(
    "Training loop is not implemented yet — see module docstring for blockers."
)

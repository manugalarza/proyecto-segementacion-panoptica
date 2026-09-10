# Architecture

Status: design notes ahead of implementation (see `models/panoptic_fcn.py`
and `models/context_fusion.py` for the planned interface and current
blockers).

## Base: Panoptic FCN

Panoptic FCN (Li et al., CVPR 2021) unifies stuff and thing prediction in
a single fully-convolutional network:

- A **kernel generator** produces one convolution kernel per instance
  (things) or per stuff class, from point-level features — which is why
  our point-supervision pipeline (`data/points.py`) matters even though
  the base architecture itself isn't implemented yet.
- A shared **feature encoder** feeds both the stuff semantic head and the
  per-kernel mask generation for things.
- Panoptic output is produced by applying each generated kernel to the
  shared feature map, then combining thing instance masks with the stuff
  semantic map (things take priority in overlap regions, standard
  panoptic-segmentation convention).

## Ablation: spatial context fusion

Planned addition (`models/context_fusion.py`), the specific architectural
contribution this project evaluates against the base model:

- **Motivation**: SDZI is a texturally ambiguous stuff class — bare,
  disturbed earth can look similar whether or not it's actually an active
  mining zone. The presence of nearby thing instances (dredges, machinery,
  camps) is strong contextual evidence one way or the other, but the base
  Panoptic FCN's stuff and thing branches don't share information before
  their respective final predictions.
- **Mechanism**: inject pooled thing-branch features (e.g. from detected
  instance regions) into the stuff-branch decoder before its final
  classification layer, and symmetrically let stuff-branch context inform
  thing-branch confidence (e.g. a candidate dredge detection sitting on
  predicted river/SDZI is more plausible than one sitting on unrelated
  terrain). Exact fusion operator (concatenation + 1x1 conv vs. attention)
  is an open implementation decision, to be settled empirically once
  training is unblocked — this document will be updated with the chosen
  design and the ablation result once available.

## Why this is evaluated as an ablation, not just "the model"

The 3-stage experimental design (`docs/methodology.md`, step 1) scores
base Panoptic FCN vs. base + context-fusion on validation before touching
the test set, so the project can report "context fusion improves PQ by X"
as an isolated, defensible claim, rather than a single black-box number
that conflates the architectural contribution with everything else
(pseudo-mask quality, split choice, data cleaning).

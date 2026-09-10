# Architecture

Status (2026-09-10): both `models/panoptic_fcn.py` (base) and
`models/context_fusion.py` (ablation) are implemented as runnable
skeletons — forward passes produce correctly-shaped outputs and
`decode_instances` runs end to end, checked with synthetic-tensor unit
tests (`tests/test_panoptic_fcn.py`, `tests/test_context_fusion.py`, both
gated on `pytest.importorskip("torch")`). Two things are explicitly *not*
resolved yet, tracked in `docs/decisions.md` rather than guessed at here:
the feature encoder is a small 3-layer conv trunk, not a real backbone
(swap-in point documented in the module docstring), and the class counts
(2 stuff / 3 thing) match what's actually labeled in
`dataset_split_completo` today, not the thesis objectives' 6 thing
categories. Nothing below has been trained yet.

## Base: Panoptic FCN

Panoptic FCN (Li et al., CVPR 2021) unifies stuff and thing prediction in
a single fully-convolutional network:

- A **kernel generator** produces one convolution kernel per instance
  (things) or per stuff class, from point-level features — which is why
  our point-supervision pipeline (`data/points.py`) matters even though
  the base architecture itself isn't trained yet.
- A shared **feature encoder** feeds both the stuff semantic head and the
  per-kernel mask generation for things.
- Panoptic output is produced by applying each generated kernel to the
  shared feature map (implemented in `PanopticFCN.decode_instances` as a
  per-pixel dot product via `torch.einsum`, after a simple max-pool NMS
  over heatmap peaks), then combining thing instance masks with the stuff
  semantic map (things take priority in overlap regions, standard
  panoptic-segmentation convention — this final combination step itself
  is not yet implemented, only per-class decoding).

## Ablation: spatial context fusion

Implemented in `models/context_fusion.py` as `ContextFusionPanopticFCN`,
the specific architectural contribution this project evaluates against
the base model:

- **Motivation**: SDZI is a texturally ambiguous stuff class — bare,
  disturbed earth can look similar whether or not it's actually an active
  mining zone. This stopped being a hypothetical concern once real-data
  testing of `data/points.py` showed GrabCut collapsing to almost nothing
  on real SDZI boxes (no color separation from surrounding terrain to key
  off). The presence of nearby thing instances (dredges, machinery, camps)
  is strong contextual evidence one way or the other, but the base
  Panoptic FCN's stuff and thing branches don't share information before
  their respective final predictions.
- **Mechanism (implemented)**: run the base heads once to get first-pass
  stuff logits and thing heatmaps, project each into a small number of
  "context" channels with a 1x1 conv (`stuff_from_things`,
  `thing_from_stuff`), concatenate that context onto the shared encoder
  features, and run a second, fusion-aware set of heads
  (`stuff_head2`, `thing_heatmap_head2`) on the concatenated tensor. This
  was chosen over an attention-based fusion for the first pass because
  it's cheap enough to unit-test and reason about on CPU; nothing rules
  out revisiting it once there's a training budget to compare options
  empirically — this document will be updated with the ablation result
  once training is unblocked.

## Why this is evaluated as an ablation, not just "the model"

The 3-stage experimental design (`docs/methodology.md`, step 1) scores
base Panoptic FCN vs. base + context-fusion on validation before touching
the test set, so the project can report "context fusion improves PQ by X"
as an isolated, defensible claim, rather than a single black-box number
that conflates the architectural contribution with everything else
(pseudo-mask quality, split choice, data cleaning).

"""Panoptic FCN + spatial context-fusion mechanism (the architectural
ablation that isolates this project's specific contribution).

Status: PLANNED — not yet implemented. Depends on panoptic_fcn.py.

Idea (see docs/architecture.md for full rationale): SDZI (Soil Disturbance
Zones of Interest) and river are large, texturally ambiguous "stuff"
regions where nearby "thing" instances (dredges, machinery) provide strong
disambiguating context — a bare-earth patch next to a dredge is far more
likely to be SDZI than an identical-looking patch with no equipment nearby.
The context-fusion mechanism injects thing-branch features into the
stuff-branch decoder (and vice versa) before the final per-pixel
classification, rather than treating the two branches as fully
independent, which is the base Panoptic FCN's assumption.

The ablation this repo evaluates is exactly:
    base Panoptic FCN  vs.  Panoptic FCN + context fusion
scored via evaluation/metrics.py (PQ/SQ/RQ, per docs/methodology.md step 4),
on the validation split, before any test-set evaluation.
"""

raise NotImplementedError(
    "Context-fusion variant is not implemented yet — depends on panoptic_fcn.py. "
    "See module docstring and docs/architecture.md."
)

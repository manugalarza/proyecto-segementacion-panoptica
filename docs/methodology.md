# Methodology

Framed as CRISP-ML(Q), consistent with the two other repos in this team's
pipeline (Jorge's `flir-leakage-pipeline`, Laura's `proyecto-FAC`), which
both scope themselves to the same four phases (no deployment/monitoring
phase for a thesis project).

## 1. Problem and data understanding

- Task: panoptic segmentation combining *stuff* classes (river, SDZI —
  Soil Disturbance Zones of Interest) with *thing* classes (machinery,
  dredges, camps, structures, containers, vehicles) on FAC aerial video of
  the Colombian Amazon.
- Baseline: "Eyes in the Sky" (Acosta-Bernal et al., 2025), YOLOv11 on the
  same 1,657-frame set, recall 0.425 for SDZI (box-level).
- **Resolved**: the source imagery is RGB, never thermal/FLIR — confirmed
  with the team/advisor (see `decisions.md`). Preprocessing assumes
  3-channel RGB (already how `models/panoptic_fcn.py`'s encoder is built);
  the antecedentes section's thermal-imagery references are no longer
  direct comparison points and need a documentation cleanup pass
  (Manuela's document edit, not a repo change).

## 2. Data preparation

- **Source**: canonically `Imagenes.zip` + `Etiquetas.zip` from the shared
  OneDrive, per Jorge's own inventory code (`"Imagenes.zip + Etiquetas.zip
  parecen el candidato más cercano al conjunto reportado de 1657
  frames"`). In practice, this project is currently working against
  `dataset_split_completo` instead (already downloaded, complete, and
  verified well-formed via `data/manifest.py` — see below), since
  `Imagenes.zip`/`Etiquetas.zip` haven't been downloaded yet; Jorge's code
  treats `dataset_split_completo` as a cross-check archive rather than the
  primary source, so this substitution is tracked as a decision to revisit
  in `decisions.md`, not a settled equivalence.
- **Manifest loading and integrity checks** (`data/manifest.py`): parses a
  YOLO-format dataset (`dataset.yaml` + per-split `images/`/`labels/`)
  into typed records, defensively handling a broken split (missing `val/`,
  images without a matching label file) rather than crashing — this is
  exactly the shape of the *other* downloaded archive,
  `flir_best_division_pr`, which is not currently usable as-is. Running
  `panoptic-mining data manifest-summary <path>` against
  `dataset_split_completo` confirmed it has no missing splits or labels,
  1702 total frames (vs. Eyes in the Sky's reported 1657 — a 45-frame
  discrepancy, unresolved), and a significant class imbalance for SDZI
  (481 boxes vs. 4120 for building) that will need addressing during
  training.
- **Independent dense ground truth**: a validation/test subset gets real
  pixel-level masks, used only for evaluation — never for training. This
  is what makes the PQ/SQ/RQ numbers trustworthy rather than an artifact
  of the pseudo-mask generation process described next.
- **Box → interior points → pseudo-mask**: our source annotations are
  boxes, not points or masks. `data/points.py` makes this transformation
  explicit and testable:
  1. `sample_interior_points` — deterministic, center-biased sampling of
     candidate foreground points inside each box.
  2. `box_to_pseudo_mask` — GrabCut seeded with the box as a region prior
     and the sampled points as sure-foreground, producing a pixel mask
     usable as a training target.
  This was flagged in an earlier external review: previous drafts
  implicitly assumed box annotations already carried mask-level
  supervision, which they do not. Real-data testing against
  `dataset_split_completo` surfaced two issues since fixed: background
  containment (the mask could leak far outside the box) and GrabCut
  collapsing to almost nothing on texturally ambiguous classes like SDZI —
  the pipeline now falls back to region-growing (flood-fill from the seed
  points) and, failing that, the full box, exposing which method was used
  via `PseudoMaskResult.method` for later auditing.
- **Splits**: the original Eyes in the Sky split (1178 train / 107 val /
  372 test) has confirmed leakage — 141 train-test and 57 train-val
  overlaps, per Jorge's manifest audit. His leakage-safe split (via
  DINOv2/CLIP embeddings + DBSCAN/OPTICS/HDBSCAN clustering) is not yet
  produced. See `decisions.md` for how this affects the evaluation design
  below.

## 3. Model engineering

- **Base model**: Panoptic FCN (Li et al.), predicting stuff via a
  semantic branch and things via per-instance kernels generated from point
  features. Implemented as a runnable skeleton in `models/panoptic_fcn.py`
  (placeholder conv-trunk encoder, not yet a real backbone; not yet
  trained) — see `docs/architecture.md` for the current implementation
  status and open items.
- **Ablation / this project's specific contribution**: a spatial
  context-fusion mechanism that lets the stuff and thing branches inform
  each other (`models/context_fusion.py`, also implemented as a runnable
  skeleton) — the intuition being that nearby machinery/dredges strongly
  disambiguate SDZI regions that would otherwise look like ordinary bare
  earth. This isolates the ablation: base Panoptic FCN vs. Panoptic FCN +
  context fusion.
- **Open class-count mismatch**: both model skeletons default to the
  classes actually labeled in `dataset_split_completo` today — 2 stuff
  (river, SDZI), 3 thing (vehicle, building, road) — not the 6 thing
  categories in the thesis objectives (machinery, dredges, camps,
  structures, containers, vehicles). See `decisions.md`; unresolved.
- **Terminology**: SDZI throughout, consistent with the baseline
  (previously "deforestación" in earlier drafts — since corrected).

## 4. Evaluation

Metrics (`evaluation/metrics.py`):
- Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality
  (RQ) — per class and mean, computed via IoU-threshold matching (0.5) per
  Kirillov et al.
- IoU / F1 / recall / precision for stuff classes (river, SDZI).
- Precision / recall / mask AP for thing classes.
- **Baseline comparability**: Eyes in the Sky reports box-level recall
  (0.425 for SDZI), not mask-level. Reporting a raw mask-IoU recall as if
  it were comparable would be a methodological error. Instead,
  `mask_based_recall_as_boxes` derives a tight bounding box from each
  predicted mask and scores box-level recall against the same
  ground-truth boxes and IoU threshold (0.5) Eyes in the Sky used — that
  number, and only that number, is what gets compared to 0.425.

### Staged experimental design (avoids repeated test-set reuse)

1. **Context ablation on validation**: base Panoptic FCN vs.
   context-fusion variant, scored on validation only.
2. **2×2 matrix**: {raw data, cleaned data (Fernanda's HUD-removal +
   denoising)} × {original split, leakage-safe split (once Jorge's
   clustering/splitting stage lands)}. Still validation-only.
3. **Final test evaluation**: only for the configuration selected from
   steps 1–2, run once on the held-out test set. The "leakage-safe" arm of
   step 2 is blocked until Jorge's split exists — see `decisions.md` for
   how that's handled in the meantime (evaluating against the baseline
   proceeds on the original split first, since Eyes in the Sky's own
   0.425 was also measured on that split).

Target: recall ≥ 0.6 for SDZI (vs. baseline's 0.425), using the
box-equivalence comparison above.

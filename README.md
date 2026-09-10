# Thesis Project — Panoptic Segmentation for Illegal Mining Detection

Code repository for Manuela Galarza's thesis project (Systems Engineering, Universidad de los Andes), advised by Juan Pablo Reyes.

## Topic

Detection of illegal mining in the Colombian Amazon from aerial video captured by the FAC (Colombian Air and Space Force), using panoptic segmentation (Panoptic FCN) that combines *stuff* classes (river, SDZI — Soil Disturbance Zones of Interest) with *thing* classes (machinery, dredges, camps, structures, containers, vehicles).

This repository covers the **panoptic segmentation** stage of a three-part team pipeline. It consumes the outputs of the other two stages and produces the final segmentation model and its evaluation against the project's baseline.

## Team pipeline

| Stage | Owner | Repo | Status |
|---|---|---|---|
| Video cleaning (HUD removal, denoising) | Fernanda / Laura Martínez-Galindo | [`proyecto-FAC`](https://github.com/Laura-Martinez-Galindo/proyecto-FAC) | frame extraction, HUD segmentation, ProPainter inpainting and 4 denoising tracks implemented |
| Leakage-aware data partitioning | Jorge Solórzano | [`flir-leakage-pipeline`](https://github.com/Kazzu00/flir-leakage-pipeline) | inventory, manifest and feature extraction (DINOv2/CLIP) implemented; clustering/splitting planned |
| **Panoptic segmentation (this repo)** | Manuela Galarza | — | see status below |

## Baseline

"Eyes in the Sky" (Acosta-Bernal et al., 2025) — YOLOv11 on 1,657 frames, 5 classes including SDZI. Reported recall 0.425 / mAP@50 0.402 for SDZI, using bounding boxes (no masks). This project's target: recall ≥ 0.6 for SDZI using panoptic masks, with a bounding-box-equivalent derived from the predicted mask so the comparison against the baseline is apples-to-apples (see `docs/decisions.md`).

## Methodology

CRISP-ML(Q): Problem and Data Understanding → Data Preparation → Model Engineering → Evaluation. Full detail in `docs/methodology.md`. Key points:

1. An independent dense ground truth (pixel-level masks on a validation/test subset, used only for evaluation, never for training).
2. Explicit box → interior points → pseudo-mask transformation (the Panoptic FCN point-supervision mechanism), rather than assuming boxes already carry masks.
3. Consistent SDZI terminology, aligned with the baseline.
4. An architectural ablation isolating this project's contribution: base Panoptic FCN vs. Panoptic FCN with a spatial context-fusion mechanism.
5. A staged, 3-step experimental design (context ablation on validation → 2×2 raw/clean × original/leakage-safe split matrix → final test evaluation only for what gets reported) to avoid repeated test-set reuse.

## Metrics

Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ), IoU/F1/recall (stuff classes: river, SDZI), precision/recall/mask AP (thing classes).

## Data consumption

**Input source:** the shared FLIR dataset on the advisor's OneDrive ("Deteccion Imagenes FLIR"). Per Jorge's own inventory findings, the canonical source matching the reported 1,657-frame set is `Imagenes.zip` + `Etiquetas.zip` (not the various `Dataset_Balanceado*` / `flir_base_*` archives, which his pipeline uses only for cross-checking). Configure the local path via `DATA_ROOT` in `.env` (see `.env.example`).

**Splits:** the original train/val/test split has confirmed data leakage (141 train–test overlaps per Jorge's manifest audit). Jorge's leakage-safe split is not yet available (clustering/splitting stage still planned in his pipeline), so the "leakage-safe" arm of the 2×2 evaluation matrix is blocked until that lands — see `docs/decisions.md`.

**Baseline weights:** `yolov11_best100.pt` from the shared OneDrive matches the YOLOv11 architecture reported by Eyes in the Sky and is used to reproduce/reevaluate the baseline on the same test split.

## Data production

- Point-level pseudo-annotations derived from box annotations (`src/panoptic_mining/data/points.py`).
- Trained Panoptic FCN checkpoints (base and context-fusion variants) — planned, pending data access.
- Evaluation reports: PQ/SQ/RQ breakdowns, per-class IoU/F1, baseline-comparable recall (planned).

## Status

- Box → points → pseudo-mask conversion implemented and unit-tested (no data dependency). Real-data testing on `dataset_split_completo` surfaced two bugs since fixed: background containment (mask was leaking far outside the box) and a missing fallback when GrabCut collapses to near-nothing on texturally ambiguous classes like SDZI.
- Evaluation metrics (PQ/SQ/RQ, IoU/F1, precision/recall/mask AP, bbox-equivalence for baseline comparison) implemented and unit-tested on synthetic inputs.
- Dataset manifest loading (`data/manifest.py`) implemented against `dataset_split_completo` — defensively handles a broken split (missing `val/`, images without a matching `labels/`), which is exactly the shape of the *other* downloaded archive, `flir_best_division_pr`. Run `panoptic-mining data manifest-summary <path>` to check a dataset before trusting it.
- Model skeletons (`models/panoptic_fcn.py` base, `models/context_fusion.py` ablation) implemented and runnable end to end on CPU — encoder is a placeholder (small conv trunk, not yet a real ResNet+FPN backbone), and class counts (2 stuff / 3 thing) match what's actually labeled today, not the thesis objectives' 6 thing categories (open, see `docs/decisions.md`). Not yet trained.
- Training loop: **planned** — `dataset_split_completo` is downloaded so this is no longer blocked on data access per se, but still pending actual implementation.
- Baseline YOLO re-evaluation: **blocked** on downloading `yolov11_best100.pt` (only `yolov26_best100.pt`, a non-matching checkpoint, has been downloaded so far — see `.env`).
- Leakage-safe split arm of the evaluation matrix: **blocked** on Jorge's splitting stage.
- RGB vs. thermal imagery: **resolved** — confirmed RGB, never thermal. See `docs/decisions.md` for the remaining (non-code) action this implies for the proposal document.

See `docs/decisions.md` for the open questions this status still depends on (thing-class count mismatch, exact data root contents).

## Installation & usage

```bash
uv sync --locked
cp .env.example .env   # set DATA_ROOT to the local Imagenes/Etiquetas path
uv run panoptic-mining points --help
uv run panoptic-mining evaluate --help
```

## Repository structure

```
repo/
├── src/panoptic_mining/
│   ├── data/          # manifest loading, box->points->pseudo-mask conversion
│   ├── models/         # Panoptic FCN base + context-fusion variant (skeleton implemented, not trained)
│   ├── training/        # training loop (planned)
│   ├── evaluation/       # PQ/SQ/RQ, IoU/F1, mask AP, baseline comparability
│   ├── baseline/        # YOLOv11 (Eyes in the Sky) re-evaluation (planned)
│   └── utils/
├── configs/             # experiment configs
├── docs/                # methodology, architecture, decision log
├── notebooks/           # exploratory notebooks (outputs not versioned)
├── scripts/             # one-off utility scripts
├── tests/               # unit tests (synthetic data, no external dependency)
├── data/                # git-ignored — local Imagenes/Etiquetas/manifests
└── artifacts/           # git-ignored — checkpoints, reports
```

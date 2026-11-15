# Evaluation protocol v1 (DRAFT, to be frozen with Jorge before opening test)

Status: draft, 2026-11-11. Test stays closed until both sides approve this file.

## 1. Partition
- Arm: `historical-S1-0` of `fixed_test_train_val_v1`. Historical frames only, no Adendo.
- Common test: 372 historical frames, frozen, identical in every arm. Quarantine (141 copies of test images found in the original train folder) is never used.
- One record per `content_id`: train 999, val 88, test 372.
- Validation is temporally adjacent to train (all 95 val rows have an immediate neighbour in train), so it is not used to choose checkpoints. Main result = final checkpoint:
  - YOLO11m: `last.pt` (epoch 50), seeds 42, 43, 44. `best.pt` is secondary, with the limitation stated.
  - Panoptic FCN: final epoch (150, cosine LR).

## 2. Label scenarios
| | Scenario B: original | Scenario A: datos_v2 |
|---|---|---|
| Labels | original FAC Etiquetas (consolidated) | datos_v2 when the frame is there, else Etiquetas, + 4 reviewed machinery boxes |
| Classes | 0 vehicle, 1 building, 2 road, 3 river, 4 SDZI | same + 5 machinery |
| Test boxes (veh / bld / mach) | 12 / 589 / 0 | 100 / 808 / 83 |
| Main metric | building AP50; vehicle AP50 reported but not conclusive (12 GT boxes) | machinery recall and AP50; vehicle + building |
| Models | Jorge YOLO11m (3 seeds); Panoptic FCN trained on B | YOLO11m trained on A with Jorge's configuration; Panoptic FCN trained on A |

Boxes are clipped to the image border ([0, 1]) when the dataset is built (rule of 2026-11-12: 7 datos_v2 river boxes in train/val and 1 road box in test exceeded it by at most 1.5% of the width; package v2).

Each model is evaluated against the labels of the scenario it was trained on. Cross-scenario numbers, if reported, are secondary.

## 3. Class mapping
- Prediction ids 0-4 are identical in both teams (0 vehicle, 1 building, 2 road, 3 river, 4 SDZI); id 5 machinery only in scenario A.
- Scored classes: vehicle, building (both scenarios) and machinery (A). Road, river: not in the box protocol.
- SDZI is excluded: its GT boxes and its predictions are dropped (`--exclude SDZI`).

## 4. Inference
- YOLO11m (to confirm with Jorge's args.yaml): imgsz = training imgsz, conf >= 0.001, NMS IoU 0.7, max_det 300. One TXT per test frame named by `nombre_frame` (`video_13min_432.txt`), lines `cls xc yc w h conf` normalized, empty file when there are no detections.
- Panoptic FCN (decoded inside the evaluator): score floor 0.01, top-k 100, mask threshold 0.5, mask NMS IoU 0.5, min area 20 px at full resolution. Box = bounding box of the largest connected component of each mask (`--box-mode largest_cc`); `--box-mode full` is secondary.

## 5. Metrics (single implementation: `scripts/eval_detection.py`)
- Per class, greedy one-to-one matching sorted by confidence. AP with all-point interpolation at IoU 0.50 (AP50) and mean over 0.50:0.05:0.95 (AP50:95).
- Recall and precision at confidence 0.25 (IoU 0.5); maximum recall over all confidences.
- 95% CI: image-level bootstrap, 1000 resamples, fixed seed. Frames are temporally correlated, so these CIs are optimistic.
- YOLO: mean and range over the 3 seeds. Panoptic FCN: one seed unless more are run (stated).
- Thesis target (scenario A): machinery recall >= 0.6 at IoU 0.5, conf 0.25.

## 6. Procedure
1. Train on train only; val only for monitoring.
2. Freeze: this file, the arm CSV hash, checkpoint hashes, evaluator commit.
3. Evaluate test once per model and scenario. No change to GT, thresholds, class map or checkpoints after looking at test.

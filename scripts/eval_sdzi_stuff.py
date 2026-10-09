"""Evaluate SDZI (and river, as a sanity check) from the STUFF branch.

SDZI is a stuff class in this model, so it is evaluated from the semantic
map, not from thing instances (the week-8 script compared thing instances
against SDZI boxes, which could never match). For every image:

  1. predict() (same preprocessing as training) -> SDZI probability map
     at the original resolution.
  2. SDZI mask = argmax == SDZI, or prob >= threshold (several thresholds).
  3. Connected components with area >= min_area (fraction of the image)
     -> one tight box per component.
  4. Greedy matching of component boxes vs GT SDZI boxes:
       - IoU >= 0.5  (main, same criterion as the YOLO baseline)
       - IoU >= 0.3  (secondary analysis)
       - coverage >= 0.3: a GT box counts as found if >= 30% of its pixels
         are predicted SDZI (region-style criterion, analysis only)
  5. Recall split by GT box size: small (< 2% of the image; mostly
     machinery) vs large (excavated ground).

Predictions are computed once per checkpoint and image and scored against
every ground truth given (the 2x2 datasets share images and split).

The operating point (mask mode + min_area) is chosen on VAL by F2 at
IoU 0.5 (recall weighted over precision) and then applied unchanged to
TEST. Precision is reported with the caveat that the GT is incomplete
(unannotated SDZI counts as false positive).

Usage (from the repo root, Windows):
    .venv\\Scripts\\python scripts\\eval_sdzi_stuff.py ^
        --checkpoint etiquetas="C:\\...\\checkpoints_2x2\\etiquetas\\epoch_150.pt" ^
        --checkpoint abril="C:\\...\\checkpoints_2x2\\abril\\epoch_150.pt" ^
        --gt etiquetas="C:\\...\\datos_2x2\\etiquetas" --gt abril="C:\\...\\datos_2x2\\abril" ^
        --out "C:\\...\\Proyecto de grado\\resultados_2x2"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT  # noqa: E402
from panoptic_mining.inference import load_model, predict  # noqa: E402

YOLO_IDS = {"vehicle": 0, "building": 1, "road": 2, "river": 3, "SDZI": 4}
MASK_MODES = ["argmax", "p>=0.2", "p>=0.3", "p>=0.4", "p>=0.5", "p>=0.6"]
MIN_AREAS = [0.0005, 0.002, 0.01]
SMALL_BOX_FRACTION = 0.02


def parse_named(values: list[str]) -> dict[str, Path]:
    out = {}
    for v in values:
        name, _, path = v.partition("=")
        out[name] = Path(path)
    return out


def read_gt_boxes(label_path: Path, class_id: int, w: int, h: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(float(parts[0])) != class_id:
            continue
        xc, yc, bw, bh = map(float, parts[1:5])
        boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h))
    return boxes


def box_iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(preds, gts, thr) -> set[int]:
    """Returns the indices of matched GT boxes."""
    pairs = sorted(((box_iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)), reverse=True)
    used_p, used_g = set(), set()
    for score, i, j in pairs:
        if score < thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
    return used_g


def component_boxes(mask: np.ndarray, min_area_px: int) -> list[tuple[float, float, float, float]]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    boxes = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if area >= min_area_px:
            boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def empty_counts() -> dict:
    keys = ["n_gt", "gt_small", "gt_large", "pred", "tp50", "tp50_small", "tp50_large", "tp30", "hit_cov30", "fp50"]
    return {k: 0 for k in keys}


def finalize(c: dict) -> dict:
    def ratio(a, b):
        return round(a / b, 4) if b else None

    r50, p50 = ratio(c["tp50"], c["n_gt"]), ratio(c["pred"] - c["fp50"], c["pred"])
    f2 = None
    if r50 is not None and p50 is not None and (4 * p50 + r50) > 0:
        f2 = round(5 * p50 * r50 / (4 * p50 + r50), 4)
    return {
        **c,
        "recall50": r50,
        "recall50_small": ratio(c["tp50_small"], c["gt_small"]),
        "recall50_large": ratio(c["tp50_large"], c["gt_large"]),
        "recall30": ratio(c["tp30"], c["n_gt"]),
        "recall_cov30": ratio(c["hit_cov30"], c["n_gt"]),
        "precision50": p50,
        "f2_50": f2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", action="append", required=True, help="name=path (repeatable)")
    ap.add_argument("--gt", action="append", required=True, help="name=dataset_root (repeatable)")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-images", type=int, default=0, help="Debug: only the first N images per split.")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    checkpoints, gts = parse_named(args.checkpoint), parse_named(args.gt)
    args.out.mkdir(parents=True, exist_ok=True)
    sdzi_stuff = DEFAULT_CLASS_SPLIT.stuff_index["SDZI"]
    river_stuff = DEFAULT_CLASS_SPLIT.stuff_index["river"]
    first_gt = next(iter(gts.values()))

    rows = []
    for ckpt_name, ckpt_path in checkpoints.items():
        model = load_model(ckpt_path)
        for split in args.splits:
            image_paths = sorted((first_gt / split / "images").glob("*.jpg"))
            if args.max_images:
                image_paths = image_paths[: args.max_images]
            counts = {(g, m, a): empty_counts() for g in gts for m in MASK_MODES for a in MIN_AREAS}
            river = {g: empty_counts() for g in gts}
            t0 = time.monotonic()
            for n_img, img_path in enumerate(image_paths, 1):
                pred = predict(model, img_path)
                h, w = pred.original_size
                probs = pred.stuff_probs[sdzi_stuff]
                masks = {"argmax": pred.stuff_argmax == sdzi_stuff}
                for m in MASK_MODES[1:]:
                    masks[m] = probs >= float(m.split(">=")[1])
                river_mask = pred.stuff_argmax == river_stuff
                comps = {(m, a): component_boxes(masks[m], int(a * h * w)) for m in MASK_MODES for a in MIN_AREAS}
                river_boxes = component_boxes(river_mask, int(0.002 * h * w))

                for g, root in gts.items():
                    label_path = root / split / "labels" / f"{img_path.stem}.txt"
                    gt_boxes = read_gt_boxes(label_path, YOLO_IDS["SDZI"], w, h)
                    small = [i for i, b in enumerate(gt_boxes) if (b[2] - b[0]) * (b[3] - b[1]) / (w * h) < SMALL_BOX_FRACTION]
                    for m in MASK_MODES:
                        mask = masks[m]
                        cov_hits = 0
                        for b in gt_boxes:
                            x1, y1, x2, y2 = (int(max(0, b[0])), int(max(0, b[1])), int(min(w, b[2])), int(min(h, b[3])))
                            region = mask[y1:y2, x1:x2]
                            if region.size and region.mean() >= 0.3:
                                cov_hits += 1
                        for a in MIN_AREAS:
                            preds = comps[(m, a)]
                            m50 = greedy_match(preds, gt_boxes, 0.5)
                            m30 = greedy_match(preds, gt_boxes, 0.3)
                            c = counts[(g, m, a)]
                            c["n_gt"] += len(gt_boxes)
                            c["gt_small"] += len(small)
                            c["gt_large"] += len(gt_boxes) - len(small)
                            c["pred"] += len(preds)
                            c["tp50"] += len(m50)
                            c["tp50_small"] += len(m50 & set(small))
                            c["tp50_large"] += len(m50 - set(small))
                            c["tp30"] += len(m30)
                            c["hit_cov30"] += cov_hits
                            c["fp50"] += len(preds) - len(m50)
                    river_gt = read_gt_boxes(label_path, YOLO_IDS["river"], w, h)
                    rm = greedy_match(river_boxes, river_gt, 0.5)
                    rc = river[g]
                    rc["n_gt"] += len(river_gt)
                    rc["pred"] += len(river_boxes)
                    rc["tp50"] += len(rm)
                    rc["fp50"] += len(river_boxes) - len(rm)
                if n_img % 50 == 0:
                    print(f"  [{ckpt_name} {split}] {n_img}/{len(image_paths)} ({time.monotonic() - t0:.0f}s)", flush=True)

            for (g, m, a), c in counts.items():
                rows.append({"checkpoint": ckpt_name, "gt": g, "split": split, "class": "SDZI", "mask": m, "min_area": a, **finalize(c)})
            for g, c in river.items():
                rows.append({"checkpoint": ckpt_name, "gt": g, "split": split, "class": "river", "mask": "argmax", "min_area": 0.002, **finalize(c)})
            print(f"[{ckpt_name}] {split}: {len(image_paths)} images in {time.monotonic() - t0:.0f}s", flush=True)

    fields = list(rows[0])
    with open(args.out / "all_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def get(ckpt, gt, split, mask, area):
        for r in rows:
            if (r["checkpoint"], r["gt"], r["split"], r["class"], r["mask"], r["min_area"]) == (ckpt, gt, split, "SDZI", mask, area):
                return r
        return None

    summary = []
    print("\n=== SDZI, operating point chosen on VAL (max F2 @ IoU 0.5), reported on TEST ===")
    print(f"{'model':<12}{'GT':<12}{'mask':<9}{'minA':>7}{'gt':>5}{'TP50':>6}{'R@0.5':>8}{'R@0.3':>8}{'Rcov':>7}{'Rsmall':>8}{'Rlarge':>8}{'P@0.5':>8}")
    for ckpt in checkpoints:
        for g in gts:
            val_rows = [r for r in rows if r["checkpoint"] == ckpt and r["gt"] == g and r["split"] == "val" and r["class"] == "SDZI"]
            if not val_rows:
                continue
            best = max(val_rows, key=lambda r: (r["f2_50"] or 0, r["recall50"] or 0))
            test = get(ckpt, g, "test", best["mask"], best["min_area"])
            if test is None:
                continue
            summary.append({"checkpoint": ckpt, "gt": g, "chosen_on_val": {"mask": best["mask"], "min_area": best["min_area"], "val_f2": best["f2_50"], "val_recall50": best["recall50"]}, "test": test})
            fmt = lambda v: "  -  " if v is None else f"{v:.3f}"  # noqa: E731
            print(f"{ckpt:<12}{g:<12}{best['mask']:<9}{best['min_area']:>7}{test['n_gt']:>5}{test['tp50']:>6}{fmt(test['recall50']):>8}{fmt(test['recall30']):>8}"
                  f"{fmt(test['recall_cov30']):>7}{fmt(test['recall50_small']):>8}{fmt(test['recall50_large']):>8}{fmt(test['precision50']):>8}")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\nRiver sanity check (argmax, IoU 0.5) on test:")
    for r in rows:
        if r["class"] == "river" and r["split"] == "test":
            print(f"  {r['checkpoint']:<12}{r['gt']:<12} gt={r['n_gt']} TP={r['tp50']} recall={r['recall50']} precision={r['precision50']}")
    print(f"\nFull grid: {args.out / 'all_results.csv'}  |  summary: {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()

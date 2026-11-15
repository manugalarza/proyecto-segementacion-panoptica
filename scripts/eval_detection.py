"""Box-level detection evaluation shared by both projects (Panoptic FCN and YOLO).

Protocol (agreed 2026-11-10):
  * Ground truth = the YOLO label boxes of the split. No SAM masks are involved.
  * One GT instance per box. Predictions are matched one-to-one, greedily by score,
    to the unmatched GT box of the same class with the highest IoU (>= threshold).
  * Exact duplicate frames (same content_id in the dataset's manifest.csv) are counted once.
  * A class can be excluded with --exclude: its GT boxes AND its predictions are dropped.
    They are not turned into ignore regions, so they do not change the other classes.
  * AP is all-points interpolated, pooled over frames per class, at IoU 0.50 and averaged
    over 0.50:0.95. Recall is reported at a fixed confidence (--conf) and at the lowest one.
  * 95% CIs come from a bootstrap over frames.

Prediction sources (repeatable, mixed freely):
  --checkpoint NAME=PATH    Panoptic FCN checkpoint: instance masks from decode_instances,
                            box = tight box around each mask (after mask NMS).
  --predictions NAME=DIR    YOLO-format prediction txt per frame: "cls xc yc w h conf",
                            normalized, file name = image stem (e.g. YOLO `save_txt` + `save_conf`).

Class ids follow the dataset.yaml of the split (0 vehicle, 1 building, 2 road, 3 river,
4 SDZI, 5 machinery). --classes picks which are scored (default: vehicle building machinery).
--pred-class-map remaps prediction ids of a --predictions source, e.g. "4:-1" to drop SDZI.

Usage (repo root, Windows):
    .venv\\Scripts\\python scripts\\eval_detection.py --data ..\\datos_hist_s3_seed0 --split val ^
        --checkpoint C=..\\checkpoints_hist_s3\\C_inst2_masks_lr3e4\\epoch_150.pt ^
        --out ..\\resultados_det\\val_C
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

NAMES = {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI", 5: "machinery"}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]


def read_boxes(path: Path, with_conf: bool) -> list[tuple]:
    """Returns [(cls, x0, y0, x1, y1, conf)] in normalized coordinates."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, xc, yc, w, h = int(p[0]), *map(float, p[1:5])
        conf = float(p[5]) if with_conf and len(p) > 5 else 1.0
        out.append((c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2, conf))
    return out


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, thr):
    """preds: [(score, box)], gts: [box]. Returns [(score, tp)] in score order."""
    used, out = set(), []
    for score, pb in sorted(preds, key=lambda p: -p[0]):
        best, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in used:
                continue
            v = box_iou(pb, gb)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= thr:
            used.add(best_j)
            out.append((score, 1))
        else:
            out.append((score, 0))
    return out


SIZE_BINS = [("small", 0.0, 0.0025), ("medium", 0.0025, 0.01), ("large", 0.01, 1.01)]  # box area / image area


def matched_gt(preds, gts, thr):
    """Indices of the GT boxes matched (same greedy one-to-one rule as match())."""
    used = set()
    for _, pb in sorted(preds, key=lambda p: -p[0]):
        best, best_j = 0.0, -1
        for j, gb in enumerate(gts):
            if j in used:
                continue
            v = box_iou(pb, gb)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= thr:
            used.add(best_j)
    return used


def covered(gb, pb) -> float:
    """Fraction of the GT box area that lies inside the predicted box."""
    ix = max(0.0, min(gb[2], pb[2]) - max(gb[0], pb[0]))
    iy = max(0.0, min(gb[3], pb[3]) - max(gb[1], pb[1]))
    a = (gb[2] - gb[0]) * (gb[3] - gb[1])
    return ix * iy / a if a > 0 else 0.0


def diagnostics(frames, class_ids, conf):
    """Where the misses come from: localization (found at IoU 0.3 but not 0.5), size,
    and merged detections (one confident box covering >= 2 GT boxes by >= 50% each)."""
    out = {}
    for c in class_ids:
        d = {"recall_iou30_all": 0, "recall_iou50_all": 0, "by_size": {}, "merged_preds_at_conf": 0,
             "preds_at_conf": 0}
        tot = {b[0]: [0, 0, 0] for b in SIZE_BINS}  # n_gt, hit50, hit30
        n_gt = h50 = h30 = 0
        for f in frames:
            gts, preds = f["gt"].get(c, []), f["pred"].get(c, [])
            m50, m30 = matched_gt(preds, gts, 0.5), matched_gt(preds, gts, 0.3)
            n_gt += len(gts); h50 += len(m50); h30 += len(m30)
            for j, g in enumerate(gts):
                a = (g[2] - g[0]) * (g[3] - g[1])
                name = next(n for n, lo, hi in SIZE_BINS if lo <= a < hi)
                tot[name][0] += 1; tot[name][1] += j in m50; tot[name][2] += j in m30
            for s, pb in preds:
                if s < conf:
                    continue
                d["preds_at_conf"] += 1
                if sum(1 for g in gts if covered(g, pb) >= 0.5) >= 2:
                    d["merged_preds_at_conf"] += 1
        d["recall_iou50_all"] = h50 / n_gt if n_gt else None
        d["recall_iou30_all"] = h30 / n_gt if n_gt else None
        d["by_size"] = {k: {"n_gt": v[0], "recall50": v[1] / v[0] if v[0] else None,
                            "recall30": v[2] / v[0] if v[0] else None} for k, v in tot.items()}
        out[NAMES[c]] = d
    return out


def average_precision(records, n_gt):
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    recs = sorted(records, key=lambda r: -r[0])
    tp = np.array([r[1] for r in recs], dtype=float)
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def unique_frames(data: Path, split: str, label_source: str | None = None) -> list[Path]:
    """Image paths of the split, one per content_id when manifest.csv has that column.
    label_source keeps only frames whose manifest label_source matches (e.g. datos_v2)."""
    img_dir = data / split / "images"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    manifest = data / "manifest.csv"
    if not manifest.exists():
        return images
    with open(manifest, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "content_id" not in rows[0] or "out_name" not in rows[0]:
        return images
    cid = {r["out_name"]: r["content_id"] for r in rows}
    if label_source:
        allowed = {r["out_name"] for r in rows if r.get("label_source") == label_source}
        images = [p for p in images if p.stem in allowed]
    seen, keep = set(), []
    for p in images:
        key = cid.get(p.stem, p.stem)
        if key in seen:
            continue
        seen.add(key)
        keep.append(p)
    return keep


def panoptic_predictions(ckpt: str, images: list[Path], class_ids: list[int], args) -> dict:
    """{stem: [(cls, x0, y0, x1, y1, score)]} from a Panoptic FCN checkpoint."""
    import cv2
    import torch
    import torch.nn.functional as F

    from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT
    from panoptic_mining.inference import load_image_like_training, load_model

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = load_model(ckpt, device=device)
    thing_index = DEFAULT_CLASS_SPLIT.thing_index
    out = {}
    with torch.no_grad():
        for k, img in enumerate(images, 1):
            tensor, (h, w) = load_image_like_training(img, getattr(model, "image_size", (512, 512)))
            output = model(tensor.to(device))
            # masks are compared at a reduced size; boxes are normalized anyway
            mh, mw = max(1, h // args.mask_downscale), max(1, w // args.mask_downscale)
            min_area = args.min_pred_area / args.mask_downscale ** 2
            dets = []
            for c in class_ids:
                name = NAMES[c]
                if name not in thing_index:
                    continue
                kept = []  # (score, mask) after mask NMS
                for score, m in model.decode_instances(output, thing_index[name], top_k=args.top_k,
                                                       score_threshold=args.score_floor):
                    mask = F.interpolate(m[None, None].float(), size=(mh, mw), mode="bilinear",
                                         align_corners=False)[0, 0].cpu().numpy() > args.mask_threshold
                    if mask.sum() < min_area:
                        continue
                    if all(np.logical_and(mask, km).sum() / np.logical_or(mask, km).sum() < args.nms_iou
                           for _, km in kept):
                        kept.append((score, mask))
                for score, mask in kept:
                    if args.box_mode == "largest_cc":
                        n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
                        if n > 2:
                            mask = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
                    ys, xs = np.nonzero(mask)
                    dets.append((c, xs.min() / mw, ys.min() / mh, (xs.max() + 1) / mw, (ys.max() + 1) / mh, score))
            out[img.stem] = dets
            if k % 25 == 0:
                print(f"  {k}/{len(images)}", flush=True)
    return out


def file_predictions(folder: Path, images: list[Path], remap: dict) -> dict:
    """Reads <stem>.txt; if absent, <nombre_frame>.txt (the dataset stem without the "__rNNNN"
    suffix added by build_split_from_csv.py), so prediction files named by the original frame
    name (e.g. from another team member's YOLO run) match. A missing file means no detections,
    so the number of matched files is printed and a run with none matched is an error."""
    out, found = {}, 0
    for img in images:
        path = folder / f"{img.stem}.txt"
        if not path.exists():
            alt = folder / f"{img.stem.split('__r')[0]}.txt"
            if alt.exists():
                path = alt
        found += path.exists()
        dets = []
        for c, *b, conf in read_boxes(path, with_conf=True):
            c = remap.get(c, c)
            if c >= 0:
                dets.append((c, *b, conf))
        out[img.stem] = dets
    print(f"  prediction files matched: {found}/{len(images)} in {folder}")
    if found == 0:
        raise SystemExit(f"No prediction file in {folder} matches the {len(images)} frames: check the file names.")
    return out


def evaluate(frames: list[dict], class_ids: list[int], conf: float) -> dict:
    """frames: [{"gt": {c: [box]}, "pred": {c: [(score, box)]}}]."""
    res = {}
    for c in class_ids:
        n_gt = sum(len(f["gt"].get(c, [])) for f in frames)
        by_thr = {}
        for thr in IOU_THRESHOLDS:
            recs = [r for f in frames for r in match(f["pred"].get(c, []), f["gt"].get(c, []), thr)]
            by_thr[thr] = (recs, average_precision(recs, n_gt))
        recs50 = by_thr[0.5][0]
        tp_all = sum(t for _, t in recs50)
        tp_conf = sum(t for s, t in recs50 if s >= conf)
        n_pred_conf = sum(1 for s, _ in recs50 if s >= conf)
        aps = [a for _, a in by_thr.values() if a is not None]
        res[NAMES[c]] = {
            "n_gt": n_gt,
            "n_pred": len(recs50),
            "AP50": by_thr[0.5][1],
            "AP50_95": float(np.mean(aps)) if aps else None,
            f"recall@conf{conf}": tp_conf / n_gt if n_gt else None,
            f"precision@conf{conf}": tp_conf / n_pred_conf if n_pred_conf else None,
            "max_recall": tp_all / n_gt if n_gt else None,
        }
    vals = [v["AP50"] for v in res.values() if v["AP50"] is not None]
    res["mAP50"] = float(np.mean(vals)) if vals else None
    vals = [v["AP50_95"] for v in res.values() if isinstance(v, dict) and v.get("AP50_95") is not None]
    res["mAP50_95"] = float(np.mean(vals)) if vals else None
    return res


def bootstrap(frames, class_ids, conf, n, seed):
    rng = np.random.default_rng(seed)
    vals = defaultdict(list)
    for _ in range(n):
        sample = [frames[i] for i in rng.integers(0, len(frames), len(frames))]
        r = evaluate_ap50_only(sample, class_ids)
        for k, v in r.items():
            if v is not None:
                vals[k].append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def evaluate_ap50_only(frames, class_ids):
    out = {}
    for c in class_ids:
        n_gt = sum(len(f["gt"].get(c, [])) for f in frames)
        recs = [r for f in frames for r in f["m50"].get(c, [])]
        out[NAMES[c]] = average_precision(recs, n_gt)
    vals = [v for v in out.values() if v is not None]
    out["mAP50"] = float(np.mean(vals)) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="Dataset root with <split>/images, <split>/labels")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH (Panoptic FCN)")
    ap.add_argument("--predictions", action="append", default=[], help="NAME=DIR (YOLO txt with conf)")
    ap.add_argument("--pred-class-map", default="", help='Remap prediction ids, e.g. "4:-1" (-1 drops)')
    ap.add_argument("--classes", nargs="+", default=["vehicle", "building", "machinery"])
    ap.add_argument("--exclude", nargs="*", default=[], help="Class names dropped from GT and predictions")
    ap.add_argument("--label-source", default=None,
                    help="Only frames whose manifest.csv label_source equals this (e.g. datos_v2)")
    ap.add_argument("--conf", type=float, default=0.25, help="Fixed confidence for recall/precision")
    ap.add_argument("--score-floor", type=float, default=0.01, help="Lowest heatmap score decoded (Panoptic FCN)")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--min-pred-area", type=int, default=20, help="px at full resolution")
    ap.add_argument("--mask-downscale", type=int, default=2)
    ap.add_argument("--box-mode", choices=["full", "largest_cc"], default="largest_cc",
                    help="largest_cc (default since 2026-11-11): box around the largest connected part of the "
                         "predicted mask. The instance kernel also fires on other, similar-looking objects of the "
                         "frame, so a box around the whole mask (full, the old behavior) spans several buildings.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    name_to_id = {v: k for k, v in NAMES.items()}
    excluded = {name_to_id[n] for n in args.exclude}
    class_ids = [name_to_id[n] for n in args.classes if name_to_id[n] not in excluded]
    remap = {int(a): int(b) for a, b in (x.split(":") for x in args.pred_class_map.split(",") if x)}

    images = unique_frames(args.data, args.split, args.label_source)
    print(f"{args.split}: {len(images)} unique frames; classes {[NAMES[c] for c in class_ids]}; excluded {sorted(args.exclude)}")
    gts = {}
    for img in images:
        g = defaultdict(list)
        for c, *b, _ in read_boxes(args.data / args.split / "labels" / f"{img.stem}.txt", with_conf=False):
            if c in class_ids:
                g[c].append(tuple(b))
        gts[img.stem] = g

    sources = {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        print(f"== {name} (checkpoint)")
        sources[name] = panoptic_predictions(path, images, class_ids, args)
    for item in args.predictions:
        name, _, path = item.partition("=")
        sources[name] = file_predictions(Path(path), images, remap)
    if not sources:
        raise SystemExit("Give at least one --checkpoint or --predictions")

    report = {"settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
              | {"n_frames": len(images), "frames": [p.stem for p in images]}, "models": {}}
    for name, preds in sources.items():
        frames = []
        for img in images:
            p = defaultdict(list)
            for c, *b, s in preds[img.stem]:
                if c in class_ids:
                    p[c].append((s, tuple(b)))
            f = {"gt": gts[img.stem], "pred": p}
            f["m50"] = {c: match(p.get(c, []), gts[img.stem].get(c, []), 0.5) for c in class_ids}
            frames.append(f)
        res = evaluate(frames, class_ids, args.conf)
        res["ci95_AP50"] = bootstrap(frames, class_ids, args.conf, args.bootstrap, args.seed)
        res["diagnostics"] = diagnostics(frames, class_ids, args.conf)
        report["models"][name] = res

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detection_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    rk, pk = f"recall@conf{args.conf}", f"precision@conf{args.conf}"
    print(f"\n{'model':<14}{'class':<11}{'n_gt':>6}{'n_pred':>8}{'AP50':>8}{'AP50:95':>9}{'R@conf':>8}{'P@conf':>8}{'maxR':>7}  CI95 AP50")
    for name, r in report["models"].items():
        for c in class_ids:
            v, ci = r[NAMES[c]], r["ci95_AP50"].get(NAMES[c])
            print(f"{name:<14}{NAMES[c]:<11}{v['n_gt']:>6}{v['n_pred']:>8}{f(v['AP50']):>8}{f(v['AP50_95']):>9}"
                  f"{f(v[rk]):>8}{f(v[pk]):>8}{f(v['max_recall']):>7}  {ci and [round(x, 3) for x in ci]}")
        ci = r["ci95_AP50"].get("mAP50")
        print(f"{name:<14}{'mAP':<11}{'':>14}{f(r['mAP50']):>8}{f(r['mAP50_95']):>9}{'':>23}  {ci and [round(x, 3) for x in ci]}")
    print("\nDiagnostics (recall over all predictions; merged = confident box covering >= 2 GT boxes)")
    print(f"{'model':<26}{'class':<11}{'R@IoU.5':>8}{'R@IoU.3':>8}  "
          + "  ".join(f"{n}(n) R.5/R.3" for n, _, _ in SIZE_BINS) + f"  {'merged/preds@conf':>18}")
    for name, r in report["models"].items():
        for c in class_ids:
            d = r["diagnostics"][NAMES[c]]
            sizes = "  ".join(f"{k}({v['n_gt']}) {f(v['recall50'])}/{f(v['recall30'])}" for k, v in d["by_size"].items())
            print(f"{name:<26}{NAMES[c]:<11}{f(d['recall_iou50_all']):>8}{f(d['recall_iou30_all']):>8}  {sizes}"
                  f"  {d['merged_preds_at_conf']:>8}/{d['preds_at_conf']}")
    print(f"\nWritten: {args.out / 'detection_summary.json'}")


if __name__ == "__main__":
    main()

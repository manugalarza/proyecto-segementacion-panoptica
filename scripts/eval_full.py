"""Full evaluation of the trained model on gt_manual.

Things (vehicle, building, machinery)
  Predicted instances come from the model's own kernel decoder
  (PanopticFCN.decode_instances): one mask per heatmap peak.
  GT instances are the connected components of each thing class in the
  gt_manual masks (components smaller than --min-gt-area px are dropped).
  Matching is COCO-style: predictions sorted by score, each greedily
  matched to the best unmatched GT with IoU >= threshold.
  AP is all-points interpolated, pooled over frames per class.
  mAP is the mean over the thing classes that have GT.
  Reported: AP50, AP75, mAP50, mAP75, mean IoU of matched instances.

Stuff (river, road)
  Pixel IoU, Dice, pixel precision and recall over all annotated pixels
  (ignore = 255 excluded). Same convention as eval_pixel.py.

95% CIs come from a bootstrap over frames (--bootstrap resamples).
With only ~29 frames the CIs are wide; read them before comparing models.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT
from panoptic_mining.inference import load_image_like_training, load_model

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
THING_GT = {"vehicle": 0, "building": 1, "machinery": 6}
STUFF_GT = {"river": 3, "road": 2}
IGNORE = 255
IOU_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]  # mAP@[50:75]


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def gt_instances(gt, cid, min_area):
    m = (gt == cid).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def match(preds, gts, thr):
    """preds: list of (score, mask). Returns (score, tp) flags and matched IoUs."""
    flags, matched_ious = [], []
    used = set()
    for score, pm in sorted(preds, key=lambda p: -p[0]):
        best, best_j = 0.0, -1
        for j, gm in enumerate(gts):
            if j in used:
                continue
            v = iou(pm, gm)
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= thr:
            used.add(best_j)
            flags.append((score, 1))
            matched_ious.append(best)
        else:
            flags.append((score, 0))
    return flags, matched_ious


def average_precision(records, n_gt):
    """All-points interpolated AP. records: list of (score, tp)."""
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    recs = sorted(records, key=lambda r: -r[0])
    tp = np.array([r[1] for r in recs], dtype=float)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def aggregate_things(frames, idxs):
    out = {}
    for name in THING_GT:
        n_gt = sum(frames[i]["things"][name]["n_gt"] for i in idxs)
        n_pred = sum(frames[i]["things"][name]["n_pred"] for i in idxs)
        ap_by_thr = {}
        for thr in IOU_THRESHOLDS:
            key = f"{thr:.2f}"
            recs = [r for i in idxs for r in frames[i]["things"][name]["flags"][key]]
            ap_by_thr[key] = average_precision(recs, n_gt)
        ious = [v for i in idxs for v in frames[i]["things"][name]["iou50"]]
        vals = [v for v in ap_by_thr.values() if v is not None]
        out[name] = {
            "n_gt": n_gt,
            "n_pred": n_pred,
            "AP50": ap_by_thr["0.50"],
            "AP75": ap_by_thr["0.75"],
            "AP50_75": float(np.mean(vals)) if vals else None,
            "AP_by_threshold": ap_by_thr,
            "mean_matched_iou": float(np.mean(ious)) if ious else None,
        }
    for key in ("AP50", "AP75", "AP50_75"):
        vals = [out[n][key] for n in THING_GT if out[n][key] is not None]
        out[f"m{key}"] = float(np.mean(vals)) if vals else None
    return out


def aggregate_stuff(frames, idxs):
    out = {}
    for name in STUFF_GT:
        inter = sum(frames[i]["stuff"][name]["inter"] for i in idxs)
        pred = sum(frames[i]["stuff"][name]["pred"] for i in idxs)
        gt = sum(frames[i]["stuff"][name]["gt"] for i in idxs)
        union = pred + gt - inter
        out[name] = {
            "IoU": inter / union if union else None,
            "Dice": 2 * inter / (pred + gt) if (pred + gt) else None,
            "pixel_precision": inter / pred if pred else None,
            "pixel_recall": inter / gt if gt else None,
        }
    return out


def bootstrap_ci(frames, n, fn, seed, key_path):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idxs = rng.integers(0, len(frames), len(frames))
        v = fn(frames, idxs)
        for k in key_path:
            v = v[k]
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--score-threshold", type=float, default=0.05, help="Heatmap peak threshold")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-pred-area", type=int, default=20, help="Predicted instances smaller than this (px) are dropped")
    ap.add_argument("--min-gt-area", type=int, default=20, help="GT components smaller than this (px) are dropped")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    frames_paths = sorted(p for p in args.images.iterdir()
                          if p.suffix.lower() in IMAGE_EXTENSIONS and (args.gt / f"{p.stem}.png").exists())
    if not frames_paths:
        raise SystemExit(f"No annotated images found in {args.images}")
    print(f"{len(frames_paths)} annotated frames, device={device}")
    args.out.mkdir(parents=True, exist_ok=True)

    thing_ids = {name: DEFAULT_CLASS_SPLIT.thing_index[name] for name in THING_GT}
    river_i = DEFAULT_CLASS_SPLIT.stuff_index["river"]
    road_i = DEFAULT_CLASS_SPLIT.stuff_index["road"]
    stuff_idx = {"river": river_i, "road": road_i}

    report = {"settings": vars(args) | {"images": str(args.images), "gt": str(args.gt), "out": str(args.out),
                                         "frames": [p.stem for p in frames_paths], "device": device},
              "models": {}}

    for item in args.checkpoint:
        name, _, path = item.partition("=")
        print(f"== {name}")
        model = load_model(path, device=device)
        model.eval()
        frames = []
        with torch.no_grad():
            for img_path in frames_paths:
                gt = cv2.imread(str(args.gt / f"{img_path.stem}.png"), cv2.IMREAD_UNCHANGED)
                if gt.ndim == 3:
                    gt = gt[..., 0]
                tensor, (h, w) = load_image_like_training(img_path)
                if gt.shape != (h, w):
                    raise SystemExit(f"Size mismatch for {img_path.stem}: mask {gt.shape} vs image {(h, w)}")
                output = model(tensor.to(device))

                rec = {"things": {}, "stuff": {}}
                for tname, cid in THING_GT.items():
                    gts = gt_instances(gt, cid, args.min_gt_area)
                    preds = []
                    for score, m in model.decode_instances(output, thing_ids[tname], top_k=args.top_k,
                                                           score_threshold=args.score_threshold):
                        up = F.interpolate(m[None, None].float(), size=(h, w), mode="bilinear",
                                           align_corners=False)[0, 0].cpu().numpy()
                        mask = up > args.mask_threshold
                        if mask.sum() >= args.min_pred_area:
                            preds.append((score, mask))
                    flags = {}
                    for thr in IOU_THRESHOLDS:
                        flags[f"{thr:.2f}"] = match(preds, gts, thr)[0]
                    _, i50 = match(preds, gts, 0.5)
                    rec["things"][tname] = {
                        "n_gt": len(gts), "n_pred": len(preds),
                        "flags": flags, "iou50": i50,
                    }

                logits = F.interpolate(output.stuff_logits, size=(h, w), mode="bilinear", align_corners=False)
                argmax = logits.argmax(dim=1)[0].cpu().numpy()
                valid = gt != IGNORE
                for sname, cid in STUFF_GT.items():
                    pred_m = (argmax == stuff_idx[sname]) & valid
                    gt_m = (gt == cid) & valid
                    rec["stuff"][sname] = {
                        "inter": int(np.logical_and(pred_m, gt_m).sum()),
                        "pred": int(pred_m.sum()),
                        "gt": int(gt_m.sum()),
                    }
                frames.append(rec)
                print(f"  {img_path.stem} done")

        things = aggregate_things(frames, list(range(len(frames))))
        stuff = aggregate_stuff(frames, list(range(len(frames))))
        ci = {
            "mAP50": bootstrap_ci(frames, args.bootstrap, aggregate_things, args.seed, ["mAP50"]),
            "mAP75": bootstrap_ci(frames, args.bootstrap, aggregate_things, args.seed, ["mAP75"]),
            "mAP50_75": bootstrap_ci(frames, args.bootstrap, aggregate_things, args.seed, ["mAP50_75"]),
        }
        for sname in STUFF_GT:
            ci[f"{sname}_IoU"] = bootstrap_ci(frames, args.bootstrap, aggregate_stuff, args.seed,
                                              [sname, "IoU"])
        report["models"][name] = {"things": things, "stuff": stuff, "ci95": ci}

    (args.out / "full_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    def f(x):
        return "  n/a" if x is None else f"{x:.3f}"

    print()
    print(f"{'model':<12}{'class':<11}{'n_gt':>6}{'n_pred':>8}{'AP50':>8}{'AP75':>8}{'AP50:75':>9}{'mIoU_m':>9}")
    for mname, m in report["models"].items():
        for cname in THING_GT:
            t = m["things"][cname]
            print(f"{mname:<12}{cname:<11}{t['n_gt']:>6}{t['n_pred']:>8}{f(t['AP50']):>8}{f(t['AP75']):>8}"
                  f"{f(t['AP50_75']):>9}{f(t['mean_matched_iou']):>9}")
        print(f"{mname:<12}{'mAP':<11}{'':>6}{'':>8}{f(m['things']['mAP50']):>8}{f(m['things']['mAP75']):>8}"
              f"{f(m['things']['mAP50_75']):>9}")
        for sname in STUFF_GT:
            s = m["stuff"][sname]
            print(f"{mname:<12}{sname:<11}  IoU {f(s['IoU'])}  Dice {f(s['Dice'])}  "
                  f"P {f(s['pixel_precision'])}  R {f(s['pixel_recall'])}")
    print(f"\nWritten: {args.out / 'full_summary.json'}")


if __name__ == "__main__":
    main()

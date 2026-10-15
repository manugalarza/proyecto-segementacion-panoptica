"""Thing-branch evaluation (vehicle / building / road) against gt_manual.

The thing branch currently predicts CENTER HEATMAPS only (no per-instance
masks: the kernel x mask-feature loss is not implemented yet). So this
measures detection-by-points, not segmentation:

  peaks    local maxima of sigmoid(thing_heatmap) above --threshold
           (3x3 max-pool NMS), mapped to original image coordinates
  GT       connected regions of each thing class in the manual masks
           (>= --min-area of the image), dilated by --tolerance pixels
           because a 1/8-resolution peak cannot land exactly on a pixel
  recall   GT regions with at least one peak of the same class inside
  precision  peaks that fall inside a GT region of their class
Touching buildings merge into one GT region, and a road is one long
region, so these are region-level numbers, not instance counts.

Usage:
    .venv\\Scripts\\python scripts\\eval_things.py --images "...\\gt_manual\\images" ^
        --gt "...\\gt_manual\\masks" --checkpoint A="...\\A\\epoch_150.pt" ^
        --checkpoint B="...\\B\\epoch_150.pt" --out "...\\resultados_things" --save-vis
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT  # noqa: E402
from panoptic_mining.inference import load_image_like_training, load_model  # noqa: E402

GT_IDS = {"vehicle": 0, "building": 1, "road": 2}
COLORS = {"vehicle": (0, 0, 255), "building": (255, 128, 0), "road": (0, 255, 255)}  # BGR
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@torch.no_grad()
def predict_peaks(model, img_path, threshold, device):
    tensor, (h, w) = load_image_like_training(img_path)
    heat = torch.sigmoid(model(tensor.to(device)).thing_heatmap)[0]  # (C, h', w')
    pooled = F.max_pool2d(heat.unsqueeze(0), 3, stride=1, padding=1)[0]
    out = {}
    hh, ww = heat.shape[-2:]
    for name in GT_IDS:
        ci = DEFAULT_CLASS_SPLIT.thing_index[name]
        ys, xs = torch.nonzero((heat[ci] == pooled[ci]) & (heat[ci] > threshold), as_tuple=True)
        out[name] = [(int((x + 0.5) * w / ww), int((y + 0.5) * h / hh), float(heat[ci, y, x]))
                     for y, x in zip(ys.tolist(), xs.tolist())]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path)
    ap.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH (repeatable)")
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.3, 0.1])
    ap.add_argument("--min-area", type=float, default=0.0001)
    ap.add_argument("--tolerance", type=int, default=15, help="GT dilation in pixels")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--save-vis", action="store_true", help="Draw peaks (first threshold) over GT contours")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    frames = sorted(p for p in args.images.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS and (args.gt / f"{p.stem}.png").exists())
    args.out.mkdir(parents=True, exist_ok=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * args.tolerance + 1, 2 * args.tolerance + 1))

    gts = {}
    for p in frames:
        gt = cv2.imread(str(args.gt / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED)
        regions = {}
        for name, gid in GT_IDS.items():
            n, lab, stats, _ = cv2.connectedComponentsWithStats((gt == gid).astype(np.uint8), connectivity=8)
            keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= args.min_area * gt.size]
            regions[name] = [cv2.dilate((lab == i).astype(np.uint8), kernel).astype(bool) for i in keep]
        gts[p.stem] = (gt, regions)

    rows, summary = [], {}
    for item in args.checkpoint:
        name, _, path = item.partition("=")
        model = load_model(path, device=device)
        for thr in args.threshold:
            tot = {c: {"gt": 0, "found": 0, "peaks": 0, "correct": 0} for c in GT_IDS}
            for p in frames:
                gt, regions = gts[p.stem]
                peaks = predict_peaks(model, p, thr, device)
                for c in GT_IDS:
                    regs = regions[c]
                    union = np.zeros(gt.shape, bool)
                    for r in regs:
                        union |= r
                    pts = peaks[c]
                    found = sum(any(r[y, x] for x, y, _ in pts) for r in regs)
                    correct = sum(bool(union[y, x]) for x, y, _ in pts)
                    t = tot[c]
                    t["gt"] += len(regs); t["found"] += found; t["peaks"] += len(pts); t["correct"] += correct
                    rows.append({"model": name, "threshold": thr, "frame": p.stem, "class": c,
                                 "gt_regions": len(regs), "found": found, "peaks": len(pts), "correct": correct})
                if args.save_vis and thr == args.threshold[0]:
                    img = cv2.imread(str(p))
                    for c, gid in GT_IDS.items():
                        cnts, _ = cv2.findContours((gt == gid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, cnts, -1, COLORS[c], 2)
                        for x, y, s in peaks[c]:
                            cv2.circle(img, (x, y), 14, (0, 0, 0), -1)
                            cv2.circle(img, (x, y), 10, COLORS[c], -1)
                    cv2.putText(img, f"{name}  thr={thr}  contour=GT  dot=prediction  red=vehicle blue=building yellow=road",
                                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    (args.out / "vis" / name).mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.out / "vis" / name / f"{p.stem}.jpg"),
                                cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1]))),
                                [cv2.IMWRITE_JPEG_QUALITY, 88])
            summary[f"{name}@{thr}"] = tot

    with open(args.out / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{len(frames)} frames; region recall = GT regions with a peak of the class inside (tolerance {args.tolerance}px)")
    print(f"{'model@thr':<16}{'class':<10}{'GT regions':>11}{'found':>7}{'recall':>8}{'peaks':>7}{'precision':>11}")
    for key, tot in summary.items():
        for c, t in tot.items():
            rec = t["found"] / t["gt"] if t["gt"] else float("nan")
            prec = t["correct"] / t["peaks"] if t["peaks"] else float("nan")
            print(f"{key:<16}{c:<10}{t['gt']:>11}{t['found']:>7}{rec:>8.2f}{t['peaks']:>7}{prec:>11.2f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()

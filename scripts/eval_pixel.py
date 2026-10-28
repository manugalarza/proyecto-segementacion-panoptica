"""Pixel-level evaluation against hand-made semantic masks (gt_manual).

Ground truth: PNG masks written by scripts/annotate_semantic.py
(pixel = class id 0 vehicle, 1 building, 2 road, 3 river, 4 SDZI, 5 background;
255 = unlabeled -> ignored). Only frames that HAVE a mask are evaluated, so a
partially annotated gt_manual can already be used.

Models (any number, all scored with the same code):
  --checkpoint NAME=PATH   Panoptic FCN checkpoint (stuff branch; encoder read
                           from the checkpoint, see inference.load_model)
  --yolo NAME=PATH         YOLO weights; predicted boxes are filled as masks
                           (river = class 3, SDZI = class 4). Box masks are
                           coarse by construction: compare YOLO mainly on the
                           region metrics, not on pixel IoU.

Metrics, for river and SDZI:
  pixel   IoU / precision / recall over all evaluated pixels
  region  GT connected components (>= --min-area of the image) counted as
          found if the prediction covers >= --cover of their pixels
          (recall_cov), and predicted components counted as correct if
          >= --cover of their pixels fall on that GT class (precision_cov)
  frames  frames with predicted SDZI but no GT SDZI (false alarms)
Recall_cov gets a 95% bootstrap CI over frames (gt_manual has ~29 frames,
so expect wide intervals; report the counts next to the rates).

The Panoptic FCN prediction is argmax by default. --sdzi-threshold t uses
P(SDZI) >= t instead; choose t on a validation split, never on gt_manual.

Usage (repo root):
    .venv\\Scripts\\python scripts\\eval_pixel.py ^
        --images "...\\gt_manual\\images" --gt "...\\gt_manual\\masks" ^
        --checkpoint A="...\\runs\\A\\epoch_150.pt" --checkpoint B="...\\runs\\B\\epoch_150.pt" ^
        --yolo yolo="...\\yolo_abril\\weights\\best.pt" --out "...\\resultados_eval_manual" --device cuda
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

GT_IDS = {"river": 3, "road": 2}
IGNORE = 255
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def components(mask: np.ndarray, min_pixels: int) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [labels == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_pixels]


def score_frame(pred: dict[str, np.ndarray], gt: np.ndarray, min_area: float, cover: float) -> dict:
    valid = gt != IGNORE
    min_pixels = max(1, int(min_area * gt.size))
    row = {}
    for name, gid in GT_IDS.items():
        g = (gt == gid) & valid
        p = pred[name] & valid
        row[f"{name}_tp"] = int((g & p).sum())
        row[f"{name}_fp"] = int((~g & p).sum())
        row[f"{name}_fn"] = int((g & ~p).sum())
        gcomp = components(g, min_pixels)
        pcomp = components(p, min_pixels)
        row[f"{name}_gt_regions"] = len(gcomp)
        row[f"{name}_gt_found"] = sum(int((c & p).sum() >= cover * c.sum()) for c in gcomp)
        row[f"{name}_pred_regions"] = len(pcomp)
        row[f"{name}_pred_correct"] = sum(int((c & g).sum() >= cover * c.sum()) for c in pcomp)
    row["sdzi_false_alarm_frame"] = int(row["road_gt_regions"] == 0 and row["road_pred_regions"] > 0)
    row["has_gt_sdzi"] = int(row["road_gt_regions"] > 0)
    return row


def summarize(rows: list[dict], n_boot: int, seed: int) -> dict:
    out = {"frames": len(rows), "frames_with_gt_sdzi": sum(r["has_gt_sdzi"] for r in rows),
           "sdzi_false_alarm_frames": sum(r["sdzi_false_alarm_frame"] for r in rows)}
    rng = np.random.default_rng(seed)
    for name in GT_IDS:
        tp, fp, fn = (sum(r[f"{name}_{k}"] for r in rows) for k in ("tp", "fp", "fn"))
        out[f"{name}_iou"] = tp / (tp + fp + fn) if tp + fp + fn else None
        out[f"{name}_pixel_precision"] = tp / (tp + fp) if tp + fp else None
        out[f"{name}_pixel_recall"] = tp / (tp + fn) if tp + fn else None
        found = np.array([r[f"{name}_gt_found"] for r in rows])
        total = np.array([r[f"{name}_gt_regions"] for r in rows])
        pc = sum(r[f"{name}_pred_correct"] for r in rows)
        pt = sum(r[f"{name}_pred_regions"] for r in rows)
        out[f"{name}_regions_found"] = f"{int(found.sum())}/{int(total.sum())}"
        out[f"{name}_recall_cov"] = found.sum() / total.sum() if total.sum() else None
        out[f"{name}_precision_cov"] = pc / pt if pt else None
        out[f"{name}_pred_regions_correct"] = f"{pc}/{pt}"
        if total.sum() and n_boot:
            idx = rng.integers(0, len(rows), size=(n_boot, len(rows)))
            f, t = found[idx].sum(1), total[idx].sum(1)
            vals = f[t > 0] / t[t > 0]
            out[f"{name}_recall_cov_ci95"] = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--gt", required=True, type=Path, help="Folder of GT PNG masks (gt_manual\\masks)")
    ap.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--yolo", action="append", default=[], help="NAME=PATH (repeatable)")
    ap.add_argument("--mask-dir", action="append", default=[],
                    help="NAME=DIR of semantic PNGs (same ids as the GT), e.g. SAM pseudo-masks: scores the LABELER itself")
    ap.add_argument("--box-labels", action="append", default=[],
                    help="NAME=DIR of YOLO .txt labels, boxes filled as masks (the 'rectangles' labeler baseline)")
    ap.add_argument("--yolo-conf", type=float, default=0.25)
    ap.add_argument("--sdzi-threshold", type=float, default=None, help="P(SDZI) >= t instead of argmax (choose on val!)")
    ap.add_argument("--min-area", type=float, default=0.0005, help="Min region size, fraction of the image")
    ap.add_argument("--cover", type=float, default=0.3, help="Coverage needed to count a region as found/correct")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--save-vis", action="store_true",
                    help="Also write <out>/vis/<frame>.jpg: image | GT | one panel per model (river cyan, SDZI magenta)")
    ap.add_argument("--vis-width", type=int, default=640)
    args = ap.parse_args()

    import torch

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    frames = sorted(p for p in args.images.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS and (args.gt / f"{p.stem}.png").exists())
    if not frames:
        raise SystemExit(f"No image in {args.images} has a mask in {args.gt}")
    print(f"{len(frames)} annotated frames, device={device}")
    args.out.mkdir(parents=True, exist_ok=True)

    predictors = {}
    if args.checkpoint:
        from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT
        from panoptic_mining.inference import load_model, predict

        river_i = DEFAULT_CLASS_SPLIT.stuff_index["river"]
        road_i = DEFAULT_CLASS_SPLIT.stuff_index["road"]
        for item in args.checkpoint:
            name, _, path = item.partition("=")
            model = load_model(path, device=device)

            def run(img_path, model=model):
                p = predict(model, img_path, device=device)
                road = (p.stuff_argmax == road_i)
                return {"river": p.stuff_argmax == river_i, "road": road}
            predictors[name] = run
    for item in args.yolo:
        from ultralytics import YOLO

        name, _, path = item.partition("=")
        yolo = YOLO(path)

        def run(img_path, yolo=yolo):
            r = yolo.predict(str(img_path), conf=args.yolo_conf, device=0 if device == "cuda" else "cpu", verbose=False)[0]
            h, w = r.orig_shape
            masks = {k: np.zeros((h, w), bool) for k in GT_IDS}
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.cls.cpu().numpy().astype(int)):
                for k, gid in GT_IDS.items():
                    if c == gid:
                        masks[k][max(0, y1):y2, max(0, x1):x2] = True
            return masks
        predictors[name] = run
    for item in args.mask_dir:
        name, _, folder = item.partition("=")

        def run(img_path, folder=Path(folder)):
            m = cv2.imread(str(folder / f"{img_path.stem}.png"), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise SystemExit(f"Missing mask {folder / (img_path.stem + '.png')}")
            return {k: m == gid for k, gid in GT_IDS.items()}
        predictors[name] = run
    for item in args.box_labels:
        name, _, folder = item.partition("=")

        def run(img_path, folder=Path(folder)):
            h, w = cv2.imread(str(img_path)).shape[:2]
            masks = {k: np.zeros((h, w), bool) for k in GT_IDS}
            lp = folder / f"{img_path.stem}.txt"
            for line in (lp.read_text().splitlines() if lp.exists() else []):
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
                x1, y1, x2, y2 = int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)
                for k, gid in GT_IDS.items():
                    if c == gid:
                        masks[k][max(0, y1):y2, max(0, x1):x2] = True
            return masks
        predictors[name] = run
    if not predictors:
        raise SystemExit("Give at least one --checkpoint, --yolo, --mask-dir or --box-labels")

    all_rows, summary = [], {}
    vis = {}  # frame -> list of (title, small label map); 0 none, 3 river, 4 SDZI, 255 ignore
    for name, run in predictors.items():
        rows = []
        for k, img_path in enumerate(frames, 1):
            gt = cv2.imread(str(args.gt / f"{img_path.stem}.png"), cv2.IMREAD_UNCHANGED)
            pred = run(img_path)
            if pred["road"].shape != gt.shape:
                pred = {key: cv2.resize(v.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                        for key, v in pred.items()}
            row = {"model": name, "frame": img_path.stem, **score_frame(pred, gt, args.min_area, args.cover)}
            if args.save_vis:
                vw, vh = args.vis_width, int(round(gt.shape[0] * args.vis_width / gt.shape[1]))
                lab = np.zeros(gt.shape, np.uint8)
                lab[pred["river"]] = 3
                lab[pred["road"]] = 2
                if img_path.stem not in vis:
                    g = np.where(np.isin(gt, (2, 3, IGNORE)), gt, 0).astype(np.uint8)
                    vis[img_path.stem] = [("ground truth (manual)", cv2.resize(g, (vw, vh), interpolation=cv2.INTER_NEAREST))]
                vis[img_path.stem].append((name, cv2.resize(lab, (vw, vh), interpolation=cv2.INTER_NEAREST)))
            rows.append(row)
            if k % 10 == 0:
                print(f"  [{name}] {k}/{len(frames)}", flush=True)
        all_rows += rows
        summary[name] = summarize(rows, args.bootstrap, args.seed)

    if args.save_vis:
        colors = {3: (255, 255, 0), 2: (0, 255, 255), IGNORE: (128, 128, 128)}  # BGR: river cyan, road yellow
        (args.out / "vis").mkdir(exist_ok=True)
        for img_path in frames:
            panels_in = vis[img_path.stem]
            h, w = panels_in[0][1].shape
            base = cv2.resize(cv2.imread(str(img_path)), (w, h))
            panels = []
            for title, lab in [("image", None)] + panels_in:
                p = base.copy()
                if lab is not None:
                    over = p.copy()
                    for k, c in colors.items():
                        over[lab == k] = c
                    p = cv2.addWeighted(p, 0.45, over, 0.55, 0)
                    for k, c in colors.items():
                        if k != IGNORE:
                            cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(p, cnts, -1, c, 2)
                band = np.zeros((34, w, 3), np.uint8)
                cv2.putText(band, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                panels.append(np.vstack([band, p]))
            cols = 3
            while len(panels) % cols:
                panels.append(np.zeros_like(panels[0]))
            rows_img = [np.hstack(panels[i:i + cols]) for i in range(0, len(panels), cols)]
            cv2.imwrite(str(args.out / "vis" / f"{img_path.stem}.jpg"), np.vstack(rows_img), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"Visualizations: {args.out / 'vis'} (river = cyan, SDZI = magenta, grey = ignored)")

    with open(args.out / "per_frame.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    meta = {"min_area": args.min_area, "cover": args.cover, "sdzi_threshold": args.sdzi_threshold,
            "yolo_conf": args.yolo_conf, "frames": [p.stem for p in frames]}
    (args.out / "summary.json").write_text(json.dumps({"settings": meta, "models": summary}, indent=2))

    def fmt(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    print(f"\n{len(frames)} frames ({summary[next(iter(summary))]['frames_with_gt_sdzi']} with GT SDZI); "
          f"region = coverage >= {args.cover}, min area {args.min_area}")
    hdr = f"{'model':<14}{'class':<7}{'IoU':>7}{'pixP':>7}{'pixR':>7}{'regions found':>15}{'R_cov':>7}{'  95% CI':<16}{'P_cov':>7}"
    print(hdr)
    for name, s in summary.items():
        for cls in GT_IDS:
            ci = s.get(f"{cls}_recall_cov_ci95")
            ci_txt = f"  [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "  -"
            print(f"{name:<14}{cls:<7}{fmt(s[f'{cls}_iou']):>7}{fmt(s[f'{cls}_pixel_precision']):>7}"
                  f"{fmt(s[f'{cls}_pixel_recall']):>7}{s[f'{cls}_regions_found']:>15}{fmt(s[f'{cls}_recall_cov']):>7}"
                  f"{ci_txt:<16}{fmt(s[f'{cls}_precision_cov']):>7}")
        print(f"{'':<14}SDZI false-alarm frames: {s['sdzi_false_alarm_frames']}/{s['frames'] - s['frames_with_gt_sdzi']}")
    print(f"\n-> {args.out / 'summary.json'} and per_frame.csv")


if __name__ == "__main__":
    main()

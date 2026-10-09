"""SAM masks from the existing YOLO boxes (box = prompt), for visual review.

These are PSEUDO-LABELS, not ground truth: SAM is told "segment what is
inside this box" and returns its best guess. The point is to compare them
with the full-rectangle pseudo-masks the model is trained on today.

For every image and every label set given with --labels name=folder it:
  - runs SAM once per image with all the boxes of that label set as prompts
  - draws a panel: box outline (thin) + SAM mask (filled, class color)
  - writes one row per box to sam_boxes.csv: class, box area, mask area,
    fill = mask / box (1.0 = SAM filled the whole rectangle, low values = the
    rectangle was mostly something else, e.g. vegetation around a road)

Usage (repo root, Windows; ultralytics downloads the SAM weights the first time):
    .venv\\Scripts\\python scripts\\sam_from_boxes.py ^
        --images "C:\\...\\datos_2x2\\etiquetas\\train\\images" ^
        --labels etiquetas="C:\\...\\datos_2x2\\etiquetas\\train\\labels" ^
        --labels abril="C:\\...\\datos_2x2\\abril\\train\\labels" ^
        --out "C:\\...\\Proyecto de grado\\sam_demo" --n 15 --model sam_b.pt
Models: mobile_sam.pt (fast), sam_b.pt (better, still OK on CPU), sam2_t.pt / sam2_b.pt.
On a machine with an NVIDIA GPU pass --device 0.

--save-masks DIR also writes one semantic PNG per image (training pseudo-labels,
same format as annotate_semantic.py): pixel = YOLO class id where SAM put a mask,
5 = background everywhere else (including the part of a box SAM left out),
255 never used. Built from ONE label set (--mask-from, default: the first
--labels). Overlaps are resolved by a fixed priority, not by file order:
river < road < building < vehicle < SDZI (SDZI painted last, so it wins over river).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from draw_yolo_labels import CLASS_COLORS, CLASS_NAMES, read_yolo_labels  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
BACKGROUND_ID = 5
# paint order for --save-masks (higher = painted later = wins): river 3, road 2, building 1, vehicle 0, SDZI 4
PAINT_PRIORITY = {3: 0, 2: 1, 1: 2, 0: 3, 4: 4}


def panel(image, boxes, masks, title):
    h, w = image.shape[:2]
    overlay = image.copy()
    for (cls, *_), m in zip(boxes, masks):
        overlay[m] = CLASS_COLORS.get(cls, (255, 255, 255))
    out = cv2.addWeighted(image, 0.45, overlay, 0.55, 0)
    t = max(2, round(min(h, w) / 400))
    for i, (cls, xc, yc, bw, bh) in enumerate(boxes, 1):
        x1, y1, x2, y2 = int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)
        cv2.putText(out, f"#{i} {CLASS_NAMES.get(cls, cls)}", (x1 + 3, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), t + 3)
        cv2.putText(out, f"#{i} {CLASS_NAMES.get(cls, cls)}", (x1 + 3, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, t)
    band = np.zeros((60, w, 3), dtype=out.dtype)
    cv2.putText(band, title, (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return np.vstack([band, out])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels", action="append", required=True, help="name=folder (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="sam_b.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=0, help="Random sample of N images (0 = all).")
    ap.add_argument("--only-class", type=int, nargs="*", default=None, help="Only frames with these classes in some label set (e.g. 4).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel-width", type=int, default=960)
    ap.add_argument("--save-masks", type=Path, default=None, help="Folder for semantic PNG pseudo-masks.")
    ap.add_argument("--mask-from", default=None, help="Label set used for --save-masks (default: first --labels).")
    ap.add_argument("--no-panels", action="store_true", help="Skip the review JPGs (faster for full runs).")
    args = ap.parse_args()

    from ultralytics import SAM  # imported here so --help works without ultralytics

    label_sets = {}
    for item in args.labels:
        name, _, folder = item.partition("=")
        label_sets[name] = Path(folder)
    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if args.only_class is not None:
        def has(p):
            return any(b[0] in args.only_class for f in label_sets.values() for b in (read_yolo_labels(f / f"{p.stem}.txt") or []))
        images = [p for p in images if has(p)]
    if args.n:
        random.Random(args.seed).shuffle(images)
        images = sorted(images[: args.n])
    args.out.mkdir(parents=True, exist_ok=True)
    mask_from = args.mask_from or next(iter(label_sets))
    if args.save_masks is not None:
        args.save_masks.mkdir(parents=True, exist_ok=True)

    model = SAM(args.model)
    rows = []
    t0 = time.monotonic()
    for k, img_path in enumerate(images, 1):
        image = cv2.imread(str(img_path))
        h, w = image.shape[:2]
        panels = []
        for name, folder in label_sets.items():
            boxes = read_yolo_labels(folder / f"{img_path.stem}.txt") or []
            masks = []
            if boxes:
                xyxy = [[(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h] for _, xc, yc, bw, bh in boxes]
                res = model(image, bboxes=xyxy, device=args.device, verbose=False)[0]
                if res.masks is not None:
                    data = res.masks.data.cpu().numpy() > 0.5
                    masks = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool) for m in data]
                masks += [np.zeros((h, w), bool)] * (len(boxes) - len(masks))
            for i, ((cls, xc, yc, bw, bh), m) in enumerate(zip(boxes, masks), 1):
                box_area = bw * bh
                mask_area = m.sum() / (h * w)
                rows.append({"frame": img_path.stem, "labels": name, "box": i, "class": CLASS_NAMES.get(cls, cls),
                             "box_area_pct": round(100 * box_area, 2), "mask_area_pct": round(100 * mask_area, 2),
                             "fill": round(mask_area / box_area, 3) if box_area else None})
            if args.save_masks is not None and name == mask_from:
                sem = np.full((h, w), BACKGROUND_ID, dtype=np.uint8)
                for (cls, *_), m in sorted(zip(boxes, masks), key=lambda bm: PAINT_PRIORITY.get(bm[0][0], 0)):
                    sem[m] = cls
                cv2.imwrite(str(args.save_masks / f"{img_path.stem}.png"), sem)
            if not args.no_panels:
                p = panel(image, boxes, masks, f"{img_path.stem} | {name}: box + SAM ({args.model})")
                panels.append(cv2.resize(p, (args.panel_width, int(p.shape[0] * args.panel_width / p.shape[1]))))
        if panels:
            sep = np.full((panels[0].shape[0], 6, 3), 255, dtype=panels[0].dtype)
            row_img = panels[0]
            for p in panels[1:]:
                row_img = np.hstack([row_img, sep, p])
            cv2.imwrite(str(args.out / f"{img_path.stem}.jpg"), row_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[{k}/{len(images)}] {img_path.stem} ({time.monotonic() - t0:.0f}s)", flush=True)

    with open(args.out / "sam_boxes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\nMedian fill (SAM mask area / box area) per label set and class:")
    for name in label_sets:
        for cls in CLASS_NAMES.values():
            vals = [r["fill"] for r in rows if r["labels"] == name and r["class"] == cls and r["fill"] is not None]
            if vals:
                print(f"  {name:<10} {cls:<9} n={len(vals):<4} median fill={np.median(vals):.2f}")
    print(f"\n{len(images)} images in {time.monotonic() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()

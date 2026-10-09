"""Draw YOLO-format box labels on top of their images for visual review.

Usage (Windows, from the repo root):
    py scripts/draw_yolo_labels.py --images "C:\\...\\Adendo" --out "C:\\...\\revision_adendo"

If --labels is omitted, each image's label is looked up next to it
(same folder, same stem, .txt) — that's how Adendo is laid out. For a
YOLO dataset with separate images/ and labels/ folders, pass --labels.

Each box is drawn with its class color and a tag "#<n> <class>", where
<n> is the line number in the .txt (starting at 1), so a specific box
can be referred to unambiguously ("frame X, box #3 is wrong").
A black header strip above the image shows the file name and per-class counts.
Images with no label file, or an empty one, are still exported and
marked as such so background frames can be reviewed too.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Class order used by Etiquetas / Adendo / the yolov11 checkpoint.
CLASS_NAMES = {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI"}
# BGR colors, chosen to be distinguishable on green/grey aerial footage.
CLASS_COLORS = {
    0: (0, 0, 255),      # vehicle  - red
    1: (255, 128, 0),    # building - blue
    2: (0, 255, 255),    # road     - yellow
    3: (255, 255, 0),    # river    - cyan
    4: (255, 0, 255),    # SDZI     - magenta
}
UNKNOWN_COLOR = (255, 255, 255)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]] | None:
    """Return [(class_id, xc, yc, w, h), ...] or None if the file doesn't exist.
    Malformed lines are reported and skipped."""
    if not label_path.is_file():
        return None
    boxes = []
    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 5:
            print(f"  WARNING {label_path.name} line {line_no}: malformed -> {line!r}")
            continue
        boxes.append((int(float(parts[0])), *map(float, parts[1:5])))
    return boxes


def draw(image: np.ndarray, boxes: list[tuple[int, float, float, float, float]] | None) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    thickness = max(2, round(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 1100)

    if boxes is None:
        _banner(out, "NO LABEL FILE", (0, 0, 255), font_scale, thickness)
        return out
    if not boxes:
        _banner(out, "EMPTY LABEL (background)", (0, 200, 0), font_scale, thickness)
        return out

    for idx, (cls, xc, yc, bw, bh) in enumerate(boxes, start=1):
        color = CLASS_COLORS.get(cls, UNKNOWN_COLOR)
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        tag = f"#{idx} {CLASS_NAMES.get(cls, f'class {cls}?')}"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = y1 - 4 if y1 - th - 6 > 0 else y1 + th + 6
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(out, tag, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    return out


def add_header(img: np.ndarray, title: str, boxes) -> np.ndarray:
    """Black strip above the image with the file name and per-class counts,
    so the legend never covers the HUD or the scene."""
    h, w = img.shape[:2]
    font_scale = max(0.5, min(h, w) / 1100)
    thickness = max(1, round(min(h, w) / 500))
    band = int(45 * font_scale * 1.6)
    header = np.zeros((band, w, 3), dtype=img.dtype)
    cv2.putText(header, title, (10, int(band * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    x = 10 + cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] + 40
    counts = Counter(b[0] for b in boxes) if boxes else Counter()
    for cls, n in sorted(counts.items()):
        text = f"{CLASS_NAMES.get(cls, f'class {cls}?')}: {n}"
        cv2.putText(header, text, (x, int(band * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, CLASS_COLORS.get(cls, UNKNOWN_COLOR), thickness)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] + 30
    return np.vstack([header, img])


def _banner(img: np.ndarray, text: str, color: tuple[int, int, int], font_scale: float, thickness: int) -> None:
    cv2.putText(img, text, (10, int(40 * font_scale * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.3, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, (10, int(40 * font_scale * 1.5)), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.3, color, thickness)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path, help="Folder with the images.")
    parser.add_argument("--labels", type=Path, default=None, help="Folder with the .txt labels (default: same as --images).")
    parser.add_argument("--out", required=True, type=Path, help="Output folder for the annotated images.")
    parser.add_argument("--max-width", type=int, default=1920, help="Downscale outputs wider than this (0 = keep size).")
    parser.add_argument("--only-class", type=int, nargs="*", default=None,
                        help="Only export images containing at least one box of these class ids (e.g. --only-class 4 for SDZI).")
    args = parser.parse_args()

    labels_dir = args.labels or args.images
    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    total = Counter()
    exported = 0
    no_label, empty = [], []
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  WARNING could not read {img_path.name}")
            continue
        boxes = read_yolo_labels(labels_dir / f"{img_path.stem}.txt")
        if args.only_class is not None and not (boxes and any(b[0] in args.only_class for b in boxes)):
            continue
        if boxes is None:
            no_label.append(img_path.name)
        elif not boxes:
            empty.append(img_path.name)
        else:
            total.update(b[0] for b in boxes)
        annotated = add_header(draw(image, boxes), img_path.name, boxes)
        if args.max_width and annotated.shape[1] > args.max_width:
            scale = args.max_width / annotated.shape[1]
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(args.out / f"{img_path.stem}.jpg"), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        exported += 1

    print(f"\nImages: {len(images)} found, {exported} exported -> {args.out}")
    print("Boxes per class: " + ", ".join(f"{CLASS_NAMES.get(c, c)}={n}" for c, n in sorted(total.items())))
    print(f"No label file: {len(no_label)} {no_label[:10]}")
    print(f"Empty label (background): {len(empty)} {empty[:10]}")


if __name__ == "__main__":
    main()

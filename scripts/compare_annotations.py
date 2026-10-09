"""Compare two YOLO annotation versions of the same frames, side by side.

Built to compare `Etiquetas` (original annotation, paper) against the
April 2026 re-annotation found in `dataset_split_completo`, but works for
any two label sets that share file stems.

Usage (Windows, from the repo root):
    .venv\\Scripts\\python scripts\\compare_annotations.py ^
        --images "C:\\...\\OneDrive_1_7-10-2026\\Imagenes" ^
        --labels-a "C:\\...\\OneDrive_1_7-10-2026\\Etiquetas" --name-a Etiquetas ^
        --labels-b "C:\\...\\OneDrive_1_7-10-2026\\dataset_split_completo" --name-b Abril ^
        --out "C:\\...\\Proyecto de grado\\comparacion_etiquetas" --n 30

What it does:
  1. Indexes every .txt under --labels-a and --labels-b recursively (any
     split/subfolder layout) and every image under --images, by file stem.
  2. For every stem present in both label sets (and with an image), counts
     boxes per class in each version and matches boxes across versions
     (same class, IoU >= --iou, greedy). Writes one row per frame to
     summary.csv and prints totals.
  3. Draws A | B side by side for a sample of --n frames (random with a
     fixed --seed, only frames where the two versions differ unless
     --include-identical). Use --all to draw every common frame.

If a stem has several label files inside the same set (e.g. Etiquetas has
the same frame in train and test with different boxes), each copy is
compared separately and the CSV says which subfolder it came from.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from draw_yolo_labels import CLASS_NAMES, add_header, draw, read_yolo_labels

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def index_files(root: Path, extensions: set[str]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            found[p.stem].append(p)
    return found


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1, bx2, by2 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0.0, min(ax2, bx2) - max(ax1, bx1)), max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def match_boxes(boxes_a, boxes_b, thr: float) -> int:
    """Greedy same-class matching; returns number of matched pairs."""
    return len(match_pairs(boxes_a, boxes_b, thr))


def match_pairs(boxes_a, boxes_b, thr: float) -> list[tuple[int, int]]:
    """Greedy same-class matching; returns matched (index_a, index_b) pairs."""
    pairs = sorted(
        ((iou(a, b), i, j) for i, a in enumerate(boxes_a) for j, b in enumerate(boxes_b) if a[0] == b[0]),
        reverse=True,
    )
    used_a, used_b, matched = set(), set(), []
    for score, i, j in pairs:
        if score < thr:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j))
    return matched


def side_by_side(image, boxes_a, boxes_b, title_a, title_b, width: int):
    panels = []
    for boxes, title in ((boxes_a, title_a), (boxes_b, title_b)):
        panel = add_header(draw(image, boxes), title, boxes)
        panels.append(cv2.resize(panel, (width, int(panel.shape[0] * width / panel.shape[1]))))
    h = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.zeros((h - p.shape[0], width, 3), p.dtype)]) for p in panels]
    sep = np.full((h, 6, 3), 255, dtype=panels[0].dtype)
    return np.hstack([panels[0], sep, panels[1]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--labels-a", required=True, type=Path)
    ap.add_argument("--labels-b", required=True, type=Path)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=30, help="How many frames to draw.")
    ap.add_argument("--all", action="store_true", help="Draw every common frame.")
    ap.add_argument("--include-identical", action="store_true", help="Also sample frames where both versions match.")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--panel-width", type=int, default=960)
    ap.add_argument("--blind", action="store_true",
                    help="Blind review: random left/right, no version names; writes clave.csv (key) and revision.csv (to fill in).")
    ap.add_argument("--stratify", action="store_true",
                    help="Sample the same number of frames per class (frames containing that class in either version).")
    args = ap.parse_args()

    images = index_files(args.images, IMAGE_EXTENSIONS)
    labels_a = index_files(args.labels_a, {".txt"})
    labels_b = index_files(args.labels_b, {".txt"})
    common = sorted(set(labels_a) & set(labels_b) & set(images))
    print(f"Label files: {args.name_a}={sum(map(len, labels_a.values()))}, {args.name_b}={sum(map(len, labels_b.values()))}")
    print(f"Frames with both annotations and an image: {len(common)}")
    if not common:
        raise SystemExit("Nothing to compare — check the paths.")

    args.out.mkdir(parents=True, exist_ok=True)
    rows, totals_a, totals_b = [], Counter(), Counter()
    cross = {c: Counter() for c in CLASS_NAMES}  # per class: matched / only_a / only_b
    for stem in common:
        boxes_b_path = labels_b[stem][0]
        boxes_b = read_yolo_labels(boxes_b_path) or []
        for path_a in labels_a[stem]:
            boxes_a = read_yolo_labels(path_a) or []
            ca, cb = Counter(b[0] for b in boxes_a), Counter(b[0] for b in boxes_b)
            totals_a.update(ca)
            totals_b.update(cb)
            pairs = match_pairs(boxes_a, boxes_b, args.iou)
            matched = len(pairs)
            matched_by_class = Counter(boxes_a[i][0] for i, _ in pairs)
            identical = matched == len(boxes_a) == len(boxes_b)
            row = {
                "frame": stem,
                f"{args.name_a}_folder": path_a.parent.relative_to(args.labels_a).as_posix(),
                f"{args.name_b}_folder": boxes_b_path.parent.relative_to(args.labels_b).as_posix(),
                f"{args.name_a}_boxes": len(boxes_a),
                f"{args.name_b}_boxes": len(boxes_b),
                "matched_boxes": matched,
                "identical": identical,
            }
            for c, name in CLASS_NAMES.items():
                m = matched_by_class.get(c, 0)
                row[f"{args.name_a}_{name}"] = ca.get(c, 0)
                row[f"{args.name_b}_{name}"] = cb.get(c, 0)
                row[f"{name}_matched"] = m
                row[f"{name}_only_{args.name_a}"] = ca.get(c, 0) - m
                row[f"{name}_only_{args.name_b}"] = cb.get(c, 0) - m
                cross[c]["matched"] += m
                cross[c]["only_a"] += ca.get(c, 0) - m
                cross[c]["only_b"] += cb.get(c, 0) - m
            row["_a"], row["_b"], row["_img"] = boxes_a, boxes_b, images[stem][0]
            rows.append(row)

    csv_cols = [k for k in rows[0] if not k.startswith("_")]
    with open(args.out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows({k: r[k] for k in csv_cols} for r in rows)

    n_ident = sum(r["identical"] for r in rows)
    more_a = sum(r[f"{args.name_a}_boxes"] > r[f"{args.name_b}_boxes"] for r in rows)
    more_b = sum(r[f"{args.name_b}_boxes"] > r[f"{args.name_a}_boxes"] for r in rows)
    print(f"\nCompared pairs: {len(rows)}  identical: {n_ident}  more boxes in {args.name_a}: {more_a}  "
          f"more boxes in {args.name_b}: {more_b}")
    print(f"{'class':<10}{args.name_a:>12}{args.name_b:>12}")
    for c, name in CLASS_NAMES.items():
        print(f"{name:<10}{totals_a.get(c, 0):>12}{totals_b.get(c, 0):>12}")

    print(f"\nBox cross-match per class (same class, IoU >= {args.iou}):")
    print(f"{'class':<10}{'matched':>9}{'only ' + args.name_a:>18}{'only ' + args.name_b:>18}{'% ' + args.name_a + ' kept':>18}")
    cross_rows = []
    for c, name in CLASS_NAMES.items():
        k = cross[c]
        total_a = k["matched"] + k["only_a"]
        kept = f"{100 * k['matched'] / total_a:.1f}%" if total_a else "-"
        print(f"{name:<10}{k['matched']:>9}{k['only_a']:>18}{k['only_b']:>18}{kept:>18}")
        cross_rows.append({"class": name, "matched": k["matched"], f"only_{args.name_a}": k["only_a"], f"only_{args.name_b}": k["only_b"]})
    with open(args.out / "cruce_por_clase.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cross_rows[0]))
        writer.writeheader()
        writer.writerows(cross_rows)

    rng = random.Random(args.seed)
    pool = rows if (args.all or args.include_identical) else [r for r in rows if not r["identical"]]
    if args.stratify and not args.all:
        per_class = max(1, args.n // len(CLASS_NAMES))
        chosen, seen = [], set()
        for c, name in CLASS_NAMES.items():
            cands = [r for r in pool if (r[f"{args.name_a}_{name}"] + r[f"{args.name_b}_{name}"]) > 0 and r["frame"] not in seen]
            rng.shuffle(cands)
            for r in cands[:per_class]:
                chosen.append(r)
                seen.add(r["frame"])
        rng.shuffle(chosen)
        pool = chosen
    elif not args.all:
        rng.shuffle(pool)
        pool = pool[: args.n]

    key_rows = []
    for idx, r in enumerate(pool, 1):
        image = cv2.imread(str(r["_img"]))
        if image is None:
            continue
        if args.blind:
            a_left = rng.random() < 0.5
            left, right = (r["_a"], r["_b"]) if a_left else (r["_b"], r["_a"])
            out = side_by_side(image, left, right, f"par {idx:03d} | Version 1", "Version 2", args.panel_width)
            fname = f"par_{idx:03d}.jpg"
            cv2.imwrite(str(args.out / fname), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
            key_rows.append({"par": f"{idx:03d}", "archivo": fname, "frame": r["frame"],
                             "version_1": args.name_a if a_left else args.name_b,
                             "version_2": args.name_b if a_left else args.name_a})
            continue
        image = cv2.imread(str(r["_img"]))
        if image is None:
            continue
        title_a = f"{args.name_a} ({r[f'{args.name_a}_folder']})"
        title_b = f"{args.name_b} ({r[f'{args.name_b}_folder']})"
        out = side_by_side(image, r["_a"], r["_b"], f"{r['frame']} | {title_a}", title_b, args.panel_width)
        suffix = r[f"{args.name_a}_folder"].replace("/", "-")
        cv2.imwrite(str(args.out / f"{r['frame']}__{suffix}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if args.blind and key_rows:
        with open(args.out / "clave.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
            writer.writeheader()
            writer.writerows(key_rows)
        with open(args.out / "revision.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["par", "archivo", "mejor (1 / 2 / igual)", "errores version 1", "errores version 2", "comentario"])
            for k in key_rows:
                writer.writerow([k["par"], k["archivo"], "", "", "", ""])
        print("Blind mode: open the par_XXX.jpg files, fill revision.csv. Do NOT open clave.csv until done.")
    print(f"\nDrew {len(pool)} comparisons + summary.csv + cruce_por_clase.csv -> {args.out}")


if __name__ == "__main__":
    main()

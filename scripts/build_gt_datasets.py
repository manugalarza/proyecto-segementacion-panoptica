"""Build the two datasets for the 2x2 ground-truth experiment.

Same images, same train/val/test split, two label versions:
  <out>/etiquetas/   labels from Etiquetas (original annotation, paper)
  <out>/abril/       labels from the April 2026 re-annotation (dataset_split_completo)

Only frames present in BOTH annotations are used (video frames
video_11min_* / video_13min_*), so the only thing that changes between the
two datasets is the annotation. Adendo and the scene frames are left out
of this experiment on purpose.

Etiquetas duplicates: 198 frames appear in two splits of Imagenes/Etiquetas
with identical images but sometimes different labels. One copy is kept:
the one with more boxes (tie -> the first one found). The choice is logged.

Split: by temporal blocks, not by frame. Each video is cut into blocks of
--block-size consecutive frame indices; whole blocks go to one split, so
neighboring frames don't end up on both sides. --gap frames at each edge
of every val/test block are dropped to separate them from adjacent train
blocks. Blocks are assigned so that both the frame count AND the SDZI count
(union of both annotations) are close to the target ratios — the original
val split had zero SDZI, which made it useless for picking thresholds.

Usage (from the repo root, Windows):
    .venv\\Scripts\\python scripts\\build_gt_datasets.py ^
        --imagenes  "C:\\...\\OneDrive_1_7-10-2026\\Imagenes" ^
        --etiquetas "C:\\...\\OneDrive_1_7-10-2026\\Etiquetas" ^
        --abril     "C:\\...\\OneDrive_1_7-10-2026\\dataset_split_completo" ^
        --out       "C:\\...\\Proyecto de grado\\datos_2x2"
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = {0: "vehicle", 1: "building", 2: "road", 3: "river", 4: "SDZI"}
SDZI = 4
VIDEO_RE = re.compile(r"^(video_1[13]min)_(\d+)$")


def read_label_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def count_class(lines: list[str], cls: int) -> int:
    return sum(1 for line in lines if int(float(line.split()[0])) == cls)


def index(root: Path, ext: str) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(root.rglob(f"*{ext}")):
        out[p.stem].append(p)
    return out


def assign_blocks(blocks: dict, ratios: dict[str, float], seed: int) -> dict:
    """Greedy: blocks with SDZI first (largest first), each to the split
    furthest below its SDZI target; then the rest, each to the split
    furthest below its frame target."""
    rng = random.Random(seed)
    keys = list(blocks)
    rng.shuffle(keys)
    total_frames = sum(len(b["frames"]) for b in blocks.values())
    total_sdzi = sum(b["sdzi"] for b in blocks.values())
    frames = {s: 0 for s in ratios}
    sdzi = {s: 0 for s in ratios}
    assignment = {}

    with_sdzi = sorted((k for k in keys if blocks[k]["sdzi"] > 0), key=lambda k: -blocks[k]["sdzi"])
    without = [k for k in keys if blocks[k]["sdzi"] == 0]
    for k in with_sdzi:
        split = min(ratios, key=lambda s: (sdzi[s] / max(total_sdzi, 1)) / ratios[s])
        assignment[k] = split
        frames[split] += len(blocks[k]["frames"])
        sdzi[split] += blocks[k]["sdzi"]
    for k in without:
        split = min(ratios, key=lambda s: (frames[s] / total_frames) / ratios[s])
        assignment[k] = split
        frames[split] += len(blocks[k]["frames"])
    return assignment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--imagenes", required=True, type=Path)
    ap.add_argument("--etiquetas", required=True, type=Path)
    ap.add_argument("--abril", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--block-size", type=int, default=20)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images = index(args.imagenes, ".jpg")
    labels_e = index(args.etiquetas, ".txt")
    labels_a = index(args.abril, ".txt")

    frames = sorted(s for s in images if s in labels_e and s in labels_a and VIDEO_RE.match(s))
    print(f"Video frames with image + Etiquetas + Abril labels: {len(frames)}")

    chosen_e: dict[str, Path] = {}
    dup_log = []
    for s in frames:
        cands = labels_e[s]
        best = max(cands, key=lambda p: len(read_label_lines(p)))
        chosen_e[s] = best
        if len(cands) > 1:
            dup_log.append((s, [f"{p.parent.name}:{len(read_label_lines(p))}" for p in cands], best.parent.name))

    blocks: dict[tuple[str, int], dict] = defaultdict(lambda: {"frames": [], "sdzi": 0})
    for s in frames:
        video, idx = VIDEO_RE.match(s).groups()
        key = (video, int(idx) // args.block_size)
        sdzi = max(count_class(read_label_lines(chosen_e[s]), SDZI), count_class(read_label_lines(labels_a[s][0]), SDZI))
        blocks[key]["frames"].append((int(idx), s))
        blocks[key]["sdzi"] += sdzi

    ratios = dict(zip(("train", "val", "test"), args.ratios))
    assignment = assign_blocks(blocks, ratios, args.seed)

    split_of: dict[str, str] = {}
    dropped = []
    for key, block in blocks.items():
        split = assignment[key]
        members = sorted(block["frames"])
        for pos, (_, s) in enumerate(members):
            at_edge = pos < args.gap or pos >= len(members) - args.gap
            if split != "train" and at_edge and len(members) > 2 * args.gap:
                dropped.append(s)
                continue
            split_of[s] = split

    for variant in ("etiquetas", "abril"):
        for split in ratios:
            (args.out / variant / split / "images").mkdir(parents=True, exist_ok=True)
            (args.out / variant / split / "labels").mkdir(parents=True, exist_ok=True)
        (args.out / variant / "dataset.yaml").write_text(
            "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 5\nnames:\n"
            + "".join(f"  {k}: {v}\n" for k, v in CLASS_NAMES.items())
        )

    rows = []
    totals = {v: {s: defaultdict(int) for s in ratios} for v in ("etiquetas", "abril")}
    for s, split in sorted(split_of.items()):
        img_src = images[s][0]
        lab = {"etiquetas": chosen_e[s], "abril": labels_a[s][0]}
        row = {"frame": s, "split": split, "block": f"{VIDEO_RE.match(s).group(1)}_{int(VIDEO_RE.match(s).group(2)) // args.block_size}"}
        for variant, label_src in lab.items():
            dst = args.out / variant / split
            shutil.copy2(img_src, dst / "images" / img_src.name)
            shutil.copy2(label_src, dst / "labels" / f"{s}.txt")
            lines = read_label_lines(label_src)
            totals[variant][split]["frames"] += 1
            for c, name in CLASS_NAMES.items():
                n = count_class(lines, c)
                totals[variant][split][name] += n
                row[f"{variant}_{name}"] = n
        rows.append(row)

    with open(args.out / "split_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / "duplicados_etiquetas.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "copias (carpeta:cajas)", "elegida"])
        for s, cands, best in dup_log:
            writer.writerow([s, " | ".join(cands), best])

    print(f"Duplicated frames in Etiquetas resolved: {len(dup_log)} (see duplicados_etiquetas.csv)")
    print(f"Frames dropped as gap at val/test block edges: {len(dropped)}")
    header = f"{'':<10}{'split':<7}{'frames':>7}" + "".join(f"{n:>10}" for n in CLASS_NAMES.values())
    print("\n" + header)
    for variant in ("etiquetas", "abril"):
        for split in ratios:
            t = totals[variant][split]
            print(f"{variant:<10}{split:<7}{t['frames']:>7}" + "".join(f"{t[n]:>10}" for n in CLASS_NAMES.values()))
    print(f"\nDatasets written to {args.out} (etiquetas/ and abril/, same images and split)")


if __name__ == "__main__":
    main()

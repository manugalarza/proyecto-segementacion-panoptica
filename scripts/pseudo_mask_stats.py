"""Answers a real question your advisor asked: does the box -> pseudo-mask
conversion (`data/points.py`) actually produce organic, non-rectangular
shapes, or does it just end up giving back the box itself ("quedan
cuadradas")?

The honest qualitative answer (see docs/decisions.md, 2026-09-21): when
GrabCut or the flood-fill fallback succeed, the result IS an irregular
shape, not the box. But there's a real third path, `full_box_fallback`,
where both heuristics fail to cover enough of the box and the function
gives up and returns the box itself — verbatim, no shape refinement at
all. This script measures how often each of the three outcomes
(`grabcut`, `flood_fill`, `full_box_fallback`) actually happens on the
real dataset, per class, so the answer to "do they end up square" has a
real number behind it instead of just "it depends".

Usage (from the repo root, inside the activated venv):

    python scripts/pseudo_mask_stats.py \\
        --data-root "..\\imagenes recortado\\dataset_split_completo\\dataset_split_completo" \\
        --sample-per-class 60 \\
        --output pseudo_mask_stats.json

Samples up to `--sample-per-class` boxes per class (across all splits,
shuffled with `--seed` for reproducibility) rather than running against
every one of the ~7,956 boxes in the dataset — GrabCut runs on the FULL
frame per box (that's how `box_to_pseudo_mask` is written, unchanged
here), so scoring every box would take a long time for not much more
statistical confidence. Prints a per-class breakdown and writes the raw
counts to `--output` as JSON, so the exact numbers can be cited (e.g. in
an email or the thesis document) without re-running this later.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from panoptic_mining.data.manifest import load_yolo_manifest
from panoptic_mining.data.points import box_to_pseudo_mask, sample_interior_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True, type=Path, help="YOLO dataset root (contains dataset.yaml).")
    parser.add_argument("--sample-per-class", type=int, default=60, help="Max boxes to sample per class.")
    parser.add_argument("--output", type=Path, default=Path("pseudo_mask_stats.json"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_yolo_manifest(args.data_root)

    # Collect (frame, box) pairs grouped by class name, across all splits.
    boxes_by_class: dict[str, list[tuple]] = defaultdict(list)
    for frame in manifest.all_frames():
        for box in frame.boxes:
            boxes_by_class[box.class_name].append((frame, box))

    import random

    rng = random.Random(args.seed)
    for class_name in boxes_by_class:
        rng.shuffle(boxes_by_class[class_name])

    results_by_class: dict[str, Counter] = {}
    image_cache: dict[Path, object] = {}
    start = time.monotonic()
    total_scored = 0

    for class_name, pairs in sorted(boxes_by_class.items()):
        sample = pairs[: args.sample_per_class]
        method_counts: Counter = Counter()
        for frame, box in sample:
            if frame.image_path not in image_cache:
                image_cache.clear()  # keep memory bounded — one image at a time is enough here
                image = cv2.imread(str(frame.image_path))
                if image is None:
                    continue
                image_cache[frame.image_path] = image
            image = image_cache[frame.image_path]

            height, width = image.shape[:2]
            box_pixels = box.to_pixels(width, height)
            try:
                points = sample_interior_points(box_pixels)
                result = box_to_pseudo_mask(image, box_pixels, points=points)
            except ValueError:
                continue  # degenerate box after clipping — skip, not counted
            method_counts[result.method] += 1
            total_scored += 1

        results_by_class[class_name] = method_counts
        n = sum(method_counts.values())
        print(f"\n{class_name} (n={n} of {len(pairs)} total boxes, sampled {len(sample)}):")
        for method in ("grabcut", "flood_fill", "full_box_fallback"):
            count = method_counts.get(method, 0)
            pct = 100 * count / n if n else 0.0
            print(f"  {method:20s} {count:4d}  ({pct:5.1f}%)")

    elapsed = time.monotonic() - start
    print(f"\nScored {total_scored} boxes in {elapsed:.1f}s.")

    output = {
        class_name: dict(counts) for class_name, counts in results_by_class.items()
    }
    args.output.write_text(json.dumps(output, indent=2))
    print(f"Raw counts written to {args.output}")

    print(
        "\nInterpretation: 'grabcut' and 'flood_fill' both produce an actual "
        "irregular shape inside the box, not the box itself. Only "
        "'full_box_fallback' means the pseudo-mask IS literally the box "
        "(both heuristics failed to find enough foreground) — that "
        "percentage is the real answer to 'do these end up square'."
    )


if __name__ == "__main__":
    main()

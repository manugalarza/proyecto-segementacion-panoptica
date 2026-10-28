"""Apply the decisions of review_sdzi.py and write a NEW dataset with a
machinery class. The source dataset is never modified.

    maquinaria -> class 5 "machinery"      vehiculo -> class 0 "vehicle"
    sdzi, ambos -> stay class 4 "SDZI"     error    -> box removed

Class ids of the output (dataset.yaml):
    0 vehicle, 1 building, 2 road, 3 river, 4 SDZI, 5 machinery
Machinery boxes drawn during the review ("ambos") are added as class 5.
"ambos" decisions without any drawn box are listed in ambos_frames.csv.
Also prints the box-area distribution per decision: it tells whether a
simple size threshold would have separated machinery from zones.

Usage:
    .venv\\Scripts\\python scripts\\apply_sdzi_review.py --data "...\\datos_2x2\\abril" ^
        --review "...\\revision_sdzi.csv" --out "...\\datos_2x2\\abril_v2"
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np

SDZI_ID, MACHINERY_ID, VEHICLE_ID = 4, 5, 0
NAMES = ["vehicle", "building", "road", "river", "SDZI", "machinery"]
NEW_CLASS = {"maquinaria": MACHINERY_ID, "vehiculo": VEHICLE_ID, "edificio": 1, "sdzi": SDZI_ID, "ambos": SDZI_ID, "error": None}


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--review", required=True, type=Path, help="CSV from review_sdzi.py or JSON from make_review_html.py")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--allow-partial", action="store_true", help="Keep undecided SDZI boxes as SDZI instead of stopping")
    args = ap.parse_args()
    if args.out.resolve() == args.data.resolve():
        raise SystemExit("--out must be a new folder, the source dataset is never overwritten")

    decisions, rows, drawn = {}, [], {}
    if args.review.suffix.lower() == ".json":  # exported by make_review_html.py
        import json
        review_rows = [{k: str(v) for k, v in r.items()} for r in json.loads(args.review.read_text(encoding="utf-8"))]
    else:
        with open(args.review, newline="", encoding="utf-8") as f:
            review_rows = list(csv.DictReader(f))
    if True:
        for r in review_rows:
            decisions[(r["split"], r["frame"], int(r["line"]))] = r["decision"]
            rows.append(r)
            for bx in filter(None, (r.get("machinery_boxes") or "").split("|")):
                drawn.setdefault((r["split"], r["frame"]), []).append(f"{MACHINERY_ID} {bx}")

    undecided, ambos, counts = [], set(), {}
    for split in args.splits:
        src_img, src_lab = args.data / split / "images", args.data / split / "labels"
        if not src_lab.is_dir():
            continue
        dst_img, dst_lab = args.out / split / "images", args.out / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lab.mkdir(parents=True, exist_ok=True)
        for p in src_img.iterdir():
            if not (dst_img / p.name).exists():
                link_or_copy(p, dst_img / p.name)
        for lp in sorted(src_lab.glob("*.txt")):
            out_lines = []
            for i, line in enumerate(lp.read_text().splitlines()):
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                if cls == SDZI_ID:
                    d = decisions.get((split, lp.stem, i))
                    if d is None:
                        undecided.append(f"{split}/{lp.stem}:{i}")
                        d = "sdzi"
                    counts[d] = counts.get(d, 0) + 1
                    if d == "ambos" and (split, lp.stem) not in drawn:
                        ambos.add((split, lp.stem))
                    new = NEW_CLASS[d]
                    if new is None:
                        continue
                    parts[0] = str(new)
                out_lines.append(" ".join(parts))
            out_lines += drawn.get((split, lp.stem), [])
            (dst_lab / lp.name).write_text("\n".join(out_lines) + ("\n" if out_lines else ""))

    if undecided and not args.allow_partial:
        shutil.rmtree(args.out)
        raise SystemExit(f"{len(undecided)} SDZI boxes have no decision (e.g. {undecided[:3]}). "
                         "Finish the review or pass --allow-partial.")

    yaml = "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 6\nnames:\n" + \
        "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES))
    (args.out / "dataset.yaml").write_text(yaml)
    with open(args.out / "ambos_frames.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "frame"])
        w.writerows(sorted(ambos))
    shutil.copy2(args.review, args.out / f"revision_sdzi{args.review.suffix}")

    print(f"Wrote {args.out}")
    print("Decisions:", counts, f"(undecided kept as SDZI: {len(undecided)})" if undecided else "")
    print(f"Machinery boxes drawn during review: {sum(len(v) for v in drawn.values())}")
    print(f"'ambos' without a drawn machinery box: {len(ambos)} frames -> ambos_frames.csv")
    print("\nBox area (% of image) per decision: median [p10, p90]")
    for d in ["maquinaria", "sdzi", "ambos", "vehiculo", "edificio", "error"]:
        a = np.array([float(r["area_pct"]) for r in rows if r["decision"] == d])
        if len(a):
            print(f"  {d:<11} n={len(a):<4} {np.median(a):6.2f}  [{np.percentile(a, 10):.2f}, {np.percentile(a, 90):.2f}]")


if __name__ == "__main__":
    main()

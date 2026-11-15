"""Apply the class-4 review of Lau's dataset (see configs/labels/lau_class4_review_2026-11-12.md).

Writes corrected 6-class label copies; the original label folders are never modified.
Class-4 boxes reviewed as machinery become class 5; everything else is copied unchanged.
"""
import argparse
import csv
import os
from collections import Counter

NAMES = ["vehicle", "building", "road", "river", "SDZI", "machinery"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--labels-in", required=True)
    ap.add_argument("--labels-out", required=True)
    a = ap.parse_args()

    review = {}
    with open(a.review_csv, newline="") as f:
        for r in csv.DictReader(f):
            stem = r["frame"].rsplit(".png", 1)[0]
            review[(stem, int(r["class4_box_index"]))] = r

    os.makedirs(a.labels_out, exist_ok=True)
    counts, changed, used = Counter(), 0, set()
    for name in sorted(os.listdir(a.labels_in)):
        if not name.endswith(".txt"):
            continue
        stem = name[:-4]
        out, k4 = [], 0
        for line in open(os.path.join(a.labels_in, name)):
            p = line.split()
            if not p:
                continue
            if p[0] == "4":
                k4 += 1
                r = review.get((stem, k4))
                if r is None:
                    raise SystemExit(f"class-4 box without review: {stem} #{k4}")
                if [f"{float(x):.6f}" for x in p[1:5]] != [f"{float(r[c]):.6f}" for c in ("xc", "yc", "w", "h")]:
                    raise SystemExit(f"coordinate mismatch: {stem} #{k4}")
                used.add((stem, k4))
                if r["new_class"] == "5":
                    p[0] = "5"
                    changed += 1
            counts[p[0]] += 1
            out.append(" ".join(p))
        with open(os.path.join(a.labels_out, name), "w") as f:
            f.write("\n".join(out) + ("\n" if out else ""))
    missing = set(review) - used
    if missing:
        raise SystemExit(f"{len(missing)} reviewed boxes not found in labels")
    print(f"changed 4->5: {changed}")
    print("boxes per class:", {NAMES[int(c)]: n for c, n in sorted(counts.items())})


if __name__ == "__main__":
    main()

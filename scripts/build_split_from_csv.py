"""Materialize one FLIR partition (one seed) into a YOLO-style dataset.

Inputs
  --partition-csv  expanded-S*-seed*-frames.csv (semicolon-separated; one row per record,
                   with columns nombre_frame, origen, original_split, content_id)
  --assignment     CSV with content_id,split for the partition being built
                   (e.g. historical_S3_seed0_content_split.csv). Only records whose
                   origen matches --origen are used.
  --images-root    folder with <original_split>/<nombre_frame>.jpg  (train|val|test)
  --labels-root    folder with the original YOLO labels (train/, test/, validation/val/)

Outputs (under --out)
  train|val|test/images/*.jpg, train|val|test/labels/*.txt, dataset.yaml, manifest.csv

Rules
  * One output file per record (the partition is balanced by records).
  * Labels come from the record's ORIGINAL split folder, so a frame that exists in two
    original splits with different annotations keeps the label of its own source copy.
  * Orphan labels (no image) are ignored.
  * Class 4 (SDZI) is KEPT as 4 (ignored in training) for frames labeled from Etiquetas.
    Review of 2026-11-10: the 11 SDZI boxes that used to be rewritten to machinery were
    zones (dunes, piles), with at most a small machine inside. Pass --sdzi-to-machinery for
    the old behavior (class 4 rewritten to 5), the same
    rule applied to datos_v2 and the other label sets.
  * --dry-run validates everything and writes the manifest but copies nothing.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict

LABEL_DIR = {"train": "train", "test": "test", "val": os.path.join("validation", "val")}
EXPECTED_HIST_S3_SEED0 = {"train": 1178, "val": 107, "test": 372}  # records
NAMES = ["vehicle", "building", "road", "river", "SDZI", "machinery"]


def read_partition(path: str, origen: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter=";") if r["origen"] == origen]
    return rows


def read_assignment(path: str) -> dict:
    """content_id -> split, plus (content_id, frame stem, original_split) -> split when the CSV has nombre_frame.

    The same content can appear twice with different splits (e.g. Jorge's fixed_test_train_val_v1:
    a test image and its duplicate copy in quarantine). Keyed by content only, the last row would
    win and a test record could silently become quarantine, so the per-frame key is used first and
    the content-only fallback prefers test > val > train > anything else.
    """
    prio = {"test": 0, "val": 1, "train": 2}
    out: dict = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            cid, split = r["content_id"], r["split"]
            if r.get("nombre_frame"):
                stem = os.path.splitext(os.path.basename(r["nombre_frame"]))[0]
                # frame names repeat across original splits (Imagenes/train vs Imagenes/test)
                out[(cid, stem, r.get("original_split", ""))] = split
            if cid not in out or prio.get(split, 9) < prio.get(out[cid], 9):
                out[cid] = split
    return out


CLIPPED: list[str] = []


def read_label(path: str, remap_sdzi: bool) -> tuple[list[str], int, int]:
    """Return (lines, n_boxes, n_sdzi_remapped)."""
    lines, n, remapped = [], 0, 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            parts = raw.split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"{path}: expected 5 fields, got {len(parts)}: {raw!r}")
            cls = int(parts[0])
            if not 0 <= cls <= 5:
                raise ValueError(f"{path}: class {cls} out of range")
            # Rule (2026-11-12, agreed with Jorge): boxes that extend past the image border are
            # clipped to [0, 1]; a few datos_v2 river/road boxes exceed it by up to 1.5% of the width.
            xc, yc, w, h = map(float, parts[1:5])
            x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
            x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
            if (x0, y0, x1, y1) != (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2):
                CLIPPED.append(f"{os.path.basename(path)} class {cls}")
                parts = [parts[0], f"{(x0 + x1) / 2:.6f}", f"{(y0 + y1) / 2:.6f}", f"{x1 - x0:.6f}", f"{y1 - y0:.6f}"]
            if remap_sdzi and cls == 4:
                cls = 5
                remapped += 1
            lines.append(" ".join([str(cls)] + parts[1:]))
            n += 1
    return lines, n, remapped


def index_datos_v2(root: str) -> dict[str, str]:
    """content sha256 -> datos_v2 label path, for every image in datos_v2/{train,val,test}."""
    import glob, hashlib
    idx = {}
    for split in ["train", "val", "test"]:
        for img in glob.glob(os.path.join(root, split, "images", "*")):
            stem = os.path.splitext(os.path.basename(img))[0]
            lab = os.path.join(root, split, "labels", stem + ".txt")
            if not os.path.isfile(lab):
                continue
            with open(img, "rb") as fh:
                sha = hashlib.sha256(fh.read()).hexdigest()
            idx.setdefault(sha, lab)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition-csv", required=True)
    ap.add_argument("--assignment", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--labels-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--origen", default="historical", choices=["historical", "adendo"])
    ap.add_argument("--sdzi-to-machinery", action="store_true",
                    help="old behavior: rewrite class 4 to 5 in Etiquetas labels (default keeps SDZI as 4)")
    ap.add_argument("--keep-sdzi", action="store_true", help="kept for compatibility; SDZI is kept by default")
    ap.add_argument("--unique-content", action="store_true",
                    help="keep one record per content_id (exact duplicate images appear in two original "
                         "splits; their labels are identical). Without it each duplicate is a separate file.")
    ap.add_argument("--extra-labels", default=None,
                    help="CSV (nombre_frame,class,xc,yc,w,h,...) of boxes appended to Etiquetas-labeled frames, "
                         "e.g. configs/labels/etiquetas_machinery_review_2026-11-10.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datos-v2", default=None,
                    help="datos_v2 root: frames found there (by content) take their label from datos_v2; "
                         "frames not found use Etiquetas and their SDZI boxes are assumed to be machinery")
    ap.add_argument("--expect", action="store_true",
                    help="check record counts against historical S3 seed 0 (train 1178, val 107, test 372)")
    args = ap.parse_args()

    rows = read_partition(args.partition_csv, args.origen)
    assign = read_assignment(args.assignment)
    remap = args.sdzi_to_machinery and not args.keep_sdzi
    extra = {}
    if args.extra_labels:
        with open(args.extra_labels, encoding="utf-8-sig", newline="") as fh:
            for e in csv.DictReader(fh):
                extra.setdefault(e["nombre_frame"], []).append(
                    " ".join([e["class"], e["xc"], e["yc"], e["w"], e["h"]]))
        print(f"extra labels: {sum(len(v) for v in extra.values())} boxes for {len(extra)} frames")

    problems: list[str] = []
    records: list[dict] = []
    per_split = Counter()
    sdzi_total = 0
    sources = Counter()
    v2_index = index_datos_v2(args.datos_v2) if args.datos_v2 else {}
    print(f"datos_v2 index: {len(v2_index)} images")
    for idx, r in enumerate(rows):
        cid, name, osplit = r["content_id"], r["nombre_frame"], r["original_split"]
        if cid not in assign:
            problems.append(f"no assignment for content {cid[:12]} ({name})")
            continue
        split = assign.get((cid, name, osplit), assign[cid])
        img = os.path.join(args.images_root, osplit, name + ".jpg")
        lab = os.path.join(args.labels_root, LABEL_DIR[osplit], name + ".txt")
        if not os.path.isfile(img):
            problems.append(f"missing image {img}")
            continue
        if not os.path.isfile(lab):
            problems.append(f"missing label {lab}")
            continue
        if v2_index and cid in v2_index:
            lab_used, source = v2_index[cid], "datos_v2"
            lines, n_boxes, n_sdzi = read_label(lab_used, False)
        else:
            lab_used, source = lab, ("etiquetas_sdzi_asumida_maquinaria" if (v2_index and remap) else "etiquetas")
            lines, n_boxes, n_sdzi = read_label(lab, remap)
            if name in extra:
                lines = lines + extra[name]
                n_boxes += len(extra[name])
        sources[source] += 1
        sdzi_total += n_sdzi
        out_name = f"{name}__r{idx:04d}"
        records.append({
            "out_name": out_name, "split": split, "nombre_frame": name, "content_id": cid,
            "original_split": osplit, "src_image": img, "src_label": lab,
            "n_boxes": n_boxes, "sdzi_remapped": n_sdzi, "lines": lines, "label_source": source,
        })
        per_split[split] += 1

    if args.unique_content:
        # Keep the record of the highest-priority split (test > val > train > others), so a test
        # image is never dropped in favour of its quarantine/train copy.
        prio = {"test": 0, "val": 1, "train": 2}
        seen, kept = set(), []
        for r in sorted(records, key=lambda r: prio.get(r["split"], 9)):
            if r["content_id"] in seen:
                per_split[r["split"]] -= 1
                continue
            seen.add(r["content_id"])
            kept.append(r)
        print(f"--unique-content: {len(records) - len(kept)} duplicate records dropped")
        records = kept

    # Labels that exist without an image (not used, reported for the audit)
    used = {(r["nombre_frame"], r["original_split"]) for r in rows}
    orphans = []
    for osplit, ldir in LABEL_DIR.items():
        base = os.path.join(args.labels_root, ldir)
        if not os.path.isdir(base):
            continue
        for f in os.listdir(base):
            if f.endswith(".txt") and (f[:-4], osplit) not in used:
                orphans.append(os.path.join(ldir, f))

    print(f"records: {len(records)} / partition rows: {len(rows)}")
    print("records per split:", dict(sorted(per_split.items())))
    print(f"boxes clipped to the image border: {len(CLIPPED)}", "; ".join(CLIPPED))
    print(f"class 4 boxes remapped to 5: {sdzi_total} (remap={remap})")
    print("label source per record:", dict(sources))
    print(f"orphan labels ignored: {len(orphans)}")
    if problems:
        print(f"PROBLEMS ({len(problems)}):")
        for p in problems[:20]:
            print("  ", p)
    if args.expect:
        for s, n in EXPECTED_HIST_S3_SEED0.items():
            if per_split[s] != n:
                problems.append(f"split {s}: {per_split[s]} records, expected {n}")
                print(f"COUNT MISMATCH {s}: {per_split[s]} != {n}")

    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "manifest.csv")
    with open(manifest, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["out_name", "split", "nombre_frame", "content_id", "original_split",
                    "n_boxes", "sdzi_remapped", "src_image", "src_label", "label_source"])
        for r in records:
            w.writerow([r["out_name"], r["split"], r["nombre_frame"], r["content_id"],
                        r["original_split"], r["n_boxes"], r["sdzi_remapped"],
                        r["src_image"], r["src_label"], r["label_source"]])
    print("manifest:", manifest)

    if args.dry_run:
        print("dry run: nothing copied")
        return 1 if problems else 0

    for r in records:
        if r["split"] not in ("train", "val", "test"):
            continue  # e.g. excluded_near_dup: listed in manifest, not copied
        d_img = os.path.join(args.out, r["split"], "images")
        d_lab = os.path.join(args.out, r["split"], "labels")
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lab, exist_ok=True)
        shutil.copyfile(r["src_image"], os.path.join(d_img, r["out_name"] + ".jpg"))
        with open(os.path.join(d_lab, r["out_name"] + ".txt"), "w", encoding="utf-8", newline="\n") as fh:
            for line in r["lines"]:
                fh.write(line + "\n")

    with open(os.path.join(args.out, "dataset.yaml"), "w", encoding="utf-8") as fh:
        fh.write("path: .\ntrain: train/images\nval: val/images\ntest: test/images\n")
        fh.write(f"nc: {len(NAMES)}\nnames:\n")
        for i, n in enumerate(NAMES):
            fh.write(f"  {i}: {n}\n")
    print("dataset written to", args.out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

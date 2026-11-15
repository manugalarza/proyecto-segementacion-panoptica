"""Build a paired HUD / HUD-free test set from the historical S3 partition (one seed).

For every historical record assigned to the test split whose frame also exists in the
HUD-free dataset, write two copies of the same frame with the SAME labels:

  out/con_hud/images/<name>.jpg   original frame (Imagenes/<original_split>/<name>.jpg)
  out/sin_hud/images/<name>.png   HUD-free frame (dataset_sin_hud/images/<imagen>.png)
  out/{con_hud,sin_hud}/labels/<name>.txt   labels from Etiquetas (the partition's labels),
                                            class 4 (SDZI) rewritten to 5 (machinery)

Using the partition's labels for both versions means the only difference between the
two sets is the HUD. The HUD-free dataset's own labels differ from Etiquetas for most
frames, so they are not used here (see the audit notes).

Standard library only. --dry-run validates and writes the manifest only.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

LABEL_DIR = {"train": "train", "test": "test", "val": os.path.join("validation", "val")}
NAMES = ["vehicle", "building", "road", "river", "SDZI", "machinery"]


def index_datos_v2(root):
    import glob, hashlib
    idx = {}
    for split in ["train", "val", "test"]:
        for img in glob.glob(os.path.join(root, split, "images", "*")):
            stem = os.path.splitext(os.path.basename(img))[0]
            lab = os.path.join(root, split, "labels", stem + ".txt")
            if os.path.isfile(lab):
                idx.setdefault(hashlib.sha256(open(img, "rb").read()).hexdigest(), lab)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partition-csv", required=True)
    ap.add_argument("--assignment", required=True, help="content_id,split CSV for the partition")
    ap.add_argument("--correspondencia", required=True, help="dataset_original/correspondencia.csv")
    ap.add_argument("--images-root", required=True, help="original/Imagenes")
    ap.add_argument("--sin-hud-images", required=True, help="dataset_sin_hud/images")
    ap.add_argument("--labels-root", required=True, help="original/Etiquetas")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datos-v2", default=None,
                    help="datos_v2 root: frames found there (by content) take their label from datos_v2; "
                         "other frames use Etiquetas with SDZI assumed to be machinery")
    args = ap.parse_args()

    with open(args.assignment, encoding="utf-8-sig", newline="") as fh:
        assign = {r["content_id"]: r["split"] for r in csv.DictReader(fh)}
    with open(args.correspondencia, encoding="utf-8-sig", newline="") as fh:
        corr = {os.path.splitext(r["nombre_original"])[0]: os.path.splitext(r["imagen"])[0]
                for r in csv.DictReader(fh)}
    with open(args.partition_csv, encoding="utf-8-sig", newline="") as fh:
        rows = [r for r in csv.DictReader(fh, delimiter=";") if r["origen"] == "historical"]

    from collections import Counter
    sources = Counter()
    v2_index = index_datos_v2(args.datos_v2) if args.datos_v2 else {}
    pairs, skipped_no_sin_hud, skipped_no_label = [], 0, 0
    seen = set()
    for r in rows:
        if assign.get(r["content_id"]) != args.split:
            continue
        name, osplit = r["nombre_frame"], r["original_split"]
        if name not in corr:
            skipped_no_sin_hud += 1
            continue
        sh = corr[name]
        lab = os.path.join(args.labels_root, LABEL_DIR[osplit], name + ".txt")
        img = os.path.join(args.images_root, osplit, name + ".jpg")
        sh_img = os.path.join(args.sin_hud_images, sh + ".png")
        if not (os.path.isfile(lab) and os.path.isfile(img) and os.path.isfile(sh_img)):
            skipped_no_label += 1
            continue
        key = (name, osplit)
        if key in seen:
            continue
        seen.add(key)
        remap = True
        if v2_index:
            import hashlib
            sha = hashlib.sha256(open(img, "rb").read()).hexdigest()
            if sha in v2_index:
                lab, remap = v2_index[sha], False
        sources[("datos_v2" if not remap else "etiquetas_sdzi_asumida_maquinaria") if v2_index else "etiquetas"] += 1
        pairs.append({"name": name, "orig_split": osplit, "sin_hud_name": sh,
                      "img": img, "sin_hud_img": sh_img, "label": lab, "remap": remap})

    print(f"split={args.split} paired records: {len(pairs)}")
    print(f"skipped (no HUD-free version): {skipped_no_sin_hud}; skipped (missing file): {skipped_no_label}")
    print("label source per paired record:", dict(sources))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "manifest.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "original_split", "sin_hud_name", "img", "sin_hud_img", "label"])
        for p in pairs:
            w.writerow([p["name"], p["orig_split"], p["sin_hud_name"], p["img"], p["sin_hud_img"], p["label"]])

    if args.dry_run:
        print("dry run: nothing copied")
        return 0

    for version in ["con_hud", "sin_hud"]:
        for d in ["images", "labels"]:
            os.makedirs(os.path.join(args.out, version, d), exist_ok=True)
    for p in pairs:
        for version, src_img, ext in [("con_hud", p["img"], ".jpg"), ("sin_hud", p["sin_hud_img"], ".png")]:
            shutil.copyfile(src_img, os.path.join(args.out, version, "images", p["name"] + ext))
            out_lab = os.path.join(args.out, version, "labels", p["name"] + ".txt")
            with open(p["label"], encoding="utf-8", errors="ignore") as src, open(out_lab, "w", encoding="utf-8", newline="\n") as dst:
                for line in src:
                    parts = line.split()
                    if not parts:
                        continue
                    if p["remap"] and parts[0] == "4":
                        parts[0] = "5"
                    dst.write(" ".join(parts) + "\n")

    for version in ["con_hud", "sin_hud"]:
        with open(os.path.join(args.out, version, "dataset.yaml"), "w", encoding="utf-8") as fh:
            fh.write(f"path: .\ntest: images\nnc: {len(NAMES)}\nnames:\n")
            for i, n in enumerate(NAMES):
                fh.write(f"  {i}: {n}\n")
    print("written to", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build datos_v2 = the reviewed video frames (datos_2x2/abril, split unchanged)
+ the scene frames of dataset_split_completo, split by SCENE GROUP.

Two steps:
  1. plan  (default): prints and writes the proposed split of the scene frames
     and a list of gt_manual candidates from the TEST scenes. Nothing is copied.
  2. --build: creates the dataset folder (after you approve the plan).

Scene frames = frames of --source that are not in --video-data and whose name
does not start with "video_" (the video gap frames stay out on purpose: they
separate the train/test blocks of the video split).

Grouping (no near-duplicates across splits):
  frame_NNNNNN_sceneSSS : sorted by frame number; a new group starts only when
                          the scene changes AND the frame gap is > --gap
  frame_NNNN (no scene) : treated as a separate source, grouped by frame-number
                          proximity (gap > --gap starts a new group)
Groups are assigned to train/val/test greedily so that frames AND machinery
boxes approach --ratios.

Labels of scene frames: SDZI boxes (class 4) are rewritten with the browser
review (--scene-review JSON): maquinaria -> 5, vehiculo -> 0, edificio -> 1,
error -> removed, sdzi/ambos -> stay 4. Class 4 (true SDZI, only a handful of
boxes) is meant to be IGNORED in training (not in the agreed class scheme).
Output classes: 0 vehicle, 1 building, 2 road, 3 river, 4 SDZI(ignored), 5 machinery.

Usage (Lenovo, repo root):
    .venv\\Scripts\\python scripts\\build_dataset_v2.py --video-data "$R\\resultados 2x2\\abril" ^
        --source "...\\dataset_split_completo" --scene-review "$R\\revision_sdzi_235.json" ^
        --out "$R\\datos_v2"                 # plan only
    ... same command + --build                # create the dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from pathlib import Path

NAMES = ["vehicle", "building", "road", "river", "SDZI", "machinery"]
NEW_CLASS = {"maquinaria": 5, "vehiculo": 0, "edificio": 1, "sdzi": 4, "ambos": 4, "error": None}
SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
SCENE_RE = re.compile(r"^frame_(\d+)_scene(\d+)$")
NOSCENE_RE = re.compile(r"^frame_(\d+)$")


def find_image(images_dir: Path, stem: str):
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def relabel_scene(lines, stem, split_src, review):
    out = []
    for i, line in enumerate(lines):
        p = line.split()
        if len(p) < 5:
            continue
        if int(p[0]) == 4:
            d = review.get((split_src, stem, i))
            if d is None:
                raise SystemExit(f"SDZI box without review decision: {split_src}/{stem} line {i}")
            new = NEW_CLASS[d["decision"]]
            if new is None:
                continue
            p[0] = str(new)
            for bx in filter(None, (d.get("machinery_boxes") or "").split("|")):
                out.append(f"5 {bx}")
        out.append(" ".join(p))
    return out


def group_frames(frames, gap, max_group):
    """frames: list of (stem, kind, number, scene). Returns ({stem: group_id}, dropped stems).
    Chains longer than max_group frames are cut into blocks; the first frame of
    each new block is dropped as a separator (same idea as the video split)."""
    groups, dropped = {}, []
    for kind in ("scene", "noscene"):
        items = sorted((f for f in frames if f[1] == kind), key=lambda f: f[2])
        gid, prev = -1, None
        for stem, _, num, scene in items:
            new = prev is None or (num - prev[0] > gap and (kind == "noscene" or scene != prev[1]))
            if kind == "scene" and prev is not None and scene == prev[1] and num - prev[0] > 10 * gap:
                new = True  # same scene id reused much later in the video
            if new:
                gid += 1
                size = 0
            elif max_group and size >= max_group:
                gid += 1
                size = 0
                dropped.append(stem)
                prev = (num, scene)
                continue
            groups[stem] = f"{kind}{gid:03d}"
            size += 1
            prev = (num, scene)
    return groups, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video-data", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--scene-review", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--gap", type=int, default=10, help="Frame-number gap that separates groups")
    ap.add_argument("--max-group", type=int, default=20,
                    help="Cut longer chains into blocks of this size (one separator frame dropped per cut); 0 = never cut")
    ap.add_argument("--n-gt", type=int, default=8, help="gt_manual candidates from test scenes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()

    review = {(r["split"], r["frame"], int(r["line"])): r
              for r in json.loads(args.scene_review.read_text(encoding="utf-8"))}
    video_stems = {p.stem for s in SPLITS for p in (args.video_data / s / "labels").glob("*.txt")}

    frames, info = [], {}
    for s in SPLITS:
        for lp in sorted((args.source / s / "labels").glob("*.txt")):
            stem = lp.stem
            if stem in video_stems or stem.startswith("video_"):
                continue
            m1, m2 = SCENE_RE.match(stem), NOSCENE_RE.match(stem)
            if m1:
                f = (stem, "scene", int(m1.group(1)), int(m1.group(2)))
            elif m2:
                f = (stem, "noscene", int(m2.group(1)), None)
            else:
                print(f"WARNING: unexpected name, skipped: {s}/{stem}")
                continue
            img = find_image(args.source / s / "images", stem)
            if img is None:
                print(f"WARNING: no image for {s}/{stem}, skipped")
                continue
            labels = relabel_scene(lp.read_text().splitlines(), stem, s, review)
            counts = [0] * len(NAMES)
            for l in labels:
                counts[int(l.split()[0])] += 1
            frames.append(f)
            info[stem] = {"src_split": s, "image": img, "labels": labels, "counts": counts}

    groups, dropped = group_frames(frames, args.gap, args.max_group)
    gstats = {}
    for stem, g in groups.items():
        st = gstats.setdefault(g, {"frames": [], "mach": 0})
        st["frames"].append(stem)
        st["mach"] += info[stem]["counts"][5]

    # greedy assignment: biggest machinery groups first, to the split furthest below target
    rng = random.Random(args.seed)
    order = sorted(gstats, key=lambda g: (-gstats[g]["mach"], -len(gstats[g]["frames"]), rng.random()))
    tot_f = sum(len(v["frames"]) for v in gstats.values())
    tot_m = max(1, sum(v["mach"] for v in gstats.values()))
    cur = {s: {"f": 0, "m": 0} for s in SPLITS}
    assign = {}
    for g in order:
        def deficit(s):
            r = args.ratios[SPLITS.index(s)]
            return (r - cur[s]["m"] / tot_m) + (r - cur[s]["f"] / tot_f)
        s = max(SPLITS, key=deficit)
        assign[g] = s
        cur[s]["f"] += len(gstats[g]["frames"])
        cur[s]["m"] += gstats[g]["mach"]

    rows = []
    for stem, g in sorted(groups.items()):
        c = info[stem]["counts"]
        rows.append({"frame": stem, "group": g, "split": assign[g], "src_split": info[stem]["src_split"],
                     **{n: c[i] for i, n in enumerate(NAMES)}})

    print(f"\nScene frames: {len(rows)} in {len(gstats)} groups "
          f"({sum(1 for g in gstats if g.startswith('scene'))} scene, {sum(1 for g in gstats if g.startswith('noscene'))} no-scene); "
          f"{len(dropped)} separator frames dropped")
    sizes = sorted(len(v["frames"]) for v in gstats.values())
    print(f"group size: min {sizes[0]}, median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"{'split':<6}{'groups':>7}{'frames':>8}" + "".join(f"{n:>11}" for n in NAMES))
    for s in SPLITS:
        sr = [r for r in rows if r["split"] == s]
        print(f"{s:<6}{len({r['group'] for r in sr}):>7}{len(sr):>8}" + "".join(f"{sum(r[n] for r in sr):>11}" for n in NAMES))

    # gt_manual candidates: from test, max one per group first, half with machinery
    test = [r for r in rows if r["split"] == "test"]
    rng.shuffle(test)
    with_m = [r for r in test if r["machinery"] > 0]
    without = [r for r in test if r["machinery"] == 0]
    picks, used = [], set()
    for pool, n in ((with_m, (args.n_gt + 1) // 2), (without, args.n_gt // 2)):
        for r in pool:
            if len([p for p in picks if p in pool]) >= n:
                break
            if r["group"] not in used:
                picks.append(r)
                used.add(r["group"])
    for r in with_m + without:  # top up if groups ran out
        if len(picks) >= args.n_gt:
            break
        if r not in picks:
            picks.append(r)
    print(f"\ngt_manual candidates ({len(picks)}):")
    for r in sorted(picks, key=lambda r: r["frame"]):
        print(f"  {r['frame']:<28} group {r['group']}  machinery {r['machinery']}  vehicle {r['vehicle']}  "
              f"building {r['building']}  road {r['road']}  river {r['river']}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "plan_escenas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "gt_manual_candidatos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(sorted(picks, key=lambda r: r["frame"]))
    print(f"\nPlan -> {args.out / 'plan_escenas.csv'}  |  candidates -> {args.out / 'gt_manual_candidatos.csv'}")

    if not args.build:
        print("Plan only. Re-run with --build to create the dataset.")
        return

    manifest = []
    for s in SPLITS:
        (args.out / s / "images").mkdir(parents=True, exist_ok=True)
        (args.out / s / "labels").mkdir(parents=True, exist_ok=True)
        src_img, src_lab = args.video_data / s / "images", args.video_data / s / "labels"
        for lp in sorted(src_lab.glob("*.txt")):
            img = find_image(src_img, lp.stem)
            if img is None:
                continue
            link_or_copy(img, args.out / s / "images" / img.name)
            shutil.copy2(lp, args.out / s / "labels" / lp.name)
            manifest.append({"frame": lp.stem, "split": s, "source": "video", "group": ""})
    for r in rows:
        it = info[r["frame"]]
        link_or_copy(it["image"], args.out / r["split"] / "images" / it["image"].name)
        lab = it["labels"]
        (args.out / r["split"] / "labels" / f"{r['frame']}.txt").write_text("\n".join(lab) + ("\n" if lab else ""))
        manifest.append({"frame": r["frame"], "split": r["split"], "source": "scene", "group": r["group"]})
    (args.out / "dataset.yaml").write_text(
        "path: .\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 6\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    with open(args.out / "split_manifest_v2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    shutil.copy2(args.scene_review, args.out / args.scene_review.name)
    for s in SPLITS:
        n = len(list((args.out / s / "images").iterdir()))
        print(f"{s}: {n} images")
    print(f"Built {args.out}. NOTE: dataset.yaml has 'path: .' - set an absolute path before using it with ultralytics.")


if __name__ == "__main__":
    main()

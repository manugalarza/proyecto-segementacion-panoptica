"""YOLO reference on a dataset built by build_split_from_csv.py (2026-11-11).

Purpose: tell apart "our model is weak" from "the data does not generalize". Trains an
Ultralytics YOLO detector on the same train/val split and writes its predictions on val and
test as YOLO txt files with confidence (cls xc yc w h conf), so eval_detection.py can score
them with exactly the same protocol as Panoptic FCN:

    python scripts/eval_detection.py --data <dataset> --split test --predictions YOLO=<out>/pred_test/labels

Note: Ultralytics applies its own augmentation (mosaic, flips, HSV, scale) by default.

Usage (repo root, Windows):
    .venv\\Scripts\\python scripts\\yolo_reference.py --data ..\\datos_escenas_v0_f0 --out ..\\yolo_escenas_v0_f0
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="Dataset root with train/val/test and dataset.yaml")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="splits to predict; use only 'val' while the test set is closed")
    ap.add_argument("--weights", choices=["best", "last"], default="best",
                    help="checkpoint used for predictions; 'last' when validation cannot be trusted for selection")
    ap.add_argument("--train-args", default="",
                    help='extra Ultralytics train arguments as key=value pairs separated by spaces, '
                         'e.g. "optimizer=SGD lr0=0.01" (to copy another run\'s configuration)')
    args = ap.parse_args()

    from ultralytics import YOLO

    data = args.data.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ["vehicle", "building", "road", "river", "SDZI", "machinery"]
    # Absolute path: a relative "path:" is resolved against Ultralytics' datasets folder, not the yaml.
    yaml = out / "data_abs.yaml"
    yaml.write_text(
        f"path: {data.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"nc: {len(names)}\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )

    weights = out / "train" / "weights" / f"{args.weights}.pt"
    if not (out / "train" / "weights" / "last.pt").exists():
        import yaml as _yaml
        extra = {k: _yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.train_args.split())}
        model = YOLO(args.model)
        model.train(data=str(yaml), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, device=args.device,
                    workers=args.workers, project=str(out), name="train", exist_ok=True, seed=args.seed,
                    verbose=False, **extra)
    else:
        print(f"Already trained: {out / 'train'}")
    suffix = "" if args.weights == "best" else "_last"

    model = YOLO(str(weights))
    for split in args.splits:
        src = data / split / "images"
        if not src.exists():
            continue
        import shutil
        shutil.rmtree(out / f"pred_{split}{suffix}", ignore_errors=True)  # save_txt appends to existing files
        for _ in model.predict(source=str(src), imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300,
                               save=False, save_txt=True, save_conf=True, device=args.device,
                               project=str(out), name=f"pred_{split}{suffix}", exist_ok=True, stream=True, verbose=False):
            pass
        print(f"{split}: predictions in {out / f'pred_{split}{suffix}' / 'labels'}")


if __name__ == "__main__":
    main()

"""Writes Ultralytics YOLO predictions for every image of a folder as TXT (cls xc yc w h conf),
in the format eval_detection.py --predictions reads (2026-11-12).

Usage (repo root):
    .venv\\Scripts\\python scripts\\yolo_predict_dir.py --weights W.pt --images DIR --out DIR [--imgsz 1024] [--device cpu]
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="folder that receives labels/<stem>.txt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from ultralytics import YOLO
    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)  # save_txt appends to existing files
    model = YOLO(args.weights)
    for _ in model.predict(source=str(args.images.resolve()), imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300,
                           save=False, save_txt=True, save_conf=True, device=args.device,
                           project=str(out.parent), name=out.name, exist_ok=True, stream=True, verbose=False):
        pass
    print(f"predictions in {out / 'labels'}")


if __name__ == "__main__":
    main()

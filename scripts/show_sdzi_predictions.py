"""Quick qualitative figure: GT SDZI boxes vs predicted SDZI probability.

For each selected frame draws three panels side by side:
  1. image + GT boxes (all classes, SDZI in magenta)
  2. SDZI probability heatmap from the stuff branch (predict(), same
     preprocessing as training) blended over the image
  3. image + predicted stuff map (river cyan, SDZI magenta)

Works with any checkpoint, including the intermediate ones saved every
10 epochs while training is still running.

Usage (from the repo root, Windows):
    .venv\\Scripts\\python scripts\\show_sdzi_predictions.py ^
        --checkpoint "C:\\...\\checkpoints_2x2\\etiquetas\\epoch_20.pt" ^
        --data "C:\\...\\datos_2x2\\abril" --split val ^
        --out "C:\\...\\Proyecto de grado\\figuras_sdzi" --n 8
By default it picks frames that contain SDZI in the GT.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from draw_yolo_labels import add_header, draw, read_yolo_labels  # noqa: E402
from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT  # noqa: E402
from panoptic_mining.inference import load_model, predict  # noqa: E402

SDZI_ID = 4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="Dataset root (datos_2x2\\abril or \\etiquetas).")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--all-frames", action="store_true", help="Sample from all frames, not only frames with SDZI.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=640, help="Width of each panel.")
    args = ap.parse_args()

    torch.set_num_threads(2)  # leave the CPU to the training run
    model = load_model(args.checkpoint)
    images = sorted((args.data / args.split / "images").glob("*.jpg"))
    labels_dir = args.data / args.split / "labels"
    if not args.all_frames:
        images = [p for p in images if any(b[0] == SDZI_ID for b in (read_yolo_labels(labels_dir / f"{p.stem}.txt") or []))]
    random.Random(args.seed).shuffle(images)
    images = images[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)

    sdzi_idx = DEFAULT_CLASS_SPLIT.stuff_index["SDZI"]
    river_idx = DEFAULT_CLASS_SPLIT.stuff_index["river"]
    tag = f"{args.checkpoint.parent.name}_{args.checkpoint.stem}"
    for p in images:
        image = cv2.imread(str(p))
        boxes = read_yolo_labels(labels_dir / f"{p.stem}.txt") or []
        pred = predict(model, p)

        gt_panel = add_header(draw(image, boxes), f"{p.stem} | GT ({args.data.name})", boxes)

        prob = (pred.stuff_probs[sdzi_idx] * 255).astype(np.uint8)
        heat = cv2.applyColorMap(prob, cv2.COLORMAP_JET)
        heat_panel = cv2.addWeighted(image, 0.5, heat, 0.5, 0)
        heat_panel = add_header(heat_panel, f"P(SDZI) max={pred.stuff_probs[sdzi_idx].max():.2f}", None)

        seg = image.copy()
        overlay = seg.copy()
        overlay[pred.stuff_argmax == sdzi_idx] = (255, 0, 255)
        overlay[pred.stuff_argmax == river_idx] = (255, 255, 0)
        seg = cv2.addWeighted(seg, 0.5, overlay, 0.5, 0)
        frac = (pred.stuff_argmax == sdzi_idx).mean()
        seg_panel = add_header(seg, f"pred stuff ({tag}) SDZI={frac:.1%}", None)

        panels = [cv2.resize(x, (args.width, int(x.shape[0] * args.width / x.shape[1]))) for x in (gt_panel, heat_panel, seg_panel)]
        h = max(x.shape[0] for x in panels)
        panels = [np.vstack([x, np.zeros((h - x.shape[0], args.width, 3), x.dtype)]) for x in panels]
        cv2.imwrite(str(args.out / f"{tag}__{p.stem}.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{p.stem}: max P(SDZI)={pred.stuff_probs[sdzi_idx].max():.2f}, SDZI pixels={frac:.1%}")
    print(f"Figures in {args.out}")


if __name__ == "__main__":
    main()

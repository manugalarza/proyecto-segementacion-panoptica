"""Train Panoptic FCN on one dataset built by build_gt_datasets.py.

Usage (from the repo root, Windows):
    .venv\\Scripts\\python scripts\\train_gt_variant.py ^
        --data "C:\\...\\Proyecto de grado\\datos_2x2\\etiquetas" ^
        --pretrained "C:\\...\\Proyecto de grado\\epoch_63.pt" ^
        --out "C:\\...\\Proyecto de grado\\checkpoints_2x2\\etiquetas" --epochs 150

Defaults match the decisions of 2026-10-08:
  - same preprocessing as inference.py (BGR, 512x512, /255)
  - stuff classes river / SDZI / none with class weights 1 / 4 / 0.3
    (missing SDZI costs more than mislabeling background; annotations are
    known to be incomplete)
  - thing classes vehicle / building / road weighted by inverse frequency
  - encoder initialized from the LandCover.ai pretraining (epoch_63.pt)
Writes train_log.csv (one row per epoch) next to the checkpoints.

Added 2026-10-14 (runs A/B, on the Lenovo GPU):
  --encoder resnet18      ResNet-18 ImageNet + FPN instead of the placeholder
                          (--pretrained is then ignored; LandCover epoch_63.pt
                          only fits the placeholder). Backbone LR = lr * 0.1.
  --masks DIR             stuff target from semantic PNGs (SAM pseudo-masks)
                          instead of filled box rectangles
  --device auto|cpu|cuda  auto = cuda if available
  Run A: --encoder resnet18                (rectangles)
  Run B: --encoder resnet18 --masks <sam_train\\masks>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from panoptic_mining.data.manifest import load_yolo_manifest  # noqa: E402
from panoptic_mining.data.torch_dataset import (  # noqa: E402
    DEFAULT_CLASS_SPLIT,
    ManifestSegmentationDataset,
    compute_thing_pos_weight,
)
from panoptic_mining.models.panoptic_fcn import FeatureEncoder, PanopticFCN, build_encoder  # noqa: E402
from panoptic_mining.training.train import TrainConfig, train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="Dataset root with dataset.yaml (e.g. datos_2x2/etiquetas).")
    ap.add_argument("--out", required=True, type=Path, help="Checkpoint folder.")
    ap.add_argument("--pretrained", type=Path, default=None, help="Checkpoint whose ENCODER is loaded (LandCover epoch_63.pt).")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--stuff-weights", type=float, nargs=3, default=[1.0, 4.0, 0.3], metavar=("RIVER", "SDZI", "NONE"))
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = all cores).")
    ap.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints in --out and start from scratch.")
    ap.add_argument("--encoder", choices=["placeholder", "resnet18"], default="placeholder")
    ap.add_argument("--resnet-weights", default="imagenet",
                    help="'imagenet' (torch hub cache / download) or a path to resnet18-f37072fd.pth")
    ap.add_argument("--encoder-lr-mult", type=float, default=None,
                    help="Backbone LR multiplier (default 0.1 for resnet18, 1.0 for placeholder).")
    ap.add_argument("--masks", type=Path, default=None, help="Folder of semantic PNG masks (e.g. SAM) for the stuff target.")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda")
    ap.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (default 4 on cuda, 0 on cpu).")
    args = ap.parse_args()

    torch.set_num_threads(args.threads or os.cpu_count() or 1)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False (check the CUDA build of torch)")
    num_workers = args.num_workers if args.num_workers is not None else (4 if device == "cuda" else 0)
    encoder_lr_mult = args.encoder_lr_mult if args.encoder_lr_mult is not None else (0.1 if args.encoder == "resnet18" else 1.0)
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    manifest = load_yolo_manifest(args.data)
    dataset = ManifestSegmentationDataset(manifest, split="train", class_split=DEFAULT_CLASS_SPLIT, stuff_mask_dir=args.masks)
    train_counts = {}
    for frame in manifest.frames_by_split["train"]:
        for box in frame.boxes:
            train_counts[box.class_name] = train_counts.get(box.class_name, 0) + 1
    thing_pos_weight = compute_thing_pos_weight(train_counts, DEFAULT_CLASS_SPLIT.thing_class_names)

    if args.encoder == "resnet18":
        encoder = build_encoder("resnet18", resnet_weights=args.resnet_weights)
        print(f"Encoder: ResNet-18 + FPN, weights={args.resnet_weights}")
        if args.pretrained is not None:
            print("NOTE: --pretrained ignored with --encoder resnet18")
            args.pretrained = None
    else:
        encoder = FeatureEncoder()
    if args.pretrained is not None:
        ckpt = torch.load(str(args.pretrained), map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        encoder.load_state_dict(enc_state)
        print(f"Loaded encoder from {args.pretrained} (epoch {ckpt.get('epoch')})")

    torch.manual_seed(args.seed)
    model = PanopticFCN(
        num_stuff_classes=len(DEFAULT_CLASS_SPLIT.stuff_class_names),
        num_thing_classes=len(DEFAULT_CLASS_SPLIT.thing_class_names),
        encoder=encoder,
    )
    start_epoch, optimizer_state = 1, None
    existing = sorted(
        (int(p.stem.split("_")[1]), p) for p in args.out.glob("epoch_*.pt") if p.stem.split("_")[1].isdigit()
    ) if args.out.is_dir() else []
    if existing and not args.no_resume:
        last_epoch, last_path = existing[-1]
        ckpt = torch.load(str(last_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer_state = ckpt.get("optimizer_state_dict")
        start_epoch = last_epoch + 1
        print(f"Resuming from {last_path} -> continuing at epoch {start_epoch}/{args.epochs}")

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.out,
        thing_pos_weight=thing_pos_weight,
        stuff_class_weight=list(args.stuff_weights),
        save_every_n_epochs=args.save_every,
        seed=args.seed,
        log_every_n_steps=50,
        start_epoch=start_epoch,
        optimizer_state=optimizer_state,
        device=device,
        encoder=args.encoder,
        encoder_lr_mult=encoder_lr_mult,
        num_workers=num_workers,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    run_info = {
        "data": str(args.data),
        "train_frames": len(dataset),
        "train_box_counts": train_counts,
        "stuff_classes": DEFAULT_CLASS_SPLIT.stuff_class_names,
        "stuff_weights": args.stuff_weights,
        "thing_classes": DEFAULT_CLASS_SPLIT.thing_class_names,
        "thing_pos_weight": thing_pos_weight,
        "pretrained": str(args.pretrained) if args.pretrained else None,
        "encoder": args.encoder,
        "resnet_weights": args.resnet_weights if args.encoder == "resnet18" else None,
        "encoder_lr_mult": encoder_lr_mult,
        "stuff_target": f"masks: {args.masks}" if args.masks else "box rectangles",
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print(json.dumps(run_info, indent=2))

    history = train(model, dataset, config)

    log_path = args.out / "train_log.csv"
    new_file = not log_path.exists() or start_epoch == 1
    with open(log_path, "w" if new_file else "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["epoch", "mean_loss", "stuff_loss", "thing_loss", "minutes"])
        for e in history.epochs:
            writer.writerow([e.epoch, f"{e.mean_loss:.5f}", f"{e.mean_stuff_loss:.5f}", f"{e.mean_thing_loss:.5f}", f"{e.elapsed_seconds / 60:.2f}"])
    print(f"Done. Checkpoints and train_log.csv in {args.out}")


if __name__ == "__main__":
    main()

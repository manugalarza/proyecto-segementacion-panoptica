"""Training report: time per epoch, total time, and learning curves for any
number of runs made with train_gt_variant.py (reads <run>/train_log.csv and
<run>/run_info.json).

Optionally computes a VALIDATION loss for every saved checkpoint
(epoch_10.pt, epoch_20.pt, ...) so train and val curves can be compared
(overfitting check). Val loss uses the same loss and class weights as
training (from run_info.json). Stuff targets on val are box rectangles unless
--val-masks is given, so for runs trained on SAM masks the val STUFF loss is
measured against a different target than the one they learned: compare
those runs on the thing loss or with eval_pixel.py instead.

--stuff-names lets old checkpoints be scored against relabelled data, e.g.
after SDZI was renamed machinery (class 5):  --stuff-names river machinery none

Usage (repo root):
    .venv\\Scripts\\python scripts\\plot_training.py --run A="C:\\runs\\A" --run B="C:\\runs\\B" ^
        --run P_rect="...\\checkpoints_2x2\\abril" --out "...\\reporte_entrenamiento" ^
        --val-data "...\\resultados 2x2\\abril" --stuff-names river machinery none
Outputs: curves.png, val_curves.png (if --val-data), timing.csv, val_loss.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def val_losses(run_dir: Path, info: dict, args, device: str) -> list[dict]:
    import torch
    from torch.utils.data import DataLoader

    from panoptic_mining.data.manifest import load_yolo_manifest
    from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT, ClassSplit, ManifestSegmentationDataset
    from panoptic_mining.inference import load_model
    from panoptic_mining.training.train import compute_losses

    split = DEFAULT_CLASS_SPLIT
    if args.stuff_names:
        split = ClassSplit(stuff_class_names=args.stuff_names,
                           thing_class_names=args.thing_names or DEFAULT_CLASS_SPLIT.thing_class_names,
                           stuff_background_class_name=args.stuff_names[-1])
    manifest = load_yolo_manifest(args.val_data)
    ds = ManifestSegmentationDataset(manifest, split="val", class_split=split, stuff_mask_dir=args.val_masks)
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    sw = torch.tensor(info["stuff_weights"], dtype=torch.float32) if info.get("stuff_weights") else None
    tw = torch.tensor(info["thing_pos_weight"], dtype=torch.float32) if info.get("thing_pos_weight") else None
    out = []
    ckpts = sorted(((int(p.stem.split("_")[1]), p) for p in run_dir.glob("epoch_*.pt")
                    if p.stem.split("_")[1].isdigit()))
    for epoch, path in ckpts:
        model = load_model(path, class_split=split, device=device)
        tot = st = th = 0.0
        n = 0
        with torch.no_grad():
            for images, stuff_t, thing_t in loader:
                t, s, h = compute_losses(model, images.to(device), stuff_t.to(device), thing_t.to(device),
                                         thing_pos_weight=tw, stuff_class_weight=sw)
                b = images.shape[0]
                tot, st, th, n = tot + t.item() * b, st + s.item() * b, th + h.item() * b, n + b
        out.append({"epoch": epoch, "val_loss": tot / n, "val_stuff": st / n, "val_thing": th / n})
        print(f"  {run_dir.name} epoch {epoch}: val stuff {st / n:.4f}  thing {th / n:.4f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=run folder (repeatable)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-data", type=Path, default=None, help="Dataset root with val/ for checkpoint val loss")
    ap.add_argument("--val-masks", type=Path, default=None, help="Semantic PNGs for val stuff targets (else rectangles)")
    ap.add_argument("--stuff-names", nargs="+", default=None)
    ap.add_argument("--thing-names", nargs="+", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {}
    for item in args.run:
        name, _, folder = item.partition("=")
        folder = Path(folder)
        log = pd.read_csv(folder / "train_log.csv")
        info_path = folder / "run_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        runs[name] = (folder, log, info)

    rows = []
    for name, (folder, log, info) in runs.items():
        rows.append({
            "run": name,
            "device": info.get("device", "cpu"),
            "encoder": info.get("encoder", "placeholder"),
            "stuff_target": "SAM masks" if str(info.get("stuff_target", "")).startswith("masks") else "rectangles",
            "train_frames": info.get("train_frames"),
            "batch_size": info.get("batch_size"),
            "epochs": int(log.epoch.max()),
            "min_per_epoch_median": round(float(log.minutes.median()), 2),
            "sec_per_image": round(float(log.minutes.median()) * 60 / max(info.get("train_frames") or 1, 1), 3),
            "total_hours": round(float(log.minutes.sum()) / 60, 2),
            "stuff_loss_first": round(float(log.stuff_loss.iloc[0]), 4),
            "stuff_loss_last": round(float(log.stuff_loss.iloc[-1]), 4),
            "thing_loss_last": round(float(log.thing_loss.iloc[-1]), 4),
        })
    timing = pd.DataFrame(rows)
    timing.to_csv(args.out / "timing.csv", index=False)
    print(timing.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, (_, log, _) in runs.items():
        axes[0].plot(log.epoch, log.stuff_loss, label=name)
        axes[1].plot(log.epoch, log.thing_loss, label=name)
    for ax, title in zip(axes, ["Stuff loss (train)", "Thing loss (train)"]):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "curves.png", dpi=150)
    print(f"-> {args.out / 'curves.png'}")

    if args.val_data:
        import torch

        device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
        all_val = []
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for k, (name, (folder, log, info)) in enumerate(runs.items()):
            v = pd.DataFrame(val_losses(folder, info, args, device))
            if v.empty:
                continue
            v.insert(0, "run", name)
            all_val.append(v)
            color = f"C{k}"
            axes[0].plot(log.epoch, log.stuff_loss, color=color, alpha=0.35, label=f"{name} train")
            axes[0].plot(v.epoch, v.val_stuff, "o-", color=color, label=f"{name} val")
            axes[1].plot(log.epoch, log.thing_loss, color=color, alpha=0.35, label=f"{name} train")
            axes[1].plot(v.epoch, v.val_thing, "o-", color=color, label=f"{name} val")
        for ax, title in zip(axes, ["Stuff loss: train (line) vs val (dots)", "Thing loss: train vs val"]):
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out / "val_curves.png", dpi=150)
        if all_val:
            pd.concat(all_val).to_csv(args.out / "val_loss.csv", index=False)
        print(f"-> {args.out / 'val_curves.png'}")


if __name__ == "__main__":
    main()

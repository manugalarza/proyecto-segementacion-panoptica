"""Qualitative sanity check for a trained checkpoint — the thing a
training-loss number alone can't answer.

Context: the 2026-09-19/20 overnight run finished fine-tuning at
mean_loss=0.1957, down from 0.6069 in the 1-epoch smoke test (see
docs/decisions.md). There is still no validation split, so that drop
could reflect real learning or could reflect overfitting on
dataset_split_completo's train split. This script doesn't settle that
(a real held-out metric still needs the leakage-safe split or the
independent dense-annotation subset), but it answers the cheaper, more
urgent question first: does the model predict anything visually
sensible at all, or has it collapsed (e.g. always predicting one stuff
class everywhere, or producing zero/thousands of thing detections)?

For each of a random sample of frames, writes one side-by-side PNG:
  - left: the frame with its ground-truth box annotations drawn
  - right: the same frame with the model's predictions overlaid —
    stuff classes as a translucent color fill, thing-class instances
    as contoured masks labeled with class + confidence score

Also prints a summary across the sample: mean predicted pixel fraction
per stuff class, and detection count per thing class. That summary is
the fastest way to catch a collapsed model without eyeballing every
image (e.g. one stuff class at ~100% everywhere, or 0 detections for
every thing class across all frames).

Usage (from the repo root, inside the activated venv):

    python scripts/visualize_predictions.py \\
        --data-root "..\\imagenes recortado\\dataset_split_completo\\dataset_split_completo" \\
        --checkpoint checkpoints\\finetune_overnight\\epoch_213.pt \\
        --split val \\
        --num-samples 8 \\
        --output-dir predictions_preview

Requires the 'vision' extra (torch, opencv) — same as `panoptic-mining train`.
Not wired into the Typer CLI (unlike the `train` subcommands) because it
writes image files as a side effect rather than a JSON-summarizable
result; kept as a standalone script alongside run_overnight.ps1, per the
README's "scripts/ - one-off utility scripts" convention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from panoptic_mining.data.manifest import load_yolo_manifest
from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT
from panoptic_mining.models.context_fusion import ContextFusionPanopticFCN
from panoptic_mining.models.panoptic_fcn import PanopticFCN

# BGR (OpenCV convention), chosen to be visually distinct from each other.
STUFF_COLORS = {
    "river": (255, 128, 0),
    "SDZI": (0, 0, 255),
}
THING_COLORS = {
    "vehicle": (0, 255, 255),
    "building": (0, 255, 0),
    "road": (255, 0, 255),
}
PREVIEW_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True, type=Path, help="YOLO dataset root (contains dataset.yaml).")
    parser.add_argument("--checkpoint", required=True, type=Path, help="A checkpoint from 'train run' (fine-tuning), e.g. checkpoints/finetune_overnight/epoch_213.pt.")
    parser.add_argument("--split", default="val", help="Which manifest split to sample from (falls back to 'train' if missing).")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions_preview"))
    parser.add_argument("--context-fusion", action="store_true", help="Load as ContextFusionPanopticFCN — must match how the checkpoint was actually trained (train run --context-fusion).")
    parser.add_argument("--score-threshold", type=float, default=0.3, help="Thing-class heatmap peak threshold, same meaning as decode_instances'.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def draw_ground_truth(image: np.ndarray, frame) -> np.ndarray:
    """Draw this frame's YOLO box annotations at their native resolution,
    resized to the preview size afterward so labels stay legible.

    Uses the same color for a class here as the prediction panel uses for
    it (stuff classes included) — a real bug in the first version of this
    script (2026-09-20): only THING_COLORS was checked, so river/SDZI
    ground-truth boxes silently fell back to plain white, indistinguishable
    from "no color assigned". That made it look like a frame's obvious
    water body simply wasn't picked up by the model, when the real
    question — was that water body even labeled 'river' in this frame's
    ground truth in the first place? — couldn't be answered from the
    image at all without checking the box's printed label text.
    """
    vis = image.copy()
    orig_h, orig_w = image.shape[:2]
    all_colors = {**STUFF_COLORS, **THING_COLORS}
    for box in frame.boxes:
        x_min, y_min, x_max, y_max = box.to_pixels(orig_w, orig_h)
        color = all_colors.get(box.class_name, (255, 255, 255))
        cv2.rectangle(vis, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
        cv2.putText(
            vis, box.class_name, (int(x_min), max(int(y_min) - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return cv2.resize(vis, (PREVIEW_SIZE, PREVIEW_SIZE))


def overlay_predictions(image: np.ndarray, model, class_split, score_threshold: float) -> tuple[np.ndarray, dict]:
    image_resized = cv2.resize(image, (PREVIEW_SIZE, PREVIEW_SIZE))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    stats = {
        "stuff_pixel_fractions": {},
        "thing_detections": {name: 0 for name in class_split.thing_class_names},
    }

    # Stuff branch: per-pixel argmax over classes, upsampled (nearest —
    # this is a class index map, not a value to interpolate) to preview size.
    stuff_probs = torch.softmax(output.stuff_logits[0], dim=0)  # (C, h, w)
    stuff_argmax = stuff_probs.argmax(dim=0).numpy().astype(np.uint8)
    stuff_argmax_full = cv2.resize(stuff_argmax, (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_NEAREST)

    overlay = image_resized.copy()
    total_pixels = stuff_argmax_full.size
    for class_name, class_idx in class_split.stuff_index.items():
        mask = stuff_argmax_full == class_idx
        overlay[mask] = STUFF_COLORS.get(class_name, (200, 200, 200))
        stats["stuff_pixel_fractions"][class_name] = float(mask.sum()) / total_pixels
    vis = cv2.addWeighted(overlay, 0.35, image_resized, 0.65, 0)

    # Thing branch: decode instances per class (kernel-mechanism decode,
    # see PanopticFCN.decode_instances), draw contour + class/score label.
    for class_name, class_idx in class_split.thing_index.items():
        instances = model.decode_instances(
            output, class_id=class_idx, top_k=10, score_threshold=score_threshold
        )
        color = THING_COLORS.get(class_name, (255, 255, 255))
        for score, mask in instances:
            mask_full = cv2.resize(mask.numpy(), (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_full > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, color, 2)
            ys, xs = np.nonzero(binary_mask)
            if len(ys) > 0:
                cv2.putText(
                    vis, f"{class_name} {score:.2f}", (int(xs.min()), max(int(ys.min()) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
                )
            stats["thing_detections"][class_name] += 1

    return vis, stats


def load_model(checkpoint_path: Path, class_split, context_fusion: bool):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cls = ContextFusionPanopticFCN if context_fusion else PanopticFCN
    model = model_cls(
        num_stuff_classes=len(class_split.stuff_class_names),
        num_thing_classes=len(class_split.thing_class_names),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, mean_loss={checkpoint['mean_loss']:.4f}")
    return model


def main() -> None:
    args = parse_args()
    manifest = load_yolo_manifest(args.data_root)

    split = args.split
    if split not in manifest.frames_by_split:
        print(f"Split {split!r} not in manifest (missing_splits={manifest.missing_splits}); falling back to 'train'.")
        split = "train"
    frames = manifest.frames_by_split[split]
    if not frames:
        raise SystemExit(f"Split {split!r} has no frames — nothing to visualize.")

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(frames), size=min(args.num_samples, len(frames)), replace=False)

    model = load_model(args.checkpoint, DEFAULT_CLASS_SPLIT, args.context_fusion)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_stuff_fractions = {name: [] for name in DEFAULT_CLASS_SPLIT.stuff_class_names}
    total_thing_detections = {name: 0 for name in DEFAULT_CLASS_SPLIT.thing_class_names}

    for i, idx in enumerate(sample_idx):
        frame = frames[idx]
        image = cv2.imread(str(frame.image_path))
        if image is None:
            print(f"Skipping unreadable image: {frame.image_path}")
            continue

        gt_vis = draw_ground_truth(image, frame)
        pred_vis, stats = overlay_predictions(image, model, DEFAULT_CLASS_SPLIT, args.score_threshold)

        for name, frac in stats["stuff_pixel_fractions"].items():
            total_stuff_fractions[name].append(frac)
        for name, count in stats["thing_detections"].items():
            total_thing_detections[name] += count

        side_by_side = np.hstack([gt_vis, pred_vis])
        cv2.putText(side_by_side, "ground truth", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_by_side, "prediction", (PREVIEW_SIZE + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = args.output_dir / f"{frame.frame_id}.png"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"[{i + 1}/{len(sample_idx)}] wrote {out_path}")

    print("\n--- Summary across sample (fastest way to spot a collapsed model) ---")
    for name, fracs in total_stuff_fractions.items():
        mean_frac = sum(fracs) / len(fracs) if fracs else 0.0
        print(f"stuff '{name}': mean predicted pixel fraction = {mean_frac:.3f}")
    for name, count in total_thing_detections.items():
        print(f"thing '{name}': {count} detections across {len(sample_idx)} frames ({count / len(sample_idx):.1f}/frame)")
    print(
        "\nIf one stuff class is near 1.0 for every frame, or every thing class shows ~0 "
        "detections/frame, the model likely collapsed — worth lowering --score-threshold "
        "first (to rule out a threshold issue) before concluding the training itself failed."
    )


if __name__ == "__main__":
    main()

"""Command-line entry point: ``uv run panoptic-mining <command> --help``.

Mirrors the command-group style of Jorge's flir-leakage-pipeline CLI
(``uv run flir-pipeline data --help``, ``... features --help``): one Typer
sub-app per pipeline stage, so stages that are implemented and stages that
are still planned are equally discoverable from the same entry point.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import typer

from panoptic_mining.data.manifest import load_yolo_manifest
from panoptic_mining.data.points import box_to_pseudo_mask, sample_interior_points
from panoptic_mining.evaluation import metrics as metrics_module

app = typer.Typer(help="Panoptic segmentation pipeline for illegal mining detection.")
data_app = typer.Typer(help="Dataset manifest loading and integrity checks.")
points_app = typer.Typer(help="Box -> interior points -> pseudo-mask utilities.")
evaluate_app = typer.Typer(help="Scoring: PQ/SQ/RQ, stuff IoU/F1, thing AP, baseline comparability.")
train_app = typer.Typer(help="Training: transfer-learning pretraining and fine-tuning (requires the 'vision' extra).")
app.add_typer(data_app, name="data")
app.add_typer(points_app, name="points")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(train_app, name="train")


@data_app.command("manifest-summary")
def data_manifest_summary(
    data_root: Path = typer.Argument(..., help="Path to the YOLO dataset root (contains dataset.yaml)."),
) -> None:
    """Load a YOLO-format dataset and print per-split/per-class counts plus
    any integrity issues (missing splits, frames without labels) — run
    this before trusting DATA_ROOT for training. Mirrors Jorge's own
    ``manifest-summary`` command in flir-leakage-pipeline."""
    manifest = load_yolo_manifest(data_root)
    summary = manifest.summary()
    typer.echo(json.dumps(summary, indent=2))
    if summary["missing_splits"]:
        typer.echo(f"WARNING: missing splits: {summary['missing_splits']}", err=True)
    for split, stats in summary["per_split"].items():
        if stats["frames_without_labels"] > 0:
            typer.echo(
                f"WARNING: {split} has {stats['frames_without_labels']} frame(s) with no labels file",
                err=True,
            )


@points_app.command("pseudo-mask")
def points_pseudo_mask(
    image_path: Path = typer.Argument(..., help="Path to the source image."),
    box: str = typer.Option(..., help="Box as 'x_min,y_min,x_max,y_max'."),
    n_points: int = typer.Option(5, help="Number of interior points to sample."),
    output_path: Path = typer.Option(Path("pseudo_mask.png"), help="Where to write the mask PNG."),
) -> None:
    """Generate a pseudo-mask for one box annotation on one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise typer.BadParameter(f"Could not read image: {image_path}")

    box_values = tuple(float(v) for v in box.split(","))
    if len(box_values) != 4:
        raise typer.BadParameter("box must have exactly 4 comma-separated values")

    seeds = sample_interior_points(box_values, n_points=n_points)
    result = box_to_pseudo_mask(image, box_values, points=seeds)
    cv2.imwrite(str(output_path), result.mask * 255)
    typer.echo(
        json.dumps(
            {
                "output_path": str(output_path),
                "method": result.method,
                "converged": result.converged,
                "n_points": len(result.points),
                "mask_pixel_count": int(result.mask.sum()),
            }
        )
    )


@evaluate_app.command("panoptic-quality")
def evaluate_panoptic_quality(
    predictions_path: Path = typer.Argument(..., help="Path to a .npz with pred masks by class."),
    ground_truth_path: Path = typer.Argument(..., help="Path to a .npz with GT masks by class."),
    iou_threshold: float = typer.Option(0.5, help="Matching IoU threshold."),
) -> None:
    """Compute per-class and mean PQ/SQ/RQ from saved mask arrays.

    Expects each .npz to contain, per class, a stacked array of shape
    (N_instances, H, W) under a key equal to the class name.
    """
    preds = np.load(predictions_path)
    gts = np.load(ground_truth_path)

    pred_by_class = {k: [preds[k][i] for i in range(preds[k].shape[0])] for k in preds.files}
    gt_by_class = {k: [gts[k][i] for i in range(gts[k].shape[0])] for k in gts.files}

    result = metrics_module.multiclass_panoptic_quality(pred_by_class, gt_by_class, iou_threshold)
    summary = {
        "mean_pq": result.mean_pq,
        "mean_sq": result.mean_sq,
        "mean_rq": result.mean_rq,
        "per_class": {
            cls: {"pq": r.pq, "sq": r.sq, "rq": r.rq, "tp": r.true_positives, "fp": r.false_positives, "fn": r.false_negatives}
            for cls, r in result.per_class.items()
        },
    }
    typer.echo(json.dumps(summary, indent=2))


@train_app.command("pretrain-transfer")
def train_pretrain_transfer(
    landcover_output_dir: Path = typer.Argument(..., help="Path to LandCover.ai's tiled 'output/' dir (produced by its bundled split.py — see docs/training.md), containing {id}.jpg + {id}_m.png pairs."),
    split_file: Path = typer.Option(None, help="Optional train.txt/val.txt/test.txt (from the same zip) to restrict to one split's tile IDs."),
    epochs: int = typer.Option(1, help="Number of pretraining epochs."),
    batch_size: int = typer.Option(4, help="Batch size."),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate."),
    checkpoint_dir: Path = typer.Option(Path("checkpoints/pretrain"), help="Where to save per-epoch checkpoints."),
    device: str = typer.Option("cpu", help="'cpu' or 'cuda'."),
) -> None:
    """Pretrain the shared encoder on LandCover.ai (forest/water/building/road
    aerial imagery) before fine-tuning on our own mining dataset — see
    data/transfer_datasets.py for why this dataset was chosen, per advisor
    feedback to use transfer learning from forest/water segmentation.
    Requires the 'vision' extra (`pip install -e ".[dev,vision]"`)."""
    from panoptic_mining.data.transfer_datasets import (
        LANDCOVER_STUFF_CLASSES,
        LANDCOVER_THING_CLASSES,
        LandCoverAIDataset,
    )
    from panoptic_mining.models.panoptic_fcn import PanopticFCN
    from panoptic_mining.training.train import TrainConfig, train as run_training

    dataset = LandCoverAIDataset(landcover_output_dir, split_file=str(split_file) if split_file else None)
    model = PanopticFCN(
        num_stuff_classes=len(LANDCOVER_STUFF_CLASSES), num_thing_classes=len(LANDCOVER_THING_CLASSES)
    )
    config = TrainConfig(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir, device=device,
    )
    history = run_training(model, dataset, config)
    typer.echo(json.dumps({"final_mean_loss": history.epochs[-1].mean_loss, "epochs_run": len(history.epochs)}))
    typer.echo(
        f"Pretrained encoder checkpoints saved under {checkpoint_dir}. "
        "Load one and pass its encoder into a fresh PanopticFCN(...) with your real class counts "
        "for fine-tuning — see docs/training.md."
    )


@train_app.command("run")
def train_run(
    data_root: Path = typer.Argument(..., help="Path to the YOLO dataset root (contains dataset.yaml)."),
    split: str = typer.Option("train", help="Which split to train on."),
    epochs: int = typer.Option(1, help="Number of epochs."),
    batch_size: int = typer.Option(4, help="Batch size."),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate."),
    checkpoint_dir: Path = typer.Option(Path("checkpoints/finetune"), help="Where to save per-epoch checkpoints."),
    pretrained_checkpoint: Path = typer.Option(None, help="Optional checkpoint from 'train pretrain-transfer' — its encoder weights are loaded, heads are re-initialized for our own class counts."),
    device: str = typer.Option("cpu", help="'cpu' or 'cuda'."),
    context_fusion: bool = typer.Option(False, help="Use ContextFusionPanopticFCN instead of the base model."),
    weight_thing_classes: bool = typer.Option(
        True,
        help="Weight the thing-branch loss by inverse class frequency (vehicle/road get more weight than "
        "building) — see data.torch_dataset.compute_thing_pos_weight and docs/decisions.md (2026-09-21). "
        "Pass --no-weight-thing-classes to restore plain unweighted BCE.",
    ),
) -> None:
    """Fine-tune on our own mining dataset. Requires the 'vision' extra."""
    import torch

    from panoptic_mining.data.manifest import load_yolo_manifest
    from panoptic_mining.data.torch_dataset import (
        DEFAULT_CLASS_SPLIT,
        ManifestSegmentationDataset,
        compute_thing_pos_weight,
    )
    from panoptic_mining.models.context_fusion import ContextFusionPanopticFCN
    from panoptic_mining.models.panoptic_fcn import FeatureEncoder, PanopticFCN
    from panoptic_mining.training.train import TrainConfig, train as run_training

    manifest = load_yolo_manifest(data_root)
    dataset = ManifestSegmentationDataset(manifest, split=split, class_split=DEFAULT_CLASS_SPLIT)

    thing_pos_weight = None
    if weight_thing_classes:
        thing_pos_weight = compute_thing_pos_weight(
            manifest.summary()["class_counts"], DEFAULT_CLASS_SPLIT.thing_class_names
        )
        typer.echo(
            f"Weighting thing classes {DEFAULT_CLASS_SPLIT.thing_class_names} by {thing_pos_weight} "
            "(inverse class frequency from this manifest)."
        )

    encoder = None
    if pretrained_checkpoint is not None:
        checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
        pretrain_model = PanopticFCN(num_stuff_classes=2, num_thing_classes=2)  # LandCover.ai head sizes
        pretrain_model.load_state_dict(checkpoint["model_state_dict"])
        encoder = pretrain_model.encoder
        typer.echo(f"Loaded pretrained encoder from {pretrained_checkpoint} (epoch {checkpoint['epoch']}).")

    model_cls = ContextFusionPanopticFCN if context_fusion else PanopticFCN
    model = model_cls(
        num_stuff_classes=len(DEFAULT_CLASS_SPLIT.stuff_class_names),
        num_thing_classes=len(DEFAULT_CLASS_SPLIT.thing_class_names),
        encoder=encoder or FeatureEncoder(),
    )
    config = TrainConfig(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir, device=device, thing_pos_weight=thing_pos_weight,
    )
    history = run_training(model, dataset, config)
    typer.echo(json.dumps({"final_mean_loss": history.epochs[-1].mean_loss, "epochs_run": len(history.epochs)}))


@app.command("status")
def status() -> None:
    """Print what's implemented vs. planned (see README for detail)."""
    typer.echo(
        "Implemented: data.points (box->points->pseudo-mask), data.manifest (YOLO dataset "
        "loading + integrity checks), data.torch_dataset + data.transfer_datasets (training "
        "targets from our data and from LandCover.ai for transfer learning), "
        "evaluation.metrics (PQ/SQ/RQ, stuff IoU/F1, thing AP, baseline box-equivalence), "
        "models.panoptic_fcn + models.context_fusion (runnable skeletons — placeholder encoder, "
        "class counts pending team/advisor resolution, see docs/decisions.md), "
        "training.train (generic loop: `panoptic-mining train pretrain-transfer` on LandCover.ai, "
        "then `panoptic-mining train run` to fine-tune — see docs/training.md; NOT yet run "
        "end-to-end against real data, this is the harness, not a trained model).\n"
        "Planned: baseline.yolo_eval (blocked on downloading yolov11_best100.pt), "
        "leakage-safe split arm of the evaluation matrix (blocked on Jorge's split), "
        "per-instance mask supervision for the kernel/mask-feature heads (training.train only "
        "supervises the stuff + thing-heatmap heads today).\n"
        "See docs/decisions.md for the open questions this depends on."
    )


if __name__ == "__main__":
    app()

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
app.add_typer(data_app, name="data")
app.add_typer(points_app, name="points")
app.add_typer(evaluate_app, name="evaluate")


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


@app.command("status")
def status() -> None:
    """Print what's implemented vs. planned (see README for detail)."""
    typer.echo(
        "Implemented: data.points (box->points->pseudo-mask), data.manifest (YOLO dataset "
        "loading + integrity checks), evaluation.metrics (PQ/SQ/RQ, stuff IoU/F1, thing AP, "
        "baseline box-equivalence), models.panoptic_fcn + models.context_fusion (runnable "
        "skeletons, not yet trained — placeholder encoder, class counts pending team/advisor "
        "resolution, see docs/decisions.md).\n"
        "Planned: training.train, baseline.yolo_eval (blocked on downloading yolov11_best100.pt), "
        "leakage-safe split arm of the evaluation matrix (blocked on Jorge's split).\n"
        "See docs/decisions.md for the open questions this depends on."
    )


if __name__ == "__main__":
    app()

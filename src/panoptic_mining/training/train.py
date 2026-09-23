"""Generic training loop for ``PanopticFCN`` / ``ContextFusionPanopticFCN``.

Deliberately dataset-agnostic: it trains against any ``Dataset`` yielding
``(image, stuff_target, thing_heatmap_target)`` tuples of the shapes
described below, so the exact same loop is used both to pretrain the
encoder on LandCover.ai (``data/transfer_datasets.py``, forest/water
transfer learning per advisor feedback 2026-09-19) and to fine-tune on our
own mining dataset (``data/torch_dataset.py``) — only the dataset and the
model's head sizes change between the two runs. See
``docs/training.md`` for the full pretrain -> fine-tune workflow.

Losses (both are standard choices, not tuned for this problem yet):
  - stuff branch: ``nn.CrossEntropyLoss(ignore_index=-1)`` against a
    per-pixel class-index target (-1 = no annotation at that pixel).
  - thing branch: ``nn.BCEWithLogitsLoss`` against a Gaussian heatmap
    target in [0, 1] (see ``data/targets.py``).
The kernel/mask-feature heads are NOT supervised by this loop yet — that
needs per-instance mask targets (from ``data/points.py``'s pseudo-masks),
which is a natural next step once this loop is confirmed to run
end-to-end, not implemented here. This is stated plainly rather than
silently training only half the model without saying so.

Status: implemented and runnable (see ``tests/test_train.py``, gated on
``pytest.importorskip("torch")``), but not yet run against real data in
this session — no GPU/long-running job was executed here. Loss curves and
resulting checkpoints are not claimed; this is the harness, not a trained
model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from panoptic_mining.models.panoptic_fcn import PanopticFCN


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-3
    stuff_loss_weight: float = 1.0
    thing_loss_weight: float = 1.0
    device: str = "cpu"
    checkpoint_dir: Path | None = None
    log_every_n_steps: int = 10
    thing_pos_weight: list[float] | None = None  # see compute_losses' docstring


@dataclass
class EpochMetrics:
    epoch: int
    mean_loss: float
    mean_stuff_loss: float
    mean_thing_loss: float
    mean_mask_loss: float = 0.0
    num_batches: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class TrainingHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)


def compute_losses(
    model: PanopticFCN,
    images: torch.Tensor,
    stuff_targets: torch.Tensor,
    thing_heatmap_targets: torch.Tensor,
    thing_masks: torch.Tensor | None = None,
    stuff_loss_weight: float = 1.0,
    thing_loss_weight: float = 1.0,
    mask_loss_weight: float = 0.5,
    thing_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass + loss computation. Returns (total, stuff, thing)
    losses, all still-differentiable scalars (caller does the backward
    pass) so this is reusable from both the training step and any future
    validation step without duplicating the loss definitions.

    ``thing_pos_weight``, if given, is a ``(num_thing_classes,)`` tensor
    passed to ``binary_cross_entropy_with_logits`` — real, rarer thing
    classes (vehicle: 717 boxes vs. building: 4120) get a higher weight
    on missed-positive errors, since the default unweighted BCE lets the
    gradient be dominated by whichever class has the most boxes. Added
    2026-09-21 after `scripts/visualize_predictions.py` showed vehicle
    and road detections had real but very low-confidence signal (see
    docs/decisions.md) — untested against real data as of this change,
    same as every other torch-dependent piece of this project (see
    training/train.py's module docstring)."""
    output = model(images)

    if output.stuff_logits.shape[-2:] != stuff_targets.shape[-2:]:
        raise ValueError(
            f"Model output resolution {tuple(output.stuff_logits.shape[-2:])} doesn't match "
            f"target resolution {tuple(stuff_targets.shape[-2:])} — check that the dataset's "
            "target_stride matches the encoder's actual downsampling factor."
        )

    # nn.functional.cross_entropy(..., ignore_index=-1) divides by the
    # count of non-ignored pixels; if a whole batch has no annotated
    # stuff pixels at all (a real, common case here — most frames have
    # vehicle/building/road boxes but no river/SDZI box, since those are
    # comparatively rare, see docs/decisions.md's class-imbalance note),
    # that count is 0 and the result is NaN (0/0), which then poisons
    # every downstream average. Caught running `train run` against real
    # data (2026-09-19): epoch mean_loss came back NaN. Guard explicitly
    # rather than let a stuff-annotation-free batch silently break
    # training — contribute zero loss (and zero gradient) for that batch
    # instead.
    has_annotated_stuff_pixels = (stuff_targets != -1).any()
    if has_annotated_stuff_pixels:
        stuff_loss = nn.functional.cross_entropy(output.stuff_logits, stuff_targets, ignore_index=-1)
    else:
        stuff_loss = output.stuff_logits.sum() * 0.0  # zero, but keeps the graph/device/dtype

    pos_weight = None
    if thing_pos_weight is not None:
        pos_weight = thing_pos_weight.to(device=output.thing_heatmap.device, dtype=output.thing_heatmap.dtype)
        pos_weight = pos_weight.view(1, -1, 1, 1)  # broadcast over (N, C, H, W)
    thing_loss = nn.functional.binary_cross_entropy_with_logits(
        output.thing_heatmap, thing_heatmap_targets, pos_weight=pos_weight
    )
    
    # Mask loss: supervise kernel/mask-feature heads if masks are provided
    if thing_masks is not None and thing_masks.numel() > 0:
        mask_loss = nn.functional.binary_cross_entropy_with_logits(
            output.mask_features, thing_masks.view_as(output.mask_features)
        )
    else:
        mask_loss = output.mask_features.sum() * 0.0  # zero loss if no masks
    
    total_loss = (
        stuff_loss_weight * stuff_loss 
        + thing_loss_weight * thing_loss 
        + (mask_loss_weight * mask_loss if thing_masks is not None else 0.0)
    )
    return total_loss, stuff_loss, thing_loss, mask_loss


def train(
    model: PanopticFCN,
    train_dataset: Dataset,
    config: TrainConfig,
) -> TrainingHistory:
    """Runs ``config.epochs`` epochs over ``train_dataset``, returning the
    per-epoch loss history. Saves a checkpoint after every epoch to
    ``config.checkpoint_dir / f"epoch_{n}.pt"`` when a checkpoint dir is
    given (creating it if needed)."""
    device = torch.device(config.device)
    model.to(device)
    model.train()

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    thing_pos_weight = (
        torch.tensor(config.thing_pos_weight, dtype=torch.float32) if config.thing_pos_weight is not None else None
    )

    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = TrainingHistory()
    training_start = time.monotonic()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.monotonic()
        total_losses, stuff_losses, thing_losses, mask_losses = [], [], [], []
        for step, batch in enumerate(loader, start=1):
            # Unpack batch (now includes thing_masks)
            if len(batch) == 4:
                images, stuff_targets, thing_heatmap_targets, thing_masks = batch
            else:
                images, stuff_targets, thing_heatmap_targets = batch
                thing_masks = None
            
            images = images.to(device)
            stuff_targets = stuff_targets.to(device)
            thing_heatmap_targets = thing_heatmap_targets.to(device)
            if thing_masks is not None:
                thing_masks = thing_masks.to(device)

            optimizer.zero_grad()
            total_loss, stuff_loss, thing_loss, mask_loss = compute_losses(
                model,
                images,
                stuff_targets,
                thing_heatmap_targets,
                config.stuff_loss_weight,
                config.thing_loss_weight,
                thing_pos_weight=thing_pos_weight,
            )
            total_loss.backward()
            optimizer.step()

            total_losses.append(total_loss.item())
            stuff_losses.append(stuff_loss.item())
            thing_losses.append(thing_loss.item())
            mask_losses.append(mask_loss.item())

            if step % config.log_every_n_steps == 0:
                elapsed = time.monotonic() - epoch_start
                seconds_per_step = elapsed / step
                print(
                    f"epoch {epoch} step {step}: loss={total_loss.item():.4f} "
                    f"({seconds_per_step:.2f}s/step, {elapsed:.0f}s elapsed this epoch)"
                )

        epoch_elapsed = time.monotonic() - epoch_start
        metrics = EpochMetrics(
            epoch=epoch,
            mean_loss=sum(total_losses) / len(total_losses),
            mean_stuff_loss=sum(stuff_losses) / len(stuff_losses),
            mean_thing_loss=sum(thing_losses) / len(thing_losses),
            num_batches=len(total_losses),
            elapsed_seconds=epoch_elapsed,
        )
        history.epochs.append(metrics)
        epochs_remaining = config.epochs - epoch
        eta_seconds = epochs_remaining * epoch_elapsed
        print(
            f"epoch {epoch}/{config.epochs}: mean_loss={metrics.mean_loss:.4f} "
            f"(stuff={metrics.mean_stuff_loss:.4f}, thing={metrics.mean_thing_loss:.4f}) "
            f"-- took {epoch_elapsed / 60:.1f} min, ETA for remaining epochs: {eta_seconds / 60:.1f} min"
        )

        if config.checkpoint_dir is not None:
            checkpoint_path = config.checkpoint_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_loss": metrics.mean_loss,
                },
                checkpoint_path,
            )

    return history

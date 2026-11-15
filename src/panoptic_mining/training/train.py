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
The kernel/mask-feature heads are NOT supervised by this loop — that
needs a per-instance (kernel x mask-feature) loss against per-instance
masks, not implemented yet. SDZI (the thesis target) is a stuff class and
is fully supervised through the stuff branch.

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
    instance_loss_weight: float = 1.0
    device: str = "cpu"
    checkpoint_dir: Path | None = None
    log_every_n_steps: int = 10
    thing_pos_weight: list[float] | None = None  # see compute_losses' docstring
    # Per-class weight for the stuff cross-entropy, in ClassSplit.stuff_class_names
    # order (e.g. [river, SDZI, none] -> [1.0, 4.0, 0.3]). None = unweighted.
    stuff_class_weight: list[float] | None = None
    # Save a checkpoint every N epochs (the last epoch is always saved).
    save_every_n_epochs: int = 1
    seed: int | None = None
    # Resume: first epoch to run (1 = from scratch) and an optional optimizer
    # state dict from the checkpoint being resumed (model weights must already
    # be loaded into the model by the caller).
    start_epoch: int = 1
    optimizer_state: dict | None = None
    # Recorded in the checkpoint so inference.load_model rebuilds the same encoder.
    encoder: str = "placeholder"
    # Learning-rate multiplier for a pretrained backbone (e.g. 0.1 for ResNet-18):
    # the backbone gets learning_rate * encoder_lr_mult, the heads learning_rate.
    encoder_lr_mult: float = 1.0
    num_workers: int = 0
    # Thing heatmap loss: "bce" (original) or "focal" (CenterNet penalty-reduced focal loss).
    heatmap_loss: str = "bce"
    # (height, width) the images are resized to; recorded so inference uses the same size.
    image_size: tuple = (512, 512)
    # "constant" (original) or "cosine": each param group's LR follows a cosine from its
    # initial value at epoch 1 to lr_min_factor * initial at the last epoch, so the final
    # checkpoint is not taken right after a loss spike. Resume-safe (computed from the epoch).
    lr_schedule: str = "constant"
    # Recorded so inference.load_model rebuilds the same kernel head (see PanopticFCN.rel_coords).
    rel_coords: bool = False
    lr_min_factor: float = 0.01


@dataclass
class EpochMetrics:
    epoch: int
    mean_loss: float
    mean_stuff_loss: float
    mean_thing_loss: float
    num_batches: int = 0
    mean_instance_loss: float = 0.0
    elapsed_seconds: float = 0.0


@dataclass
class TrainingHistory:
    epochs: list[EpochMetrics] = field(default_factory=list)


def instance_dice_loss(output, inst_map, inst_ids, inst_centers, mask_logits_fn=None) -> torch.Tensor:
    """Per-instance BCE + Dice (Panoptic FCN kernel x feature loss).

    BCE is added because Dice alone collapsed to empty masks on a one-frame
    overfit test (dice stuck at 0.987); BCE gives the sigmoid a gradient.

    For each thing instance in the frame, its kernel is read at its heatmap
    peak (inst_centers), applied to the shared mask-feature map, and the
    sigmoid mask is compared with the instance's target mask (inst_map == id).
    Instances whose target vanished at feature resolution are skipped.
    """
    feats_all = output.mask_features
    kernels_all = output.kernels
    dices = []
    for b in range(kernels_all.shape[0]):
        valid = inst_ids[b] >= 0
        if not bool(valid.any()):
            continue
        ids = inst_ids[b][valid]
        cs = inst_centers[b][valid]
        kernels = kernels_all[b][:, cs[:, 0], cs[:, 1]].T  # (n, K)
        if mask_logits_fn is not None:
            logits = mask_logits_fn(kernels, feats_all[b], cs)
        else:
            logits = torch.einsum("nk,khw->nhw", kernels, feats_all[b])
        pred = torch.sigmoid(logits)
        target = (inst_map[b].unsqueeze(0) == ids.view(-1, 1, 1)).to(pred.dtype)
        keep = target.flatten(1).sum(1) > 0
        if not bool(keep.any()):
            continue
        logits, target = logits[keep], target[keep]
        pred = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
        inter = (pred * target).flatten(1).sum(1)
        denom = pred.flatten(1).sum(1) + target.flatten(1).sum(1)
        dices.append(bce + 1.0 - (2.0 * inter + 1.0) / (denom + 1.0))
    if not dices:
        return feats_all.sum() * 0.0
    return torch.cat(dices).mean()


def centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (Zhou et al., CenterNet, 2019).
    Positives are the cells where the target equals 1 (needs integer_center heatmaps);
    negatives near a center are down-weighted by (1 - target)^beta. Normalized by the
    number of positives, so sparse heatmaps are not dominated by background."""
    pred = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1.0 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - target) ** beta * neg
    num_pos = pos.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos.clamp(min=1.0)


def compute_losses(
    model: PanopticFCN,
    images: torch.Tensor,
    stuff_targets: torch.Tensor,
    thing_heatmap_targets: torch.Tensor,
    stuff_loss_weight: float = 1.0,
    thing_loss_weight: float = 1.0,
    *,
    thing_pos_weight: torch.Tensor | None = None,
    stuff_class_weight: torch.Tensor | None = None,
    instance_targets: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    instance_loss_weight: float = 1.0,
    heatmap_loss: str = "bce",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One forward pass + loss computation. Returns (total, stuff, thing),
    all still-differentiable scalars (caller does the backward pass).

    Everything after ``thing_loss_weight`` is keyword-only on purpose: an
    earlier version accepted ``thing_masks`` as the 5th positional argument
    and ``train()`` passed ``config.stuff_loss_weight`` into it by position,
    which crashed on the first batch (2026-10-08 review). That mask loss was
    also not the Panoptic FCN mask loss (it supervised the raw 64-channel
    mask-feature map, so the kernel head never got gradient) and has been
    removed until a proper per-instance kernel x feature loss is written.

    ``thing_pos_weight``: ``(num_thing_classes,)`` tensor for
    ``binary_cross_entropy_with_logits`` — rarer thing classes get a higher
    weight on missed positives.

    ``stuff_class_weight``: ``(num_stuff_classes,)`` tensor for the stuff
    cross-entropy. Used to make missing SDZI more expensive than mislabeling
    background, since the annotations are known to be incomplete (unannotated
    SDZI is labeled "none"). Lowering the "none" weight raises recall and
    lowers precision — report both.
    """
    output = model(images)

    if output.stuff_logits.shape[-2:] != stuff_targets.shape[-2:]:
        raise ValueError(
            f"Model output resolution {tuple(output.stuff_logits.shape[-2:])} doesn't match "
            f"target resolution {tuple(stuff_targets.shape[-2:])} — check that the dataset's "
            "target_stride matches the encoder's actual downsampling factor."
        )

    # A batch with no annotated stuff pixels at all makes cross_entropy
    # (ignore_index=-1) divide by zero -> NaN. Contribute an exact zero instead.
    has_annotated_stuff_pixels = (stuff_targets != -1).any()
    if has_annotated_stuff_pixels:
        weight = None
        if stuff_class_weight is not None:
            weight = stuff_class_weight.to(device=output.stuff_logits.device, dtype=output.stuff_logits.dtype)
        stuff_loss = nn.functional.cross_entropy(
            output.stuff_logits, stuff_targets, weight=weight, ignore_index=-1
        )
    else:
        stuff_loss = output.stuff_logits.sum() * 0.0

    pos_weight = None
    if thing_pos_weight is not None:
        pos_weight = thing_pos_weight.to(device=output.thing_heatmap.device, dtype=output.thing_heatmap.dtype)
        pos_weight = pos_weight.view(1, -1, 1, 1)
    if heatmap_loss == "focal":
        thing_loss = centernet_focal_loss(output.thing_heatmap, thing_heatmap_targets)
    else:
        thing_loss = nn.functional.binary_cross_entropy_with_logits(
            output.thing_heatmap, thing_heatmap_targets, pos_weight=pos_weight
        )

    if instance_targets is not None:
        inst_map, inst_ids, inst_centers = instance_targets
        instance_loss = instance_dice_loss(output, inst_map, inst_ids, inst_centers,
                                           mask_logits_fn=getattr(model, "instance_mask_logits", None))
    else:
        instance_loss = output.mask_features.sum() * 0.0

    total_loss = (stuff_loss_weight * stuff_loss + thing_loss_weight * thing_loss
                  + instance_loss_weight * instance_loss)
    return total_loss, stuff_loss, thing_loss, instance_loss


def train(
    model: PanopticFCN,
    train_dataset: Dataset,
    config: TrainConfig,
) -> TrainingHistory:
    """Runs ``config.epochs`` epochs over ``train_dataset`` and returns the
    per-epoch loss history. Checkpoints go to
    ``config.checkpoint_dir / f"epoch_{n}.pt"`` every
    ``config.save_every_n_epochs`` epochs and always at the last epoch.
    The dataset must yield ``(image, stuff_target, thing_heatmap)`` tuples;
    extra trailing items (e.g. thing masks) are ignored."""
    if config.seed is not None:
        torch.manual_seed(config.seed)
    device = torch.device(config.device)
    model.to(device)
    model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    backbone = getattr(model.encoder, "backbone_parameters", None)
    if backbone is not None and config.encoder_lr_mult != 1.0:
        backbone_ids = {id(p) for p in backbone()}
        param_groups = [
            {"params": [p for p in model.parameters() if id(p) in backbone_ids],
             "lr": config.learning_rate * config.encoder_lr_mult},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": config.learning_rate},
        ]
    else:
        param_groups = [{"params": list(model.parameters()), "lr": config.learning_rate}]
    optimizer = torch.optim.Adam(param_groups, lr=config.learning_rate)
    base_lrs = [g["lr"] for g in optimizer.param_groups]  # before loading a resumed state
    if config.optimizer_state is not None:
        optimizer.load_state_dict(config.optimizer_state)
    thing_pos_weight = (
        torch.tensor(config.thing_pos_weight, dtype=torch.float32) if config.thing_pos_weight is not None else None
    )
    stuff_class_weight = (
        torch.tensor(config.stuff_class_weight, dtype=torch.float32) if config.stuff_class_weight is not None else None
    )

    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = TrainingHistory()
    for epoch in range(config.start_epoch, config.epochs + 1):
        epoch_start = time.monotonic()
        if config.lr_schedule == "cosine":
            import math
            progress = (epoch - 1) / max(1, config.epochs - 1)
            factor = config.lr_min_factor + (1 - config.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            for g, base in zip(optimizer.param_groups, base_lrs):
                g["lr"] = base * factor
        total_losses, stuff_losses, thing_losses, instance_losses = [], [], [], []
        for step, batch in enumerate(loader, start=1):
            images, stuff_targets, thing_heatmap_targets = batch[0], batch[1], batch[2]
            instance_targets = None
            if len(batch) == 6:
                instance_targets = tuple(t.to(device, non_blocking=True) for t in batch[3:6])
            images = images.to(device, non_blocking=True)
            stuff_targets = stuff_targets.to(device, non_blocking=True)
            thing_heatmap_targets = thing_heatmap_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            total_loss, stuff_loss, thing_loss, instance_loss = compute_losses(
                model,
                images,
                stuff_targets,
                thing_heatmap_targets,
                stuff_loss_weight=config.stuff_loss_weight,
                thing_loss_weight=config.thing_loss_weight,
                thing_pos_weight=thing_pos_weight,
                stuff_class_weight=stuff_class_weight,
                instance_targets=instance_targets,
                instance_loss_weight=config.instance_loss_weight,
                heatmap_loss=config.heatmap_loss,
            )
            total_loss.backward()
            optimizer.step()

            total_losses.append(total_loss.item())
            stuff_losses.append(stuff_loss.item())
            thing_losses.append(thing_loss.item())
            instance_losses.append(instance_loss.item())

            if step % config.log_every_n_steps == 0:
                elapsed = time.monotonic() - epoch_start
                print(
                    f"epoch {epoch} step {step}: loss={total_loss.item():.4f} "
                    f"({elapsed / step:.2f}s/step, {elapsed:.0f}s elapsed this epoch)",
                    flush=True,
                )

        epoch_elapsed = time.monotonic() - epoch_start
        metrics = EpochMetrics(
            epoch=epoch,
            mean_loss=sum(total_losses) / len(total_losses),
            mean_stuff_loss=sum(stuff_losses) / len(stuff_losses),
            mean_thing_loss=sum(thing_losses) / len(thing_losses),
            mean_instance_loss=sum(instance_losses) / len(instance_losses),
            num_batches=len(total_losses),
            elapsed_seconds=epoch_elapsed,
        )
        history.epochs.append(metrics)
        eta_seconds = (config.epochs - epoch) * epoch_elapsed
        print(
            f"epoch {epoch}/{config.epochs}: mean_loss={metrics.mean_loss:.4f} "
            f"(stuff={metrics.mean_stuff_loss:.4f}, thing={metrics.mean_thing_loss:.4f}) "
            f"-- took {epoch_elapsed / 60:.1f} min, ETA {eta_seconds / 60:.1f} min",
            flush=True,
        )

        is_last = epoch == config.epochs
        if config.checkpoint_dir is not None and (is_last or epoch % config.save_every_n_epochs == 0):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "mean_loss": metrics.mean_loss,
                    "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in config.__dict__.items() if k != "optimizer_state"},
                },
                config.checkpoint_dir / f"epoch_{epoch}.pt",
            )

    return history

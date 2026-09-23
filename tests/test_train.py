"""Unit tests for training/train.py — a tiny synthetic in-memory dataset,
one epoch, CPU only. Not a claim that training converges to anything
useful; just that the loop runs, losses are finite, and a checkpoint gets
written where expected."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset  # noqa: E402

from panoptic_mining.models.panoptic_fcn import FeatureEncoder, PanopticFCN  # noqa: E402
from panoptic_mining.training.train import TrainConfig, compute_losses, train  # noqa: E402


class _TinySyntheticDataset(Dataset):
    """4 samples of 32x32 RGB images with random stuff/thing targets at
    the encoder's 1/8 resolution (4x4) — big enough to exercise batching
    without being slow."""

    def __init__(self, num_stuff_classes=2, num_thing_classes=3, num_samples=4):
        self.num_samples = num_samples
        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_thing_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.rand(3, 32, 32)
        stuff_target = torch.randint(0, self.num_stuff_classes, (4, 4))
        thing_heatmap = torch.rand(self.num_thing_classes, 4, 4)
        return image, stuff_target, thing_heatmap


def _tiny_model(num_stuff_classes=2, num_thing_classes=3):
    encoder = FeatureEncoder(in_channels=3, base_channels=4)
    return PanopticFCN(
        num_stuff_classes=num_stuff_classes, num_thing_classes=num_thing_classes,
        kernel_dim=8, encoder=encoder,
    )


def test_compute_losses_returns_finite_scalars():
    model = _tiny_model()
    images = torch.rand(2, 3, 32, 32)
    stuff_targets = torch.randint(0, 2, (2, 4, 4))
    thing_heatmap_targets = torch.rand(2, 3, 4, 4)

    total, stuff, thing = compute_losses(model, images, stuff_targets, thing_heatmap_targets, 1.0, 1.0)
    assert torch.isfinite(total)
    assert torch.isfinite(stuff)
    assert torch.isfinite(thing)


def test_compute_losses_ignores_minus_one_stuff_targets():
    model = _tiny_model()
    images = torch.rand(1, 3, 32, 32)
    stuff_targets = torch.full((1, 4, 4), fill_value=-1, dtype=torch.long)  # all "no annotation"
    thing_heatmap_targets = torch.zeros(1, 3, 4, 4)

    # Should not raise despite every pixel being ignore_index.
    total, stuff, thing = compute_losses(model, images, stuff_targets, thing_heatmap_targets, 1.0, 1.0)
    assert torch.isfinite(total)


def test_compute_losses_whole_batch_with_no_stuff_annotations_is_zero_not_nan():
    """Regression test: when EVERY sample in a batch has an all -1 stuff
    target (a real, common case — most frames have vehicle/building/road
    boxes but no river/SDZI box, since those are comparatively rare, see
    docs/decisions.md), nn.functional.cross_entropy(ignore_index=-1)
    divides by zero non-ignored pixels and returns NaN, which then
    poisons the epoch's mean loss. Caught running `train run` against
    real data (2026-09-19: 'epoch 1/1: mean_loss=nan'). The stuff loss
    for such a batch should come back as an exact, finite zero instead."""
    model = _tiny_model()
    images = torch.rand(3, 3, 32, 32)
    stuff_targets = torch.full((3, 4, 4), fill_value=-1, dtype=torch.long)
    thing_heatmap_targets = torch.rand(3, 3, 4, 4)

    total, stuff, thing = compute_losses(model, images, stuff_targets, thing_heatmap_targets, 1.0, 1.0)

    assert torch.isfinite(total)
    assert stuff.item() == 0.0
    assert torch.isfinite(thing)


def test_train_does_not_produce_nan_when_some_batches_lack_stuff_annotations():
    """End-to-end regression test for the same NaN bug, exercised through
    the full train() loop rather than just compute_losses directly — some
    samples have real stuff targets, some are all -1, mimicking a real
    dataset where most frames lack a river/SDZI box."""

    class _MixedAnnotationDataset(Dataset):
        def __len__(self):
            return 6

        def __getitem__(self, idx):
            image = torch.rand(3, 32, 32)
            if idx % 2 == 0:
                stuff_target = torch.full((4, 4), fill_value=-1, dtype=torch.long)  # no stuff boxes
            else:
                stuff_target = torch.randint(0, 2, (4, 4))
            thing_heatmap = torch.rand(3, 4, 4)
            return image, stuff_target, thing_heatmap

    model = _tiny_model()
    history = train(model, _MixedAnnotationDataset(), TrainConfig(epochs=1, batch_size=3, device="cpu"))

    assert not (history.epochs[0].mean_loss != history.epochs[0].mean_loss)  # NaN != NaN
    assert not (history.epochs[0].mean_stuff_loss != history.epochs[0].mean_stuff_loss)


def test_compute_losses_raises_on_resolution_mismatch():
    model = _tiny_model()
    images = torch.rand(1, 3, 32, 32)
    wrong_size_target = torch.randint(0, 2, (1, 999, 999))
    thing_heatmap_targets = torch.rand(1, 3, 4, 4)
    with pytest.raises(ValueError):
        compute_losses(model, images, wrong_size_target, thing_heatmap_targets, 1.0, 1.0)


def test_train_runs_one_epoch_and_returns_history():
    model = _tiny_model()
    dataset = _TinySyntheticDataset()
    config = TrainConfig(epochs=1, batch_size=2, device="cpu")

    history = train(model, dataset, config)

    assert len(history.epochs) == 1
    assert history.epochs[0].num_batches == 2  # 4 samples / batch_size 2
    assert history.epochs[0].mean_loss >= 0.0


def test_train_saves_checkpoint_per_epoch(tmp_path):
    model = _tiny_model()
    dataset = _TinySyntheticDataset()
    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainConfig(epochs=2, batch_size=2, device="cpu", checkpoint_dir=checkpoint_dir)

    train(model, dataset, config)

    assert (checkpoint_dir / "epoch_1.pt").exists()
    assert (checkpoint_dir / "epoch_2.pt").exists()
    checkpoint = torch.load(checkpoint_dir / "epoch_2.pt", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert "model_state_dict" in checkpoint


def test_train_updates_model_weights():
    model = _tiny_model()
    dataset = _TinySyntheticDataset()
    before = model.stuff_head.weight.clone()

    train(model, dataset, TrainConfig(epochs=1, batch_size=2, device="cpu"))

    after = model.stuff_head.weight
    assert not torch.equal(before, after)


def test_compute_losses_with_thing_pos_weight_penalizes_missed_rare_class_more():
    """Regression test for the 2026-09-21 class-weighting fix (see
    docs/decisions.md): with a heavily skewed pos_weight, a missed
    positive in the up-weighted (rare) class should contribute more loss
    than the same miss with no weighting at all."""
    model = _tiny_model()
    images = torch.rand(1, 3, 32, 32)
    stuff_targets = torch.randint(0, 2, (1, 4, 4))
    # Class 0 has a strong positive (a real box) the model isn't already
    # predicting; classes 1/2 are all-zero (nothing to miss there).
    thing_heatmap_targets = torch.zeros(1, 3, 4, 4)
    thing_heatmap_targets[0, 0, 2, 2] = 1.0

    _, _, thing_loss_unweighted = compute_losses(model, images, stuff_targets, thing_heatmap_targets, 1.0, 1.0)
    _, _, thing_loss_weighted = compute_losses(
        model, images, stuff_targets, thing_heatmap_targets, 1.0, 1.0,
        thing_pos_weight=torch.tensor([10.0, 1.0, 1.0]),
    )

    assert thing_loss_weighted > thing_loss_unweighted


def test_train_accepts_thing_pos_weight_from_config():
    """End-to-end smoke test: TrainConfig.thing_pos_weight should flow
    through train() without raising, and should actually change the loss
    trajectory (not just be silently ignored)."""
    model_a = _tiny_model()
    model_b = _tiny_model()
    dataset = _TinySyntheticDataset()

    torch.manual_seed(0)
    history_unweighted = train(model_a, dataset, TrainConfig(epochs=1, batch_size=2, device="cpu"))
    torch.manual_seed(0)
    history_weighted = train(
        model_b, dataset, TrainConfig(epochs=1, batch_size=2, device="cpu", thing_pos_weight=[5.0, 1.0, 1.0])
    )

    assert torch.isfinite(torch.tensor(history_weighted.epochs[0].mean_thing_loss))
    # Different weighting should produce a different loss trajectory, not
    # be silently ignored.
    assert history_unweighted.epochs[0].mean_thing_loss != history_weighted.epochs[0].mean_thing_loss

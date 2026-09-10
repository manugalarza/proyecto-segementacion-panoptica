"""Unit tests for models/panoptic_fcn.py.

These require torch (the ``vision`` extra). They're skipped automatically
in environments without it (e.g. this sandbox) via ``pytest.importorskip``,
but run for real once ``pip install -e ".[dev,vision]"`` is done locally.
All inputs are tiny synthetic tensors — the point is to check shapes and
wiring, not model quality.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from panoptic_mining.models.panoptic_fcn import (  # noqa: E402
    FeatureEncoder,
    PanopticFCN,
    PanopticFCNOutput,
)


def _tiny_model(num_stuff_classes=2, num_thing_classes=3, kernel_dim=8, base_channels=4):
    encoder = FeatureEncoder(in_channels=3, base_channels=base_channels)
    return PanopticFCN(
        num_stuff_classes=num_stuff_classes,
        num_thing_classes=num_thing_classes,
        kernel_dim=kernel_dim,
        encoder=encoder,
    )


def test_feature_encoder_downsamples_by_eight():
    encoder = FeatureEncoder(in_channels=3, base_channels=4)
    images = torch.zeros(2, 3, 64, 64)
    feats = encoder(images)
    assert feats.shape == (2, 16, 8, 8)  # base_channels * 4 channels, /8 spatial


def test_panoptic_fcn_forward_output_shapes():
    model = _tiny_model(num_stuff_classes=2, num_thing_classes=3, kernel_dim=8, base_channels=4)
    images = torch.zeros(2, 3, 64, 64)
    output = model(images)

    assert isinstance(output, PanopticFCNOutput)
    assert output.stuff_logits.shape == (2, 2, 8, 8)
    assert output.thing_heatmap.shape == (2, 3, 8, 8)
    assert output.kernels.shape == (2, 8, 8, 8)
    assert output.mask_features.shape == (2, 8, 8, 8)


def test_panoptic_fcn_forward_is_deterministic_in_eval_mode():
    model = _tiny_model()
    model.eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        out1 = model(images)
        out2 = model(images)
    assert torch.equal(out1.stuff_logits, out2.stuff_logits)
    assert torch.equal(out1.thing_heatmap, out2.thing_heatmap)


def test_decode_instances_rejects_batch_size_greater_than_one():
    model = _tiny_model()
    images = torch.zeros(2, 3, 32, 32)
    output = model(images)
    with pytest.raises(ValueError):
        model.decode_instances(output, class_id=0)


def test_decode_instances_returns_masks_matching_encoder_resolution():
    model = _tiny_model(kernel_dim=8, base_channels=4)
    model.eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(images)
        # Force at least one peak above threshold regardless of random init.
        output.thing_heatmap[0, 0, 2, 2] = 10.0
        instances = model.decode_instances(output, class_id=0, top_k=5, score_threshold=0.3)

    assert len(instances) >= 1
    score, mask = instances[0]
    assert 0.0 <= score <= 1.0
    assert mask.shape == output.mask_features.shape[-2:]
    assert torch.all((mask >= 0.0) & (mask <= 1.0))


def test_decode_instances_respects_top_k():
    model = _tiny_model(kernel_dim=8, base_channels=4)
    model.eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        output = model(images)
        output.thing_heatmap[0, 0] = 10.0  # every location above threshold
        instances = model.decode_instances(output, class_id=0, top_k=3, score_threshold=0.3)

    assert len(instances) <= 3

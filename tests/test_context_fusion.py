"""Unit tests for models/context_fusion.py.

Requires torch — see test_panoptic_fcn.py's docstring for the
importorskip pattern used here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from panoptic_mining.models.context_fusion import ContextFusionPanopticFCN  # noqa: E402
from panoptic_mining.models.panoptic_fcn import FeatureEncoder, PanopticFCNOutput  # noqa: E402


def _tiny_fusion_model(fusion_channels=4, base_channels=4, kernel_dim=8):
    encoder = FeatureEncoder(in_channels=3, base_channels=base_channels)
    return ContextFusionPanopticFCN(
        num_stuff_classes=2,
        num_thing_classes=3,
        kernel_dim=kernel_dim,
        encoder=encoder,
        fusion_channels=fusion_channels,
    )


def test_context_fusion_forward_output_shapes():
    model = _tiny_fusion_model()
    images = torch.zeros(2, 3, 64, 64)
    output = model(images)

    assert isinstance(output, PanopticFCNOutput)
    assert output.stuff_logits.shape == (2, 2, 8, 8)
    assert output.thing_heatmap.shape == (2, 3, 8, 8)
    assert output.kernels.shape == (2, 8, 8, 8)
    assert output.mask_features.shape == (2, 8, 8, 8)


def test_context_fusion_uses_second_pass_heads_not_base_heads():
    """The second-pass heads take feat_channels + fusion_channels inputs,
    so if forward() accidentally routed through the base (first-pass) heads
    instead, this would raise a shape-mismatch error rather than silently
    passing — this test exists to make that failure mode explicit."""
    model = _tiny_fusion_model(fusion_channels=4, base_channels=4)
    assert model.stuff_head2.in_channels == model.feat_channels + model.fusion_channels
    assert model.thing_heatmap_head2.in_channels == model.feat_channels + model.fusion_channels

    images = torch.randn(1, 3, 32, 32)
    output = model(images)
    assert output.stuff_logits.shape[1] == model.num_stuff_classes
    assert output.thing_heatmap.shape[1] == model.num_thing_classes


def test_context_fusion_gradients_flow_through_both_branches():
    """Sanity check that the fusion cross-connections are actually wired
    into the graph (not detached) — a broken fusion path would leave one
    branch's context conv with no gradient."""
    model = _tiny_fusion_model()
    images = torch.randn(1, 3, 32, 32)
    output = model(images)
    loss = output.stuff_logits.sum() + output.thing_heatmap.sum()
    loss.backward()

    assert model.stuff_from_things.weight.grad is not None
    assert model.thing_from_stuff.weight.grad is not None
    assert not torch.all(model.stuff_from_things.weight.grad == 0)
    assert not torch.all(model.thing_from_stuff.weight.grad == 0)


def test_context_fusion_differs_from_base_first_pass():
    """The second-pass stuff logits should generally differ from what a
    plain first-pass head would produce on the same features, since they
    see extra context channels — regression guard against forward()
    accidentally short-circuiting to the first pass."""
    model = _tiny_fusion_model()
    model.eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        feats = model.encoder(images)
        first_pass_stuff = model.stuff_head(feats)
        output = model(images)

    assert output.stuff_logits.shape == first_pass_stuff.shape
    assert not torch.allclose(output.stuff_logits, first_pass_stuff)

"""Base Panoptic FCN architecture.

Status: PLANNED — not yet implemented.

Blocked on:
  - Access to Imagenes.zip / Etiquetas.zip (see .env.example / README data
    consumption section) to determine actual input resolution, channel
    count, and class distribution before committing to a backbone config.
  - Confirmation of the RGB-vs-thermal question (docs/decisions.md); the
    input stem (3-channel vs 1-channel, normalization stats) depends on it.

Planned interface, so downstream code (training loop, ablation variant in
context_fusion.py) can be written against a stable contract before the
model itself exists:

    class PanopticFCN(nn.Module):
        def __init__(self, num_stuff_classes: int, num_thing_classes: int, backbone: str = "resnet50"): ...
        def forward(self, images: Tensor) -> PanopticFCNOutput: ...

PanopticFCNOutput is expected to expose:
  - stuff_logits: (B, num_stuff_classes, H, W)
  - thing_kernels: per-instance convolution kernels generated from point
    features (Li et al., "Fully Convolutional Networks for Panoptic
    Segmentation")
  - thing_masks: (B, N_instances, H, W) after kernel application

See docs/architecture.md for the full design and how this differs from
the context-fusion ablation in context_fusion.py.
"""

raise NotImplementedError(
    "PanopticFCN is not implemented yet — see module docstring for blockers. "
    "data/points.py and evaluation/metrics.py are ready and unit-tested "
    "ahead of this, so the pseudo-mask and scoring machinery is available "
    "as soon as training data is."
)

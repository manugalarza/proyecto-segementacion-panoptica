"""Base Panoptic FCN architecture.

Implements the mechanism from Li et al., "Fully Convolutional Networks for
Panoptic Segmentation" (CVPR 2021): a semantic branch for stuff classes,
and a kernel-generator branch for things — at each spatial location, a
per-instance convolution kernel is predicted and then applied (as a 1x1
conv, i.e. a per-pixel dot product) to a shared mask-feature map to
produce that instance's mask. This is why the point-supervision pipeline
in ``data/points.py`` matters: kernel generation is naturally driven by
point-level supervision, which is exactly what box annotations get
converted into.

Status: runnable skeleton, not yet trained or tuned. Two things are
explicitly deferred rather than guessed at:

  1. ``FeatureEncoder`` here is a small plain conv trunk, not a real
     ResNet+FPN backbone. It exists so the architecture is testable end to
     end on CPU without pulling in a large pretrained network before
     there's a training budget/compute plan to justify the choice. Swap
     it for ``torchvision.models.detection.backbone_utils.resnet_fpn_backbone``
     (or similar) once that's decided — the rest of the model only
     depends on getting a single feature map of shape
     ``(B, feat_channels, H', W')`` out of it.
  2. ``num_stuff_classes`` / ``num_thing_classes`` defaults below (2 and
     3) match what's actually annotated in the downloaded
     ``dataset_split_completo`` (stuff: river, SDZI; things: vehicle,
     building, road) — see ``data/manifest.py`` output. This does NOT
     match the 6 thing categories listed in the thesis objectives
     (machinery, dredges, camps, structures, containers, vehicles) or the
     antecedentes' 5-vs-6 class framing; only 3 thing classes are actually
     labeled in what's been downloaded so far. This mismatch is tracked
     in docs/decisions.md and needs resolving with the team/advisor before
     it's load-bearing for anything beyond this skeleton.

Both are meant to be reconfigured, not silently trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PanopticFCNOutput:
    stuff_logits: torch.Tensor  # (B, num_stuff_classes, H', W') — pre-softmax
    thing_heatmap: torch.Tensor  # (B, num_thing_classes, H', W') — pre-sigmoid
    kernels: torch.Tensor  # (B, kernel_dim, H', W')
    mask_features: torch.Tensor  # (B, kernel_dim, H', W')


class FeatureEncoder(nn.Module):
    """Lightweight conv trunk producing one feature map at 1/8 input
    resolution. See module docstring point 1 — this is a placeholder for
    a real backbone, kept small so the rest of the architecture can be
    exercised (including on CPU, in unit tests) before a backbone choice
    is finalized."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.out_channels = base_channels * 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.stem(images)


class ResNetFPNEncoder(nn.Module):
    """ResNet-18 (torchvision) + a minimal FPN top-down path, producing ONE
    feature map at 1/8 input resolution so it is a drop-in replacement for
    ``FeatureEncoder`` (same stride, so the dataset's ``target_stride=8``
    and every head stay unchanged). Added 2026-10-14 after the 2x2
    experiment showed the placeholder trunk could not learn even river.

    Input convention is unchanged: BGR in [0, 1] (see ``inference.py``).
    The ImageNet channel flip (BGR -> RGB) and mean/std normalization are
    done here, inside the model, so training and inference preprocessing
    stay identical by construction.

    ``weights``: "imagenet" (torchvision IMAGENET1K_V1; downloaded to the
    torch hub cache, or found there if copied by hand), a path to a
    ``resnet18-*.pth`` state dict, or None (random init, for tests).
    """

    def __init__(self, weights: str | None = "imagenet", out_channels: int = 128):
        super().__init__()
        from torchvision.models import resnet18

        if weights == "imagenet":
            from torchvision.models import ResNet18_Weights

            net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            net = resnet18(weights=None)
            if weights is not None:
                net.load_state_dict(torch.load(str(weights), map_location="cpu"))
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)  # 1/4
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4
        self.lat2 = nn.Conv2d(128, out_channels, 1)  # 1/8
        self.lat3 = nn.Conv2d(256, out_channels, 1)  # 1/16
        self.lat4 = nn.Conv2d(512, out_channels, 1)  # 1/32
        self.smooth = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True))
        self.out_channels = out_channels
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def backbone_parameters(self):
        for m in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            yield from m.parameters()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images.flip(1) - self.mean) / self.std  # BGR -> RGB, ImageNet normalization
        c2 = self.layer2(self.layer1(self.stem(x)))
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        return self.smooth(p2)


def build_encoder(name: str = "placeholder", resnet_weights: str | None = "imagenet") -> nn.Module:
    """``name`` is stored in every checkpoint's config so ``inference.load_model``
    can rebuild the same architecture."""
    if name == "placeholder":
        return FeatureEncoder()
    if name == "resnet18":
        return ResNetFPNEncoder(weights=resnet_weights)
    raise ValueError(f"Unknown encoder {name!r} (expected 'placeholder' or 'resnet18')")


class PanopticFCN(nn.Module):
    """Base model: independent stuff (semantic) and thing (kernel-based
    instance) branches over a shared encoder. See ``context_fusion.py``
    for the ablation that lets these branches inform each other."""

    def __init__(
        self,
        num_stuff_classes: int = 3,  # river, SDZI, none (explicit background — see docs/decisions.md, 2026-09-20)
        num_thing_classes: int = 3,  # vehicle, building, road — see module docstring point 2
        kernel_dim: int = 64,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder or FeatureEncoder()
        feat_channels = self.encoder.out_channels

        self.num_stuff_classes = num_stuff_classes
        self.num_thing_classes = num_thing_classes
        self.kernel_dim = kernel_dim
        self.feat_channels = feat_channels

        self.stuff_head = nn.Conv2d(feat_channels, num_stuff_classes, kernel_size=1)
        self.thing_heatmap_head = nn.Conv2d(feat_channels, num_thing_classes, kernel_size=1)
        self.kernel_head = nn.Conv2d(feat_channels, kernel_dim, kernel_size=1)
        self.mask_feature_head = nn.Sequential(
            nn.Conv2d(feat_channels, kernel_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_dim, kernel_dim, kernel_size=3, padding=1),
        )

    def forward(self, images: torch.Tensor) -> PanopticFCNOutput:
        feats = self.encoder(images)
        return PanopticFCNOutput(
            stuff_logits=self.stuff_head(feats),
            thing_heatmap=self.thing_heatmap_head(feats),
            kernels=self.kernel_head(feats),
            mask_features=self.mask_feature_head(feats),
        )

    def decode_instances(
        self,
        output: PanopticFCNOutput,
        class_id: int,
        top_k: int = 20,
        score_threshold: float = 0.3,
    ) -> list[tuple[float, torch.Tensor]]:
        """Decode instance masks for one thing class from its heatmap
        peaks, following Panoptic FCN's kernel mechanism: each selected
        location's kernel vector is applied to the shared mask-feature map
        (a per-pixel dot product) to produce that instance's mask.

        Operates on a single image (``output`` must have batch size 1).
        Peak selection is a simple max-pool NMS — a placeholder for
        whatever peak-selection/NMS strategy training experiments settle
        on, not a claim that this is the final approach.

        Returns a list of ``(score, mask)`` where ``mask`` is an
        ``(H', W')`` tensor of sigmoid probabilities (same spatial
        resolution as the encoder output, i.e. NOT upsampled to input
        resolution — that's the caller's job once a target size is known).
        """
        if output.thing_heatmap.shape[0] != 1:
            raise ValueError("decode_instances operates on one image at a time (batch size 1)")

        heatmap = torch.sigmoid(output.thing_heatmap[0, class_id])  # (H', W')
        pooled = F.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)
        pooled = pooled[0, 0]
        is_peak = (heatmap == pooled) & (heatmap > score_threshold)
        ys, xs = torch.nonzero(is_peak, as_tuple=True)
        scores = heatmap[ys, xs]

        if len(scores) > top_k:
            top_scores, top_idx = torch.topk(scores, top_k)
            ys, xs, scores = ys[top_idx], xs[top_idx], top_scores

        mask_features = output.mask_features[0]  # (K, H', W')
        instances = []
        for y, x, score in zip(ys.tolist(), xs.tolist(), scores.tolist()):
            kernel = output.kernels[0, :, y, x]  # (K,)
            mask_logits = torch.einsum("k,khw->hw", kernel, mask_features)
            instances.append((score, torch.sigmoid(mask_logits)))
        return instances

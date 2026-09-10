"""Context-fusion ablation on top of the base Panoptic FCN.

Motivation (see ``docs/architecture.md``): SDZI (Soil Disturbance Zones of
Interest) is texturally ambiguous — disturbed earth often has no strong
color/texture separation from surrounding terrain, which is exactly the
failure mode that showed up as a real GrabCut limitation in
``data/points.py`` (mask collapsing to a handful of pixels for SDZI boxes).
The hypothesis this ablation tests: nearby *thing* detections (machinery,
dredges, vehicles) are strong indirect evidence of *stuff* disturbance, and
vice versa (a detected disturbed-earth region raises confidence that a
nearby faint detection is real mining equipment, not noise). This is the
project's architectural contribution beyond a stock Panoptic FCN — the
comparison between this class and the base ``PanopticFCN`` is the ablation
the evaluation plan (docs/methodology.md, step 4) is built around.

Mechanism: run the base heads once to get first-pass stuff logits and thing
heatmaps, project each into a small number of "context" channels with a 1x1
conv, concatenate that context onto the shared encoder features, and run a
second, fusion-aware set of heads on the concatenated tensor. This keeps the
fusion cheap (1x1 convs) and keeps the first-pass heads from the base class
reusable as-is — nothing about ``PanopticFCN.__init__`` needs to change for
this subclass to work.

Status: runnable skeleton, not yet trained or tuned — same caveats as
``panoptic_fcn.py`` (placeholder encoder, class counts pending team/advisor
resolution). Whether this actually helps SDZI recall is an empirical
question for the ablation, not something to assume from the design alone.
"""

from __future__ import annotations

import torch
from torch import nn

from panoptic_mining.models.panoptic_fcn import PanopticFCN, PanopticFCNOutput


class ContextFusionPanopticFCN(PanopticFCN):
    """Panoptic FCN with a second fusion stage: each branch's first-pass
    output is projected and fed into the other branch's second-pass head,
    alongside the shared encoder features."""

    def __init__(self, *args, fusion_channels: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.fusion_channels = fusion_channels

        # First-pass thing heatmap -> context channels consumed by the
        # second-pass stuff head (things inform stuff).
        self.stuff_from_things = nn.Conv2d(self.num_thing_classes, fusion_channels, kernel_size=1)
        # First-pass stuff logits -> context channels consumed by the
        # second-pass thing head (stuff informs things).
        self.thing_from_stuff = nn.Conv2d(self.num_stuff_classes, fusion_channels, kernel_size=1)

        self.stuff_head2 = nn.Conv2d(
            self.feat_channels + fusion_channels, self.num_stuff_classes, kernel_size=1
        )
        self.thing_heatmap_head2 = nn.Conv2d(
            self.feat_channels + fusion_channels, self.num_thing_classes, kernel_size=1
        )

    def forward(self, images: torch.Tensor) -> PanopticFCNOutput:
        feats = self.encoder(images)

        # First pass: same heads as the base model, used only to produce
        # context for the second pass (not returned directly).
        stuff_logits_pass1 = self.stuff_head(feats)
        thing_heatmap_pass1 = self.thing_heatmap_head(feats)

        thing_context = self.stuff_from_things(thing_heatmap_pass1)
        stuff_context = self.thing_from_stuff(stuff_logits_pass1)

        stuff_logits = self.stuff_head2(torch.cat([feats, thing_context], dim=1))
        thing_heatmap = self.thing_heatmap_head2(torch.cat([feats, stuff_context], dim=1))

        return PanopticFCNOutput(
            stuff_logits=stuff_logits,
            thing_heatmap=thing_heatmap,
            kernels=self.kernel_head(feats),
            mask_features=self.mask_feature_head(feats),
        )

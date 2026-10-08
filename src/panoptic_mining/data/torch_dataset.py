"""PyTorch ``Dataset`` wrapping ``data/manifest.py`` — turns box
annotations from ``dataset_split_completo`` into the (image, stuff
target, thing heatmap target) tuples ``training/train.py`` trains on.

Requires torch (``vision`` extra) — this whole module is skipped by
anything that imports it without torch installed, same convention as
``models/panoptic_fcn.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from panoptic_mining.data.manifest import DatasetManifest, FrameRecord
from panoptic_mining.data.targets import gaussian_heatmap_targets, rasterize_boxes_to_class_map, generate_thing_pseudo_masks


@dataclass
class ClassSplit:
    """Which annotated class names go to the stuff branch vs. the thing
    branch, and in what index order — this is the mapping documented as
    an open question in ``docs/decisions.md`` (the 3 thing / 2 stuff
    classes actually labeled vs. the thesis objectives' 6 thing
    categories), made explicit and swappable here rather than hard-coded.

    ``stuff_background_class_name``, if set, must also appear in
    ``stuff_class_names`` — it's the explicit "none of the stuff classes"
    label given to pixels outside every river/SDZI box, instead of the
    old behavior of just ignoring them. Added 2026-09-20 after the first
    full overnight fine-tuning run showed the stuff branch had learned no
    spatial structure at all (see ``rasterize_boxes_to_class_map``'s
    docstring in ``data/targets.py`` and ``docs/decisions.md`` for the
    real evidence). Set to ``None`` to restore the old ignore-index
    behavior."""

    stuff_class_names: list[str]
    thing_class_names: list[str]
    stuff_background_class_name: str | None = None

    @property
    def stuff_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.stuff_class_names)}

    @property
    def thing_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.thing_class_names)}

    @property
    def stuff_background_index(self) -> int | None:
        if self.stuff_background_class_name is None:
            return None
        return self.stuff_index[self.stuff_background_class_name]


DEFAULT_CLASS_SPLIT = ClassSplit(
    stuff_class_names=["river", "SDZI", "none"],
    thing_class_names=["vehicle", "building", "road"],
    stuff_background_class_name="none",
)


def compute_thing_pos_weight(class_counts: dict[str, int], thing_class_names: list[str]) -> list[float]:
    """Inverse-frequency weight per thing class, for
    ``training.train.TrainConfig.thing_pos_weight`` — added 2026-09-21
    after real predictions showed vehicle/road detections had much lower
    confidence than building (see docs/decisions.md). Unweighted BCE lets
    the gradient be dominated by whichever class has the most boxes
    (building: 4120 vs. vehicle: 717 in ``dataset_split_completo``), so a
    rarer class is penalized less for being missed than a common class is
    — weighting by ``max(class_counts) / class_counts[name]`` makes a
    missed rare-class positive cost as much as a missed common-class one,
    on average. Classes missing from ``class_counts`` (count 0) get
    weight 1.0 rather than dividing by zero — nothing to reweight against
    if it was never annotated at all."""
    counts = [class_counts.get(name, 0) for name in thing_class_names]
    max_count = max(counts) if counts else 0
    return [max_count / count if count > 0 else 1.0 for count in counts]


class ManifestSegmentationDataset(Dataset):
    """One sample = one frame from a ``DatasetManifest`` split, resized to
    ``image_size`` and downsampled by ``target_stride`` (matching
    ``FeatureEncoder``'s 1/8 spatial reduction by default) for the stuff
    class map and thing heatmap targets.

    Deliberately does not use ``data/points.py``'s GrabCut/flood-fill
    pseudo-masks as the per-pixel stuff target — see ``data/targets.py``'s
    module docstring for why (too slow per training sample as-is). This
    means the stuff branch is currently trained against coarse
    box-rectangle targets, not the refined pseudo-masks; tightening that
    gap (e.g. a precomputed pseudo-mask cache) is listed as follow-up work
    in the training loop docstring, not assumed away here.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        split: str,
        class_split: ClassSplit = DEFAULT_CLASS_SPLIT,
        image_size: tuple[int, int] = (512, 512),
        target_stride: int = 8,
    ):
        if split not in manifest.frames_by_split:
            raise ValueError(
                f"Split {split!r} not present in manifest (missing_splits={manifest.missing_splits})"
            )
        self.frames: list[FrameRecord] = manifest.frames_by_split[split]
        self.class_split = class_split
        self.image_height, self.image_width = image_size
        self.target_height = self.image_height // target_stride
        self.target_width = self.image_width // target_stride

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frame = self.frames[idx]
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {frame.image_path}")
        orig_height, orig_width = image.shape[:2]
        image_resized = cv2.resize(image, (self.image_width, self.image_height))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        scale_x, scale_y = self.target_width / orig_width, self.target_height / orig_height
        boxes_pixels = []
        for box in frame.boxes:
            x_min, y_min, x_max, y_max = box.to_pixels(orig_width, orig_height)
            boxes_pixels.append(
                (box.class_name, (x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y))
            )

        stuff_target = rasterize_boxes_to_class_map(
            boxes_pixels,
            self.target_height,
            self.target_width,
            self.class_split.stuff_index,
            background_index=self.class_split.stuff_background_index,
        )
        thing_heatmap = gaussian_heatmap_targets(
            boxes_pixels, self.target_height, self.target_width, self.class_split.thing_index
        )

        # Generate pseudo-masks for thing instances
        thing_masks = generate_thing_pseudo_masks(
            image_resized,
            boxes_pixels,
            self.target_height,
            self.target_width,
            self.class_split.thing_index,
        )
        
        return (
            image_tensor,
            torch.from_numpy(stuff_target).long(),
            torch.from_numpy(thing_heatmap).float(),
            torch.from_numpy(thing_masks).float(),
        )

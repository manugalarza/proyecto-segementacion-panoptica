"""PyTorch ``Dataset`` wrapping ``data/manifest.py`` — turns box
annotations from ``dataset_split_completo`` into the (image, stuff
target, thing heatmap target) tuples ``training/train.py`` trains on.

Requires torch (``vision`` extra) — this whole module is skipped by
anything that imports it without torch installed, same convention as
``models/panoptic_fcn.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    # Annotated classes that are neither stuff nor thing for training: their pixels get
    # ignore (-1) in the stuff target and no heatmap. Taxonomy agreed with Juan Pablo
    # (2026-10-22): SDZI is ignored for now.
    ignore_class_names: tuple = ()

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
    stuff_class_names=["river", "road", "none"],
    thing_class_names=["vehicle", "building", "machinery"],
    stuff_background_class_name="none",
    ignore_class_names=("SDZI",),
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


# Class ids of the semantic PNG masks written by scripts/sam_from_boxes.py
# (--save-masks) and scripts/annotate_semantic.py: the YOLO ids of the
# project plus 5 = background; 255 = unlabeled/ignore.
# gt_manual ids: 0 vehicle, 1 building, 2 road, 3 river, 4 SDZI, 5 background, 6 machinery.
SEMANTIC_MASK_CLASS_NAMES = ["vehicle", "building", "road", "river", "SDZI", "background", "machinery"]
SEMANTIC_IGNORE = 255


def semantic_mask_to_stuff_target(
    mask: np.ndarray,
    target_height: int,
    target_width: int,
    class_split: ClassSplit,
    mask_class_names: list[str] = SEMANTIC_MASK_CLASS_NAMES,
) -> np.ndarray:
    """Full-resolution semantic PNG -> (target_height, target_width) stuff
    class map with -1 = ignore. Non-stuff classes (things, background)
    become the stuff background class ("none"). Downsampling takes, per
    target cell, the label covering the largest area (one-hot + INTER_AREA
    + argmax) instead of nearest-neighbour, so small SDZI regions are not
    dropped just because they miss the sampled pixel."""
    background = class_split.stuff_background_index
    if background is None:
        raise ValueError("Mask targets need an explicit stuff background class")
    lut = np.full(256, -1, dtype=np.int64)
    for mask_id, name in enumerate(mask_class_names):
        if name in class_split.ignore_class_names:
            continue  # stays -1 (ignored)
        lut[mask_id] = class_split.stuff_index.get(name, background)
    lut[SEMANTIC_IGNORE] = -1
    full = lut[mask]
    labels = [-1] + list(range(len(class_split.stuff_class_names)))
    fractions = np.stack(
        [cv2.resize((full == lab).astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_AREA)
         for lab in labels]
    )
    return np.asarray(labels, dtype=np.int64)[fractions.argmax(axis=0)]


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
        return_thing_masks: bool = False,
        stuff_mask_dir: str | Path | None = None,
        instance_mask_dir: str | Path | None = None,
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
        # Thing masks have a variable count per frame, which the default
        # DataLoader collate can't batch; they're also not used by the
        # current training loop. Off by default (2026-10-08).
        self.return_thing_masks = return_thing_masks
        # If set, <instance_mask_dir>/<frame>.png is a uint16 instance-id map
        # (see scripts/sam_from_boxes.py --save-instances). Id k = frame.boxes[k-1].
        self.instance_mask_dir = Path(instance_mask_dir) if instance_mask_dir is not None else None
        if self.instance_mask_dir is not None:
            missing = [f.image_path for f in self.frames
                       if not (self.instance_mask_dir / f"{Path(f.image_path).stem}.png").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} frames have no instance mask in {self.instance_mask_dir} (e.g. {missing[:3]})")
        # If set, the stuff target comes from <stuff_mask_dir>/<frame>.png
        # (e.g. SAM pseudo-masks) instead of filled box rectangles. Thing
        # heatmaps still come from the boxes. Every frame must have a mask.
        self.stuff_mask_dir = Path(stuff_mask_dir) if stuff_mask_dir is not None else None
        if self.stuff_mask_dir is not None:
            missing = [f.image_path.stem for f in self.frames
                       if not (self.stuff_mask_dir / f"{Path(f.image_path).stem}.png").exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} frames have no mask in {self.stuff_mask_dir} (e.g. {missing[:3]})"
                )

    def __len__(self) -> int:
        return len(self.frames)

    MAX_INSTANCES_PER_FRAME = 64

    def _instance_sample(self, frame, boxes_pixels, image_tensor, stuff_target, thing_heatmap):
        """Returns the 6-tuple (image, stuff, heatmap, inst_map, inst_ids, inst_centers).

        inst_map: (H', W') long, instance ids at feature resolution (0 = none).
        inst_ids: (MAX,) long, ids present in the frame, -1 padded.
        inst_centers: (MAX, 2) long, (row, col) of each instance's heatmap peak
        in feature coordinates (same center as gaussian_heatmap_targets), -1 padded.
        """
        path = self.instance_mask_dir / f"{Path(frame.image_path).stem}.png"
        inst = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if inst is None or inst.ndim != 2:
            raise RuntimeError(f"Could not read instance mask: {path}")
        inst_small = cv2.resize(inst, (self.target_width, self.target_height),
                                interpolation=cv2.INTER_NEAREST).astype(np.int64)
        if inst_small.max(initial=0) > len(boxes_pixels):
            raise RuntimeError(f"{path}: instance id {int(inst.max())} > {len(boxes_pixels)} boxes in the label file")

        ids, centers = [], []
        thing_names = self.class_split.thing_index
        for k, (name, (x0, y0, x1, y1)) in enumerate(boxes_pixels, 1):
            if name not in thing_names:
                continue
            row = int(np.clip(round((y0 + y1) / 2.0), 0, self.target_height - 1))
            col = int(np.clip(round((x0 + x1) / 2.0), 0, self.target_width - 1))
            ids.append(k)
            centers.append((row, col))
        if len(ids) > self.MAX_INSTANCES_PER_FRAME:
            raise RuntimeError(f"{frame.image_path}: {len(ids)} thing instances > MAX_INSTANCES_PER_FRAME")

        inst_ids = torch.full((self.MAX_INSTANCES_PER_FRAME,), -1, dtype=torch.long)
        inst_centers = torch.full((self.MAX_INSTANCES_PER_FRAME, 2), -1, dtype=torch.long)
        if ids:
            inst_ids[:len(ids)] = torch.tensor(ids, dtype=torch.long)
            inst_centers[:len(ids)] = torch.tensor(centers, dtype=torch.long)
        return (
            image_tensor,
            torch.from_numpy(stuff_target).long(),
            torch.from_numpy(thing_heatmap).float(),
            torch.from_numpy(inst_small),
            inst_ids,
            inst_centers,
        )

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

        if self.stuff_mask_dir is not None:
            mask_path = self.stuff_mask_dir / f"{Path(frame.image_path).stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 2:
                raise RuntimeError(f"Could not read single-channel mask: {mask_path}")
            stuff_target = semantic_mask_to_stuff_target(
                mask, self.target_height, self.target_width, self.class_split
            )
        else:
            stuff_target = rasterize_boxes_to_class_map(
                boxes_pixels,
                self.target_height,
                self.target_width,
                self.class_split.stuff_index,
                background_index=self.class_split.stuff_background_index,
            )
            ignore_boxes = [b for b in boxes_pixels if b[0] in self.class_split.ignore_class_names]
            if ignore_boxes:
                ign = rasterize_boxes_to_class_map(
                    ignore_boxes, self.target_height, self.target_width,
                    {n: 0 for n in self.class_split.ignore_class_names}, background_index=None)
                stuff_target[ign == 0] = -1
        thing_heatmap = gaussian_heatmap_targets(
            boxes_pixels, self.target_height, self.target_width, self.class_split.thing_index
        )

        if self.instance_mask_dir is not None:
            return self._instance_sample(frame, boxes_pixels, image_tensor, stuff_target, thing_heatmap)

        if not self.return_thing_masks:
            return (
                image_tensor,
                torch.from_numpy(stuff_target).long(),
                torch.from_numpy(thing_heatmap).float(),
            )

        # Pseudo-masks for thing instances (currently box rectangles, see targets.py)
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

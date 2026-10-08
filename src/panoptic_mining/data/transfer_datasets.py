"""Transfer-learning dataset adapters: pretrain the shared encoder on a
public dataset with a similar domain (aerial/satellite land-cover) and
overlapping classes (water, woodland) before fine-tuning on our own small,
imbalanced mining dataset. This is a direct response to advisor feedback
(2026-09-19): find a forest/water-body segmentation dataset for transfer
learning, since river and SDZI are exactly the two classes hurting most
from limited/imbalanced data (see ``docs/decisions.md`` — SDZI has only
481 boxes vs. 4120 for building).

Two candidates were evaluated (see chat/decision log, not duplicated
here) and this module supports the one actually implemented:

**LandCover.ai** (Boguszewski et al., CVPR-W 2021) — chosen over DeepGlobe
because it's aerial orthophoto imagery (25-50cm/px), the same imaging
regime as the FAC video, not satellite; and because its 4 classes
(building, woodland, water, road) line up almost one-to-one with what we
already label (building/road as things, water ~ river as stuff), so the
encoder learns features in a domain and class space close to ours before
ever seeing the mining dataset. "Woodland" has no direct counterpart in
our labels but is kept as an auxiliary pretraining class (dropped when we
swap in our own heads for fine-tuning) since general vegetation-vs-not
texture discrimination plausibly helps distinguish SDZI (bare, disturbed
earth) from surrounding vegetation — that transfer benefit is a hypothesis
this pretraining step is meant to test, not a guarantee.

Dataset format — corrected 2026-09-19 after checking the official
download page, since the first version of this module guessed wrong:
the v1 zip (https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip)
ships large, full-size orthophotos and masks plus a bundled ``split.py``
that tiles them into 512x512 pieces and writes ``train.txt``/``val.txt``/
``test.txt`` listing which tile IDs go where. The *tiled* output (what
this loader actually reads, after running ``split.py`` once — see
``docs/training.md``) is a single ``output/`` directory containing, per
tile, an ``{id}.jpg`` image and a matching ``{id}_m.png`` single-channel
mask (pixel value = class index: 0=background, 1=building, 2=woodland,
3=water, 4=road) — confirmed against both torchgeo's own LandCoverAI
loader and its Hugging Face mirror, since the raw zip's pre-tiling layout
isn't fully documented on the download page itself. This is a *pixel*
mask, not boxes, so unlike ``data/torch_dataset.py`` there is no
per-annotation rasterization step for the stuff classes — the mask is
already what we want, just remapped. Thing-class (building, road) heatmap
targets are still derived the same CenterNet-style way as our own data,
using connected-component centroids of each thing class's mask region as
proxy box centers (LandCover.ai's masks are semantic, not instance-level,
so a single heatmap peak per connected blob is the closest available
proxy — two touching buildings would count as one instance here, a known
limitation of a semantic-only source dataset for pretraining a
kernel/instance branch).

Status: run against the real download for the first time 2026-09-19 (by
Manuela, after downloading the zip and running its bundled ``split.py``
herself) — this surfaced one real bug, since fixed: the real ``_m.png``
mask files are 3-channel (the class index repeated across R/G/B), not
single-channel grayscale as this module originally assumed, which raised
``IndexError: too many indices for array`` the first time it ran against
real data. ``__getitem__`` now collapses a 3-channel mask to one channel
before use; see ``tests/test_transfer_datasets.py``'s regression test.
Still not run for enough epochs/data to say anything about whether
pretraining helps — this confirms the loader works against real files,
not that pretraining is beneficial yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from panoptic_mining.data.targets import gaussian_heatmap_targets

LANDCOVER_CLASS_NAMES = {0: "background", 1: "building", 2: "woodland", 3: "water", 4: "road"}
LANDCOVER_STUFF_CLASSES = ["woodland", "water"]  # pretraining-only stuff head
LANDCOVER_THING_CLASSES = ["building", "road"]  # pretraining-only thing head


def _connected_component_boxes(binary_mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Proxy "boxes" (really just centers via bounding rects) for each
    connected blob of a binary mask, for feeding into
    ``gaussian_heatmap_targets`` — see module docstring for why this is a
    semantic-to-instance approximation, not real instance segmentation."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    boxes = []
    for label in range(1, num_labels):  # label 0 is background
        x, y, w, h, area = stats[label]
        if area < 4:  # drop single-pixel noise blobs
            continue
        boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


@dataclass
class LandCoverAISample:
    image_path: Path
    mask_path: Path


def index_landcover_ai(root: Path, split_file: str | None = None) -> list[LandCoverAISample]:
    """Pair up ``{id}.jpg`` with ``{id}_m.png`` inside ``root`` (the tiled
    ``output/`` directory produced by LandCover.ai's bundled ``split.py``
    — pass that directory as ``root``, not the zip's top-level extraction
    folder). Raises if no pairs are found, since a silently-empty
    pretraining dataset is worse than an explicit error here (unlike
    ``data/manifest.py``'s per-split tolerance, which exists specifically
    because our own dataset has known-broken splits to handle gracefully —
    LandCover.ai's tiled output does not have that problem).

    ``split_file`` optionally restricts to the IDs listed in one of
    LandCover.ai's own ``train.txt``/``val.txt``/``test.txt`` (one tile ID
    per line, no extension) — pass its path to train only on that split
    instead of everything under ``root``.
    """
    root = Path(root)
    allowed_ids: set[str] | None = None
    if split_file is not None:
        allowed_ids = {line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()}

    samples = []
    for image_path in sorted(root.glob("*.jpg")):
        tile_id = image_path.stem
        if allowed_ids is not None and tile_id not in allowed_ids:
            continue
        mask_path = root / f"{tile_id}_m.png"
        if mask_path.exists():
            samples.append(LandCoverAISample(image_path=image_path, mask_path=mask_path))
    if not samples:
        raise ValueError(
            f"No {{id}}.jpg/{{id}}_m.png pairs found under {root} — did you run LandCover.ai's "
            "bundled split.py first? See docs/training.md."
        )
    return samples


class LandCoverAIDataset(Dataset):
    """Pretraining dataset: yields the same ``(image, stuff_target,
    thing_heatmap)`` tuple shape as ``ManifestSegmentationDataset``, so
    ``training/train.py``'s loop is unchanged between pretraining and
    fine-tuning — only the dataset and the model's head sizes differ."""

    def __init__(
        self,
        root: Path,
        split_file: str | None = None,
        image_size: tuple[int, int] = (512, 512),
        target_stride: int = 8,
    ):
        self.samples = index_landcover_ai(Path(root), split_file=split_file)
        self.image_height, self.image_width = image_size
        self.target_height = self.image_height // target_stride
        self.target_width = self.image_width // target_stride

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path))
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise RuntimeError(f"Could not read {sample.image_path} / {sample.mask_path}")
        if mask.ndim == 3:
            # LandCover.ai's real _m.png files are 3-channel (each channel
            # holding the same class-index value), not the single-channel
            # grayscale this loader originally assumed — caught via a real
            # IndexError running against the actual download (2026-09-19).
            # All channels carry the same value, so any one of them works.
            mask = mask[:, :, 0]

        image_resized = cv2.resize(image, (self.image_width, self.image_height))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        mask_resized = cv2.resize(
            mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST
        )

        stuff_target = np.full((self.target_height, self.target_width), fill_value=-1, dtype=np.int32)
        for class_id, class_name in LANDCOVER_CLASS_NAMES.items():
            if class_name in LANDCOVER_STUFF_CLASSES:
                stuff_index = LANDCOVER_STUFF_CLASSES.index(class_name)
                stuff_target[mask_resized == class_id] = stuff_index

        # Thing-class instance boxes: detect connected components on the
        # NATIVE-resolution mask, not the already-downsampled mask_resized
        # above, then scale the resulting boxes down to target resolution.
        # Real bug (caught 2026-09-20 running the full test suite with
        # torch installed for the first time since this module was
        # written): detecting components *after* downsampling to the
        # coarse target grid (e.g. 64x64 for a 512x512 tile at stride 8)
        # could shrink a real, fully-visible building blob to 0-1 pixels
        # via nearest-neighbor resizing, which the noise-area filter in
        # _connected_component_boxes then silently dropped as noise —
        # losing real instances, not just cleaning up noise. Detecting on
        # the native mask first avoids that; see the regression test.
        native_height, native_width = mask.shape[:2]
        scale_x = self.target_width / native_width
        scale_y = self.target_height / native_height

        thing_boxes = []
        for class_id, class_name in LANDCOVER_CLASS_NAMES.items():
            if class_name in LANDCOVER_THING_CLASSES:
                for x_min, y_min, x_max, y_max in _connected_component_boxes(mask == class_id):
                    thing_boxes.append(
                        (class_name, (x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y))
                    )

        thing_index = {name: i for i, name in enumerate(LANDCOVER_THING_CLASSES)}
        thing_heatmap = gaussian_heatmap_targets(
            thing_boxes, self.target_height, self.target_width, thing_index
        )

        return (
            image_tensor,
            torch.from_numpy(stuff_target).long(),
            torch.from_numpy(thing_heatmap).float(),
        )

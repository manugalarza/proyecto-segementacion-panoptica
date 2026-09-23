"""Unit tests for data/transfer_datasets.py (LandCover.ai adapter) — uses
small synthetic image/mask pairs written to tmp_path, since the real
~1GB+ LandCover.ai download was not fetched in this session (see module
docstring). Gated on torch + cv2."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from panoptic_mining.data.transfer_datasets import (  # noqa: E402
    LANDCOVER_STUFF_CLASSES,
    LANDCOVER_THING_CLASSES,
    LandCoverAIDataset,
    _connected_component_boxes,
    index_landcover_ai,
)


def _write_synthetic_landcover_sample(root, tile_id="tile_0000", size=64, three_channel_mask=True):
    """Mimics the tiled `output/` directory LandCover.ai's bundled
    split.py produces: {id}.jpg + {id}_m.png side by side, not separate
    images/masks subfolders — see transfer_datasets.py's module docstring
    for why this was corrected from an earlier, wrong assumption.

    ``three_channel_mask`` defaults to True because that's what the real
    download's _m.png files actually are (caught via a real IndexError
    running against the real dataset, 2026-09-19) — set it False only to
    exercise the single-channel code path too."""
    root.mkdir(parents=True, exist_ok=True)

    image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    cv2.imwrite(str(root / f"{tile_id}.jpg"), image)

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:30, 10:20] = 2  # woodland
    mask[40:55, 5:15] = 3  # water
    mask[0:8, 0:8] = 1  # building blob
    if three_channel_mask:
        mask = np.stack([mask, mask, mask], axis=-1)
    cv2.imwrite(str(root / f"{tile_id}_m.png"), mask)
    return root / f"{tile_id}.jpg", root / f"{tile_id}_m.png"


def test_index_landcover_ai_pairs_images_and_masks(tmp_path):
    _write_synthetic_landcover_sample(tmp_path)
    samples = index_landcover_ai(tmp_path)
    assert len(samples) == 1
    assert samples[0].image_path.stem == samples[0].mask_path.stem.removesuffix("_m")


def test_index_landcover_ai_raises_when_no_pairs_found(tmp_path):
    with pytest.raises(ValueError):
        index_landcover_ai(tmp_path)


def test_index_landcover_ai_respects_split_file(tmp_path):
    _write_synthetic_landcover_sample(tmp_path, tile_id="tile_0000")
    _write_synthetic_landcover_sample(tmp_path, tile_id="tile_0001")
    split_file = tmp_path / "train.txt"
    split_file.write_text("tile_0000\n")

    samples = index_landcover_ai(tmp_path, split_file=str(split_file))

    assert len(samples) == 1
    assert samples[0].image_path.stem == "tile_0000"


def test_connected_component_boxes_finds_one_blob():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:10, 5:10] = 1
    boxes = _connected_component_boxes(mask)
    assert len(boxes) == 1
    x_min, y_min, x_max, y_max = boxes[0]
    assert (x_min, y_min) == (5.0, 5.0)
    assert (x_max, y_max) == (10.0, 10.0)


def test_connected_component_boxes_ignores_tiny_noise():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0, 0] = 1  # single pixel, below the area-4 floor
    boxes = _connected_component_boxes(mask)
    assert boxes == []


def test_landcover_dataset_item_shapes(tmp_path):
    _write_synthetic_landcover_sample(tmp_path, size=64)
    dataset = LandCoverAIDataset(tmp_path, image_size=(64, 64), target_stride=8)
    assert len(dataset) == 1

    image, stuff_target, thing_heatmap = dataset[0]
    assert image.shape == (3, 64, 64)
    assert stuff_target.shape == (8, 8)
    assert thing_heatmap.shape == (len(LANDCOVER_THING_CLASSES), 8, 8)


def test_landcover_dataset_stuff_target_has_water_and_woodland(tmp_path):
    _write_synthetic_landcover_sample(tmp_path, size=64)
    dataset = LandCoverAIDataset(tmp_path, image_size=(64, 64), target_stride=8)
    _, stuff_target, _ = dataset[0]

    woodland_index = LANDCOVER_STUFF_CLASSES.index("woodland")
    water_index = LANDCOVER_STUFF_CLASSES.index("water")
    assert (stuff_target == woodland_index).any()
    assert (stuff_target == water_index).any()


def test_landcover_dataset_thing_heatmap_has_building_peak(tmp_path):
    """Regression test (real bug caught 2026-09-20, see
    transfer_datasets.py's __getitem__ comment): connected-component
    detection used to run on the mask AFTER downsampling to the coarse
    target grid, which could shrink this test's 8x8-pixel building blob
    (in a 64x64 image, downsampled to an 8x8 target — an 8x factor) to a
    single pixel via nearest-neighbor resizing, then get dropped entirely
    by the noise-area filter. Detecting on the native-resolution mask
    first and scaling the resulting box down fixes this."""
    _write_synthetic_landcover_sample(tmp_path, size=64)
    dataset = LandCoverAIDataset(tmp_path, image_size=(64, 64), target_stride=8)
    _, _, thing_heatmap = dataset[0]

    building_index = LANDCOVER_THING_CLASSES.index("building")
    assert thing_heatmap[building_index].max() > 0.5


def test_landcover_dataset_handles_three_channel_mask_files():
    """Regression test: the real LandCover.ai _m.png files are 3-channel
    (same value repeated per channel), not single-channel grayscale as
    originally assumed — this raised a real IndexError
    ('too many indices for array: array is 2-dimensional, but 3 were
    indexed') when run against the real download (2026-09-19). This test
    fails again if that handling regresses."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_synthetic_landcover_sample(root, size=64, three_channel_mask=True)
        dataset = LandCoverAIDataset(root, image_size=(64, 64), target_stride=8)
        # Must not raise IndexError when indexing the 2D stuff_target with
        # a boolean array derived from a 3-channel mask.
        image, stuff_target, thing_heatmap = dataset[0]
        assert stuff_target.shape == (8, 8)


def test_landcover_dataset_single_channel_mask_still_works(tmp_path):
    """The single-channel code path (mask.ndim == 2) must keep working
    too, in case a different LandCover.ai release/tool ever writes true
    grayscale masks."""
    _write_synthetic_landcover_sample(tmp_path, size=64, three_channel_mask=False)
    dataset = LandCoverAIDataset(tmp_path, image_size=(64, 64), target_stride=8)
    _, stuff_target, _ = dataset[0]
    assert stuff_target.shape == (8, 8)

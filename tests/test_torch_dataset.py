"""Unit tests for data/torch_dataset.py — needs torch + cv2 with real
files on disk (synthetic tiny images written to tmp_path), gated on torch."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

from panoptic_mining.data.manifest import (  # noqa: E402
    BoxAnnotation,
    DatasetManifest,
    FrameRecord,
)
from panoptic_mining.data.torch_dataset import (  # noqa: E402
    DEFAULT_CLASS_SPLIT,
    ClassSplit,
    ManifestSegmentationDataset,
    compute_thing_pos_weight,
)


def _write_dummy_image(path, size=(64, 64)):
    image = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _manifest_with_one_frame(tmp_path):
    image_path = tmp_path / "frame_0000.jpg"
    _write_dummy_image(image_path)

    boxes = [
        BoxAnnotation(class_id=4, class_name="river", x_center=0.5, y_center=0.5, width=0.4, height=0.3),
        BoxAnnotation(class_id=0, class_name="vehicle", x_center=0.2, y_center=0.2, width=0.1, height=0.1),
    ]
    frame = FrameRecord(frame_id="frame_0000", split="train", image_path=image_path, label_path=None, boxes=boxes)
    return DatasetManifest(
        root=tmp_path,
        class_names={0: "vehicle", 1: "building", 2: "road", 3: "SDZI", 4: "river"},
        frames_by_split={"train": [frame]},
    )


def test_dataset_length_matches_split_frame_count(tmp_path):
    manifest = _manifest_with_one_frame(tmp_path)
    dataset = ManifestSegmentationDataset(manifest, split="train")
    assert len(dataset) == 1


def test_dataset_raises_for_missing_split(tmp_path):
    manifest = _manifest_with_one_frame(tmp_path)
    with pytest.raises(ValueError):
        ManifestSegmentationDataset(manifest, split="val")


def test_dataset_item_shapes(tmp_path):
    manifest = _manifest_with_one_frame(tmp_path)
    dataset = ManifestSegmentationDataset(
        manifest,
        split="train",
        class_split=ClassSplit(stuff_class_names=["river", "SDZI"], thing_class_names=["vehicle", "building", "road"]),
        image_size=(64, 64),
        target_stride=8,
    )
    image, stuff_target, thing_heatmap = dataset[0]

    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert stuff_target.shape == (8, 8)
    assert thing_heatmap.shape == (3, 8, 8)


def test_dataset_stuff_target_has_river_class_somewhere(tmp_path):
    manifest = _manifest_with_one_frame(tmp_path)
    dataset = ManifestSegmentationDataset(manifest, split="train", image_size=(64, 64), target_stride=8)
    _, stuff_target, _ = dataset[0]
    # river is index 0 in DEFAULT_CLASS_SPLIT's stuff_class_names
    assert (stuff_target == 0).any()


def test_dataset_thing_heatmap_has_a_peak_for_vehicle(tmp_path):
    manifest = _manifest_with_one_frame(tmp_path)
    dataset = ManifestSegmentationDataset(manifest, split="train", image_size=(64, 64), target_stride=8)
    _, _, thing_heatmap = dataset[0]
    # vehicle is index 0 in DEFAULT_CLASS_SPLIT's thing_class_names
    assert thing_heatmap[0].max() > 0.5


def test_default_class_split_has_explicit_background_class():
    """Regression test for the 2026-09-20 fix (see docs/decisions.md):
    DEFAULT_CLASS_SPLIT must label unannotated stuff pixels with an
    explicit background class, not silently ignore them — that silent
    ignoring is what let the stuff branch learn zero spatial structure in
    the first full overnight fine-tuning run."""
    assert DEFAULT_CLASS_SPLIT.stuff_background_class_name == "none"
    assert DEFAULT_CLASS_SPLIT.stuff_background_index is not None
    assert "none" in DEFAULT_CLASS_SPLIT.stuff_class_names


def test_dataset_stuff_target_has_no_ignored_pixels_with_default_class_split(tmp_path):
    """With the default class split (explicit 'none' background class),
    every pixel of the stuff target should get a real label — no -1s left
    over from the old ignore-index behavior."""
    manifest = _manifest_with_one_frame(tmp_path)
    dataset = ManifestSegmentationDataset(manifest, split="train", image_size=(64, 64), target_stride=8)
    _, stuff_target, _ = dataset[0]
    assert not (stuff_target == -1).any()
    # Most of a 64x64 frame is outside the single small river box, so the
    # background class should dominate — this is exactly the pixels that
    # used to get no supervision at all.
    background_index = dataset.class_split.stuff_background_index
    assert (stuff_target == background_index).any()


def test_compute_thing_pos_weight_matches_real_class_imbalance():
    """Regression test for the 2026-09-21 thing-branch weighting fix (see
    docs/decisions.md): the rarest class should get the highest weight,
    the most common class weight 1.0, using the real dataset_split_completo
    counts as the concrete example this was built for."""
    class_counts = {"vehicle": 717, "building": 4120, "road": 1958, "river": 680, "SDZI": 481}
    weights = compute_thing_pos_weight(class_counts, ["vehicle", "building", "road"])

    assert weights[1] == pytest.approx(1.0)  # building is the most common thing class
    assert weights[0] == pytest.approx(4120 / 717)  # vehicle, the rarest, gets the highest weight
    assert weights[0] > weights[2] > weights[1]


def test_compute_thing_pos_weight_defaults_to_one_for_missing_class():
    weights = compute_thing_pos_weight({"building": 100}, ["vehicle", "building"])
    assert weights == [1.0, 1.0]

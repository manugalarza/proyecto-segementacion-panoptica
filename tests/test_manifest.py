"""Unit tests for data/manifest.py — builds tiny synthetic YOLO-format
datasets on disk (via pytest's tmp_path) rather than depending on the real
downloaded dataset, so these run anywhere."""

from __future__ import annotations

import numpy as np
import cv2
import pytest

from panoptic_mining.data.manifest import load_yolo_manifest


def _write_image(path, size=(20, 20)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_dataset_yaml(root, splits=("train", "val", "test")):
    lines = [f"path: {root}"]
    for split in splits:
        lines.append(f"{split}: {split}/images")
    lines.append("nc: 5")
    lines.append("names:")
    lines.append("  0: vehicle")
    lines.append("  1: building")
    lines.append("  2: road")
    lines.append("  3: river")
    lines.append("  4: SDZI")
    (root / "dataset.yaml").write_text("\n".join(lines))


def test_load_well_formed_dataset(tmp_path):
    _write_dataset_yaml(tmp_path)
    for split in ("train", "val", "test"):
        _write_image(tmp_path / split / "images" / "frame_0.jpg")
        (tmp_path / split / "labels").mkdir(parents=True, exist_ok=True)
        (tmp_path / split / "labels" / "frame_0.txt").write_text("4 0.5 0.5 0.2 0.2\n")

    manifest = load_yolo_manifest(tmp_path)

    assert manifest.missing_splits == []
    assert set(manifest.frames_by_split) == {"train", "val", "test"}
    for split, frames in manifest.frames_by_split.items():
        assert len(frames) == 1
        assert frames[0].labels_available is True
        assert len(frames[0].boxes) == 1
        assert frames[0].boxes[0].class_name == "SDZI"


def test_load_dataset_missing_val_split(tmp_path):
    """Mirrors flir_best_division_pr: dataset.yaml references val/images
    but the val/ directory was never created."""
    _write_dataset_yaml(tmp_path, splits=("train", "val", "test"))
    for split in ("train", "test"):
        _write_image(tmp_path / split / "images" / "frame_0.jpg")
        (tmp_path / split / "labels").mkdir(parents=True, exist_ok=True)
        (tmp_path / split / "labels" / "frame_0.txt").write_text("3 0.5 0.5 0.2 0.2\n")
    # no tmp_path / "val" directory at all

    manifest = load_yolo_manifest(tmp_path)

    assert manifest.missing_splits == ["val"]
    assert set(manifest.frames_by_split) == {"train", "test"}


def test_load_dataset_train_images_without_labels_dir(tmp_path):
    """Mirrors flir_best_division_pr/train: images/ exists, labels/ does not."""
    _write_dataset_yaml(tmp_path)
    _write_image(tmp_path / "train" / "images" / "frame_0.jpg")
    # no train/labels directory at all
    for split in ("val", "test"):
        _write_image(tmp_path / split / "images" / "frame_0.jpg")
        (tmp_path / split / "labels").mkdir(parents=True, exist_ok=True)
        (tmp_path / split / "labels" / "frame_0.txt").write_text("")

    manifest = load_yolo_manifest(tmp_path)

    train_frames = manifest.frames_by_split["train"]
    assert len(train_frames) == 1
    assert train_frames[0].labels_available is False
    assert train_frames[0].boxes == []


def test_load_dataset_frame_with_empty_label_file(tmp_path):
    """An empty label file means zero annotated instances, not a missing-labels problem."""
    _write_dataset_yaml(tmp_path, splits=("train",))
    _write_image(tmp_path / "train" / "images" / "frame_0.jpg")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "frame_0.txt").write_text("")

    manifest = load_yolo_manifest(tmp_path)
    frame = manifest.frames_by_split["train"][0]
    assert frame.labels_available is True
    assert frame.boxes == []


def test_missing_dataset_yaml_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yolo_manifest(tmp_path)


def test_malformed_label_line_raises(tmp_path):
    _write_dataset_yaml(tmp_path, splits=("train",))
    _write_image(tmp_path / "train" / "images" / "frame_0.jpg")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "frame_0.txt").write_text("4 0.5 0.5\n")  # missing 2 fields

    with pytest.raises(ValueError):
        load_yolo_manifest(tmp_path)


def test_box_annotation_to_pixels():
    from panoptic_mining.data.manifest import BoxAnnotation

    box = BoxAnnotation(class_id=4, class_name="SDZI", x_center=0.5, y_center=0.5, width=0.2, height=0.4)
    x_min, y_min, x_max, y_max = box.to_pixels(image_width=100, image_height=200)
    assert (x_min, y_min, x_max, y_max) == pytest.approx((40.0, 60.0, 60.0, 140.0))


def test_frame_record_pixel_boxes(tmp_path):
    _write_dataset_yaml(tmp_path, splits=("train",))
    _write_image(tmp_path / "train" / "images" / "frame_0.jpg", size=(100, 200))
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "frame_0.txt").write_text("4 0.5 0.5 0.2 0.4\n")

    manifest = load_yolo_manifest(tmp_path)
    frame = manifest.frames_by_split["train"][0]
    pixel_boxes = frame.pixel_boxes()

    assert len(pixel_boxes) == 1
    class_name, box = pixel_boxes[0]
    assert class_name == "SDZI"
    assert box == pytest.approx((40.0, 60.0, 60.0, 140.0))


def test_summary_reports_counts_and_missing_splits(tmp_path):
    _write_dataset_yaml(tmp_path, splits=("train", "val", "test"))
    _write_image(tmp_path / "train" / "images" / "frame_0.jpg")
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "labels" / "frame_0.txt").write_text("4 0.5 0.5 0.2 0.2\n3 0.1 0.1 0.1 0.1\n")
    _write_image(tmp_path / "test" / "images" / "frame_0.jpg")
    (tmp_path / "test" / "labels").mkdir(parents=True)
    (tmp_path / "test" / "labels" / "frame_0.txt").write_text("4 0.5 0.5 0.2 0.2\n")
    # val/ missing entirely

    manifest = load_yolo_manifest(tmp_path)
    summary = manifest.summary()

    assert summary["missing_splits"] == ["val"]
    assert summary["per_split"]["train"]["frame_count"] == 1
    assert summary["per_split"]["train"]["box_count"] == 2
    assert summary["class_counts"]["SDZI"] == 2
    assert summary["class_counts"]["river"] == 1
    assert summary["class_counts"]["vehicle"] == 0

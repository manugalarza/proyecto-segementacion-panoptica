"""Dataset manifest loading for YOLO-format splits.

Loads the dataset actually downloaded so far (``dataset_split_completo``:
train/val/test, each with images/ + labels/, declared via a
``dataset.yaml`` — see README "Data consumption" and
``docs/decisions.md``), into a structure the rest of the pipeline
(box->points->pseudo-mask, training, evaluation) can consume without
knowing about the raw folder layout.

This does NOT assume every split is well-formed. Real inspection of the
shared OneDrive turned up ``flir_best_division_pr``, which is missing its
``val/`` split entirely (despite its own dataset.yaml referencing
``val/images``) and has a ``train/images`` with no matching ``train/labels``.
Rather than crash on that kind of dataset, loading here is defensive: a
missing labels directory yields frames with no boxes and
``labels_available=False``, and a missing split directory is simply
absent from the manifest — both are reported, not hidden, so a caller (or
the CLI's ``manifest-summary`` command) can catch a broken dataset before
training silently runs on it.

Still explicitly NOT handled here (tracked in docs/decisions.md):
  - Aligning frame IDs against Jorge's leakage-safe split, once it exists.
  - Anything about the RGB-vs-thermal question — this module only reads
    box annotations and file paths, not pixel content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import yaml

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class BoxAnnotation:
    """One YOLO-format box annotation, kept in normalized coordinates as
    written on disk (0..1 relative to image width/height)."""

    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def to_pixels(self, image_width: int, image_height: int) -> tuple[float, float, float, float]:
        """Convert to ``(x_min, y_min, x_max, y_max)`` in pixel coordinates
        for a specific image size, e.g. to feed data.points.box_to_pseudo_mask."""
        x_min = (self.x_center - self.width / 2) * image_width
        y_min = (self.y_center - self.height / 2) * image_height
        x_max = (self.x_center + self.width / 2) * image_width
        y_max = (self.y_center + self.height / 2) * image_height
        return x_min, y_min, x_max, y_max


@dataclass
class FrameRecord:
    """One frame: its image path, split, and box annotations (empty if the
    dataset had no matching label file — see ``labels_available``)."""

    frame_id: str
    split: str
    image_path: Path
    label_path: Path | None
    boxes: list[BoxAnnotation] = field(default_factory=list)
    labels_available: bool = True

    def image_size(self) -> tuple[int, int]:
        """Return (width, height) by reading the image header. Raises if
        the image can't be read — callers that only need annotations
        (e.g. dataset statistics) should avoid calling this."""
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        h, w = image.shape[:2]
        return w, h

    def pixel_boxes(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        """Convenience: (class_name, pixel_box) for every annotation,
        reading this frame's actual image size once."""
        width, height = self.image_size()
        return [(box.class_name, box.to_pixels(width, height)) for box in self.boxes]


@dataclass
class DatasetManifest:
    root: Path
    class_names: dict[int, str]
    frames_by_split: dict[str, list[FrameRecord]] = field(default_factory=dict)
    missing_splits: list[str] = field(default_factory=list)

    def all_frames(self) -> list[FrameRecord]:
        return [frame for frames in self.frames_by_split.values() for frame in frames]

    def summary(self) -> dict:
        """Per-split frame/box counts, per-class box counts, and any
        integrity issues found (missing splits, frames without labels) —
        the CLI's ``data manifest-summary`` command surfaces exactly this."""
        per_split = {}
        for split, frames in self.frames_by_split.items():
            frames_without_labels = sum(1 for f in frames if not f.labels_available)
            box_count = sum(len(f.boxes) for f in frames)
            per_split[split] = {
                "frame_count": len(frames),
                "box_count": box_count,
                "frames_without_labels": frames_without_labels,
            }

        class_counts: dict[str, int] = {name: 0 for name in self.class_names.values()}
        for frame in self.all_frames():
            for box in frame.boxes:
                class_counts[box.class_name] = class_counts.get(box.class_name, 0) + 1

        return {
            "root": str(self.root),
            "per_split": per_split,
            "class_counts": class_counts,
            "missing_splits": self.missing_splits,
        }


def _load_split(
    root: Path, split: str, class_names: dict[int, str]
) -> list[FrameRecord] | None:
    """Load one split ('train' | 'val' | 'test'). Returns None if the
    split directory doesn't exist at all (as with flir_best_division_pr's
    missing val/), rather than raising, so the caller can record it as a
    missing split and continue with whatever splits ARE present."""
    split_dir = root / split
    images_dir = split_dir / "images"
    if not split_dir.is_dir() or not images_dir.is_dir():
        return None

    labels_dir = split_dir / "labels"
    labels_available = labels_dir.is_dir()

    frames: list[FrameRecord] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        frame_id = image_path.stem
        label_path = labels_dir / f"{frame_id}.txt" if labels_available else None

        boxes: list[BoxAnnotation] = []
        frame_labels_available = labels_available
        if label_path is not None and label_path.is_file():
            boxes = _parse_label_file(label_path, class_names)
        elif labels_available:
            # labels/ exists as a directory but this specific frame has no
            # matching .txt — treat as "no annotated instances", not as a
            # missing-labels problem for the whole frame.
            boxes = []
        else:
            frame_labels_available = False

        frames.append(
            FrameRecord(
                frame_id=frame_id,
                split=split,
                image_path=image_path,
                label_path=label_path,
                boxes=boxes,
                labels_available=frame_labels_available,
            )
        )
    return frames


def _parse_label_file(label_path: Path, class_names: dict[int, str]) -> list[BoxAnnotation]:
    boxes = []
    text = label_path.read_text().strip()
    if not text:
        return boxes
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected 5 fields, got {len(parts)}")
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(v) for v in parts[1:])
        boxes.append(
            BoxAnnotation(
                class_id=class_id,
                class_name=class_names.get(class_id, str(class_id)),
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return boxes


def load_yolo_manifest(root: Path | str) -> DatasetManifest:
    """Load a YOLO-format dataset rooted at ``root`` (the directory
    containing ``dataset.yaml`` and the split subdirectories).

    Example: ``load_yolo_manifest(Path(os.environ["DATA_ROOT"]))`` once
    DATA_ROOT (see .env.example) points at dataset_split_completo.
    """
    root = Path(root)
    yaml_path = root / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"No dataset.yaml found at {yaml_path}")

    with yaml_path.open() as f:
        config = yaml.safe_load(f)

    raw_names = config.get("names", {})
    class_names = {int(k): v for k, v in raw_names.items()}

    frames_by_split: dict[str, list[FrameRecord]] = {}
    missing_splits: list[str] = []
    for split in ("train", "val", "test"):
        frames = _load_split(root, split, class_names)
        if frames is None:
            missing_splits.append(split)
        else:
            frames_by_split[split] = frames

    return DatasetManifest(
        root=root,
        class_names=class_names,
        frames_by_split=frames_by_split,
        missing_splits=missing_splits,
    )

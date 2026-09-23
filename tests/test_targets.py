"""Unit tests for data/targets.py — pure numpy, no torch dependency."""

from __future__ import annotations

import numpy as np
import pytest

from panoptic_mining.data.targets import (
    gaussian_heatmap_targets,
    rasterize_boxes_to_class_map,
    resize_class_map_nearest,
)


def test_rasterize_boxes_paints_class_index_inside_box():
    boxes = [("river", (10.0, 10.0, 30.0, 20.0))]
    class_map = rasterize_boxes_to_class_map(boxes, height=40, width=40, class_name_to_index={"river": 0})
    assert np.all(class_map[10:20, 10:30] == 0)


def test_rasterize_boxes_background_is_minus_one():
    class_map = rasterize_boxes_to_class_map([], height=10, width=10, class_name_to_index={"river": 0})
    assert np.all(class_map == -1)


def test_rasterize_boxes_skips_unknown_class_names():
    boxes = [("vehicle", (0.0, 0.0, 5.0, 5.0))]
    class_map = rasterize_boxes_to_class_map(boxes, height=10, width=10, class_name_to_index={"river": 0})
    assert np.all(class_map == -1)


def test_rasterize_boxes_background_index_fills_unannotated_pixels():
    """Regression test for the 2026-09-20 fix: passing background_index
    should label unannotated pixels with that class instead of -1, so the
    stuff branch gets real negative supervision instead of none at all
    (see docs/decisions.md for why this mattered in practice)."""
    boxes = [("river", (10.0, 10.0, 20.0, 20.0))]
    class_map = rasterize_boxes_to_class_map(
        boxes, height=40, width=40, class_name_to_index={"river": 0, "none": 1}, background_index=1
    )
    assert np.all(class_map[10:20, 10:20] == 0)
    assert class_map[0, 0] == 1  # unannotated corner is the explicit background class, not -1
    assert not np.any(class_map == -1)


def test_rasterize_boxes_later_box_wins_on_overlap():
    boxes = [("river", (0.0, 0.0, 10.0, 10.0)), ("SDZI", (5.0, 5.0, 15.0, 15.0))]
    class_map = rasterize_boxes_to_class_map(
        boxes, height=20, width=20, class_name_to_index={"river": 0, "SDZI": 1}
    )
    assert class_map[7, 7] == 1  # overlap region got the later box's class
    assert class_map[1, 1] == 0  # non-overlap region kept the first box's class


def test_gaussian_heatmap_peak_is_at_box_center():
    boxes = [("vehicle", (10.0, 10.0, 20.0, 20.0))]
    heatmap = gaussian_heatmap_targets(boxes, height=30, width=30, class_name_to_index={"vehicle": 0})
    assert heatmap.shape == (1, 30, 30)
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap[0]), heatmap[0].shape)
    assert (peak_y, peak_x) == (15, 15)
    assert heatmap[0, 15, 15] == pytest.approx(1.0, abs=1e-5)


def test_gaussian_heatmap_values_bounded_zero_to_one():
    boxes = [("vehicle", (5.0, 5.0, 25.0, 25.0)), ("vehicle", (40.0, 40.0, 60.0, 60.0))]
    heatmap = gaussian_heatmap_targets(boxes, height=80, width=80, class_name_to_index={"vehicle": 0})
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_gaussian_heatmap_empty_classes_returns_empty_array():
    heatmap = gaussian_heatmap_targets([], height=10, width=10, class_name_to_index={})
    assert heatmap.shape == (0, 10, 10)


def test_resize_class_map_nearest_preserves_class_values():
    class_map = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, -1, -1], [2, 2, -1, -1]], dtype=np.int32)
    resized = resize_class_map_nearest(class_map, out_height=2, out_width=2)
    assert set(np.unique(resized)).issubset({0, 1, 2, -1})
    assert resized.shape == (2, 2)


def test_resize_class_map_nearest_upsamples_without_interpolating():
    class_map = np.array([[0, 1]], dtype=np.int32)
    resized = resize_class_map_nearest(class_map, out_height=4, out_width=4)
    # Only the original two class values should appear — no averaged/interpolated values.
    assert set(np.unique(resized)).issubset({0, 1})

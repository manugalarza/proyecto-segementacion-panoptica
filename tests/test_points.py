"""Unit tests for data/points.py — all synthetic, no real dataset needed."""

from __future__ import annotations

import numpy as np
import pytest

from panoptic_mining.data.points import box_to_pseudo_mask, sample_interior_points


def test_sample_interior_points_returns_requested_count():
    box = (10.0, 20.0, 50.0, 80.0)
    points = sample_interior_points(box, n_points=7, rng=np.random.default_rng(0))
    assert points.shape == (7, 2)


def test_sample_interior_points_first_point_is_center():
    box = (0.0, 0.0, 10.0, 20.0)
    points = sample_interior_points(box, n_points=3, rng=np.random.default_rng(0))
    assert points[0] == pytest.approx((5.0, 10.0))


def test_sample_interior_points_all_points_inside_box():
    box = (5.0, 5.0, 25.0, 15.0)
    points = sample_interior_points(box, n_points=20, rng=np.random.default_rng(1))
    x_min, y_min, x_max, y_max = box
    assert np.all(points[:, 0] >= x_min) and np.all(points[:, 0] <= x_max)
    assert np.all(points[:, 1] >= y_min) and np.all(points[:, 1] <= y_max)


def test_sample_interior_points_rejects_degenerate_box():
    with pytest.raises(ValueError):
        sample_interior_points((10.0, 10.0, 10.0, 20.0), n_points=3)


def test_sample_interior_points_rejects_zero_points():
    with pytest.raises(ValueError):
        sample_interior_points((0.0, 0.0, 10.0, 10.0), n_points=0)


def _synthetic_image_with_bright_square(size=100, square_box=(30, 30, 70, 70)):
    """A dark background with a bright square — an easy case for GrabCut
    to separate foreground from background, so the test checks the
    plumbing (shapes, dtypes, seed usage) rather than segmentation
    quality on a hard real image."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    x_min, y_min, x_max, y_max = square_box
    image[y_min:y_max, x_min:x_max] = 220
    return image


def test_box_to_pseudo_mask_shape_matches_image():
    image = _synthetic_image_with_bright_square()
    box = (25.0, 25.0, 75.0, 75.0)
    result = box_to_pseudo_mask(image, box)
    assert result.mask.shape == image.shape[:2]
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 1})


def test_box_to_pseudo_mask_recovers_bright_square_reasonably():
    image = _synthetic_image_with_bright_square(square_box=(30, 30, 70, 70))
    box = (25.0, 25.0, 75.0, 75.0)
    result = box_to_pseudo_mask(image, box)

    true_square = np.zeros(image.shape[:2], dtype=np.uint8)
    true_square[30:70, 30:70] = 1

    intersection = np.logical_and(result.mask, true_square).sum()
    union = np.logical_or(result.mask, true_square).sum()
    iou = intersection / union
    assert iou > 0.5, f"Expected reasonable overlap with the true square, got IoU={iou:.3f}"


def test_box_to_pseudo_mask_rejects_box_outside_image():
    image = _synthetic_image_with_bright_square()
    with pytest.raises(ValueError):
        box_to_pseudo_mask(image, (200.0, 200.0, 300.0, 300.0))


def test_box_to_pseudo_mask_never_extends_past_box_plus_margin():
    """Regression test: an earlier bug marked the area outside the box as
    'probable' background (GC_PR_BGD) instead of certain (GC_BGD), letting
    GrabCut's color model reclassify pixels far outside the box as
    foreground whenever their color resembled the seed points. Observed on
    real data as a mask ~5x larger than its box. A uniformly-colored image
    (no color contrast at all) is an adversarial case for this: if
    background containment isn't enforced structurally, GrabCut's GMM has
    nothing to key off and can hand back nearly the whole image."""
    size = 100
    image = np.full((size, size, 3), 128, dtype=np.uint8)  # uniform gray, no contrast
    box = (30.0, 30.0, 60.0, 60.0)
    box_margin = 2
    result = box_to_pseudo_mask(image, box, box_margin=box_margin)

    x_min, y_min, x_max, y_max = box
    allowed = np.zeros((size, size), dtype=np.uint8)
    allowed[
        max(0, int(y_min) - box_margin) : min(size, int(y_max) + box_margin),
        max(0, int(x_min) - box_margin) : min(size, int(x_max) + box_margin),
    ] = 1

    outside_box_pixels = np.logical_and(result.mask, np.logical_not(allowed)).sum()
    assert outside_box_pixels == 0, (
        f"Mask leaked {outside_box_pixels} px outside the box+margin region "
        "— background containment regressed."
    )


def test_box_to_pseudo_mask_falls_back_to_flood_fill_on_uniform_color():
    """Regression scenario: on real SDZI annotations, GrabCut sometimes
    reclassifies almost the entire box as background because disturbed
    earth doesn't stand out in color from the surrounding terrain, leaving
    only the seed pixels as 'foreground' (observed: 5 px out of a ~17,000
    px box). A uniform-color image (no contrast at all, the extreme case
    of 'no color separation') triggers exactly that GrabCut failure — but
    flood-fill's local, neighbor-by-neighbor growing has no trouble here
    (every neighbor is an exact color match), so the pipeline should
    recover via method='flood_fill' rather than giving up on the full box."""
    size = 100
    image = np.full((size, size, 3), 128, dtype=np.uint8)  # uniform gray, no contrast
    box = (30.0, 30.0, 60.0, 60.0)
    box_margin = 2
    result = box_to_pseudo_mask(image, box, box_margin=box_margin)

    assert result.method == "flood_fill"
    assert result.converged is True
    x_min, y_min, x_max, y_max = box
    expected_full_box = np.zeros((size, size), dtype=np.uint8)
    expected_full_box[
        max(0, int(y_min) - box_margin) : min(size, int(y_max) + box_margin),
        max(0, int(x_min) - box_margin) : min(size, int(x_max) + box_margin),
    ] = 1
    assert np.array_equal(result.mask, expected_full_box)


def test_box_to_pseudo_mask_full_fallback_when_both_methods_fail():
    """When even flood-fill can't find a coherent region — every pixel has
    a distinct value (via a deterministic index pattern, no randomness),
    so no neighbor is ever an exact/near match and GrabCut's global color
    model has nothing consistent to key off either — the function should
    give up and return the full box with method='full_box_fallback'."""
    size = 100
    idx = np.arange(size * size, dtype=np.uint8).reshape(size, size)
    image = np.stack([idx] * 3, axis=-1)
    box = (30.0, 30.0, 60.0, 60.0)
    box_margin = 2
    result = box_to_pseudo_mask(image, box, box_margin=box_margin)

    assert result.method == "full_box_fallback"
    assert result.converged is False
    x_min, y_min, x_max, y_max = box
    expected_full_box = np.zeros((size, size), dtype=np.uint8)
    expected_full_box[
        max(0, int(y_min) - box_margin) : min(size, int(y_max) + box_margin),
        max(0, int(x_min) - box_margin) : min(size, int(x_max) + box_margin),
    ] = 1
    assert np.array_equal(result.mask, expected_full_box)


def test_flood_fill_region_growing_stays_within_box():
    """Direct unit test of the private flood-fill helper: a uniform image
    (guaranteed full growth from any seed) must still respect the box
    bounds passed in, independent of whatever box_to_pseudo_mask does with
    margins."""
    from panoptic_mining.data.points import _flood_fill_region_growing

    size = 50
    image = np.full((size, size, 3), 200, dtype=np.uint8)
    points = np.array([[25.0, 25.0]])
    grown = _flood_fill_region_growing(
        image, points, x_min=10, y_min=10, x_max=40, y_max=40, tolerance=10
    )

    assert grown.shape == (size, size)
    assert grown[10:40, 10:40].all()  # fully grown inside a uniform box
    assert not grown[:10, :].any()
    assert not grown[40:, :].any()
    assert not grown[:, :10].any()
    assert not grown[:, 40:].any()

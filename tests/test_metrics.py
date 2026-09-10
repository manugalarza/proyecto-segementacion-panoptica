"""Unit tests for evaluation/metrics.py — all synthetic, no real dataset needed."""

from __future__ import annotations

import numpy as np
import pytest

from panoptic_mining.evaluation.metrics import (
    average_precision,
    box_iou,
    dice_f1,
    iou,
    mask_based_recall_as_boxes,
    mask_to_bbox,
    match_instances,
    multiclass_panoptic_quality,
    panoptic_quality,
    precision_recall,
    score_stuff_class,
)


def _square_mask(size, box):
    mask = np.zeros((size, size), dtype=np.uint8)
    x_min, y_min, x_max, y_max = box
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def test_iou_identical_masks_is_one():
    m = _square_mask(20, (2, 2, 10, 10))
    assert iou(m, m) == pytest.approx(1.0)


def test_iou_disjoint_masks_is_zero():
    a = _square_mask(20, (0, 0, 5, 5))
    b = _square_mask(20, (10, 10, 15, 15))
    assert iou(a, b) == pytest.approx(0.0)


def test_iou_both_empty_is_one():
    empty = np.zeros((10, 10), dtype=np.uint8)
    assert iou(empty, empty) == pytest.approx(1.0)


def test_dice_f1_identical_masks_is_one():
    m = _square_mask(20, (2, 2, 10, 10))
    assert dice_f1(m, m) == pytest.approx(1.0)


def test_score_stuff_class_perfect_prediction():
    m = _square_mask(20, (2, 2, 10, 10))
    result = score_stuff_class(m, m, "river")
    assert result.iou == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_score_stuff_class_partial_overlap():
    pred = _square_mask(20, (0, 0, 10, 10))  # 100 px
    gt = _square_mask(20, (5, 5, 15, 15))  # 100 px, overlap 5x5=25
    result = score_stuff_class(pred, gt, "sdzi")
    assert result.iou == pytest.approx(25 / (100 + 100 - 25))


def test_match_instances_exact_matches():
    a = _square_mask(20, (0, 0, 5, 5))
    b = _square_mask(20, (10, 10, 15, 15))
    result = match_instances([a, b], [a, b], iou_threshold=0.5)
    assert len(result.matches) == 2
    assert result.unmatched_preds == []
    assert result.unmatched_gts == []


def test_match_instances_below_threshold_unmatched():
    pred = _square_mask(20, (0, 0, 3, 3))
    gt = _square_mask(20, (0, 0, 10, 10))
    result = match_instances([pred], [gt], iou_threshold=0.5)
    assert result.matches == []
    assert result.unmatched_preds == [0]
    assert result.unmatched_gts == [0]


def test_precision_recall_perfect():
    a = _square_mask(20, (0, 0, 5, 5))
    b = _square_mask(20, (10, 10, 15, 15))
    precision, recall = precision_recall([a, b], [a, b])
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)


def test_precision_recall_false_positive_and_negative():
    pred = [
        _square_mask(20, (0, 0, 5, 5)),
        _square_mask(20, (15, 15, 19, 19)),  # spurious FP
    ]
    gt = [
        _square_mask(20, (0, 0, 5, 5)),
        _square_mask(20, (10, 10, 15, 15)),  # missed -> FN
    ]
    precision, recall = precision_recall(pred, gt)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)


def test_average_precision_perfect_ranking_is_one():
    gt = [_square_mask(20, (0, 0, 5, 5)), _square_mask(20, (10, 10, 15, 15))]
    pred = gt
    scores = [0.9, 0.8]
    ap = average_precision(pred, scores, gt, iou_threshold=0.5)
    assert ap == pytest.approx(1.0)


def test_average_precision_no_ground_truth_no_predictions_is_one():
    assert average_precision([], [], [], iou_threshold=0.5) == pytest.approx(1.0)


def test_average_precision_no_ground_truth_with_false_positive_is_zero():
    pred = [_square_mask(20, (0, 0, 5, 5))]
    assert average_precision(pred, [0.9], [], iou_threshold=0.5) == pytest.approx(0.0)


def test_panoptic_quality_perfect_prediction():
    a = _square_mask(20, (0, 0, 5, 5))
    b = _square_mask(20, (10, 10, 15, 15))
    result = panoptic_quality([a, b], [a, b])
    assert result.sq == pytest.approx(1.0)
    assert result.rq == pytest.approx(1.0)
    assert result.pq == pytest.approx(1.0)
    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_panoptic_quality_matches_sq_times_rq():
    pred = [_square_mask(20, (0, 0, 8, 8))]
    gt = [_square_mask(20, (2, 2, 10, 10))]
    result = panoptic_quality(pred, gt, iou_threshold=0.3)
    assert result.pq == pytest.approx(result.sq * result.rq)


def test_multiclass_panoptic_quality_averages_across_classes():
    perfect = [_square_mask(20, (0, 0, 5, 5))]
    preds = {"river": perfect, "sdzi": []}
    gts = {"river": perfect, "sdzi": [_square_mask(20, (10, 10, 15, 15))]}
    result = multiclass_panoptic_quality(preds, gts)
    assert result.per_class["river"].pq == pytest.approx(1.0)
    assert result.per_class["sdzi"].pq == pytest.approx(0.0)  # 1 FN, 0 TP
    assert result.mean_pq == pytest.approx(0.5)


def test_mask_to_bbox_matches_known_square():
    mask = _square_mask(20, (3, 4, 10, 12))
    box = mask_to_bbox(mask)
    assert box == (3, 4, 10, 12)


def test_mask_to_bbox_empty_mask_returns_none():
    assert mask_to_bbox(np.zeros((10, 10), dtype=np.uint8)) is None


def test_box_iou_identical_boxes_is_one():
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_disjoint_boxes_is_zero():
    assert box_iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 15.0, 15.0)) == pytest.approx(0.0)


def test_mask_based_recall_as_boxes_matches_baseline_style_matching():
    # Predicted mask whose tight bbox exactly matches one of two GT boxes;
    # the other GT box has no corresponding prediction (a miss).
    pred_masks = [_square_mask(30, (2, 2, 10, 10))]
    gt_boxes = [(2.0, 2.0, 10.0, 10.0), (20.0, 20.0, 25.0, 25.0)]
    recall = mask_based_recall_as_boxes(pred_masks, gt_boxes, iou_threshold=0.5)
    assert recall == pytest.approx(0.5)


def test_mask_based_recall_as_boxes_no_ground_truth_is_one():
    assert mask_based_recall_as_boxes([], [], iou_threshold=0.5) == pytest.approx(1.0)

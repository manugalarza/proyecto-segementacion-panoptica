"""Evaluation metrics: PQ/SQ/RQ, stuff IoU/F1, thing precision/recall/mask AP,
and the bounding-box-equivalence helper used to compare against the
"Eyes in the Sky" baseline (which reports box-level recall, not mask-level).

All functions operate on plain numpy arrays so they can be unit-tested with
synthetic masks — none of this depends on the real dataset being downloaded.

References
----------
- Kirillov et al., "Panoptic Segmentation" (CVPR 2019) for PQ/SQ/RQ.
- Everingham et al. (PASCAL VOC) / Lin et al. (COCO) for AP via
  precision-recall interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Mask = np.ndarray  # binary array, shape (H, W), values in {0, 1}


def iou(mask_a: Mask, mask_b: Mask) -> float:
    """Intersection-over-Union between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def dice_f1(mask_a: Mask, mask_b: Mask) -> float:
    """Dice coefficient / F1 score between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 1.0
    return float(2 * intersection) / float(total)


# ---------------------------------------------------------------------------
# Stuff classes (river, SDZI): single semantic mask per class per image.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StuffScore:
    class_name: str
    iou: float
    f1: float
    recall: float
    precision: float


def score_stuff_class(pred_mask: Mask, gt_mask: Mask, class_name: str) -> StuffScore:
    """IoU/F1/precision/recall for one stuff class on one image (or a stack
    of images already flattened into a single mask pair)."""
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 1.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 1.0

    return StuffScore(
        class_name=class_name,
        iou=iou(pred_mask, gt_mask),
        f1=dice_f1(pred_mask, gt_mask),
        recall=recall,
        precision=precision,
    )


# ---------------------------------------------------------------------------
# Thing classes: instance-level, matched greedily by descending IoU (a
# simplification of Hungarian matching that is exact whenever the optimal
# assignment is also the greedy one, which holds for the well-separated
# instances typical of aerial mining imagery; documented here as a known
# simplification rather than silently assumed).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchResult:
    matches: list[tuple[int, int, float]]  # (pred_idx, gt_idx, iou)
    unmatched_preds: list[int]
    unmatched_gts: list[int]


def match_instances(
    pred_masks: list[Mask], gt_masks: list[Mask], iou_threshold: float = 0.5
) -> MatchResult:
    """Greedily match predicted instances to ground-truth instances by IoU."""
    pairs: list[tuple[float, int, int]] = []
    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_masks):
            score = iou(pm, gm)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pi, gi in pairs:
        if pi in matched_preds or gi in matched_gts:
            continue
        matched_preds.add(pi)
        matched_gts.add(gi)
        matches.append((pi, gi, score))

    unmatched_preds = [i for i in range(len(pred_masks)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_masks)) if i not in matched_gts]
    return MatchResult(matches=matches, unmatched_preds=unmatched_preds, unmatched_gts=unmatched_gts)


def precision_recall(
    pred_masks: list[Mask], gt_masks: list[Mask], iou_threshold: float = 0.5
) -> tuple[float, float]:
    """Instance-level precision/recall at a single IoU threshold."""
    result = match_instances(pred_masks, gt_masks, iou_threshold)
    tp = len(result.matches)
    fp = len(result.unmatched_preds)
    fn = len(result.unmatched_gts)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return precision, recall


def average_precision(
    pred_masks: list[Mask],
    pred_scores: list[float],
    gt_masks: list[Mask],
    iou_threshold: float = 0.5,
) -> float:
    """Mask AP at a single IoU threshold, via all-points (COCO-style)
    precision-recall interpolation.

    Predictions are ranked by confidence score (descending); for each rank
    we mark a hit (matches an as-yet-unmatched GT at >= iou_threshold) or a
    miss, accumulate precision/recall, then integrate the interpolated
    precision-recall curve.
    """
    if len(pred_masks) != len(pred_scores):
        raise ValueError("pred_masks and pred_scores must be the same length")
    n_gt = len(gt_masks)
    if n_gt == 0:
        return 1.0 if len(pred_masks) == 0 else 0.0

    order = np.argsort(pred_scores)[::-1]
    matched_gt: set[int] = set()
    tps = np.zeros(len(order))
    fps = np.zeros(len(order))

    for rank, idx in enumerate(order):
        pm = pred_masks[idx]
        best_iou, best_gi = 0.0, -1
        for gi, gm in enumerate(gt_masks):
            if gi in matched_gt:
                continue
            score = iou(pm, gm)
            if score > best_iou:
                best_iou, best_gi = score, gi
        if best_iou >= iou_threshold:
            tps[rank] = 1
            matched_gt.add(best_gi)
        else:
            fps[rank] = 1

    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(fps)
    recalls = cum_tp / n_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)

    # All-points interpolation: precision envelope is monotonically
    # non-increasing when read right-to-left, then integrate over recall.
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    change_points = np.where(recalls[1:] != recalls[:-1])[0]
    ap = float(np.sum((recalls[change_points + 1] - recalls[change_points]) * precisions[change_points + 1]))
    return ap


# ---------------------------------------------------------------------------
# Panoptic Quality (PQ = SQ x RQ), computed per class then averaged.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanopticQualityResult:
    pq: float
    sq: float
    rq: float
    true_positives: int
    false_positives: int
    false_negatives: int


def panoptic_quality(
    pred_masks: list[Mask], gt_masks: list[Mask], iou_threshold: float = 0.5
) -> PanopticQualityResult:
    """PQ/SQ/RQ for a single class, following Kirillov et al.: matches are
    valid only above ``iou_threshold`` (canonically 0.5, which also
    guarantees at most one valid match per ground-truth segment)."""
    result = match_instances(pred_masks, gt_masks, iou_threshold)
    tp = len(result.matches)
    fp = len(result.unmatched_preds)
    fn = len(result.unmatched_gts)

    sum_iou = sum(score for _, _, score in result.matches)
    sq = sum_iou / tp if tp > 0 else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = tp / denom if denom > 0 else 0.0
    pq = sq * rq

    return PanopticQualityResult(pq=pq, sq=sq, rq=rq, true_positives=tp, false_positives=fp, false_negatives=fn)


@dataclass
class MultiClassPanopticQuality:
    per_class: dict[str, PanopticQualityResult] = field(default_factory=dict)

    @property
    def mean_pq(self) -> float:
        if not self.per_class:
            return 0.0
        return float(np.mean([r.pq for r in self.per_class.values()]))

    @property
    def mean_sq(self) -> float:
        if not self.per_class:
            return 0.0
        return float(np.mean([r.sq for r in self.per_class.values()]))

    @property
    def mean_rq(self) -> float:
        if not self.per_class:
            return 0.0
        return float(np.mean([r.rq for r in self.per_class.values()]))


def multiclass_panoptic_quality(
    pred_masks_by_class: dict[str, list[Mask]],
    gt_masks_by_class: dict[str, list[Mask]],
    iou_threshold: float = 0.5,
) -> MultiClassPanopticQuality:
    """PQ/SQ/RQ per class over the union of classes present in either
    predictions or ground truth (a class missing from one side is scored,
    not silently skipped, since silently dropping FN-only classes would
    inflate the mean)."""
    classes = sorted(set(pred_masks_by_class) | set(gt_masks_by_class))
    result = MultiClassPanopticQuality()
    for cls in classes:
        preds = pred_masks_by_class.get(cls, [])
        gts = gt_masks_by_class.get(cls, [])
        result.per_class[cls] = panoptic_quality(preds, gts, iou_threshold)
    return result


# ---------------------------------------------------------------------------
# Baseline comparability: Eyes in the Sky reports box-level recall, not
# mask-level. To compare fairly we derive a tight bounding box from each
# predicted mask and re-run box-based matching against the baseline's own
# box ground truth, rather than claiming "we beat 0.425" from a
# mask-vs-mask number that was never computed the same way.
# ---------------------------------------------------------------------------


def mask_to_bbox(mask: Mask) -> tuple[int, int, int, int] | None:
    """Tight (x_min, y_min, x_max, y_max) bounding box of a binary mask's
    foreground pixels, or None if the mask is empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax_min, ay_min, ax_max, ay_max = box_a
    bx_min, by_min, bx_max, by_max = box_b

    inter_x_min, inter_y_min = max(ax_min, bx_min), max(ay_min, by_min)
    inter_x_max, inter_y_max = min(ax_max, bx_max), min(ay_max, by_max)
    inter_w, inter_h = max(0.0, inter_x_max - inter_x_min), max(0.0, inter_y_max - inter_y_min)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax_max - ax_min) * max(0.0, ay_max - ay_min)
    area_b = max(0.0, bx_max - bx_min) * max(0.0, by_max - by_min)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def mask_based_recall_as_boxes(
    pred_masks: list[Mask], gt_boxes: list[tuple[float, float, float, float]], iou_threshold: float = 0.5
) -> float:
    """Recall of predicted masks (converted to tight boxes) against the
    baseline's original box-level ground truth, at the same IoU threshold
    Eyes in the Sky used (0.5). This is the number that is directly
    comparable to their reported 0.425 for SDZI — a raw mask-IoU recall is
    NOT comparable and should not be reported as such."""
    pred_boxes = [b for b in (mask_to_bbox(m) for m in pred_masks) if b is not None]
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0

    matched_gt: set[int] = set()
    pairs = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            score = box_iou(pb, gb)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))
    pairs.sort(reverse=True)

    matched_pred: set[int] = set()
    for score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    return len(matched_gt) / len(gt_boxes)

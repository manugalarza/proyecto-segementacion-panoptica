"""
baseline/yolo_eval.py
Reproduce Eyes in the Sky YOLOv11 recall on test set.
Calculates recall for SDZI (class 4) using IoU@0.5.
"""

import json
from pathlib import Path
from collections import defaultdict

import torch
from ultralytics import YOLO
import numpy as np


def iou_box(box1, box2):
    """IoU between two boxes in [x_min, y_min, x_max, y_max] format."""
    x_min_i = max(box1[0], box2[0])
    y_min_i = max(box1[1], box2[1])
    x_max_i = min(box1[2], box2[2])
    y_max_i = min(box1[3], box2[3])

    if x_max_i < x_min_i or y_max_i < y_min_i:
        return 0.0

    inter = (x_max_i - x_min_i) * (y_max_i - y_min_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def yolo_to_corner(x_center, y_center, width, height, img_w, img_h):
    """Convert YOLO format (normalized center) to corner format [x_min, y_min, x_max, y_max]."""
    x_min = (x_center - width / 2) * img_w
    y_min = (y_center - height / 2) * img_h
    x_max = (x_center + width / 2) * img_w
    y_max = (y_center + height / 2) * img_h
    return [x_min, y_min, x_max, y_max]


def load_gt_boxes(label_path, img_w, img_h):
    """Load ground truth boxes from YOLO .txt file."""
    boxes_by_class = defaultdict(list)
    if not label_path.exists():
        return boxes_by_class

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_c, y_c, w, h = map(float, parts[1:5])
            box = yolo_to_corner(x_c, y_c, w, h, img_w, img_h)
            boxes_by_class[class_id].append(box)

    return boxes_by_class


def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Match predictions to GT boxes using greedy IoU matching."""
    matched_gt = set()
    tp = 0

    # Sort predictions by confidence (descending)
    pred_boxes_sorted = sorted(pred_boxes, key=lambda x: x[1], reverse=True)

    for pred_box, conf in pred_boxes_sorted:
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = iou_box(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fn


def main():
    dataset_root = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo")
    test_images_dir = dataset_root / "test" / "images"
    test_labels_dir = dataset_root / "test" / "labels"
    model_path = r"..\imagenes recortado\yolov11_best100.pt"

    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # SDZI is class 4
    target_class = 4

    total_tp = 0
    total_fn = 0
    image_results = []

    image_files = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    print(f"Found {len(image_files)} test images.")

    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(image_files)}]")

        # Run inference
        results = model(str(img_path), conf=0.25, verbose=False)
        result = results[0]

        img_w, img_h = result.orig_shape[1], result.orig_shape[0]

        # Extract predictions for target class
        pred_boxes = []
        if result.boxes is not None:
            for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if int(cls_id) == target_class:
                    pred_boxes.append((box.cpu().numpy(), float(conf)))

        # Load ground truth
        label_path = test_labels_dir / img_path.stem / ".txt"
        if not label_path.exists():
            label_path = test_labels_dir / (img_path.stem + ".txt")

        gt_boxes_all = load_gt_boxes(label_path, img_w, img_h)
        gt_boxes = gt_boxes_all.get(target_class, [])

        # Match
        tp, fn = match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5)
        total_tp += tp
        total_fn += fn

        image_results.append({
            "image": img_path.name,
            "pred_count": len(pred_boxes),
            "gt_count": len(gt_boxes),
            "tp": tp,
            "fn": fn
        })

    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print(f"\n=== YOLOv11 Baseline Results (SDZI, IoU@0.5) ===")
    print(f"Total TP: {total_tp}")
    print(f"Total FN: {total_fn}")
    print(f"Recall: {recall:.4f}")
    print(f"\nExpected (Eyes in the Sky): 0.425")
    print(f"Difference: {recall - 0.425:.4f}")

    # Save detailed results
    output = {
        "model": "yolov11_best100.pt",
        "target_class": "SDZI (4)",
        "iou_threshold": 0.5,
        "total_tp": total_tp,
        "total_fn": total_fn,
        "recall": recall,
        "per_image": image_results
    }

    with open("baseline_yolo_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to baseline_yolo_results.json")


if __name__ == "__main__":
    main()
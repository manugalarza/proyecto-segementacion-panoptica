"""Debug script: test YOLO with different confidence thresholds."""
import json
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
import numpy as np

def iou_box(box1, box2):
    """IoU between two boxes."""
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
    """Convert YOLO format to corner format."""
    x_min = (x_center - width / 2) * img_w
    y_min = (y_center - height / 2) * img_h
    x_max = (x_center + width / 2) * img_w
    y_max = (y_center + height / 2) * img_h
    return [x_min, y_min, x_max, y_max]

def load_gt_boxes(label_path, img_w, img_h):
    """Load GT boxes."""
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

def test_confidence(model, dataset_root, target_class, conf_threshold):
    """Test at a specific confidence threshold."""
    test_images_dir = dataset_root / "test" / "images"
    test_labels_dir = dataset_root / "test" / "labels"
    
    total_tp = 0
    total_fn = 0
    total_pred = 0
    total_gt = 0
    class_dist = defaultdict(int)
    conf_dist = []
    
    image_files = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    
    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0:
            print(f"    [{idx + 1}/{len(image_files)}]")
        
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        result = results[0]
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
        
        # Count all predictions by class
        if result.boxes is not None:
            for cls_id, conf in zip(result.boxes.cls, result.boxes.conf):
                class_dist[int(cls_id)] += 1
                conf_dist.append(float(conf))
        
        # Extract predictions for target class
        pred_boxes = []
        if result.boxes is not None:
            for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if int(cls_id) == target_class:
                    pred_boxes.append((box.cpu().numpy(), float(conf)))
        
        total_pred += len(pred_boxes)
        
        # Load GT
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_boxes_all = load_gt_boxes(label_path, img_w, img_h)
        gt_boxes = gt_boxes_all.get(target_class, [])
        total_gt += len(gt_boxes)
        
        # Match (greedy)
        matched_gt = set()
        for pred_box, conf in sorted(pred_boxes, key=lambda x: x[1], reverse=True):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = iou_box(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= 0.5 and best_gt_idx >= 0:
                total_tp += 1
                matched_gt.add(best_gt_idx)
        
        total_fn += len(gt_boxes) - len(matched_gt)
    
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    print(f"\n  conf={conf_threshold}:")
    print(f"    Recall: {recall:.4f}")
    print(f"    TP: {total_tp}, FN: {total_fn}")
    print(f"    Total predictions (class {target_class}): {total_pred}")
    print(f"    Total GT (class {target_class}): {total_gt}")
    print(f"    All predictions by class: {dict(class_dist)}")
    if conf_dist:
        print(f"    Confidence stats (all classes): min={min(conf_dist):.4f}, max={max(conf_dist):.4f}, mean={np.mean(conf_dist):.4f}")

def main():
    dataset_root = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo")
    model_path = r"..\imagenes recortado\yolov11_best100.pt"
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    target_class = 4  # SDZI
    print(f"\nTesting different confidence thresholds for SDZI (class {target_class}):")
    print(f"Expected recall (Eyes in the Sky): 0.425\n")
    
    for conf in [0.25, 0.15, 0.10, 0.05]:
        test_confidence(model, dataset_root, target_class, conf)

if __name__ == "__main__":
    main()

"""Evaluation script for Panoptic FCN using mask_based_recall_as_boxes."""

import json
import yaml
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, "src")
from panoptic_mining.evaluation.metrics import mask_based_recall_as_boxes


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Load GT boxes for SDZI (class 4)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            if class_id != 4:  # SDZI only
                continue
            x_c, y_c, w, h = map(float, parts[1:5])
            x_min = (x_c - w / 2) * img_w
            y_min = (y_c - h / 2) * img_h
            x_max = (x_c + w / 2) * img_w
            y_max = (y_c + h / 2) * img_h
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes


def extract_thing_masks(panoptic_map: np.ndarray, thing_class_id: int, num_stuff: int) -> list[np.ndarray]:
    """Extract binary masks for thing instances."""
    masks = []
    for panoptic_id in np.unique(panoptic_map):
        panoptic_id = int(panoptic_id)
        if panoptic_id // 1000 == thing_class_id + num_stuff:
            mask = (panoptic_map == panoptic_id).astype(np.uint8)
            masks.append(mask)
    return masks


def main():
    repo_root = Path(".")
    predictions_dir = repo_root / "predictions" / "epoch_500"
    dataset_yaml = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo\dataset.yaml")
    dataset_root = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo")
    test_images_dir = dataset_root / "test" / "images"
    test_labels_dir = dataset_root / "test" / "labels"
    
    metadata_path = predictions_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    print(f"Stuff classes: {metadata['stuff_classes']}")
    print(f"Thing classes: {metadata['thing_classes']}")
    print(f"\nEvaluating on {len(metadata['images'])} test images\n")
    
    num_stuff = len(metadata['stuff_classes'])
    thing_classes = metadata['thing_classes']
    
    total_tp = 0
    total_fn = 0
    image_results = []
    
    for idx, img_path in enumerate(sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(metadata['images'])}]")
        
        original_img = Image.open(img_path).convert("RGB")
        img_w, img_h = original_img.size
        
        panoptic_path = predictions_dir / "panoptic" / f"{img_path.stem}.npy"
        if not panoptic_path.exists():
            continue
        
        panoptic_map = np.load(panoptic_path)
        
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)
        
        # Evaluate first thing class (vehicle, id=0) as proxy for SDZI detection
        pred_masks = extract_thing_masks(panoptic_map, thing_class_id=0, num_stuff=num_stuff)
        
        recall = mask_based_recall_as_boxes(pred_masks, gt_boxes, iou_threshold=0.5)
        tp = int(recall * len(gt_boxes)) if len(gt_boxes) > 0 else 0
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fn += fn
        
        image_results.append({
            "image": img_path.name,
            "pred_count": len(pred_masks),
            "gt_count": len(gt_boxes),
            "tp": tp,
            "fn": fn,
            "recall": recall,
        })
    
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    print(f"\n=== Panoptic FCN Results (Thing Class 0, IoU@0.5) ===")
    print(f"Total TP: {total_tp}")
    print(f"Total FN: {total_fn}")
    print(f"Recall: {recall:.4f}")
    print(f"\nYOLOv11 Baseline Recall: 0.0274")
    print(f"Difference: {recall - 0.0274:.4f}")
    
    output = {
        "model": "panoptic_fcn_epoch_500",
        "thing_class_evaluated": "0 (vehicle)",
        "iou_threshold": 0.5,
        "total_tp": total_tp,
        "total_fn": total_fn,
        "recall": recall,
        "per_image": image_results
    }
    
    output_path = repo_root / "panoptic_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to panoptic_eval_results.json")


if __name__ == "__main__":
    main()

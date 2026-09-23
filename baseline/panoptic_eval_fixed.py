"""Corrected evaluation: all thing instances (any class) vs SDZI boxes."""

import json
from pathlib import Path
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


def extract_all_thing_masks(panoptic_map: np.ndarray, num_stuff: int) -> list[np.ndarray]:
    """Extract binary masks for ALL thing instances (any class)."""
    masks = []
    for panoptic_id in np.unique(panoptic_map):
        panoptic_id = int(panoptic_id)
        # thing_id = (thing_class_id + num_stuff) * 1000 + instance_idx
        # Extract if it's a thing (panoptic_id >= num_stuff * 1000)
        if panoptic_id >= num_stuff * 1000:
            mask = (panoptic_map == panoptic_id).astype(np.uint8)
            masks.append(mask)
    return masks


def main():
    predictions_dir = Path("predictions/epoch_500")
    dataset_root = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo")
    test_images_dir = dataset_root / "test" / "images"
    test_labels_dir = dataset_root / "test" / "labels"
    
    metadata_path = predictions_dir / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    num_stuff = len(metadata['stuff_classes'])
    
    print(f"Evaluating Panoptic FCN (ALL thing instances vs SDZI boxes)")
    print(f"Stuff classes: {metadata['stuff_classes']}")
    print(f"Thing classes: {metadata['thing_classes']}\n")
    
    total_tp = 0
    total_fn = 0
    total_instances_generated = 0
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
        
        # Extract ALL thing instances (any class)
        pred_masks = extract_all_thing_masks(panoptic_map, num_stuff)
        total_instances_generated += len(pred_masks)
        
        # Load GT boxes for SDZI
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(label_path, img_w, img_h)
        
        # Compute recall
        recall = mask_based_recall_as_boxes(pred_masks, gt_boxes, iou_threshold=0.5)
        tp = int(recall * len(gt_boxes)) if len(gt_boxes) > 0 else 0
        fn = len(gt_boxes) - tp
        
        total_tp += tp
        total_fn += fn
        
        image_results.append({
            "image": img_path.name,
            "pred_instances": len(pred_masks),
            "gt_sdzi_boxes": len(gt_boxes),
            "tp": tp,
            "fn": fn,
            "recall": recall,
        })
    
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    
    print(f"\n=== Panoptic FCN Results (ALL instances vs SDZI, IoU@0.5) ===")
    print(f"Total instances generated: {total_instances_generated}")
    print(f"Total TP: {total_tp}")
    print(f"Total FN: {total_fn}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"\nYOLOv11 Baseline Recall: 0.0274")
    print(f"Difference: {overall_recall - 0.0274:.4f}")
    
    output = {
        "model": "panoptic_fcn_epoch_500",
        "evaluation_method": "all_thing_instances vs SDZI_boxes",
        "iou_threshold": 0.5,
        "total_instances_generated": total_instances_generated,
        "total_tp": total_tp,
        "total_fn": total_fn,
        "recall": overall_recall,
        "caveat": "Evaluated in dataset with ~141 overlaps train-test. Will recalculate in leakage-safe split.",
        "per_image": image_results
    }
    
    output_path = Path("panoptic_eval_fixed.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nDetailed results saved to panoptic_eval_fixed.json")
    print(f"⚠️  CAVEAT: Results use dataset with leakage. Will be recalculated in split leakage-safe.")


if __name__ == "__main__":
    main()

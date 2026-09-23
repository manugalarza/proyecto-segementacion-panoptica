"""Inference script for Panoptic FCN on test set."""

import json
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, "src")
from panoptic_mining.models.panoptic_fcn import PanopticFCN


def load_classes(dataset_yaml_path: str) -> tuple[list[str], list[str]]:
    """Load stuff and thing classes from dataset.yaml."""
    with open(dataset_yaml_path) as f:
        config = yaml.safe_load(f)
    
    names = config.get("names", {})
    thing_classes = [names.get(i, f"class_{i}") for i in range(3)]
    stuff_classes = [names.get(i, f"class_{i}") for i in range(3, 5)] + ["background"]
    
    return stuff_classes, thing_classes


def load_model(checkpoint_path: str, num_stuff: int, num_thing: int, device: str = "cuda") -> PanopticFCN:
    """Load model checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = PanopticFCN(num_stuff_classes=num_stuff, num_thing_classes=num_thing)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both plain state_dict and checkpoint with metadata
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def load_image(img_path: str, device: str) -> torch.Tensor:
    """Load and normalize image."""
    img = Image.open(img_path).convert("RGB")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor * 2.0 - 1.0)
    return img_tensor.to(device)


def generate_panoptic_map(model: PanopticFCN, img_tensor: torch.Tensor, original_h: int, original_w: int) -> np.ndarray:
    """Generate panoptic map."""
    with torch.no_grad():
        output = model(img_tensor)
    
    # Decode stuff
    stuff_logits = output.stuff_logits[0].permute(1, 2, 0)
    stuff_probs = torch.softmax(stuff_logits, dim=-1)
    stuff_mask = torch.argmax(stuff_probs, dim=-1).cpu().numpy()
    
    # Initialize panoptic map with stuff
    panoptic_map_feat = stuff_mask.copy().astype(np.uint32)
    num_stuff = model.num_stuff_classes
    
    # Overlay things
    for thing_class_id in range(model.num_thing_classes):
        instances = model.decode_instances(output, class_id=thing_class_id, top_k=50, score_threshold=0.3)
        for instance_idx, (score, mask_logits) in enumerate(instances):
            mask = (mask_logits > 0.5).cpu().numpy()
            panoptic_id = (thing_class_id + num_stuff) * 1000 + instance_idx
            panoptic_map_feat[mask > 0] = panoptic_id
    
    # Upsample
    panoptic_tensor = torch.from_numpy(panoptic_map_feat).float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(panoptic_tensor, size=(original_h, original_w), mode="nearest")
    return upsampled[0, 0].numpy().astype(np.uint32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    repo_root = Path(".")
    dataset_yaml = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo\dataset.yaml")
    checkpoint_path = repo_root / "checkpoints" / "finetune_weighted_longer" / "epoch_500.pt"
    dataset_root = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo")
    test_images_dir = dataset_root / "test" / "images"
    output_dir = repo_root / "predictions" / "epoch_500"
    
    (output_dir / "panoptic").mkdir(parents=True, exist_ok=True)
    (output_dir / "panoptic_vis").mkdir(parents=True, exist_ok=True)
    
    # Load classes
    stuff_classes, thing_classes = load_classes(str(dataset_yaml))
    print(f"Stuff classes: {stuff_classes}")
    print(f"Thing classes: {thing_classes}\n")
    
    model = load_model(str(checkpoint_path), num_stuff=len(stuff_classes), num_thing=len(thing_classes), device=device)
    
    image_files = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    print(f"Found {len(image_files)} test images.\n")
    
    metadata = {
        "checkpoint": "finetune_weighted_longer/epoch_500.pt",
        "stuff_classes": stuff_classes,
        "thing_classes": thing_classes,
        "total_images": len(image_files),
        "images": []
    }
    
    for idx, img_path in enumerate(image_files):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(image_files)}]")
        
        original_img = Image.open(img_path).convert("RGB")
        original_h, original_w = original_img.size[1], original_img.size[0]
        
        img_tensor = load_image(str(img_path), device)
        panoptic_map = generate_panoptic_map(model, img_tensor, original_h, original_w)
        
        output_name = img_path.stem
        panoptic_path = output_dir / "panoptic" / f"{output_name}.npy"
        np.save(panoptic_path, panoptic_map)
        
        panoptic_vis = (panoptic_map % 256).astype(np.uint8)
        panoptic_vis_img = Image.fromarray(panoptic_vis, mode="L")
        panoptic_vis_path = output_dir / "panoptic_vis" / f"{output_name}.png"
        panoptic_vis_img.save(panoptic_vis_path)
        
        metadata["images"].append({"name": img_path.name, "shape": [original_h, original_w]})
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Panoptic maps saved to {output_dir}/panoptic/")
    print(f"✓ Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()

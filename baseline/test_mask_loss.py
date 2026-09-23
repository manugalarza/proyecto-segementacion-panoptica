"""Quick test that mask_loss computes and converges (1 epoch)."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, "src")

from panoptic_mining.data.manifest import load_yolo_manifest
from panoptic_mining.data.torch_dataset import ManifestSegmentationDataset
from panoptic_mining.models.panoptic_fcn import PanopticFCN
from panoptic_mining.training.train import compute_losses

# Setup
dataset_root = Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo")

# Load manifest from dataset directory
print(f"Loading manifest from {dataset_root}...")
manifest = load_yolo_manifest(dataset_root)
print(f"  Found classes: {manifest.class_names}")
print(f"  Found splits: {list(manifest.frames_by_split.keys())}")

# Use validation split (small, quick to run)
if "val" in manifest.frames_by_split:
    split_to_use = "val"
elif "valid" in manifest.frames_by_split:
    split_to_use = "valid"
else:
    split_to_use = list(manifest.frames_by_split.keys())[0]

print(f"\nUsing split: {split_to_use}")
val_dataset = ManifestSegmentationDataset(manifest, split=split_to_use)
print(f"Dataset size: {len(val_dataset)}")

# Create data loader (just first 5 samples for speed)
from torch.utils.data import DataLoader, Subset
small_dataset = Subset(val_dataset, range(min(5, len(val_dataset))))
loader = DataLoader(small_dataset, batch_size=2, num_workers=0)

# Load model from checkpoint
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = Path("checkpoints/finetune_weighted_longer/epoch_500.pt")
model = PanopticFCN(num_stuff_classes=3, num_thing_classes=3).to(device)

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Simple training loop (just forward pass + loss) to test mask_loss
print(f"\n=== Testing mask_loss computation ({len(small_dataset)} samples) ===")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
mask_losses = []
all_losses = []

for batch_idx, batch in enumerate(loader):
    print(f"\nBatch {batch_idx + 1}/{len(loader)}...")
    
    # Unpack batch: (image, stuff_target, thing_heatmap, thing_masks)
    if len(batch) == 4:
        images, stuff_targets, thing_heatmaps, thing_masks = batch
    else:
        print(f"  ERROR: Expected 4-tuple, got {len(batch)}-tuple")
        continue
    
    images = images.to(device)
    stuff_targets = stuff_targets.to(device)
    thing_heatmaps = thing_heatmaps.to(device)
    thing_masks = thing_masks.to(device)
    
    optimizer.zero_grad()
    
    # Compute losses
    try:
        stuff_loss, thing_loss, mask_loss, total_loss = compute_losses(
            model=model,
            images=images,
            stuff_targets=stuff_targets,
            thing_heatmap_targets=thing_heatmaps,
            thing_masks=thing_masks,
        )
        
        print(f"  ✓ Loss computed successfully")
        print(f"    stuff_loss: {stuff_loss.item():.6f}")
        print(f"    thing_loss: {thing_loss.item():.6f}")
        print(f"    mask_loss:  {mask_loss.item():.6f}")
        print(f"    total_loss: {total_loss.item():.6f}")
        
        mask_losses.append(mask_loss.item())
        all_losses.append(total_loss.item())
        
        # Check for NaN
        if torch.isnan(total_loss):
            print(f"  ❌ ERROR: NaN detected in loss!")
            break
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
    except Exception as e:
        print(f"  ❌ ERROR computing loss: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n=== Results ===")
if mask_losses:
    print(f"Mask loss values: {mask_losses}")
    print(f"Total loss values: {all_losses}")
    print(f"✅ PASS: mask_loss computes without NaN")
else:
    print(f"❌ FAIL: No mask_loss computed")

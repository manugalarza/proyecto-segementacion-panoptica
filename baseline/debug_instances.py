"""Debug: ¿el modelo está generando instancias de things?"""
import torch
from pathlib import Path
import sys
sys.path.insert(0, "src")
from panoptic_mining.models.panoptic_fcn import PanopticFCN
from PIL import Image
import numpy as np

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = Path("checkpoints/finetune_weighted_longer/epoch_500.pt")
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint["model_state_dict"]

model = PanopticFCN(num_stuff_classes=3, num_thing_classes=3)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Test on first image
test_img = list(Path(r"..\imagenes recortado\dataset_split_completo\dataset_split_completo\test\images").glob("*.jpg"))[0]
img = Image.open(test_img).convert("RGB")
img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
img_tensor = (img_tensor * 2.0 - 1.0).to(device)

print(f"Testing on: {test_img.name}")
print(f"Image shape: {img_tensor.shape}\n")

with torch.no_grad():
    output = model(img_tensor)

print("=== THING INSTANCES ===")
for thing_class_id in range(model.num_thing_classes):
    instances = model.decode_instances(output, class_id=thing_class_id, top_k=100, score_threshold=0.1)
    print(f"Thing class {thing_class_id}: {len(instances)} instances (threshold=0.1)")
    if instances:
        for i, (score, mask) in enumerate(instances[:3]):
            print(f"  Instance {i}: score={score:.4f}, mask_pixels={mask.sum().item()}")

print("\n=== STUFF LOGITS ===")
stuff_logits = output.stuff_logits[0]
stuff_probs = torch.softmax(stuff_logits, dim=0)
for stuff_class_id in range(model.num_stuff_classes):
    prob = stuff_probs[stuff_class_id].mean().item()
    print(f"Stuff class {stuff_class_id}: avg_prob={prob:.4f}")

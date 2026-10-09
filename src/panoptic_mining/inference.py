"""Single inference path shared by every evaluation / visualization script.

It reproduces EXACTLY the preprocessing of ``data/torch_dataset.py``
(the training path):

    cv2.imread  -> BGR, no color conversion
    cv2.resize  -> (image_size[1], image_size[0]) i.e. 512x512 by default
    / 255.0     -> range [0, 1], no mean/std normalization

Background (2026-10-08 review): ``baseline/inference_panoptic_fcn.py`` and
``baseline/debug_instances.py`` used PIL RGB, no resize and a [-1, 1]
range, so every number produced through them came from inputs the model
never saw during training. Any RGB conversion belongs in visualization
code only, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from panoptic_mining.data.torch_dataset import DEFAULT_CLASS_SPLIT, ClassSplit
from panoptic_mining.models.panoptic_fcn import PanopticFCN


@dataclass
class Prediction:
    stuff_probs: np.ndarray  # (num_stuff_classes, H, W) float32 softmax, ORIGINAL image resolution
    stuff_argmax: np.ndarray  # (H, W) int, ORIGINAL image resolution
    original_size: tuple[int, int]  # (height, width)


def load_image_like_training(image_path: str | Path, image_size: tuple[int, int] = (512, 512)) -> tuple[torch.Tensor, tuple[int, int]]:
    """Returns ((1, 3, H, W) float tensor in [0, 1], BGR order) and the
    original (height, width)."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    original_size = image.shape[:2]
    height, width = image_size
    resized = cv2.resize(image, (width, height))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor, original_size


def load_model(checkpoint_path: str | Path, class_split: ClassSplit = DEFAULT_CLASS_SPLIT, device: str = "cpu") -> PanopticFCN:
    model = PanopticFCN(
        num_stuff_classes=len(class_split.stuff_class_names),
        num_thing_classes=len(class_split.thing_class_names),
    )
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model: PanopticFCN, image_path: str | Path, image_size: tuple[int, int] = (512, 512), device: str = "cpu") -> Prediction:
    """Stuff-branch prediction upsampled (bilinear on logits) back to the
    original image resolution."""
    tensor, (orig_h, orig_w) = load_image_like_training(image_path, image_size)
    output = model(tensor.to(device))
    logits = F.interpolate(output.stuff_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)
    return Prediction(stuff_probs=probs, stuff_argmax=probs.argmax(axis=0), original_size=(orig_h, orig_w))

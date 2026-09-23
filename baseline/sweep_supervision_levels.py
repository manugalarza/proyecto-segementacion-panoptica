"""
Skeleton: Panoptic FCN with varying levels of point supervision.
Experiments for semanas 9-10: train with P5, P10, P20, P30 of pseudo-masks.

NOT YET IMPLEMENTED: requires
1. Connecting pseudo-masks to training loop (kernel/mask-feature loss)
2. Leakage-safe split from Jorge
3. Config system to control supervision level P
"""

import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SupervisionSweepConfig:
    """Sweep configuration: P5, P10, P20, P30 point supervision."""
    supervision_levels: list[int] = None  # [5, 10, 20, 30]
    epochs: int = 500
    batch_size: int = 4
    learning_rate: float = 1e-3
    checkpoint_dir: Path = Path("checkpoints/sweep_supervision")
    
    def __post_init__(self):
        if self.supervision_levels is None:
            self.supervision_levels = [5, 10, 20, 30]
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


def train_with_supervision_level(p: int, config: SupervisionSweepConfig):
    """
    Train Panoptic FCN with P% of point supervision.
    
    Steps (TODO):
    1. Load train split (leakage-safe from Jorge)
    2. Subsample to P% of points randomly
    3. Generate pseudo-masks for those points only
    4. Connect pseudo-masks to kernel/mask-feature loss
    5. Train model
    6. Save checkpoint: checkpoint_dir / f"panoptic_p{p}_epoch{epoch}.pt"
    
    Args:
        p: Supervision level (5, 10, 20, or 30)
        config: SupervisionSweepConfig
    """
    print(f"Training Panoptic FCN with P={p}% supervision...")
    print(f"  Loading dataset (leakage-safe split required from Jorge)")
    print(f"  Subsampling to {p}% of points")
    print(f"  Connecting pseudo-masks to loss")
    print(f"  Training for {config.epochs} epochs...")
    
    # TODO: implement actual training loop
    # model = PanopticFCN(...)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # for epoch in range(config.epochs):
    #     for batch in train_loader:
    #         loss = model_forward(...) + mask_loss(pseudo_masks)  # <- CRITICAL
    #         loss.backward()
    #         optimizer.step()


def evaluate_on_validation(p: int, config: SupervisionSweepConfig):
    """
    Evaluate trained model on leakage-safe validation set.
    
    Metrics:
    - Panoptic Quality (PQ)
    - Segmentation Quality (SQ) 
    - Recognition Quality (RQ)
    - recall@0.5 (converted to boxes like baseline)
    
    TODO: use src/panoptic_mining/evaluation/metrics.py
    """
    print(f"Evaluating P={p}% model on validation set...")
    # results = {...}
    # return results


def main():
    config = SupervisionSweepConfig()
    results = {}
    
    print("=== Panoptic FCN Supervision Level Sweep ===")
    print("Status: SKELETON ONLY (requires implementation of steps below)\n")
    
    print("Blockers before execution:")
    print("1. Jorge provides leakage-safe split (train/val/test)")
    print("2. Pseudo-mask loss connected to training loop (kernel/mask-feature heads)")
    print("3. Supervision subsampling logic implemented in data loader\n")
    
    for p in config.supervision_levels:
        print(f"\n--- P={p}% ---")
        # train_with_supervision_level(p, config)
        # val_results = evaluate_on_validation(p, config)
        # results[f"p{p}"] = val_results
        print("[SKIPPED: awaiting blockers]")
    
    # Save results
    output_path = Path("sweep_supervision_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

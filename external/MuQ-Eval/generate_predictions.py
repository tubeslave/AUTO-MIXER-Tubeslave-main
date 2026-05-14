"""
Generate test_predictions.json for each experiment and fold.

Loads best_model.pt checkpoints, runs inference on the validation set,
and saves per-clip predictions, targets, and clip IDs.

Usage:
    python generate_predictions.py --output-dir ./outputs
    python generate_predictions.py --output-dir ./outputs --experiment A3b_contrastive
    python generate_predictions.py --output-dir ./outputs --experiment A3c --fold 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train import load_config
from src.data import build_dataloaders
from src.model import MusicQualityModel


EXPERIMENT_CONFIGS = {
    "A1_frozen_mlp": "configs/A1_frozen_mlp.yaml",
    "A2_frozen_ordinal": "configs/A2_frozen_ordinal.yaml",
    "A3a_lora": "configs/A3a_lora.yaml",
    "A3b_contrastive": "configs/A3b_contrastive.yaml",
    "A3c": "configs/A3c.yaml",
    "A6_mert95m": "configs/A6_mert.yaml",
}


def generate_fold_predictions(cfg, fold_idx: int, output_dir: Path) -> dict:
    """Load best model and generate predictions on validation set."""
    fold_dir = output_dir / cfg.experiment.name / f"fold{fold_idx}"
    ckpt_path = fold_dir / "best_model.pt"

    if not ckpt_path.exists():
        print(f"  SKIP fold {fold_idx}: {ckpt_path} not found")
        return None

    # Set up fold config
    fold_cfg = OmegaConf.create(dict(cfg))
    fold_cfg.paths.output_dir = str(fold_dir)

    # Build dataloaders (we only need val_loader)
    _, val_loader, _ = build_dataloaders(fold_cfg, fold_idx)

    # Build model and load checkpoint
    model = MusicQualityModel(fold_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    use_amp = cfg.training.mixed_precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if cfg.training.mixed_precision == "bf16" else torch.float16

    all_preds = {name: [] for name in model.head_names}
    all_targets = {"MI": [], "TA": []}
    all_clip_ids = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"  Fold {fold_idx}", leave=False):
            waveforms = batch["waveform"].to(device)
            dataset_ids = batch.get("dataset_id", torch.zeros(len(waveforms), dtype=torch.long)).to(device)

            with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                predictions = model(waveforms, dataset_ids)

            for name in model.head_names:
                if name in predictions:
                    pred = predictions[name]
                    if pred.dim() > 1:
                        # Ordinal: convert logits to expected score
                        probs = torch.softmax(pred, dim=-1)
                        bins = torch.linspace(1, 5, pred.shape[-1], device=pred.device)
                        pred = (probs * bins.unsqueeze(0)).sum(-1)
                    all_preds[name].append(pred.float().cpu())

            all_targets["MI"].append(batch["mi_score"])
            all_targets["TA"].append(batch["ta_score"])
            all_clip_ids.extend(batch["clip_id"].tolist())

    # Concatenate
    result = {
        "fold": fold_idx,
        "experiment": cfg.experiment.name,
        "n_samples": len(all_clip_ids),
        "clip_ids": all_clip_ids,
    }

    for name in model.head_names:
        if all_preds[name]:
            result[f"predictions_{name}"] = torch.cat(all_preds[name]).numpy().tolist()

    for name in ["MI", "TA"]:
        if all_targets[name]:
            result[f"targets_{name}"] = torch.cat(all_targets[name]).numpy().tolist()

    # Also save in the format analyze_results.py expects
    if all_preds.get("MI"):
        result["predictions"] = result["predictions_MI"]
        result["targets"] = result["targets_MI"]

    # Save
    pred_path = fold_dir / "test_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {pred_path} ({result['n_samples']} samples)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate test predictions from saved checkpoints")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--experiment", type=str, default=None,
                       help="Run only this experiment (e.g., A3b_contrastive)")
    parser.add_argument("--fold", type=int, default=None,
                       help="Run only this fold (0-4)")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.experiment:
        experiments = {args.experiment: EXPERIMENT_CONFIGS[args.experiment]}
    else:
        experiments = EXPERIMENT_CONFIGS

    print("=" * 60)
    print("Generating Test Predictions")
    print("=" * 60)

    for exp_name, config_path in experiments.items():
        print(f"\n{'─' * 60}")
        print(f"Experiment: {exp_name}")
        print(f"{'─' * 60}")

        if not Path(config_path).exists():
            print(f"  SKIP: config {config_path} not found")
            continue

        cfg = load_config(config_path)

        folds = [args.fold] if args.fold is not None else range(args.n_folds)
        for fold_idx in folds:
            generate_fold_predictions(cfg, fold_idx, output_dir)

    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

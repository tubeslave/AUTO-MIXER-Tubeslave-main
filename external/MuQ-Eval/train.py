"""
Main training entry point for the neural music quality evaluator.

Usage:
    # A1 baseline (frozen MuQ + MLP + MSE)
    python train.py --config configs/A1_frozen_mlp.yaml --fold 0

    # A2 go/no-go (frozen MuQ + ordinal CE)
    python train.py --config configs/A2_frozen_ordinal.yaml --fold 0

    # A3a ablation (MuQ + LoRA + ordinal CE)
    python train.py --config configs/A3a_lora.yaml --fold 0

    # A3 full (LoRA + contrastive + multi-dataset + calibration)
    python train.py --config configs/A3_full.yaml --fold 0

    # Run all 5 folds for a config
    python train.py --config configs/A2_frozen_ordinal.yaml --all-folds

    # Run baselines on test set of fold 0
    python train.py --config configs/A1_frozen_mlp.yaml --fold 0 --baselines-only
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from datasets import Audio, load_dataset

from src.data import (
    AudioProcessor, MusicEvalDataset, SongEvalDataset,
    get_musiceval_cv_splits, build_dataloaders,
)
from src.model import MusicQualityModel
from src.losses import CombinedLoss
from src.trainer import Trainer
from src.baselines import compute_all_baselines
from src.evaluation import full_evaluation


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> OmegaConf:
    """Load experiment config with base config inheritance."""
    cfg = OmegaConf.load(config_path)

    # Load base config if specified
    if "defaults" in cfg:
        defaults = OmegaConf.to_container(cfg.defaults) if hasattr(cfg.defaults, '__iter__') else cfg.defaults
        base_name = defaults[0] if isinstance(defaults, list) else defaults
        base_path = Path(config_path).parent / f"{base_name}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(str(base_path))
            cfg = OmegaConf.merge(base_cfg, cfg)

    # Remove defaults key (not a real config field)
    if "defaults" in cfg:
        del cfg["defaults"]

    return cfg


def run_fold(cfg, fold_idx: int) -> dict:
    """Run training for a single CV fold."""
    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.experiment.name} | Fold {fold_idx}")
    print(f"{'='*60}")

    set_seed(cfg.seed + fold_idx)

    # Update output directory for this fold
    fold_cfg = OmegaConf.create(dict(cfg))
    fold_cfg.paths.output_dir = str(
        Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold{fold_idx}"
    )

    # Build dataloaders
    train_loader, val_loader, songeval_loader = build_dataloaders(fold_cfg, fold_idx)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    if songeval_loader:
        print(f"SongEval: {len(songeval_loader.dataset)} samples")

    # Build model
    model = MusicQualityModel(fold_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} params total, {n_trainable:,} trainable "
          f"({100*n_trainable/n_params:.1f}%)")

    # Build loss
    loss_fn = CombinedLoss(fold_cfg)

    # Train
    trainer = Trainer(
        fold_cfg, model, train_loader, val_loader, loss_fn, songeval_loader
    )
    results = trainer.train()

    # Post-training evaluation with full stats
    print("\n--- Full evaluation on best model ---")
    best_ckpt = torch.load(
        Path(fold_cfg.paths.output_dir) / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    return results


def run_baselines(cfg, fold_idx: int) -> dict:
    """Run baseline metrics on the test set of a given fold."""
    print(f"\n{'='*60}")
    print(f"Baselines | Fold {fold_idx}")
    print(f"{'='*60}")

    # Load dataset and get split
    me_dataset = load_dataset(cfg.data.musiceval_id, split="train")
    me_dataset = me_dataset.cast_column("audio", Audio(decode=False))
    splits = get_musiceval_cv_splits(me_dataset, cfg.data.cv_folds, cfg.seed)
    train_indices, test_indices = splits[fold_idx]

    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold{fold_idx}"
    results = compute_all_baselines(
        me_dataset, test_indices, output_dir, reference_indices=train_indices
    )
    return results


def aggregate_folds(cfg, n_folds: int) -> dict:
    """Aggregate results across all CV folds."""
    all_results = []
    base_dir = Path(cfg.paths.output_dir) / cfg.experiment.name

    for fold_idx in range(n_folds):
        results_path = base_dir / f"fold{fold_idx}" / "training_results.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results.append(json.load(f))

    if not all_results:
        print("No fold results found.")
        return {}

    # Aggregate best metrics
    aggregated = {"n_folds": len(all_results)}
    metric_keys = all_results[0].get("best_metrics", {}).keys()
    for key in metric_keys:
        values = [r["best_metrics"].get(key, 0) for r in all_results]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    aggregated["best_epochs"] = [r["best_epoch"] for r in all_results]

    print("\n--- Aggregated Results ---")
    for key in metric_keys:
        mean = aggregated[f"{key}_mean"]
        std = aggregated[f"{key}_std"]
        print(f"  {key}: {mean:.4f} +/- {std:.4f}")

    # Save
    agg_path = base_dir / "aggregated_results.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Train neural music quality evaluator")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to experiment config YAML")
    parser.add_argument("--fold", type=int, default=0,
                       help="CV fold index (0-4)")
    parser.add_argument("--all-folds", action="store_true",
                       help="Run all CV folds sequentially")
    parser.add_argument("--baselines-only", action="store_true",
                       help="Only compute baseline metrics on test set")
    parser.add_argument("--aggregate-only", action="store_true",
                       help="Only aggregate existing fold results")
    parser.add_argument("--subsample", type=float, default=None,
                       help="Override training set subsample fraction (0.0-1.0)")
    parser.add_argument("--subsample-n", type=int, default=None,
                       help="Override training set size as absolute count (e.g. 100, 150, 250)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override subsample if provided via CLI (absolute count takes priority)
    if args.subsample_n is not None:
        cfg.data.train_subsample_n = args.subsample_n
    elif args.subsample is not None:
        cfg.data.train_subsample = args.subsample
    print(f"Config: {args.config}")
    print(f"Experiment: {cfg.experiment.name}")
    print(f"Tuning mode: {cfg.model.tuning_mode}")
    print(f"Loss: {cfg.loss.type}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if args.aggregate_only:
        aggregate_folds(cfg, cfg.data.cv_folds)
        return

    if args.baselines_only:
        run_baselines(cfg, args.fold)
        return

    if args.all_folds:
        for fold in range(cfg.data.cv_folds):
            run_fold(cfg, fold)
        aggregate_folds(cfg, cfg.data.cv_folds)
    else:
        run_fold(cfg, args.fold)


if __name__ == "__main__":
    main()

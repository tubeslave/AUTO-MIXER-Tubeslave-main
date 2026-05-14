"""
Run baseline metrics (FAD VGGish) on test folds.

Usage:
    python run_baselines.py --config configs/A1_frozen_mlp.yaml --fold 0
    python run_baselines.py --config configs/A1_frozen_mlp.yaml --all-folds
"""

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Run baseline metrics")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--all-folds", action="store_true")
args = parser.parse_args()

# Monkey-patch torch.load to allow legacy checkpoints (FAD VGGish)
import torch
_original_torch_load = torch.load
def _patched_torch_load(*a, **kw):
    if "weights_only" not in kw:
        kw["weights_only"] = False
    return _original_torch_load(*a, **kw)
torch.load = _patched_torch_load

# Now safe to import everything else
sys.path.insert(0, str(Path(__file__).parent))
from omegaconf import OmegaConf
from datasets import Audio, load_dataset
from src.data import get_musiceval_cv_splits
from src.baselines import compute_all_baselines


def load_config(config_path: str):
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        defaults = OmegaConf.to_container(cfg.defaults) if hasattr(cfg.defaults, '__iter__') else cfg.defaults
        base_name = defaults[0] if isinstance(defaults, list) else defaults
        base_path = Path(config_path).parent / f"{base_name}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(str(base_path))
            cfg = OmegaConf.merge(base_cfg, cfg)
    if "defaults" in cfg:
        del cfg["defaults"]
    return cfg


def run_fold_baselines(cfg, fold_idx: int):
    print(f"\n{'=' * 60}")
    print(f"Baselines | Fold {fold_idx}")
    print(f"{'=' * 60}")

    me_dataset = load_dataset(cfg.data.musiceval_id, split="train")
    me_dataset = me_dataset.cast_column("audio", Audio(decode=False))
    splits = get_musiceval_cv_splits(me_dataset, cfg.data.cv_folds, cfg.seed)
    train_indices, test_indices = splits[fold_idx]

    output_dir = Path(cfg.paths.output_dir) / cfg.experiment.name / f"fold{fold_idx}"
    results = compute_all_baselines(
        me_dataset, test_indices, output_dir, reference_indices=train_indices
    )

    print(f"\nBaseline results for fold {fold_idx}:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


def main():
    cfg = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Experiment: {cfg.experiment.name}")

    if args.all_folds:
        for fold in range(cfg.data.cv_folds):
            run_fold_baselines(cfg, fold)
    else:
        run_fold_baselines(cfg, args.fold)

    print("\nDone!")


if __name__ == "__main__":
    main()

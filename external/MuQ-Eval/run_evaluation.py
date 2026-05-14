"""
Evaluation benchmark for MuQ-Eval.

Loads a trained checkpoint and evaluates on MusicEval test folds,
reproducing Tables 1-2 from the paper.

Usage:
    # Evaluate a single fold
    python run_evaluation.py --checkpoint outputs/A1_frozen_mlp/fold0/best_model.pt \
                             --config configs/A1_frozen_mlp.yaml --fold 0

    # Evaluate all 5 folds and report mean +/- std
    python run_evaluation.py --checkpoint-dir outputs/A1_frozen_mlp \
                             --config configs/A1_frozen_mlp.yaml --all-folds

    # Evaluate with bootstrap CIs
    python run_evaluation.py --checkpoint-dir outputs/A1_frozen_mlp \
                             --config configs/A1_frozen_mlp.yaml --all-folds --bootstrap
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from src.data import build_dataloaders
from src.model import MusicQualityModel
from src.evaluation import (
    compute_system_level_correlations,
    compute_utterance_level_correlations,
    bootstrap_ci,
)


def evaluate_fold(checkpoint_path, config, fold, device="cuda", bootstrap=False):
    """Evaluate a single fold and return metrics."""
    # Load model
    model = MusicQualityModel(config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model = model.to(device).eval()

    # Build test dataloader
    _, val_loader = build_dataloaders(config, fold=fold)

    # Run inference
    all_preds_mi, all_preds_ta = [], []
    all_targets_mi, all_targets_ta = [], []
    all_model_ids = []

    with torch.no_grad():
        for batch in val_loader:
            waveforms = batch["waveform"].to(device)
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(waveforms)

            all_preds_mi.extend(outputs["MI"].cpu().numpy().tolist())
            if "TA" in outputs:
                all_preds_ta.extend(outputs["TA"].cpu().numpy().tolist())
            all_targets_mi.extend(batch["mi_score"].numpy().tolist())
            if "ta_score" in batch:
                all_targets_ta.extend(batch["ta_score"].numpy().tolist())
            if "model_id" in batch:
                all_model_ids.extend(batch["model_id"])

    preds_mi = np.array(all_preds_mi)
    targets_mi = np.array(all_targets_mi)
    model_ids = np.array(all_model_ids) if all_model_ids else None

    # Compute correlations
    results = {}

    # System-level
    if model_ids is not None and len(np.unique(model_ids)) > 1:
        sys_metrics = compute_system_level_correlations(
            preds_mi, targets_mi, model_ids
        )
        results["system_level"] = sys_metrics

    # Utterance-level
    utt_metrics = compute_utterance_level_correlations(preds_mi, targets_mi)
    results["utterance_level_MI"] = utt_metrics

    if all_preds_ta:
        preds_ta = np.array(all_preds_ta)
        targets_ta = np.array(all_targets_ta)
        utt_ta = compute_utterance_level_correlations(preds_ta, targets_ta)
        results["utterance_level_TA"] = utt_ta

    # Bootstrap CIs if requested
    if bootstrap and model_ids is not None:
        ci = bootstrap_ci(preds_mi, targets_mi, model_ids, B=1000, seed=42)
        results["bootstrap_ci"] = ci

    return results


def main():
    parser = argparse.ArgumentParser(description="MuQ-Eval evaluation benchmark")
    parser.add_argument("--checkpoint", type=str, help="Path to single checkpoint")
    parser.add_argument("--checkpoint-dir", type=str,
                        help="Directory with fold0/, fold1/, ... subdirectories")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate")
    parser.add_argument("--all-folds", action="store_true",
                        help="Evaluate all 5 folds")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Compute bootstrap CIs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: stdout)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.all_folds:
        assert args.checkpoint_dir, "--checkpoint-dir required with --all-folds"
        fold_results = []
        for fold in range(5):
            ckpt = Path(args.checkpoint_dir) / f"fold{fold}" / "best_model.pt"
            if not ckpt.exists():
                print(f"WARNING: {ckpt} not found, skipping fold {fold}")
                continue
            print(f"Evaluating fold {fold}...")
            result = evaluate_fold(
                str(ckpt), config, fold, args.device, args.bootstrap
            )
            result["fold"] = fold
            fold_results.append(result)

        # Aggregate across folds
        aggregate = aggregate_fold_results(fold_results)
        output = {"folds": fold_results, "aggregate": aggregate}
    else:
        ckpt = args.checkpoint or str(
            Path(args.checkpoint_dir) / f"fold{args.fold}" / "best_model.pt"
        )
        output = evaluate_fold(ckpt, config, args.fold, args.device, args.bootstrap)
        output["fold"] = args.fold

    # Output
    output_str = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_str)
        print(f"Results saved to {args.output}")
    else:
        print(output_str)


def aggregate_fold_results(fold_results):
    """Compute mean +/- std across folds."""
    agg = {}
    metric_keys = ["system_level", "utterance_level_MI", "utterance_level_TA"]

    for key in metric_keys:
        values_by_metric = {}
        for fr in fold_results:
            if key not in fr:
                continue
            for metric_name, value in fr[key].items():
                if isinstance(value, (int, float)):
                    values_by_metric.setdefault(metric_name, []).append(value)

        if values_by_metric:
            agg[key] = {}
            for metric_name, values in values_by_metric.items():
                arr = np.array(values)
                agg[key][metric_name] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "values": [float(v) for v in arr],
                }

    return agg


if __name__ == "__main__":
    main()

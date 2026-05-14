"""
Summarize data efficiency (subsample) experiments into subsample_summary.json.

Reads training_results.json from each experiment's fold directories
and aggregates best metrics across folds.

Usage:
    python summarize_subsample.py --output-dir ./outputs
"""

import argparse
import json
from pathlib import Path

import numpy as np


SUBSAMPLE_EXPERIMENTS = [
    ("A1_n100", "A1", 100),
    ("A1_n150", "A1", 150),
    ("A1_n250", "A1", 250),
    ("A1_n500", "A1", 500),
    ("A1_n750", "A1", 750),
    ("A1_n1000", "A1", 1000),
    ("A1_frozen_mlp", "A1", "full"),
    ("A3a_n100", "A3a", 100),
    ("A3a_n150", "A3a", 150),
    ("A3a_n250", "A3a", 250),
    ("A3a_n500", "A3a", 500),
    ("A3a_n750", "A3a", 750),
    ("A3a_n1000", "A3a", 1000),
    ("A3a_lora", "A3a", "full"),
]


def load_fold_best_metrics(output_dir: Path, exp_name: str) -> list[dict]:
    """Load best_metrics from each fold's training_results.json."""
    results = []
    fold_idx = 0
    while True:
        path = output_dir / exp_name / f"fold{fold_idx}" / "training_results.json"
        if not path.exists():
            break
        with open(path) as f:
            data = json.load(f)
        best = data.get("best_metrics", {})
        best["best_epoch"] = data.get("best_epoch", None)
        results.append(best)
        fold_idx += 1
    return results


def aggregate(fold_results: list[dict], key: str) -> tuple:
    """Return (mean, std) for a metric across folds."""
    vals = [fr.get(key) for fr in fold_results if fr.get(key) is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def main():
    parser = argparse.ArgumentParser(description="Summarize subsample experiments")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    summary = {}

    for exp_name, model_type, n_samples in SUBSAMPLE_EXPERIMENTS:
        exp_dir = output_dir / exp_name
        if not exp_dir.exists():
            print(f"  SKIP: {exp_name} (not found)")
            continue

        fold_results = load_fold_best_metrics(output_dir, exp_name)
        if not fold_results:
            print(f"  SKIP: {exp_name} (no fold results)")
            continue

        mi_pcc_mean, mi_pcc_std = aggregate(fold_results, "val/MI_pcc")
        mi_srcc_mean, mi_srcc_std = aggregate(fold_results, "val/MI_srcc")
        ta_pcc_mean, ta_pcc_std = aggregate(fold_results, "val/TA_pcc")
        ta_srcc_mean, ta_srcc_std = aggregate(fold_results, "val/TA_srcc")
        loss_mean, loss_std = aggregate(fold_results, "val/loss")
        best_epochs = [fr.get("best_epoch") for fr in fold_results]

        summary[exp_name] = {
            "model_type": model_type,
            "n_samples": n_samples,
            "n_folds": len(fold_results),
            "val/MI_pcc_mean": mi_pcc_mean,
            "val/MI_pcc_std": mi_pcc_std,
            "val/MI_srcc_mean": mi_srcc_mean,
            "val/MI_srcc_std": mi_srcc_std,
            "val/TA_pcc_mean": ta_pcc_mean,
            "val/TA_pcc_std": ta_pcc_std,
            "val/TA_srcc_mean": ta_srcc_mean,
            "val/TA_srcc_std": ta_srcc_std,
            "val/loss_mean": loss_mean,
            "val/loss_std": loss_std,
            "best_epochs": best_epochs,
        }

        print(f"  {exp_name} ({model_type}, n={n_samples}): "
              f"MI_SRCC={mi_srcc_mean:.4f}±{mi_srcc_std:.4f}, "
              f"MI_PCC={mi_pcc_mean:.4f}±{mi_pcc_std:.4f} "
              f"[{len(fold_results)} folds]")

    # Save
    out_path = output_dir / "subsample_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Print table
    print(f"\n{'='*80}")
    print(f"{'Experiment':<20} {'Model':<6} {'N':>6} {'MI_SRCC':>10} {'MI_PCC':>10} {'TA_SRCC':>10}")
    print(f"{'-'*80}")
    for exp_name, data in summary.items():
        n = str(data["n_samples"])
        mi_srcc = f"{data['val/MI_srcc_mean']:.4f}" if data["val/MI_srcc_mean"] else "N/A"
        mi_pcc = f"{data['val/MI_pcc_mean']:.4f}" if data["val/MI_pcc_mean"] else "N/A"
        ta_srcc = f"{data['val/TA_srcc_mean']:.4f}" if data["val/TA_srcc_mean"] else "N/A"
        print(f"{exp_name:<20} {data['model_type']:<6} {n:>6} {mi_srcc:>10} {mi_pcc:>10} {ta_srcc:>10}")


if __name__ == "__main__":
    main()

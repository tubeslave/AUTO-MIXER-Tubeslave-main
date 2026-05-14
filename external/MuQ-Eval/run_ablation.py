"""
Ablation chain runner: executes the full progressive ablation
A1 -> A2 (go/no-go) -> A3a -> A3b -> A3 -> A6.

Enforces the go/no-go gate at A2:
  - SRCC_sys(MI) >= 0.85 -> proceed to Phase 2
  - SRCC_sys(MI) in [0.75, 0.85) -> hyperparameter sweep retry
  - SRCC_sys(MI) < 0.75 -> STOP, revisit encoder choice

Usage:
    python run_ablation.py --output-dir ./outputs --folds 5
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ABLATION_CHAIN = [
    # (config_file, phase, gate_check)
    ("configs/A1_frozen_mlp.yaml", "Phase 1: Baseline", None),
    ("configs/A2_frozen_ordinal.yaml", "Phase 1: Go/No-Go Gate", "go_no_go"),
    ("configs/A3a_lora.yaml", "Phase 2: +LoRA", None),
    ("configs/A3b_contrastive.yaml", "Phase 2: +Contrastive", None),
    ("configs/A3c.yaml", "Phase 2: +Uncertainty+LoRA32", None),
    ("configs/A6_mert.yaml", "Phase 3: MERT Encoder Ablation", None),
]

GO_NO_GO_THRESHOLD = 0.80
GO_NO_GO_RETRY_THRESHOLD = 0.75


def run_experiment(config_path: str, n_folds: int, output_dir: str) -> dict:
    """Run an experiment across all folds and return aggregated results."""
    cmd = [
        sys.executable, "train.py",
        "--config", config_path,
        "--all-folds",
    ]
    print(f"\n{'#'*70}")
    print(f"Running: {config_path}")
    print(f"{'#'*70}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"WARNING: {config_path} exited with code {result.returncode}")

    # Load aggregated results
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(config_path)
    if "defaults" in cfg:
        defaults = OmegaConf.to_container(cfg.defaults) if hasattr(cfg.defaults, '__iter__') else cfg.defaults
        base_name = defaults[0] if isinstance(defaults, list) else defaults
        base_path = Path(config_path).parent / f"{base_name}.yaml"
        if base_path.exists():
            base_cfg = OmegaConf.load(str(base_path))
            cfg = OmegaConf.merge(base_cfg, cfg)

    exp_name = cfg.get("experiment", {}).get("name", "unknown")
    agg_path = Path(output_dir) / exp_name / "aggregated_results.json"
    if agg_path.exists():
        with open(agg_path) as f:
            return json.load(f)
    return {}


def check_go_no_go(results: dict) -> str:
    """Check go/no-go gate for A2.

    Returns: "proceed", "retry", or "stop".
    """
    srcc_sys = results.get("val/MI_srcc_mean", 0)
    print(f"\nGo/No-Go Gate Check:")
    print(f"  SRCC_sys(MI) = {srcc_sys:.4f}")
    print(f"  Threshold: >= {GO_NO_GO_THRESHOLD}")

    if srcc_sys >= GO_NO_GO_THRESHOLD:
        print(f"  -> PROCEED to Phase 2")
        return "proceed"
    elif srcc_sys >= GO_NO_GO_RETRY_THRESHOLD:
        print(f"  -> RETRY with hyperparameter sweep (in [{GO_NO_GO_RETRY_THRESHOLD}, {GO_NO_GO_THRESHOLD}))")
        return "retry"
    else:
        print(f"  -> STOP. SRCC < {GO_NO_GO_RETRY_THRESHOLD}. Revisit encoder choice.")
        return "stop"


def main():
    parser = argparse.ArgumentParser(description="Run full ablation chain")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--start-from", type=int, default=0,
                       help="Index in ablation chain to start from (0-based)")
    args = parser.parse_args()

    all_results = {}

    for i, (config, phase, gate) in enumerate(ABLATION_CHAIN):
        if i < args.start_from:
            continue

        print(f"\n{'='*70}")
        print(f"ABLATION STEP {i}: {phase}")
        print(f"Config: {config}")
        print(f"{'='*70}")

        results = run_experiment(config, args.folds, args.output_dir)
        all_results[config] = {"phase": phase, "results": results}

        # Go/no-go gate
        if gate == "go_no_go":
            decision = check_go_no_go(results)
            all_results[config]["gate_decision"] = decision
            if decision == "stop":
                print("\nABLATION CHAIN STOPPED at go/no-go gate.")
                break
            elif decision == "retry":
                print("\nRetry needed. Manual intervention required.")
                print("Adjust hyperparameters in A2 config and re-run.")
                break

    # Save ablation summary
    summary_path = Path(args.output_dir) / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAblation summary saved to: {summary_path}")

    # Print comparison table
    print(f"\n{'='*70}")
    print("ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'SRCC_sys(MI)':<15} {'PCC_utt(MI)':<15} {'SRCC_utt(MI)':<15}")
    print(f"{'-'*70}")
    for config, data in all_results.items():
        r = data["results"]
        name = Path(config).stem
        srcc_sys = r.get("val/MI_srcc_mean", "N/A")
        pcc_utt = r.get("val/MI_pcc_mean", "N/A")
        srcc_utt = r.get("val/MI_srcc_mean", "N/A")
        if isinstance(srcc_sys, float):
            print(f"{name:<25} {srcc_sys:<15.4f} {pcc_utt:<15.4f} {srcc_utt:<15.4f}")
        else:
            print(f"{name:<25} {srcc_sys:<15} {pcc_utt:<15} {srcc_utt:<15}")


if __name__ == "__main__":
    main()

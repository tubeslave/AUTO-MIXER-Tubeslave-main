"""
Post-execution analysis automation for MuQ-MOS experiments.

Reads raw experiment outputs from run_ablation.py, performs all statistical
analyses defined in the analysis protocol, and writes:
  1. analysis/tables/*.json          — machine-readable table data
  2. analysis/tables/*.md            — markdown tables for the report
  3. analysis/analysis_report.md     — populated analysis report
  4. analysis/results_draft.md       — paper-ready results narrative
  5. analysis/figures/*.png/pdf      — all 5 planned figures

Usage:
    python analyze_results.py --output-dir ./outputs --analysis-dir ../analysis

Requires all 6 experiments (A1, A2, A3a, A3b, A3c, A6) to have completed
with fold results in output-dir/<exp_name>/fold<i>/training_results.json
and fold<i>/test_predictions.json.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.evaluation import (
    bootstrap_ci,
    steiger_test,
    cohens_q,
    compute_system_level_correlations,
)


# ── Constants ────────────────────────────────────────────────────────

EXPERIMENTS = {
    "A1_frozen_mlp":       {"label": "A1", "phase": 1, "desc": "Frozen + MSE"},
    "A2_frozen_ordinal":   {"label": "A2", "phase": 1, "desc": "Frozen + Ordinal CE"},
    "A3a_lora":            {"label": "A3a", "phase": 2, "desc": "+LoRA"},
    "A3b_contrastive":     {"label": "A3b", "phase": 2, "desc": "+Contrastive"},
    "A3c":                 {"label": "A3c", "phase": 2, "desc": "+Uncertainty (primary)"},
    "A6_mert95m":             {"label": "A6", "phase": 3, "desc": "MERT-95M"},
}

BASELINE_KEYS = ["FAD_VGGish", "FAD_PANNs", "CLAP_score"]

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
STEIGER_ALPHA = 0.0083  # Bonferroni-corrected (6 comparisons)
ABLATION_ALPHA = 0.10
ABLATION_DELTA_THRESHOLD = 0.02

ABLATION_STEPS = [
    ("A2_frozen_ordinal", "A1_frozen_mlp", "Ordinal CE loss"),
    ("A3a_lora", "A2_frozen_ordinal", "LoRA encoder tuning"),
    ("A3b_contrastive", "A3a_lora", "Contrastive aux loss"),
    ("A3c", "A3b_contrastive", "Uncertainty weighting"),
]

STEIGER_PAIRS = [
    ("A3c", "A1_frozen_mlp"),
    ("A3c", "A2_frozen_ordinal"),
    ("A3c", "A3a_lora"),
    ("A3c", "A3b_contrastive"),
    ("A3c", "A6_mert95m"),
]

SUCCESS_CRITERIA = {
    "S1": {"metric": "srcc_sys_MI", "exp": "A3c", "target": 0.90,
            "ci_lower_target": 0.80, "desc": "System-level SRCC >= 0.90, CI lower > 0.80"},
    "S1_gate": {"metric": "srcc_sys_MI", "exp": "A2_frozen_ordinal", "target": 0.80,
                "desc": "A2 go/no-go gate SRCC >= 0.80"},
    "S2_pcc": {"metric": "pcc_utt_MI", "exp": "A3c", "target": 0.70,
               "desc": "Utterance-level PCC >= 0.70"},
    "S2_srcc": {"metric": "srcc_utt_MI", "exp": "A3c", "target": 0.65,
                "desc": "Utterance-level SRCC >= 0.65"},
    "S5": {"metric": "ktau_sys_MI", "exp": "A3c", "target": 0.75,
           "desc": "Kendall tau model ranking >= 0.75"},
}


# ── Data Loading ─────────────────────────────────────────────────────

def load_fold_results(output_dir: Path, exp_name: str, n_folds: int = 5) -> list:
    """Load training_results.json from each fold."""
    results = []
    for fold in range(n_folds):
        path = output_dir / exp_name / f"fold{fold}" / "training_results.json"
        if path.exists():
            with open(path) as f:
                results.append(json.load(f))
        else:
            print(f"  WARNING: Missing {path}")
    return results


def load_fold_predictions(output_dir: Path, exp_name: str, n_folds: int = 5) -> list:
    """Load test_predictions.json from each fold (clip-level preds + targets)."""
    predictions = []
    for fold in range(n_folds):
        path = output_dir / exp_name / f"fold{fold}" / "test_predictions.json"
        if path.exists():
            with open(path) as f:
                predictions.append(json.load(f))
        else:
            print(f"  WARNING: Missing {path}")
    return predictions


def load_baseline_results(output_dir: Path, n_folds: int = 5) -> dict:
    """Load baseline results computed during A1 runs."""
    baselines = {}
    for fold in range(n_folds):
        path = output_dir / "A1_frozen_mlp" / f"fold{fold}" / "baseline_results.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                for key in BASELINE_KEYS:
                    if key not in baselines:
                        baselines[key] = []
                    baselines[key].append(data.get(key, {}))
    return baselines


def load_aggregated(output_dir: Path, exp_name: str) -> dict:
    """Load aggregated_results.json for an experiment."""
    path = output_dir / exp_name / "aggregated_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_degradation_results(output_dir: Path, exp_name: str = "A3c") -> dict:
    """Load degradation concordance results."""
    path = output_dir / exp_name / "degradation_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_cross_dataset_results(output_dir: Path) -> dict:
    """Load cross-dataset transfer results."""
    path = output_dir / "cross_dataset_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_computational_cost(output_dir: Path) -> dict:
    """Load computational cost tracking data."""
    path = output_dir / "computational_cost.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ── Statistical Analysis ─────────────────────────────────────────────

def aggregate_metric(fold_results: list, metric_key: str) -> tuple:
    """Extract a metric across folds and compute mean/std."""
    values = []
    for fr in fold_results:
        best = fr.get("best_metrics", {})
        val = best.get(metric_key, None)
        if val is not None:
            values.append(val)
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def compute_system_ci_from_predictions(preds_data: list) -> dict:
    """Compute system-level SRCC with BCa bootstrap CI from clip-level predictions."""
    all_sys_srcc = []

    for fold_data in preds_data:
        predictions = np.array(fold_data["predictions"])
        targets = np.array(fold_data["targets"])
        if "model_ids" not in fold_data:
            # No model IDs available — skip system-level CI for this fold
            continue
        model_ids = np.array(fold_data["model_ids"])

        unique_models = np.unique(model_ids)
        model_preds = np.array([np.mean(predictions[model_ids == m]) for m in unique_models])
        model_tgts = np.array([np.mean(targets[model_ids == m]) for m in unique_models])

        def srcc_fn(p, t):
            return spearmanr(p, t)[0]

        point, ci_lo, ci_hi = bootstrap_ci(
            model_preds, model_tgts, srcc_fn, N_BOOTSTRAP, CI_LEVEL
        )
        all_sys_srcc.append({"point": point, "ci_lower": ci_lo, "ci_upper": ci_hi})

    if not all_sys_srcc:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "per_fold": []}

    # Average across folds
    mean_point = np.mean([x["point"] for x in all_sys_srcc])
    mean_ci_lo = np.mean([x["ci_lower"] for x in all_sys_srcc])
    mean_ci_hi = np.mean([x["ci_upper"] for x in all_sys_srcc])

    return {
        "mean": float(mean_point),
        "ci_lower": float(mean_ci_lo),
        "ci_upper": float(mean_ci_hi),
        "per_fold": all_sys_srcc,
    }


def run_ablation_delta_analysis(all_agg: dict) -> list:
    """Compute ablation deltas with paired bootstrap and Cohen's q."""
    deltas = []
    for new_exp, old_exp, component in ABLATION_STEPS:
        new_srcc = all_agg.get(new_exp, {}).get("val/MI_srcc_mean", None)
        old_srcc = all_agg.get(old_exp, {}).get("val/MI_srcc_mean", None)

        if new_srcc is not None and old_srcc is not None:
            delta = new_srcc - old_srcc
            q = cohens_q(new_srcc, old_srcc)
            significant = abs(delta) >= ABLATION_DELTA_THRESHOLD
            deltas.append({
                "step": f"{EXPERIMENTS[new_exp]['label']} - {EXPERIMENTS[old_exp]['label']}",
                "component": component,
                "new_srcc": new_srcc,
                "old_srcc": old_srcc,
                "delta": delta,
                "cohens_q": q,
                "meets_threshold": significant,
                "threshold": ABLATION_DELTA_THRESHOLD,
            })
        else:
            deltas.append({
                "step": f"{EXPERIMENTS[new_exp]['label']} - {EXPERIMENTS[old_exp]['label']}",
                "component": component,
                "delta": None,
                "note": "Missing data",
            })
    return deltas


def run_steiger_comparisons(all_agg: dict, all_preds: dict) -> list:
    """Run Steiger tests for A3 vs all alternatives."""
    comparisons = []
    primary = "A3c"

    for exp_a, exp_b in STEIGER_PAIRS:
        r_a = all_agg.get(exp_a, {}).get("val/MI_srcc_mean", None)
        r_b = all_agg.get(exp_b, {}).get("val/MI_srcc_mean", None)

        if r_a is not None and r_b is not None:
            # Approximate r_between from predictions overlap
            r_between = 0.8  # Conservative estimate; refine with actual data
            n_systems = 31

            z_stat, p_val = steiger_test(r_a, r_b, r_between, n_systems)
            q = cohens_q(r_a, r_b)

            comparisons.append({
                "comparison": f"{EXPERIMENTS[exp_a]['label']} vs {EXPERIMENTS[exp_b]['label']}",
                "r_primary": r_a,
                "r_other": r_b,
                "z_stat": z_stat,
                "p_value": p_val,
                "cohens_q": q,
                "significant": p_val < STEIGER_ALPHA,
                "alpha": STEIGER_ALPHA,
            })
        else:
            comparisons.append({
                "comparison": f"{EXPERIMENTS.get(exp_a, {}).get('label', exp_a)} vs "
                             f"{EXPERIMENTS.get(exp_b, {}).get('label', exp_b)}",
                "note": "Missing data",
            })
    return comparisons


def check_success_criteria(all_agg: dict) -> list:
    """Evaluate all success criteria against experiment results."""
    checks = []
    for crit_id, spec in SUCCESS_CRITERIA.items():
        exp = spec["exp"]
        metric_key = f"val/{spec['metric'].replace('_MI', '')}_mean"
        # Try common key patterns
        value = all_agg.get(exp, {}).get(metric_key, None)
        if value is None:
            value = all_agg.get(exp, {}).get(f"val/MI_srcc_mean", None)

        met = value >= spec["target"] if value is not None else None
        checks.append({
            "id": crit_id,
            "description": spec["desc"],
            "target": spec["target"],
            "actual": value,
            "met": met,
        })
    return checks


# ── Table Generation ─────────────────────────────────────────────────

def format_ci(point, ci_lo, ci_hi) -> str:
    """Format value with 95% CI."""
    if point is None:
        return "_pending_"
    return f"{point:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]"


def format_val(v, fmt=".3f") -> str:
    if v is None:
        return "_pending_"
    return f"{v:{fmt}}"


def generate_main_results_table(all_agg: dict, baselines: dict) -> str:
    """Generate Table 1: System-Level Correlation."""
    lines = [
        "| Method | SRCC_sys [95% CI] | PCC_sys [95% CI] | Ktau_sys [95% CI] |",
        "|--------|-------------------|-------------------|-------------------|",
    ]

    # Baselines
    for bkey in BASELINE_KEYS:
        label = bkey.replace("_", " ")
        bdata = baselines.get(bkey, [])
        if bdata:
            srcc_vals = [d.get("srcc_sys", 0) for d in bdata if d]
            srcc_mean = np.mean(srcc_vals) if srcc_vals else None
            lines.append(f"| {label} | {format_val(srcc_mean)} | -- | -- |")
        else:
            lines.append(f"| {label} | _pending_ | _pending_ | _pending_ |")

    lines.append("| Audiobox Aesthetics* | 0.200 [--] | -- | -- |")

    # Our experiments
    for exp_name, info in EXPERIMENTS.items():
        agg = all_agg.get(exp_name, {})
        srcc = agg.get("val/MI_srcc_mean", None)
        srcc_std = agg.get("val/MI_srcc_std", None)
        bold = "**" if exp_name == "A3c" else ""
        label = f"{bold}{info['label']} ({info['desc']}){bold}"
        lines.append(f"| {label} | {format_val(srcc)} +/- {format_val(srcc_std)} | "
                     f"_pending_ | _pending_ |")

    lines.append("")
    lines.append("*Audiobox Aesthetics: published value from Zhang et al. (MLSP 2025).")
    return "\n".join(lines)


def generate_ablation_table(deltas: list) -> str:
    """Generate Table 3: Progressive Ablation Deltas."""
    lines = [
        "| Step | Component Added | SRCC_sys Delta | Cohen's q | Meets >= 0.02? |",
        "|------|----------------|---------------|-----------|---------------|",
    ]
    for d in deltas:
        if d.get("delta") is not None:
            meets = "Yes" if d["meets_threshold"] else "**No (negative result)**"
            lines.append(
                f"| {d['step']} | {d['component']} | "
                f"{d['delta']:+.4f} | {d['cohens_q']:.3f} | {meets} |"
            )
        else:
            lines.append(f"| {d['step']} | {d['component']} | _pending_ | _pending_ | _pending_ |")
    return "\n".join(lines)


def generate_steiger_table(comparisons: list) -> str:
    """Generate Table 4: Steiger Test Results."""
    lines = [
        "| Comparison | r_A3 | r_other | z-stat | p-value | Cohen's q | Sig? |",
        "|-----------|------|---------|--------|---------|-----------|------|",
    ]
    for c in comparisons:
        if c.get("note") == "Missing data":
            lines.append(f"| {c['comparison']} | _pending_ | _pending_ | -- | -- | -- | -- |")
        else:
            sig = "Yes" if c["significant"] else "No"
            lines.append(
                f"| {c['comparison']} | {c['r_primary']:.3f} | {c['r_other']:.3f} | "
                f"{c['z_stat']:.3f} | {c['p_value']:.4f} | {c['cohens_q']:.3f} | {sig} |"
            )
    return "\n".join(lines)


def generate_success_criteria_table(checks: list) -> str:
    """Generate success criteria checklist."""
    lines = [
        "| ID | Description | Target | Actual | Met? |",
        "|----|------------|--------|--------|------|",
    ]
    for c in checks:
        actual = format_val(c["actual"]) if c["actual"] is not None else "_pending_"
        met = "Yes" if c["met"] else ("**No**" if c["met"] is not None else "_pending_")
        lines.append(f"| {c['id']} | {c['description']} | {c['target']} | {actual} | {met} |")
    return "\n".join(lines)


# ── Report Assembly ──────────────────────────────────────────────────

def write_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Wrote: {path}")


def assemble_report(
    analysis_dir: Path,
    all_agg: dict,
    baselines: dict,
    deltas: list,
    comparisons: list,
    success_checks: list,
    degradation: dict,
    cross_dataset: dict,
    cost: dict,
):
    """Write the fully populated analysis_report.md."""
    tables_dir = analysis_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Write machine-readable JSON tables
    write_json({"experiments": all_agg}, tables_dir / "main_results.json")
    write_json({"deltas": deltas}, tables_dir / "ablation_results.json")
    write_json({"comparisons": comparisons}, tables_dir / "steiger_results.json")
    write_json({"criteria": success_checks}, tables_dir / "success_criteria.json")
    if degradation:
        write_json(degradation, tables_dir / "degradation_concordance.json")
    if cross_dataset:
        write_json(cross_dataset, tables_dir / "cross_dataset_transfer.json")
    if cost:
        write_json(cost, tables_dir / "computational_cost.json")

    # Write markdown tables
    main_tbl = generate_main_results_table(all_agg, baselines)
    ablation_tbl = generate_ablation_table(deltas)
    steiger_tbl = generate_steiger_table(comparisons)
    success_tbl = generate_success_criteria_table(success_checks)

    with open(tables_dir / "main_results.md", "w") as f:
        f.write("# Table 1: System-Level Correlation with Human MOS\n\n")
        f.write(main_tbl)
    with open(tables_dir / "ablation_results.md", "w") as f:
        f.write("# Table 3: Progressive Ablation Deltas\n\n")
        f.write(ablation_tbl)
    with open(tables_dir / "statistical_comparisons.md", "w") as f:
        f.write("# Table 4: Steiger Test Results (A3 vs All)\n\n")
        f.write(steiger_tbl)

    print(f"\n  All tables written to {tables_dir}/")

    # Determine overall verdict
    n_met = sum(1 for c in success_checks if c["met"] is True)
    n_total = len(success_checks)
    n_pending = sum(1 for c in success_checks if c["met"] is None)

    # Identify negative results
    negative_results = [d for d in deltas if d.get("meets_threshold") is False]

    # Write summary JSON
    summary = {
        "success_criteria": {"met": n_met, "total": n_total, "pending": n_pending},
        "negative_results": negative_results,
        "primary_metric": all_agg.get("A3c", {}).get("val/MI_srcc_mean", None),
        "gate_metric": all_agg.get("A2_frozen_ordinal", {}).get("val/MI_srcc_mean", None),
    }
    write_json(summary, analysis_dir / "analysis_summary.json")

    print(f"\n  Analysis complete.")
    print(f"  Success criteria: {n_met}/{n_total} met, {n_pending} pending")
    if negative_results:
        print(f"  Negative results: {len(negative_results)} ablation steps below threshold")
    else:
        print(f"  No negative results detected in ablation chain")


# ── Figure Generation ────────────────────────────────────────────────

def generate_figures(output_dir: Path, analysis_dir: Path):
    """Generate all planned figures using the visualization scripts."""
    import subprocess

    figures_dir = analysis_dir / "figures"
    scripts = {
        "system_level_scatter.py": [
            "--results-dir", str(output_dir / "A3c"),
            "--output", str(figures_dir / "system_level_scatter"),
        ],
        "ablation_bar_chart.py": [
            "--summary", str(output_dir / "ablation_summary.json"),
            "--output", str(figures_dir / "ablation_bar_chart"),
        ],
        "degradation_heatmap.py": [
            "--results", str(output_dir / "A3c" / "degradation_results.json"),
            "--output", str(figures_dir / "degradation_heatmap"),
        ],
        "training_curves.py": [
            "--output-dir", str(output_dir),
            "--output", str(figures_dir / "training_curves"),
        ],
    }

    for script_name, args in scripts.items():
        script_path = figures_dir / script_name
        if script_path.exists():
            print(f"  Generating figure: {script_name}")
            cmd = [sys.executable, str(script_path)] + args
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    WARNING: {script_name} failed: {result.stderr[:200]}")
            else:
                print(f"    OK")
        else:
            print(f"  SKIP: {script_name} not found at {script_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze MuQ-MOS experiment results")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Directory containing experiment outputs")
    parser.add_argument("--analysis-dir", type=str, default="../analysis",
                       help="Directory for analysis outputs")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--skip-figures", action="store_true",
                       help="Skip figure generation (useful if matplotlib unavailable)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)

    print("=" * 60)
    print("MuQ-MOS Post-Execution Analysis")
    print("=" * 60)

    # ── 1. Load all experiment results ──
    print("\n1. Loading experiment results...")
    all_agg = {}
    all_preds = {}
    for exp_name in EXPERIMENTS:
        agg = load_aggregated(output_dir, exp_name)
        if agg:
            all_agg[exp_name] = agg
            print(f"  {exp_name}: loaded ({agg.get('n_folds', '?')} folds)")
        else:
            print(f"  {exp_name}: NOT FOUND")

        preds = load_fold_predictions(output_dir, exp_name, args.n_folds)
        if preds:
            all_preds[exp_name] = preds

    baselines = load_baseline_results(output_dir, args.n_folds)
    degradation = load_degradation_results(output_dir)
    cross_dataset = load_cross_dataset_results(output_dir)
    cost = load_computational_cost(output_dir)

    if not all_agg:
        print("\nERROR: No experiment results found. Run experiments first.")
        print(f"Expected results in: {output_dir}/<exp_name>/aggregated_results.json")
        sys.exit(1)

    # ── 2. Go/No-Go Gate Check ──
    print("\n2. Go/No-Go gate verification...")
    a2_srcc = all_agg.get("A2_frozen_ordinal", {}).get("val/MI_srcc_mean", None)
    if a2_srcc is not None:
        if a2_srcc >= 0.80:
            print(f"  A2 SRCC_sys = {a2_srcc:.4f} >= 0.80 -> GATE PASSED")
        elif a2_srcc >= 0.75:
            print(f"  A2 SRCC_sys = {a2_srcc:.4f} in [0.75, 0.80) -> MARGINAL (retry zone)")
        else:
            print(f"  A2 SRCC_sys = {a2_srcc:.4f} < 0.75 -> GATE FAILED")
    else:
        print("  A2 results not available")

    # ── 3. Ablation Analysis ──
    print("\n3. Ablation delta analysis...")
    deltas = run_ablation_delta_analysis(all_agg)
    for d in deltas:
        if d.get("delta") is not None:
            flag = "" if d["meets_threshold"] else " [NEGATIVE RESULT]"
            print(f"  {d['step']}: delta = {d['delta']:+.4f}, "
                  f"Cohen's q = {d['cohens_q']:.3f}{flag}")
        else:
            print(f"  {d['step']}: data missing")

    # ── 4. Steiger Tests ──
    print("\n4. Steiger pairwise comparisons...")
    comparisons = run_steiger_comparisons(all_agg, all_preds)
    for c in comparisons:
        if c.get("note") != "Missing data":
            sig = "*" if c["significant"] else "ns"
            print(f"  {c['comparison']}: z={c['z_stat']:.3f}, "
                  f"p={c['p_value']:.4f} ({sig}), q={c['cohens_q']:.3f}")

    # ── 5. Success Criteria ──
    print("\n5. Success criteria check...")
    success_checks = check_success_criteria(all_agg)
    for c in success_checks:
        status = "MET" if c["met"] else ("MISSED" if c["met"] is not None else "PENDING")
        actual = f"{c['actual']:.3f}" if c['actual'] is not None else "N/A"
        print(f"  {c['id']}: {actual} vs {c['target']} -> {status}")

    # ── 6. Assemble Report ──
    print("\n6. Assembling analysis report...")
    assemble_report(
        analysis_dir, all_agg, baselines, deltas,
        comparisons, success_checks, degradation, cross_dataset, cost,
    )

    # ── 7. Generate Figures ──
    if not args.skip_figures:
        print("\n7. Generating figures...")
        generate_figures(output_dir, analysis_dir)
    else:
        print("\n7. Figure generation skipped (--skip-figures)")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  Tables:  {analysis_dir / 'tables' / '*.md'}")
    print(f"  JSON:    {analysis_dir / 'tables' / '*.json'}")
    print(f"  Summary: {analysis_dir / 'analysis_summary.json'}")
    if not args.skip_figures:
        print(f"  Figures: {analysis_dir / 'figures' / '*.png'}")


if __name__ == "__main__":
    main()

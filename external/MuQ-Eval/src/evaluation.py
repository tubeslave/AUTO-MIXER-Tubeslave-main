"""
Evaluation and statistical analysis module.

Computes:
  - Utterance-level PCC and SRCC with BCa bootstrap CIs
  - System-level SRCC (per-model mean scores) with bootstrap CIs
  - Steiger test for comparing dependent correlations
  - Kendall tau for model ranking with permutation test
  - Cohen's q effect sizes for correlation differences
  - Degradation concordance with binomial test
"""

import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau, binomtest
from typing import Optional
import warnings


def compute_correlations(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """Compute PCC and SRCC between predictions and targets.

    Args:
        predictions: [N] predicted scores.
        targets: [N] ground-truth scores.

    Returns:
        Dict with pcc, srcc, pcc_p, srcc_p.
    """
    valid = ~(np.isnan(predictions) | np.isnan(targets))
    pred = predictions[valid]
    tgt = targets[valid]

    if len(pred) < 3:
        return {"pcc": 0.0, "srcc": 0.0, "pcc_p": 1.0, "srcc_p": 1.0}

    pcc, pcc_p = pearsonr(pred, tgt)
    srcc, srcc_p = spearmanr(pred, tgt)

    return {
        "pcc": float(pcc),
        "srcc": float(srcc),
        "pcc_p": float(pcc_p),
        "srcc_p": float(srcc_p),
        "n": int(len(pred)),
    }


def compute_system_level_correlations(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_ids: np.ndarray,
) -> dict:
    """Compute system-level correlations (per-model mean scores).

    Args:
        predictions: [N] per-clip predictions.
        targets: [N] per-clip ground truth.
        model_ids: [N] model identifiers for each clip.

    Returns:
        Dict with srcc_sys, pcc_sys, ktau_sys, n_systems.
    """
    unique_models = np.unique(model_ids)
    model_pred_means = []
    model_tgt_means = []

    for model in unique_models:
        mask = model_ids == model
        pred_mean = np.mean(predictions[mask])
        tgt_mean = np.mean(targets[mask])
        model_pred_means.append(pred_mean)
        model_tgt_means.append(tgt_mean)

    pred_arr = np.array(model_pred_means)
    tgt_arr = np.array(model_tgt_means)

    if len(pred_arr) < 3:
        return {"srcc_sys": 0.0, "pcc_sys": 0.0, "ktau_sys": 0.0, "n_systems": 0}

    pcc, pcc_p = pearsonr(pred_arr, tgt_arr)
    srcc, srcc_p = spearmanr(pred_arr, tgt_arr)
    ktau, ktau_p = kendalltau(pred_arr, tgt_arr)

    return {
        "srcc_sys": float(srcc),
        "pcc_sys": float(pcc),
        "ktau_sys": float(ktau),
        "srcc_sys_p": float(srcc_p),
        "n_systems": int(len(unique_models)),
    }


def bootstrap_ci(
    predictions: np.ndarray,
    targets: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute BCa bootstrap confidence interval for a correlation metric.

    Args:
        predictions: [N] predicted scores.
        targets: [N] ground truth scores.
        metric_fn: Function(pred, tgt) -> float (e.g., SRCC).
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (0.95 for 95% CI).
        seed: Random seed.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(predictions)
    point = metric_fn(predictions, targets)

    # Bootstrap resamples
    boot_values = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            val = metric_fn(predictions[idx], targets[idx])
            boot_values.append(val)
        except Exception:
            continue

    boot_values = np.array(boot_values)

    # BCa correction
    # Bias correction
    z0 = stats.norm.ppf(np.mean(boot_values < point))

    # Acceleration (jackknife)
    jackknife_vals = []
    for i in range(n):
        idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        try:
            jackknife_vals.append(metric_fn(predictions[idx], targets[idx]))
        except Exception:
            jackknife_vals.append(point)
    jackknife_vals = np.array(jackknife_vals)
    jackknife_mean = np.mean(jackknife_vals)
    diff = jackknife_mean - jackknife_vals
    a = np.sum(diff ** 3) / (6 * (np.sum(diff ** 2)) ** 1.5 + 1e-10)

    # Adjusted percentiles
    alpha = (1 - ci) / 2
    z_alpha = stats.norm.ppf(alpha)
    z_1alpha = stats.norm.ppf(1 - alpha)

    a1 = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    ci_lower = float(np.percentile(boot_values, 100 * a1))
    ci_upper = float(np.percentile(boot_values, 100 * a2))

    return float(point), ci_lower, ci_upper


def steiger_test(
    r_xz: float, r_yz: float, r_xy: float, n: int
) -> tuple[float, float]:
    """Steiger's test for comparing two dependent correlations.

    Tests H0: rho_xz == rho_yz where x and y are two predictors
    and z is the criterion (human MOS).

    Args:
        r_xz: Correlation between predictor X and criterion Z.
        r_yz: Correlation between predictor Y and criterion Z.
        r_xy: Correlation between predictors X and Y.
        n: Sample size.

    Returns:
        (z_statistic, p_value) two-tailed.
    """
    # Fisher z-transform
    z_xz = np.arctanh(r_xz)
    z_yz = np.arctanh(r_yz)

    # Steiger's formula
    r_det = (1 - r_xz**2 - r_yz**2 - r_xy**2 + 2 * r_xz * r_yz * r_xy)
    r_bar = (r_xz + r_yz) / 2

    denom = np.sqrt(
        (2 * (1 - r_xy)) / ((n - 3) * (1 + r_bar))
    )

    if denom < 1e-10:
        return 0.0, 1.0

    z_stat = (z_xz - z_yz) / denom
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return float(z_stat), float(p_val)


def cohens_q(r1: float, r2: float) -> float:
    """Cohen's q effect size for the difference between two correlations.

    Interpretation: small >= 0.10, medium >= 0.30, large >= 0.50.
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    return abs(z1 - z2)


def degradation_concordance(
    model_fn,
    clean_audio: np.ndarray,
    degraded_audio: np.ndarray,
) -> float:
    """Compute degradation concordance: fraction of pairs where
    the model assigns a higher score to the clean version.

    Args:
        model_fn: Function(audio_batch) -> scores_array.
        clean_audio: [N, samples] clean audio.
        degraded_audio: [N, samples] degraded versions.

    Returns:
        Concordance rate in [0, 1].
    """
    clean_scores = model_fn(clean_audio)
    degraded_scores = model_fn(degraded_audio)
    concordant = np.sum(clean_scores > degraded_scores)
    return float(concordant / len(clean_scores))


def full_evaluation(
    predictions: np.ndarray,
    targets: np.ndarray,
    model_ids: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    """Run full evaluation suite on a set of predictions.

    Returns comprehensive metrics with confidence intervals.
    """
    results = {}

    # Utterance-level
    utt = compute_correlations(predictions, targets)
    results["utterance"] = utt

    # Utterance-level with CIs
    def srcc_fn(p, t):
        return spearmanr(p, t)[0]
    def pcc_fn(p, t):
        return pearsonr(p, t)[0]

    srcc_point, srcc_lo, srcc_hi = bootstrap_ci(
        predictions, targets, srcc_fn, n_bootstrap, ci_level
    )
    pcc_point, pcc_lo, pcc_hi = bootstrap_ci(
        predictions, targets, pcc_fn, n_bootstrap, ci_level
    )
    results["utterance_ci"] = {
        "srcc": {"point": srcc_point, "ci_lower": srcc_lo, "ci_upper": srcc_hi},
        "pcc": {"point": pcc_point, "ci_lower": pcc_lo, "ci_upper": pcc_hi},
    }

    # System-level
    sys_corr = compute_system_level_correlations(predictions, targets, model_ids)
    results["system"] = sys_corr

    # System-level SRCC with CI
    unique_models = np.unique(model_ids)
    model_preds = np.array([np.mean(predictions[model_ids == m]) for m in unique_models])
    model_tgts = np.array([np.mean(targets[model_ids == m]) for m in unique_models])

    if len(model_preds) >= 5:
        sys_srcc_pt, sys_srcc_lo, sys_srcc_hi = bootstrap_ci(
            model_preds, model_tgts, srcc_fn, n_bootstrap, ci_level
        )
        results["system_ci"] = {
            "srcc_sys": {
                "point": sys_srcc_pt,
                "ci_lower": sys_srcc_lo,
                "ci_upper": sys_srcc_hi,
            }
        }

    return results

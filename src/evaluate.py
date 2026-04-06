"""Evaluation utilities — metrics, bootstrap CIs, and statistical tests."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)
from itertools import combinations


def compute_metrics(y_true, y_proba, threshold=0.5):
    """Standard classification metrics given probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "auc": roc_auc_score(y_true, y_proba),
        "ap": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": cm,
        "threshold": threshold,
    }


def find_optimal_threshold(y_true, y_proba, n_thresholds=200):
    """Find threshold that maximises F1 score."""
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_f1, best_t = 0.0, 0.5

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t)


def bootstrap_ci(y_true, y_proba, metric_fn, n_bootstrap=1000, ci=0.95,
                 seed=42):
    """Bootstrap confidence interval for a scalar metric.

    metric_fn(y_true, y_proba) -> float
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        # skip degenerate samples
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(metric_fn(y_true[idx], y_proba[idx]))

    scores = np.array(scores)
    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))

    return float(np.mean(scores)), float(lower), float(upper)


def bootstrap_all_metrics(y_true, y_proba, threshold=0.5, n_bootstrap=1000):
    """Bootstrap CIs for AUC, AP, F1, precision, recall."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    results = {}

    # AUC
    point, lo, hi = bootstrap_ci(y_true, y_proba, roc_auc_score, n_bootstrap)
    results["auc"] = {"point": point, "ci_lower": lo, "ci_upper": hi}

    # AP
    point, lo, hi = bootstrap_ci(
        y_true, y_proba, average_precision_score, n_bootstrap
    )
    results["ap"] = {"point": point, "ci_lower": lo, "ci_upper": hi}

    # metrics that need thresholding
    def _f1(y, p):
        return f1_score(y, (p >= threshold).astype(int), zero_division=0)

    def _prec(y, p):
        return precision_score(y, (p >= threshold).astype(int), zero_division=0)

    def _rec(y, p):
        return recall_score(y, (p >= threshold).astype(int), zero_division=0)

    for name, fn in [("f1", _f1), ("precision", _prec), ("recall", _rec)]:
        point, lo, hi = bootstrap_ci(y_true, y_proba, fn, n_bootstrap)
        results[name] = {"point": point, "ci_lower": lo, "ci_upper": hi}

    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test with continuity correction.

    Compares two classifiers' binary predictions on the same test set.
    """
    # contingency: b01 = A wrong & B right, b10 = A right & B wrong
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    b01 = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    b10 = int(np.sum(correct_a & ~correct_b))    # A right, B wrong

    # continuity-corrected chi-squared
    denom = b01 + b10
    if denom == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b01": b01, "b10": b10}

    chi2 = (abs(b01 - b10) - 1) ** 2 / denom

    # p-value from chi2 with 1 df — using scipy if available, else approx
    try:
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
    except ImportError:
        # rough approximation — good enough for reporting
        import math
        p_value = math.erfc(math.sqrt(chi2 / 2))

    return {"chi2": float(chi2), "p_value": float(p_value),
            "b01": b01, "b10": b10}


# ---------------------------------------------------------------------------
# Curve data helpers
# ---------------------------------------------------------------------------

def get_roc_data(y_true, y_proba):
    """Returns (fpr, tpr, thresholds) for plotting."""
    return roc_curve(y_true, y_proba)


def get_pr_data(y_true, y_proba):
    """Returns (precision, recall, thresholds) for plotting."""
    return precision_recall_curve(y_true, y_proba)


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def full_evaluation(y_true, predictions, n_bootstrap=1000):
    """Evaluate all models and run pairwise McNemar tests.

    predictions: dict of {model_name: y_proba}
    Returns dict with per-model metrics and pairwise comparisons.
    """
    y_true = np.asarray(y_true)
    results = {}

    # per-model metrics
    for name, proba in predictions.items():
        proba = np.asarray(proba)
        opt_t = find_optimal_threshold(y_true, proba)
        metrics = compute_metrics(y_true, proba, threshold=opt_t)
        ci = bootstrap_all_metrics(y_true, proba, threshold=opt_t,
                                   n_bootstrap=n_bootstrap)
        results[name] = {
            "metrics": metrics,
            "bootstrap_ci": ci,
            "optimal_threshold": opt_t,
        }

    # pairwise McNemar tests
    model_names = list(predictions.keys())
    pairwise = {}
    for a, b in combinations(model_names, 2):
        proba_a = np.asarray(predictions[a])
        proba_b = np.asarray(predictions[b])
        t_a = results[a]["optimal_threshold"]
        t_b = results[b]["optimal_threshold"]

        pred_a = (proba_a >= t_a).astype(int)
        pred_b = (proba_b >= t_b).astype(int)

        test = mcnemar_test(y_true, pred_a, pred_b)
        pairwise[f"{a}_vs_{b}"] = test

    results["pairwise_mcnemar"] = pairwise
    return results

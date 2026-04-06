#!/usr/bin/env python3
"""Sensitivity analysis for wallet threshold and co-trading degree.

Tests the FeaturelessHeteroGAT's robustness to two critical hyperparameters:
  1. min_tokens (wallet activity threshold): {10, 20, 50}
  2. target_mean_degree (co-trading graph density): {5, 15, 30}

For each setting the full pipeline is re-run: rebuild graph -> retrain -> evaluate.
Results are saved to results/sensitivity_results.json.

Usage:
    python scripts/run_sensitivity.py
"""

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.graph import build_all_graphs
from src.models.featureless_gat import FeaturelessHeteroGAT
from src.train import train_model, get_device
from src.evaluate import bootstrap_ci
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sensitivity")

RESULTS_DIR = ROOT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"

# Hyperparameter grids
MIN_TOKENS_VALUES = [10, 20, 50]
TARGET_DEGREE_VALUES = [5, 15, 30]


def load_config() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_evaluate(cfg: dict, device: str) -> dict:
    """Build graphs from config, train FeaturelessHeteroGAT, evaluate on test.

    Returns dict with val_auc, test_auc, and bootstrap CI.
    """
    set_seed(cfg["seed"])

    # Build graphs with the modified config
    log.info("Building graphs (min_tokens=%d, target_mean_degree=%.1f) ...",
             cfg["data"]["min_tokens"], cfg["graph"]["target_mean_degree"])
    graphs, _ = build_all_graphs(cfg)

    train_data = graphs["train"]
    val_data = graphs["val"]
    test_data = graphs["test"]

    mcfg = cfg["model"]
    tcfg = cfg["training"]

    # Train
    model = FeaturelessHeteroGAT(
        embed_dim=mcfg["embed_dim"],
        hidden_dim=mcfg["hidden_dim"],
        gat_heads=mcfg["gat_heads"],
        gat_head_dim=mcfg["gat_head_dim"],
        metadata=train_data.metadata(),
        dropout=mcfg["dropout"],
    )
    log.info("GAT parameters: %s", f"{model.count_parameters():,}")

    result = train_model(
        model, train_data, val_data,
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        max_epochs=tcfg["max_epochs"],
        patience=tcfg["patience"],
        grad_clip=tcfg["grad_clip"],
        device=device,
    )
    val_auc = result["best_val_auc"]
    log.info("Val AUC: %.4f (epoch %d)", val_auc, result["epochs_trained"])

    # Evaluate on test set
    model.to(device)
    model.eval()
    test_dev = test_data.to(device)
    with torch.no_grad():
        logits, _ = model(
            {k: test_dev[k].x for k in ["token", "wallet"]},
            {et: test_dev[et].edge_index for et in test_dev.edge_types},
        )
        proba = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

    y_true = test_data["token"].y.cpu().numpy()
    test_auc = roc_auc_score(y_true, proba)

    # Bootstrap CI
    ecfg = cfg["evaluation"]
    point, ci_lo, ci_hi = bootstrap_ci(
        y_true, proba, roc_auc_score, n_bootstrap=ecfg["n_bootstrap"],
    )

    # Graph statistics
    n_tokens = train_data["token"].x.shape[0]
    n_wallets = train_data["wallet"].x.shape[0]

    return {
        "val_auc": round(val_auc, 4),
        "test_auc": round(test_auc, 4),
        "test_auc_ci_lower": round(ci_lo, 4),
        "test_auc_ci_upper": round(ci_hi, 4),
        "n_tokens_train": n_tokens,
        "n_wallets_train": n_wallets,
        "epochs_trained": result["epochs_trained"],
    }


def main():
    cfg = load_config()
    device = get_device()
    log.info("Device: %s | Base seed: %d", device, cfg["seed"])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {"min_tokens": {}, "target_mean_degree": {}}

    # --- Sweep 1: min_tokens ---
    log.info("=" * 60)
    log.info("SWEEP 1: min_tokens values = %s", MIN_TOKENS_VALUES)
    log.info("=" * 60)

    for mt in MIN_TOKENS_VALUES:
        log.info("-" * 40)
        log.info("min_tokens = %d", mt)
        t0 = time.time()

        sweep_cfg = copy.deepcopy(cfg)
        sweep_cfg["data"]["min_tokens"] = mt

        metrics = build_train_evaluate(sweep_cfg, device)
        metrics["time_s"] = round(time.time() - t0, 1)
        all_results["min_tokens"][str(mt)] = metrics

        log.info("min_tokens=%d -> test AUC=%.4f [%.4f, %.4f] (%.0fs)",
                 mt, metrics["test_auc"],
                 metrics["test_auc_ci_lower"], metrics["test_auc_ci_upper"],
                 metrics["time_s"])

    # --- Sweep 2: target_mean_degree ---
    log.info("=" * 60)
    log.info("SWEEP 2: target_mean_degree values = %s", TARGET_DEGREE_VALUES)
    log.info("=" * 60)

    for deg in TARGET_DEGREE_VALUES:
        log.info("-" * 40)
        log.info("target_mean_degree = %d", deg)
        t0 = time.time()

        sweep_cfg = copy.deepcopy(cfg)
        sweep_cfg["graph"]["target_mean_degree"] = float(deg)

        metrics = build_train_evaluate(sweep_cfg, device)
        metrics["time_s"] = round(time.time() - t0, 1)
        all_results["target_mean_degree"][str(deg)] = metrics

        log.info("target_mean_degree=%d -> test AUC=%.4f [%.4f, %.4f] (%.0fs)",
                 deg, metrics["test_auc"],
                 metrics["test_auc_ci_lower"], metrics["test_auc_ci_upper"],
                 metrics["time_s"])

    # --- Save results ---
    out_path = RESULTS_DIR / "sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_path)

    # --- Print summary table ---
    log.info("\n" + "=" * 60)
    log.info("SENSITIVITY ANALYSIS SUMMARY")
    log.info("=" * 60)

    log.info("\nmin_tokens sweep (target_mean_degree=%.1f fixed):",
             cfg["graph"]["target_mean_degree"])
    log.info("%-12s %8s %8s %10s %10s", "min_tokens", "val_AUC", "test_AUC",
             "CI_lower", "CI_upper")
    for mt in MIN_TOKENS_VALUES:
        r = all_results["min_tokens"][str(mt)]
        log.info("%-12d %8.4f %8.4f %10.4f %10.4f",
                 mt, r["val_auc"], r["test_auc"],
                 r["test_auc_ci_lower"], r["test_auc_ci_upper"])

    log.info("\ntarget_mean_degree sweep (min_tokens=%d fixed):",
             cfg["data"]["min_tokens"])
    log.info("%-12s %8s %8s %10s %10s", "degree", "val_AUC", "test_AUC",
             "CI_lower", "CI_upper")
    for deg in TARGET_DEGREE_VALUES:
        r = all_results["target_mean_degree"][str(deg)]
        log.info("%-12d %8.4f %8.4f %10.4f %10.4f",
                 deg, r["val_auc"], r["test_auc"],
                 r["test_auc_ci_lower"], r["test_auc_ci_upper"])


if __name__ == "__main__":
    main()

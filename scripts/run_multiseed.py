#!/usr/bin/env python3
"""Multi-seed training for variance estimation.

Trains the FeaturelessHeteroGAT and MLP baseline across multiple seeds
to report mean +/- std of test AUC, quantifying model stability.

Seeds: {42, 123, 7}
Results saved to results/multiseed_results.json.

Usage:
    python scripts/run_multiseed.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.featureless_gat import FeaturelessHeteroGAT
from src.models.baselines import MLPBaseline, train_mlp
from src.train import train_model, get_device
from src.evaluate import bootstrap_ci

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("multiseed")

RESULTS_DIR = ROOT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"
CKPT_DIR = RESULTS_DIR / "checkpoints"

SEEDS = [42, 123, 7]


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


def run_gat_inference(model, test_data, device):
    """Run GAT inference on test set, return (y_true, y_proba)."""
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
    return y_true, proba


def run_mlp_inference(model, test_X, device):
    """Run MLP inference on test set, return y_proba."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        proba = model.predict_proba(
            torch.tensor(test_X, dtype=torch.float32).to(device)
        ).cpu().numpy()
    return proba


def main():
    cfg = load_config()
    device = get_device()
    log.info("Device: %s | Seeds: %s", device, SEEDS)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load pre-built graphs (shared across seeds -- only model init varies)
    log.info("Loading pre-built graphs ...")
    train_data = torch.load(GRAPH_DIR / "hetero_graph_train.pt", weights_only=False)
    val_data = torch.load(GRAPH_DIR / "hetero_graph_val.pt", weights_only=False)
    test_data = torch.load(GRAPH_DIR / "hetero_graph_test.pt", weights_only=False)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    ecfg = cfg["evaluation"]

    # Prepare tabular data for MLP
    train_X = train_data["token"].x.cpu().numpy()
    train_y = train_data["token"].y.cpu().numpy()
    val_X = val_data["token"].x.cpu().numpy()
    val_y = val_data["token"].y.cpu().numpy()
    test_X = test_data["token"].x.cpu().numpy()
    y_true = test_data["token"].y.cpu().numpy()

    gat_results = []
    mlp_results = []

    for seed in SEEDS:
        log.info("=" * 50)
        log.info("SEED: %d", seed)
        log.info("=" * 50)

        set_seed(seed)

        # --- FeaturelessHeteroGAT ---
        log.info("Training FeaturelessHeteroGAT (seed=%d) ...", seed)
        t0 = time.time()

        model = FeaturelessHeteroGAT(
            embed_dim=mcfg["embed_dim"],
            hidden_dim=mcfg["hidden_dim"],
            gat_heads=mcfg["gat_heads"],
            gat_head_dim=mcfg["gat_head_dim"],
            metadata=train_data.metadata(),
            dropout=mcfg["dropout"],
        )

        res = train_model(
            model, train_data, val_data,
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
            max_epochs=tcfg["max_epochs"],
            patience=tcfg["patience"],
            grad_clip=tcfg["grad_clip"],
            device=device,
        )

        _, gat_proba = run_gat_inference(model, test_data, device)
        gat_auc = roc_auc_score(y_true, gat_proba)

        point, ci_lo, ci_hi = bootstrap_ci(
            y_true, gat_proba, roc_auc_score,
            n_bootstrap=ecfg["n_bootstrap"],
        )

        gat_entry = {
            "seed": seed,
            "val_auc": round(res["best_val_auc"], 4),
            "test_auc": round(gat_auc, 4),
            "test_auc_ci_lower": round(ci_lo, 4),
            "test_auc_ci_upper": round(ci_hi, 4),
            "epochs_trained": res["epochs_trained"],
            "time_s": round(time.time() - t0, 1),
        }
        gat_results.append(gat_entry)
        log.info("GAT seed=%d: val_AUC=%.4f  test_AUC=%.4f [%.4f, %.4f] (%.0fs)",
                 seed, res["best_val_auc"], gat_auc, ci_lo, ci_hi,
                 gat_entry["time_s"])

        # --- MLP Baseline ---
        log.info("Training MLP baseline (seed=%d) ...", seed)
        set_seed(seed)
        t0 = time.time()

        in_dim = train_X.shape[1]
        mlp = MLPBaseline(input_dim=in_dim)

        mlp_res = train_mlp(
            mlp, train_X, train_y, val_X, val_y,
            lr=tcfg["lr"],
            epochs=tcfg["max_epochs"],
            patience=tcfg["patience"],
            device=device,
        )

        mlp_proba = run_mlp_inference(mlp, test_X, device)
        mlp_auc = roc_auc_score(y_true, mlp_proba)

        point, ci_lo, ci_hi = bootstrap_ci(
            y_true, mlp_proba, roc_auc_score,
            n_bootstrap=ecfg["n_bootstrap"],
        )

        mlp_entry = {
            "seed": seed,
            "val_auc": round(mlp_res["best_val_auc"], 4),
            "test_auc": round(mlp_auc, 4),
            "test_auc_ci_lower": round(ci_lo, 4),
            "test_auc_ci_upper": round(ci_hi, 4),
            "epochs_trained": mlp_res["epochs_trained"],
            "time_s": round(time.time() - t0, 1),
        }
        mlp_results.append(mlp_entry)
        log.info("MLP seed=%d: val_AUC=%.4f  test_AUC=%.4f [%.4f, %.4f] (%.0fs)",
                 seed, mlp_res["best_val_auc"], mlp_auc, ci_lo, ci_hi,
                 mlp_entry["time_s"])

    # --- Aggregate statistics ---
    gat_aucs = np.array([r["test_auc"] for r in gat_results])
    mlp_aucs = np.array([r["test_auc"] for r in mlp_results])

    summary = {
        "seeds": SEEDS,
        "featureless_gat": {
            "runs": gat_results,
            "mean_test_auc": round(float(np.mean(gat_aucs)), 4),
            "std_test_auc": round(float(np.std(gat_aucs)), 4),
        },
        "mlp_baseline": {
            "runs": mlp_results,
            "mean_test_auc": round(float(np.mean(mlp_aucs)), 4),
            "std_test_auc": round(float(np.std(mlp_aucs)), 4),
        },
    }

    # Save
    out_path = RESULTS_DIR / "multiseed_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Print summary
    log.info("\n" + "=" * 60)
    log.info("MULTI-SEED SUMMARY (%d seeds: %s)", len(SEEDS), SEEDS)
    log.info("=" * 60)

    log.info("\nFeaturelessHeteroGAT:")
    log.info("  %-6s %8s %8s", "Seed", "val_AUC", "test_AUC")
    for r in gat_results:
        log.info("  %-6d %8.4f %8.4f", r["seed"], r["val_auc"], r["test_auc"])
    log.info("  Mean +/- Std: %.4f +/- %.4f",
             summary["featureless_gat"]["mean_test_auc"],
             summary["featureless_gat"]["std_test_auc"])

    log.info("\nMLP Baseline:")
    log.info("  %-6s %8s %8s", "Seed", "val_AUC", "test_AUC")
    for r in mlp_results:
        log.info("  %-6d %8.4f %8.4f", r["seed"], r["val_auc"], r["test_auc"])
    log.info("  Mean +/- Std: %.4f +/- %.4f",
             summary["mlp_baseline"]["mean_test_auc"],
             summary["mlp_baseline"]["std_test_auc"])


if __name__ == "__main__":
    main()

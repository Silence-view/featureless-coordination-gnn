#!/usr/bin/env python3
"""
COMP0162 Advanced Machine Learning — Coursework Pipeline
=========================================================

End-to-end pipeline for training and evaluating a featureless heterogeneous
GAT on the Solana wallet-token network. The model learns entirely from graph
topology (no hand-crafted node features) and predicts whether a token is
high-risk based on the trading patterns of its associated wallets.

Usage:
    python run_experiments.py --step all
    python run_experiments.py --step build
    python run_experiments.py --step train
    python run_experiments.py --step evaluate
    python run_experiments.py --step analyse
    python run_experiments.py --step figures
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# Project modules
from src.data.graph import build_all_graphs
from src.models.featureless_gat import FeaturelessHeteroGAT
from src.models.featured_hetero_gat import FeaturedHeteroGAT
from src.models.baselines import (
    LogisticRegressionBaseline, SVMBaseline,
    RandomForestBaseline, GradientBoostingBaseline,
    MLPBaseline, train_mlp,
)
from src.train import train_model, get_device
from src.evaluate import full_evaluation, find_optimal_threshold
from src.visualise import plot_roc_curves, plot_tsne
from src.analysis import extract_embeddings, cluster_wallets, profile_clusters, compute_tsne

# Paths
ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "params.yaml"
RESULTS_DIR = ROOT / "results"
GRAPH_DIR = RESULTS_DIR / "graphs"
CKPT_DIR = RESULTS_DIR / "checkpoints"
FIG_DIR = RESULTS_DIR / "figures"

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load experiment parameters from params.yaml."""
    with open(PARAMS_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg


def set_seed(seed: int):
    """Fix random seeds everywhere for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic mode helps but can slow things down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Step 1: Build graphs
# ---------------------------------------------------------------------------

def step_build(cfg: dict):
    log.info("=== Step 1: Building heterogeneous graphs + temporal events ===")
    os.makedirs(GRAPH_DIR, exist_ok=True)

    graphs, temporal_events = build_all_graphs(cfg)

    for split, g in graphs.items():
        out = GRAPH_DIR / f"hetero_graph_{split}.pt"
        torch.save(g, str(out))
        log.info("Saved %s  (%d tokens, %d wallets)",
                 out.name, g["token"].x.shape[0], g["wallet"].x.shape[0])

    # Save temporal events for link prediction
    for split, events in temporal_events.items():
        out = GRAPH_DIR / f"temporal_events_{split}.parquet"
        events.to_parquet(str(out))
        log.info("Saved %s  (%s events)", out.name, f"{len(events):,}")

    return graphs


# ---------------------------------------------------------------------------
# Step 2: Train models
# ---------------------------------------------------------------------------

def step_train(cfg: dict, device: str):
    log.info("=== Step 2: Training models ===")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Load pre-built graphs
    train_data = torch.load(GRAPH_DIR / "hetero_graph_train.pt", weights_only=False)
    val_data = torch.load(GRAPH_DIR / "hetero_graph_val.pt", weights_only=False)

    mcfg = cfg["model"]
    tcfg = cfg["training"]

    # --- FeaturelessHeteroGAT ---
    log.info("Training FeaturelessHeteroGAT ...")
    model = FeaturelessHeteroGAT(
        embed_dim=mcfg["embed_dim"],
        hidden_dim=mcfg["hidden_dim"],
        gat_heads=mcfg["gat_heads"],
        gat_head_dim=mcfg["gat_head_dim"],
        metadata=train_data.metadata(),
        dropout=mcfg["dropout"],
    )
    log.info("GAT parameters: %s", f"{model.count_parameters():,}")

    gat_result = train_model(
        model, train_data, val_data,
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        max_epochs=tcfg["max_epochs"],
        patience=tcfg["patience"],
        grad_clip=tcfg["grad_clip"],
        device=device,
        save_dir=str(CKPT_DIR),
    )
    log.info("GAT best val AUC: %.4f (epoch %d)", gat_result["best_val_auc"],
             gat_result["epochs_trained"])

    # --- FeaturedHeteroGAT ---
    log.info("Training FeaturedHeteroGAT ...")
    token_feat_dim = train_data["token"].x.shape[1]
    wallet_feat_dim = train_data["wallet"].x.shape[1]
    feat_model = FeaturedHeteroGAT(
        token_feat_dim=token_feat_dim,
        wallet_feat_dim=wallet_feat_dim,
        embed_dim=mcfg["embed_dim"],
        hidden_dim=mcfg["hidden_dim"],
        gat_heads=mcfg["gat_heads"],
        gat_head_dim=mcfg["gat_head_dim"],
        metadata=train_data.metadata(),
        dropout=mcfg["dropout"],
    )
    log.info("FeaturedGAT parameters: %s", f"{feat_model.count_parameters():,}")

    feat_result = train_model(
        feat_model, train_data, val_data,
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        max_epochs=tcfg["max_epochs"],
        patience=tcfg["patience"],
        grad_clip=tcfg["grad_clip"],
        device=device,
        save_dir=str(CKPT_DIR),
    )
    # Save with distinct name (train_model saves as best_hetero_gat.pt by default)
    torch.save(feat_model.state_dict(), CKPT_DIR / "best_featured_gat.pt")
    log.info("FeaturedGAT best val AUC: %.4f (epoch %d)", feat_result["best_val_auc"],
             feat_result["epochs_trained"])

    # --- Logistic regression baseline ---
    log.info("Training logistic regression baseline ...")
    train_X = train_data["token"].x.cpu().numpy()
    train_y = train_data["token"].y.cpu().numpy()
    val_X = val_data["token"].x.cpu().numpy()
    val_y = val_data["token"].y.cpu().numpy()

    lr_model = LogisticRegressionBaseline(C=1.0)
    lr_model.fit(train_X, train_y)
    lr_val = lr_model.assess(val_X, val_y)
    log.info("LR val AUC: %.4f, F1: %.4f", lr_val["auc"], lr_val["f1"])

    # Save the LR model
    import pickle
    with open(CKPT_DIR / "lr_baseline.pkl", "wb") as f:
        pickle.dump(lr_model, f)

    # --- SVM baseline ---
    log.info("Training SVM (RBF) baseline ...")
    svm_model = SVMBaseline(C=1.0)
    svm_model.fit(train_X, train_y)
    svm_val = svm_model.assess(val_X, val_y)
    log.info("SVM val AUC: %.4f, F1: %.4f", svm_val["auc"], svm_val["f1"])

    with open(CKPT_DIR / "svm_baseline.pkl", "wb") as f:
        pickle.dump(svm_model, f)

    # --- Random Forest baseline ---
    log.info("Training Random Forest baseline (200 trees) ...")
    rf_model = RandomForestBaseline(n_estimators=200)
    rf_model.fit(train_X, train_y)
    rf_val = rf_model.assess(val_X, val_y)
    log.info("RF val AUC: %.4f, F1: %.4f", rf_val["auc"], rf_val["f1"])

    with open(CKPT_DIR / "rf_baseline.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    # --- Gradient Boosting baseline ---
    log.info("Training Gradient Boosting baseline (200 estimators) ...")
    gb_model = GradientBoostingBaseline(n_estimators=200)
    gb_model.fit(train_X, train_y)
    gb_val = gb_model.assess(val_X, val_y)
    log.info("GB val AUC: %.4f, F1: %.4f", gb_val["auc"], gb_val["f1"])

    with open(CKPT_DIR / "gb_baseline.pkl", "wb") as f:
        pickle.dump(gb_model, f)

    # --- MLP baseline ---
    log.info("Training MLP baseline ...")
    in_dim = train_X.shape[1]
    mlp = MLPBaseline(input_dim=in_dim)
    log.info("MLP parameters: %s", f"{mlp.count_parameters():,}")

    # Compute pos_weight from training labels
    n_pos = train_y.sum()
    n_neg = len(train_y) - n_pos
    pw = n_neg / max(n_pos, 1)

    mlp_result = train_mlp(
        mlp, train_X, train_y, val_X, val_y,
        lr=tcfg["lr"],
        epochs=tcfg["max_epochs"],
        patience=tcfg["patience"],
        device=device,
    )
    log.info("MLP best val AUC: %.4f", mlp_result["best_val_auc"])

    # Save MLP checkpoint
    torch.save(mlp.state_dict(), CKPT_DIR / "mlp_baseline.pt")

    summary = {
        "gat_val_auc": gat_result["best_val_auc"],
        "feat_gat_val_auc": feat_result["best_val_auc"],
        "lr_val_auc": lr_val["auc"],
        "svm_val_auc": svm_val["auc"],
        "rf_val_auc": rf_val["auc"],
        "gb_val_auc": gb_val["auc"],
        "mlp_val_auc": mlp_result["best_val_auc"],
    }
    with open(RESULTS_DIR / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------------
# Step 3: Evaluate on held-out test set
# ---------------------------------------------------------------------------

def step_evaluate(cfg: dict, device: str):
    log.info("=== Step 3: Evaluating on test set ===")
    import pickle

    test_data = torch.load(GRAPH_DIR / "hetero_graph_test.pt", weights_only=False)
    train_data = torch.load(GRAPH_DIR / "hetero_graph_train.pt", weights_only=False)

    y_true = test_data["token"].y.cpu().numpy()
    test_X = test_data["token"].x.cpu().numpy()

    # --- GAT predictions ---
    model = FeaturelessHeteroGAT(
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        gat_heads=cfg["model"]["gat_heads"],
        gat_head_dim=cfg["model"]["gat_head_dim"],
        metadata=test_data.metadata(),
        dropout=cfg["model"]["dropout"],
    )
    state = torch.load(CKPT_DIR / "best_hetero_gat.pt", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_data_dev = test_data.to(device)
        gat_proba = model.predict_proba(
            {k: test_data_dev[k].x for k in ["token", "wallet"]},
            {et: test_data_dev[et].edge_index for et in test_data_dev.edge_types},
        ).cpu().numpy()

    # --- FeaturedGAT predictions ---
    token_feat_dim = test_X.shape[1]
    wallet_feat_dim = test_data["wallet"].x.shape[1]
    feat_model = FeaturedHeteroGAT(
        token_feat_dim=token_feat_dim,
        wallet_feat_dim=wallet_feat_dim,
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        gat_heads=cfg["model"]["gat_heads"],
        gat_head_dim=cfg["model"]["gat_head_dim"],
        metadata=test_data.metadata(),
        dropout=cfg["model"]["dropout"],
    )
    feat_state = torch.load(CKPT_DIR / "best_featured_gat.pt", weights_only=False)
    feat_model.load_state_dict(feat_state)
    feat_model.to(device)
    feat_model.eval()

    with torch.no_grad():
        feat_proba = feat_model.predict_proba(
            {k: test_data_dev[k].x for k in ["token", "wallet"]},
            {et: test_data_dev[et].edge_index for et in test_data_dev.edge_types},
        ).cpu().numpy()

    # --- LR predictions ---
    with open(CKPT_DIR / "lr_baseline.pkl", "rb") as f:
        lr_model = pickle.load(f)
    lr_proba = lr_model.predict_proba(test_X)

    # --- SVM predictions ---
    with open(CKPT_DIR / "svm_baseline.pkl", "rb") as f:
        svm_model = pickle.load(f)
    svm_proba = svm_model.predict_proba(test_X)

    # --- RF predictions ---
    with open(CKPT_DIR / "rf_baseline.pkl", "rb") as f:
        rf_model = pickle.load(f)
    rf_proba = rf_model.predict_proba(test_X)

    # --- GB predictions ---
    with open(CKPT_DIR / "gb_baseline.pkl", "rb") as f:
        gb_model = pickle.load(f)
    gb_proba = gb_model.predict_proba(test_X)

    # --- MLP predictions ---
    in_dim = test_X.shape[1]
    mlp = MLPBaseline(input_dim=in_dim)
    mlp.load_state_dict(torch.load(CKPT_DIR / "mlp_baseline.pt", weights_only=False))
    mlp.to(device)
    mlp.eval()
    with torch.no_grad():
        mlp_proba = mlp.predict_proba(
            torch.tensor(test_X, dtype=torch.float32).to(device)
        ).cpu().numpy()

    # --- Full evaluation with bootstrap CIs + McNemar ---
    predictions = {
        "FeaturelessGAT": gat_proba,
        "FeaturedGAT": feat_proba,
        "LogisticRegression": lr_proba,
        "SVM": svm_proba,
        "RandomForest": rf_proba,
        "GradientBoosting": gb_proba,
        "MLP": mlp_proba,
    }

    ecfg = cfg["evaluation"]
    results = full_evaluation(y_true, predictions, n_bootstrap=ecfg["n_bootstrap"])

    # Save predictions and results
    os.makedirs(RESULTS_DIR / "predictions", exist_ok=True)
    for name, proba in predictions.items():
        np.save(RESULTS_DIR / "predictions" / f"{name}_proba.npy", proba)
    np.save(RESULTS_DIR / "predictions" / "y_true.npy", y_true)

    # Serialise results (drop confusion matrices which aren't JSON-friendly)
    serialisable = {}
    for k, v in results.items():
        if k == "pairwise_mcnemar":
            serialisable[k] = v
        elif isinstance(v, dict):
            serialisable[k] = {
                "bootstrap_ci": v.get("bootstrap_ci", {}),
                "optimal_threshold": v.get("optimal_threshold", 0.5),
            }
    with open(RESULTS_DIR / "evaluation_results.json", "w") as f:
        json.dump(serialisable, f, indent=2, default=str)

    return results


# ---------------------------------------------------------------------------
# Step 4: Post-hoc analysis (clustering, t-SNE)
# ---------------------------------------------------------------------------

def step_analyse(cfg: dict, device: str):
    log.info("=== Step 4: Wallet embedding analysis ===")
    import hdbscan
    from sklearn.manifold import TSNE

    test_data = torch.load(GRAPH_DIR / "hetero_graph_test.pt", weights_only=False)

    # Reload trained GAT
    model = FeaturelessHeteroGAT(
        embed_dim=cfg["model"]["embed_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        gat_heads=cfg["model"]["gat_heads"],
        gat_head_dim=cfg["model"]["gat_head_dim"],
        metadata=test_data.metadata(),
        dropout=cfg["model"]["dropout"],
    )
    state = torch.load(CKPT_DIR / "best_hetero_gat.pt", weights_only=False)
    model.load_state_dict(state)

    # Extract wallet embeddings from the trained model
    log.info("Extracting wallet embeddings ...")
    embeddings = extract_embeddings(model, test_data, device)
    log.info("Wallet embeddings shape: %s", embeddings.shape)

    # HDBSCAN clustering
    log.info("Clustering wallets with HDBSCAN ...")
    labels = cluster_wallets(embeddings, min_cluster_size=50)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    log.info("Found %d clusters, %d noise points (%.1f%%)",
             n_clusters, n_noise, 100 * n_noise / len(labels))

    # Profile clusters using wallet feature data
    wallet_feat_path = RESULTS_DIR / "processed" / "wallet_features_test.parquet"
    if wallet_feat_path.exists():
        wf = pd.read_parquet(wallet_feat_path)
        wallet_addrs = test_data["wallet"].wallet_address
        profiles = profile_clusters(labels, wf, wallet_addrs, test_data)
        profiles.to_csv(RESULTS_DIR / "cluster_profiles.csv", index=False)
        log.info("Cluster profiles saved")
    else:
        log.warning("Wallet features not found at %s, skipping profiling", wallet_feat_path)

    # t-SNE for visualisation
    log.info("Computing t-SNE projection ...")
    coords = compute_tsne(embeddings, perplexity=30)

    # Save everything
    os.makedirs(RESULTS_DIR / "analysis", exist_ok=True)
    np.save(RESULTS_DIR / "analysis" / "wallet_embeddings.npy", embeddings)
    np.save(RESULTS_DIR / "analysis" / "cluster_labels.npy", labels)
    np.save(RESULTS_DIR / "analysis" / "tsne_coords.npy", coords)

    log.info("Analysis artefacts saved to %s", RESULTS_DIR / "analysis")
    return {"n_clusters": n_clusters, "n_noise": int(n_noise)}


# ---------------------------------------------------------------------------
# Step 5: Generate figures for the report
# ---------------------------------------------------------------------------

def step_figures(cfg: dict):
    log.info("=== Step 5: Generating figures ===")
    os.makedirs(FIG_DIR, exist_ok=True)

    # Load saved predictions
    pred_dir = RESULTS_DIR / "predictions"
    y_true = np.load(pred_dir / "y_true.npy")
    preds = {}
    for name in ["FeaturelessGAT", "FeaturedGAT", "LogisticRegression",
                  "SVM", "RandomForest", "GradientBoosting", "MLP"]:
        path = pred_dir / f"{name}_proba.npy"
        if path.exists():
            preds[name] = np.load(path)

    # ROC curves
    plot_roc_curves(y_true, preds, save_dir=str(FIG_DIR))
    log.info("Saved roc_curves.pdf")

    # t-SNE plot (coloured by cluster)
    analysis_dir = RESULTS_DIR / "analysis"
    if (analysis_dir / "tsne_coords.npy").exists():
        coords = np.load(analysis_dir / "tsne_coords.npy")
        labels = np.load(analysis_dir / "cluster_labels.npy")
        plot_tsne(coords, labels, save_dir=str(FIG_DIR))
        log.info("Saved tsne_wallets.pdf")
    else:
        log.warning("t-SNE coords not found; run the analyse step first")

    log.info("All figures written to %s", FIG_DIR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="COMP0162 coursework: featureless HeteroGAT on Solana wallet-token graphs",
    )
    parser.add_argument(
        "--step",
        choices=["all", "build", "train", "evaluate", "analyse", "figures"],
        default="all",
        help="Which pipeline step to run (default: all)",
    )
    args = parser.parse_args()

    # Logging: console + file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "pipeline.log", mode="a"),
        ],
    )

    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device()
    log.info("Device: %s | Seed: %d", device, cfg["seed"])

    t_start = time.time()

    steps_to_run = (
        ["build", "train", "evaluate", "analyse", "figures"]
        if args.step == "all"
        else [args.step]
    )

    results = {}
    for step in steps_to_run:
        t0 = time.time()
        if step == "build":
            step_build(cfg)
        elif step == "train":
            results["train"] = step_train(cfg, device)
        elif step == "evaluate":
            results["evaluate"] = step_evaluate(cfg, device)
        elif step == "analyse":
            results["analyse"] = step_analyse(cfg, device)
        elif step == "figures":
            step_figures(cfg)
        log.info("Step '%s' completed in %.1f s", step, time.time() - t0)

    elapsed = time.time() - t_start

    # Print summary
    log.info("=" * 60)
    log.info("Pipeline finished in %.1f s", elapsed)
    if "train" in results:
        r = results["train"]
        log.info("  GAT val AUC:  %.4f", r["gat_val_auc"])
        log.info("  FeatGAT AUC:  %.4f", r["feat_gat_val_auc"])
        log.info("  LR  val AUC:  %.4f", r["lr_val_auc"])
        log.info("  SVM val AUC:  %.4f", r["svm_val_auc"])
        log.info("  RF  val AUC:  %.4f", r["rf_val_auc"])
        log.info("  GB  val AUC:  %.4f", r["gb_val_auc"])
        log.info("  MLP val AUC:  %.4f", r["mlp_val_auc"])
    if "evaluate" in results:
        for name in ["FeaturelessGAT", "FeaturedGAT", "LogisticRegression",
                      "SVM", "RandomForest", "GradientBoosting", "MLP"]:
            if name in results["evaluate"]:
                b = results["evaluate"][name].get("bootstrap_ci", {})
                if "auc" in b:
                    log.info("  %s test AUC: %.4f [%.4f, %.4f]",
                             name,
                             b["auc"]["point"],
                             b["auc"]["ci_lower"],
                             b["auc"]["ci_upper"])
    if "analyse" in results:
        a = results["analyse"]
        log.info("  Wallet clusters: %d (noise: %d)", a["n_clusters"], a["n_noise"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()

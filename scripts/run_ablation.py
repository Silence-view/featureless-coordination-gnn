#!/usr/bin/env python3
"""Edge-type ablation study for the FeaturelessHeteroGAT.

Trains the model with different subsets of edge types to determine
which coordination signals matter most for classification.

Ablation variants:
  1. full          - all 4 edge types (baseline)
  2. no_same_tx    - remove same-transaction edges
  3. no_co_trade   - remove co-trading edges
  4. no_coord      - remove ALL wallet-wallet edges (same_tx + co_trade)
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.featureless_gat import FeaturelessHeteroGAT
from src.train import train_model, get_device
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ablation")

GRAPH_DIR = ROOT / "results" / "graphs"
RESULTS_DIR = ROOT / "results"


def load_config():
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def filter_edges(data, keep_edge_types):
    """Return a copy of data with only the specified edge types."""
    from torch_geometric.data import HeteroData
    new_data = HeteroData()
    for ntype in data.node_types:
        for key, val in data[ntype].items():
            new_data[ntype][key] = val
    for etype in data.edge_types:
        if etype in keep_edge_types:
            for key, val in data[etype].items():
                new_data[etype][key] = val
    return new_data


def run_metrics(model, test_data, device):
    """Get predictions and compute metrics."""
    model.to(device)
    test_dev = test_data.to(device)
    model.eval()
    with torch.no_grad():
        logits, _ = model(
            {k: test_dev[k].x for k in ["token", "wallet"]},
            {et: test_dev[et].edge_index for et in test_dev.edge_types},
        )
        proba = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
    y_true = test_data["token"].y.cpu().numpy()
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    best_f1 = 0
    for thr in np.arange(0.01, 0.99, 0.01):
        f1 = f1_score(y_true, (proba >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return {"auc": round(auc, 4), "f1": round(best_f1, 4), "ap": round(ap, 4)}


ALL_EDGES = [
    ("wallet", "trades", "token"),
    ("token", "traded_by", "wallet"),
    ("wallet", "co_trades", "wallet"),
    ("wallet", "same_tx", "wallet"),
]

ABLATIONS = {
    "full": ALL_EDGES,
    "no_same_tx": [e for e in ALL_EDGES if e[1] != "same_tx"],
    "no_co_trade": [e for e in ALL_EDGES if e[1] != "co_trades"],
    "no_coordination": [e for e in ALL_EDGES if e[1] in ("trades", "traded_by")],
}


def main():
    cfg = load_config()
    device = get_device()
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    log.info("Device: %s", device)

    train_data = torch.load(GRAPH_DIR / "hetero_graph_train.pt", weights_only=False)
    val_data = torch.load(GRAPH_DIR / "hetero_graph_val.pt", weights_only=False)
    test_data = torch.load(GRAPH_DIR / "hetero_graph_test.pt", weights_only=False)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    results = {}

    for name, keep in ABLATIONS.items():
        log.info("=" * 50)
        log.info("ABLATION: %s  edges=%s", name, [e[1] for e in keep])
        t0 = time.time()

        tr = filter_edges(train_data, keep)
        va = filter_edges(val_data, keep)
        te = filter_edges(test_data, keep)

        model = FeaturelessHeteroGAT(
            embed_dim=mcfg["embed_dim"],
            hidden_dim=mcfg["hidden_dim"],
            gat_heads=mcfg["gat_heads"],
            gat_head_dim=mcfg["gat_head_dim"],
            metadata=tr.metadata(),
            dropout=mcfg["dropout"],
        )
        log.info("Params: %s", f"{model.count_parameters():,}")

        res = train_model(
            model, tr, va,
            lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
            max_epochs=tcfg["max_epochs"], patience=tcfg["patience"],
            grad_clip=tcfg["grad_clip"], device=device,
        )

        metrics = run_metrics(model, te, device)
        elapsed = time.time() - t0

        results[name] = {
            "edges": [e[1] for e in keep],
            "params": model.count_parameters(),
            "val_auc": round(res["best_val_auc"], 4),
            "test": metrics,
            "time_s": round(elapsed, 1),
        }
        log.info("Test: AUC=%.4f F1=%.4f AP=%.4f (%.0fs)",
                 metrics["auc"], metrics["f1"], metrics["ap"], elapsed)

    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info("\nSUMMARY:")
    log.info("%-20s %6s %6s %6s", "Variant", "AUC", "F1", "AP")
    for name, r in results.items():
        t = r["test"]
        log.info("%-20s %.4f %.4f %.4f", name, t["auc"], t["f1"], t["ap"])


if __name__ == "__main__":
    main()

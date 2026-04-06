"""Early detection evaluation — how quickly can the model spot risky tokens?

The idea: evaluate the trained model on progressively larger subgraphs
built from the first 1h, 6h, 24h, and 7d of trading activity per token.
If the graph-based model reaches high AUC faster than tabular baselines,
it proves that structural signal is available early — before retail money
flows in.

This is the main original contribution beyond the temporal model itself.
"""

import logging
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.evaluate import compute_metrics, find_optimal_threshold

log = logging.getLogger(__name__)

# horizons in seconds
HORIZONS = {
    "1h":  3600,
    "6h":  6 * 3600,
    "24h": 24 * 3600,
    "3d":  3 * 24 * 3600,
    "7d":  7 * 24 * 3600,
}


def filter_edges_by_horizon(data: HeteroData, tx_timestamps: dict,
                            token_launch_times: dict,
                            horizon_sec: int) -> HeteroData:
    """Keep only edges where the interaction happened within horizon_sec
    of the token's launch time.

    tx_timestamps: {edge_type: Tensor[E] of unix seconds}
    token_launch_times: {token_idx: unix_sec} — first observed tx per token
    """
    data = deepcopy(data)

    for etype in data.edge_types:
        ei = data[etype].edge_index
        src_type, _, dst_type = etype

        if etype not in tx_timestamps:
            continue  # skip edges without timestamps (same_tx etc.)

        ts = tx_timestamps[etype]

        # figure out which end is the token
        if dst_type == "token":
            tok_indices = ei[1]
        elif src_type == "token":
            tok_indices = ei[0]
        else:
            continue  # wallet-wallet edges, keep as-is

        # keep edge if ts < launch_time + horizon
        launch = torch.tensor([
            token_launch_times.get(int(t), 0) for t in tok_indices
        ], dtype=ts.dtype)
        keep = ts <= (launch + horizon_sec)

        data[etype].edge_index = ei[:, keep]
        if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None:
            data[etype].edge_attr = data[etype].edge_attr[keep]

    return data


def get_token_launch_times(tx_timestamps: dict, edge_index_dict: dict) -> dict:
    """Infer launch time per token as the earliest observed transaction."""
    launch = {}

    for etype, ts in tx_timestamps.items():
        src_type, _, dst_type = etype
        ei = edge_index_dict[etype]

        if dst_type == "token":
            tok_idx = ei[1]
        elif src_type == "token":
            tok_idx = ei[0]
        else:
            continue

        for i in range(len(tok_idx)):
            t = int(tok_idx[i])
            s = float(ts[i])
            if t not in launch or s < launch[t]:
                launch[t] = s

    return launch


@torch.no_grad()
def early_detection_curve(model, full_data, tx_timestamps,
                          device="cpu"):
    """Run model at multiple time horizons.

    Returns dict: {horizon_name: {auc, f1, precision, recall, threshold, n_edges}}
    """
    # NOTE: this works with FeaturelessHeteroGAT (classification).
    # For TemporalHeteroGAT, use link_eval.py instead.
    model.to(device)
    model.eval()

    y_true = full_data["token"].y.numpy()

    launch_times = get_token_launch_times(
        tx_timestamps,
        {et: full_data[et].edge_index for et in full_data.edge_types}
    )

    results = {}

    for name, sec in HORIZONS.items():
        partial = filter_edges_by_horizon(
            full_data, tx_timestamps, launch_times, sec
        )
        partial = partial.to(device)

        n_edges = sum(
            partial[et].edge_index.shape[1] for et in partial.edge_types
        )

        logits, _ = model(
            {k: partial[k].x for k in ["token", "wallet"]},
            {et: partial[et].edge_index for et in partial.edge_types},
        )
        proba = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()

        thr = find_optimal_threshold(y_true, proba)
        metrics = compute_metrics(y_true, proba, threshold=thr)

        results[name] = {
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "threshold": thr,
            "n_edges": n_edges,
        }
        log.info("  %s: AUC=%.4f  F1=%.4f  edges=%s",
                 name, metrics["auc"], metrics["f1"], f"{n_edges:,}")

    return results


def plot_early_detection(model_results: dict, save_dir: str):
    """Plot early detection curves for multiple models.

    model_results: {model_name: {horizon: {auc: float}}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "coral", "forestgreen", "goldenrod"]
    horizons = list(HORIZONS.keys())
    x = range(len(horizons))

    for i, (name, hrs) in enumerate(model_results.items()):
        aucs = [hrs[h]["auc"] for h in horizons]
        ax.plot(x, aucs, "o-", color=colors[i % len(colors)],
                label=name, linewidth=2, markersize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(horizons)
    ax.set_xlabel("Time since token launch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Early Detection: How Fast Can We Spot Risk?")
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "early_detection.pdf"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "early_detection.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved early detection plot to %s", save_dir)

"""
Post-training analysis: wallet embeddings, clustering, and t-SNE.

Extracts learned wallet representations from the final GNN layer,
clusters them with HDBSCAN to discover behavioural archetypes, and
projects to 2D via t-SNE for the report figures.
"""

import logging

import numpy as np
import pandas as pd
import torch
import hdbscan
from sklearn.manifold import TSNE

log = logging.getLogger(__name__)


def extract_embeddings(model, data, device: str = "cpu") -> np.ndarray:
    """Get wallet node embeddings from a trained FeaturelessHeteroGAT.

    Runs a forward pass and grabs the wallet representations after the
    second (attention) layer, before the prediction head.

    Returns
    -------
    np.ndarray, shape [num_wallets, hidden_dim]
    """
    model = model.to(device)
    data = data.to(device)
    model.eval()

    with torch.no_grad():
        _, x_dict = model(
            {k: data[k].x for k in ["token", "wallet"]},
            {et: data[et].edge_index for et in data.edge_types},
        )
    return x_dict["wallet"].cpu().numpy()


def cluster_wallets(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> np.ndarray:
    """Cluster wallet embeddings with HDBSCAN.

    Returns cluster labels; -1 denotes noise.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    log.info("HDBSCAN: %d clusters, %d noise (%.1f%%)",
             n_clusters, n_noise, 100 * n_noise / len(labels))
    return labels


def profile_clusters(
    labels: np.ndarray,
    wallet_features: pd.DataFrame,
    wallet_addresses: list[str],
    data,
) -> pd.DataFrame:
    """Aggregate wallet-level features per cluster.

    Computes the mean of each wallet feature column within each cluster,
    plus the fraction of high-risk tokens each cluster's wallets trade.

    Returns a DataFrame with one row per non-noise cluster.
    """
    token_y = data["token"].y.cpu().numpy()
    wt_edge = data["wallet", "trades", "token"].edge_index.cpu().numpy()

    # Per-wallet high-risk token fraction
    high_frac = np.zeros(len(wallet_addresses))
    for w_idx in range(len(wallet_addresses)):
        tok_idx = wt_edge[1, wt_edge[0] == w_idx]
        if len(tok_idx) > 0:
            high_frac[w_idx] = token_y[tok_idx].mean()

    wdf = wallet_features.copy()
    wdf["cluster"] = labels
    wdf["high_risk_frac"] = high_frac[: len(wdf)]

    rows = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        sub = wdf[wdf["cluster"] == cid]
        row = {"cluster": cid, "n_wallets": len(sub)}
        # Average every numeric column
        for col in sub.select_dtypes(include=[np.number]).columns:
            if col != "cluster":
                row[f"mean_{col}"] = sub[col].mean()
        rows.append(row)

    result = pd.DataFrame(rows)
    log.info("Cluster profiles:\n%s", result.to_string(index=False))
    return result


def compute_tsne(
    embeddings: np.ndarray,
    perplexity: float = 30.0,
    seed: int = 42,
) -> np.ndarray:
    """Project embeddings to 2D with t-SNE.

    Returns array of shape [N, 2].
    """
    log.info("t-SNE: perplexity=%.1f, n=%d", perplexity, len(embeddings))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=seed,
        max_iter=1000,
    )
    return tsne.fit_transform(embeddings)

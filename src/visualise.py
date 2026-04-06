"""Plotting utilities — ROC curves, t-SNE, cluster profiles."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, average_precision_score

from src.evaluate import get_roc_data, get_pr_data

# matplotlib defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid")


def _save(fig, save_dir, name):
    """Save as both PDF (vector) and PNG (raster)."""
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.pdf"), dpi=300)
    fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=300)
    plt.close(fig)


def plot_roc_curves(y_true, predictions, save_dir="figures"):
    """ROC curves for multiple models on one plot.

    predictions: dict of {model_name: y_proba}
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    colours = sns.color_palette("colorblind", n_colors=len(predictions))
    for (name, proba), colour in zip(predictions.items(), colours):
        fpr, tpr, _ = get_roc_data(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=colour, linewidth=1.8)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    _save(fig, save_dir, "roc_curves")
    return fig


def plot_tsne(embeddings, labels, save_dir="figures", perplexity=30,
              seed=42):
    """t-SNE visualisation of wallet embeddings coloured by cluster.

    embeddings: np.array (n_wallets, dim)
    labels: np.array of cluster IDs
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, init="pca", learning_rate="auto")
    coords = tsne.fit_transform(embeddings)

    n_clusters = len(np.unique(labels))
    palette = sns.color_palette("husl", n_colors=n_clusters)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(n_clusters):
        mask = labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                   color=palette[i], label=f"Cluster {i}")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Wallet Embeddings (t-SNE)")
    ax.legend(markerscale=3, fontsize=8, loc="best")
    # empirically looks cleaner without grid on scatter
    ax.grid(False)

    _save(fig, save_dir, "tsne_embeddings")
    return fig


def plot_cluster_profiles(profiles, save_dir="figures"):
    """Bar chart comparing cluster characteristics.

    profiles: dict of {cluster_id: {feature_name: value}}
    """
    cluster_ids = sorted(profiles.keys())
    features = list(profiles[cluster_ids[0]].keys())
    n_clusters = len(cluster_ids)
    n_features = len(features)

    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4),
                             sharey=False)
    if n_features == 1:
        axes = [axes]

    palette = sns.color_palette("husl", n_colors=n_clusters)

    for ax, feat in zip(axes, features):
        vals = [profiles[c][feat] for c in cluster_ids]
        bars = ax.bar(range(n_clusters), vals, color=palette)
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels([f"C{c}" for c in cluster_ids])
        ax.set_title(feat.replace("_", " ").title(), fontsize=10)

    fig.suptitle("Cluster Profiles", fontsize=13, y=1.02)
    fig.tight_layout()

    _save(fig, save_dir, "cluster_profiles")
    return fig

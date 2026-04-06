"""Evaluation metrics for temporal link prediction.

Standard metrics following TGB (Huang et al. 2024) and DyGLib conventions:
    - MRR  (Mean Reciprocal Rank)
    - Hits@K  (K = 1, 3, 10)
    - AP   (Average Precision on pos/neg pairs)

All ranking metrics treat each positive edge as a query: we rank it among
its n_neg corrupted negatives and report where the true destination lands.
"""

import logging

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from src.data.temporal_edges import build_temporal_batches, sample_negatives_temporal

log = logging.getLogger(__name__)


def compute_mrr(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Mean Reciprocal Rank.

    For each positive, rank it among its negatives. A positive that beats
    all 49 negatives gets rank 1 (RR = 1.0); one that loses to all of them
    gets rank 50 (RR = 0.02).

    Args:
        pos_scores: [N] score for the true destination
        neg_scores: [N, n_neg] scores for corrupted destinations
    Returns:
        MRR as a float in (0, 1].
    """
    # how many negatives score >= the positive?
    ranks = (neg_scores >= pos_scores[:, None]).sum(axis=1) + 1
    return float(np.mean(1.0 / ranks))


def compute_hits_at_k(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    k: int = 10,
) -> float:
    """Fraction of positives ranked in top-K among negatives.

    Args:
        pos_scores: [N]
        neg_scores: [N, n_neg]
        k: cutoff (1, 3, or 10 typically)
    Returns:
        Hits@K as a float in [0, 1].
    """
    ranks = (neg_scores >= pos_scores[:, None]).sum(axis=1) + 1
    return float(np.mean(ranks <= k))


def compute_link_ap(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Average Precision treating each (pos, neg_set) as a retrieval task.

    For each query we build a binary label vector (1 positive, n_neg negatives)
    and compute AP, then average across all queries.

    This is the "filtered" AP used in TGB -- it measures how well the model
    separates the true destination from random corruptions.
    """
    n = len(pos_scores)
    n_neg = neg_scores.shape[1]
    aps = np.empty(n)

    for i in range(n):
        scores = np.concatenate([[pos_scores[i]], neg_scores[i]])
        labels = np.concatenate([[1], np.zeros(n_neg)])
        aps[i] = average_precision_score(labels, scores)

    return float(np.mean(aps))


@torch.no_grad()
def run_link_prediction_eval(
    model,
    decoder,
    bochner,
    events,
    graph,
    n_neg: int = 49,
    batch_size: int = 1024,
    device: str = "cpu",
) -> dict:
    """Full link prediction assessment on a set of temporal events.

    Runs the GNN encoder once per batch to get node embeddings at the
    batch's query time, then scores positive and negative edges through
    the decoder. After each batch the observed edges could be added to
    the graph for live-update (transductive setting), but we keep it
    simple here and use a static snapshot.

    Args:
        model: TemporalHeteroGAT encoder
        decoder: LinkDecoder
        bochner: BochnerTimeEncoding (shared instance)
        events: DataFrame with columns [src, dst, ts]
        graph: HeteroData snapshot used for message-passing
        n_neg: negatives per positive
        batch_size: temporal batch size
        device: torch device string

    Returns:
        dict with keys: mrr, hits@1, hits@3, hits@10, ap
    """
    model.eval()
    decoder.eval()

    n_tokens = graph["token"].x.size(0)
    all_pos_scores = []
    all_neg_scores = []

    for batch in build_temporal_batches(events, batch_size):
        aug = sample_negatives_temporal(batch, n_tokens, n_neg=n_neg, seed=None)
        t_query = float(batch["ts"].max())

        # encode graph at query time
        x_dict = {k: graph[k].x.to(device) for k in graph.node_types}
        ei_dict = {et: graph[et].edge_index.to(device) for et in graph.edge_types}
        et_dict = {et: graph[et].edge_time.to(device)
                   for et in graph.edge_types if hasattr(graph[et], "edge_time")}

        node_embs = model(x_dict, ei_dict, et_dict, t_query=t_query)
        z_w = node_embs["wallet"]
        z_t = node_embs["token"]

        # score positives
        src_pos = torch.tensor(batch["src"], device=device, dtype=torch.long)
        dst_pos = torch.tensor(batch["dst"], device=device, dtype=torch.long)
        dt_w = torch.full((len(src_pos),), 0.0, device=device)
        dt_t = torch.full((len(dst_pos),), 0.0, device=device)

        pos_logits = decoder(
            z_w[src_pos], z_t[dst_pos],
            bochner(dt_w), bochner(dt_t),
        ).cpu().numpy()

        # score negatives -- reshape to [n_pos, n_neg]
        neg_dst_mat = aug["neg_dst"]  # [n_pos, n_neg]
        n_pos = len(batch["src"])
        neg_logits = np.empty((n_pos, n_neg), dtype=np.float32)

        for j in range(n_neg):
            dst_neg_j = torch.tensor(neg_dst_mat[:, j], device=device, dtype=torch.long)
            src_rep = src_pos  # same wallets
            scores_j = decoder(
                z_w[src_rep], z_t[dst_neg_j],
                bochner(dt_w), bochner(dt_t),
            ).cpu().numpy()
            neg_logits[:, j] = scores_j

        all_pos_scores.append(pos_logits)
        all_neg_scores.append(neg_logits)

    pos_all = np.concatenate(all_pos_scores)
    neg_all = np.concatenate(all_neg_scores, axis=0)

    metrics = {
        "mrr": compute_mrr(pos_all, neg_all),
        "hits@1": compute_hits_at_k(pos_all, neg_all, k=1),
        "hits@3": compute_hits_at_k(pos_all, neg_all, k=3),
        "hits@10": compute_hits_at_k(pos_all, neg_all, k=10),
        "ap": compute_link_ap(pos_all, neg_all),
    }

    log.info("Link prediction:  MRR=%.4f  H@1=%.4f  H@3=%.4f  H@10=%.4f  AP=%.4f",
             metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
             metrics["hits@10"], metrics["ap"])
    return metrics


# ---------------------------------------------------------------------------
# Heuristic baselines for link prediction
# ---------------------------------------------------------------------------

def run_edgebank(
    train_events,
    test_events,
    n_tokens: int,
    n_neg: int = 19,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict:
    """EdgeBank baseline (Poursafaei et al. 2022).

    Memorises all (wallet, token) pairs from training. At test time,
    scores 1.0 for previously-seen pairs and 0.0 for unseen ones.
    """
    # Build memory: set of (wallet, token) pairs
    memory = set(zip(train_events["src"].values, train_events["dst"].values))
    log.info("EdgeBank memory: %d unique (wallet, token) pairs", len(memory))

    rng = np.random.RandomState(seed)
    all_pos_scores = []
    all_neg_scores = []

    for batch in build_temporal_batches(test_events, batch_size):
        src = batch["src"]
        dst_pos = batch["dst"]
        n_pos = len(src)

        neg_dst = rng.randint(0, n_tokens, size=(n_pos, n_neg))

        # Score positives
        pos_scores = np.array([1.0 if (s, d) in memory else 0.0
                               for s, d in zip(src, dst_pos)], dtype=np.float32)

        # Score negatives
        neg_scores = np.zeros((n_pos, n_neg), dtype=np.float32)
        for i in range(n_pos):
            for j in range(n_neg):
                if (src[i], neg_dst[i, j]) in memory:
                    neg_scores[i, j] = 1.0

        all_pos_scores.append(pos_scores)
        all_neg_scores.append(neg_scores)

    pos_all = np.concatenate(all_pos_scores)
    neg_all = np.concatenate(all_neg_scores, axis=0)

    metrics = {
        "mrr": compute_mrr(pos_all, neg_all),
        "hits@1": compute_hits_at_k(pos_all, neg_all, k=1),
        "hits@3": compute_hits_at_k(pos_all, neg_all, k=3),
        "hits@10": compute_hits_at_k(pos_all, neg_all, k=10),
        "ap": compute_link_ap(pos_all, neg_all),
    }
    log.info("EdgeBank:  MRR=%.4f  H@1=%.4f  H@3=%.4f  H@10=%.4f  AP=%.4f",
             metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
             metrics["hits@10"], metrics["ap"])
    return metrics


def run_popularity(
    train_events,
    test_events,
    n_tokens: int,
    n_neg: int = 19,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict:
    """Most-popular-token baseline.

    Scores each candidate token by its normalised frequency in
    training data. Ignores the query wallet entirely.
    """
    from collections import Counter

    token_freq = Counter(train_events["dst"].values)
    max_freq = max(token_freq.values()) if token_freq else 1

    # Pre-compute normalised scores for all tokens
    scores_lookup = np.zeros(n_tokens, dtype=np.float32)
    for tok, cnt in token_freq.items():
        if tok < n_tokens:
            scores_lookup[tok] = cnt / max_freq

    rng = np.random.RandomState(seed)
    all_pos_scores = []
    all_neg_scores = []

    for batch in build_temporal_batches(test_events, batch_size):
        dst_pos = batch["dst"]
        n_pos = len(dst_pos)
        neg_dst = rng.randint(0, n_tokens, size=(n_pos, n_neg))

        pos_scores = scores_lookup[dst_pos]
        neg_scores = scores_lookup[neg_dst]

        all_pos_scores.append(pos_scores)
        all_neg_scores.append(neg_scores)

    pos_all = np.concatenate(all_pos_scores)
    neg_all = np.concatenate(all_neg_scores, axis=0)

    metrics = {
        "mrr": compute_mrr(pos_all, neg_all),
        "hits@1": compute_hits_at_k(pos_all, neg_all, k=1),
        "hits@3": compute_hits_at_k(pos_all, neg_all, k=3),
        "hits@10": compute_hits_at_k(pos_all, neg_all, k=10),
        "ap": compute_link_ap(pos_all, neg_all),
    }
    log.info("Popularity:  MRR=%.4f  H@1=%.4f  H@3=%.4f  H@10=%.4f  AP=%.4f",
             metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
             metrics["hits@10"], metrics["ap"])
    return metrics


def run_cosine_similarity(
    wallet_embeddings: np.ndarray,
    token_embeddings: np.ndarray,
    test_events,
    n_neg: int = 19,
    batch_size: int = 1024,
    seed: int = 42,
) -> dict:
    """Cosine similarity baseline for link prediction.

    Uses pre-computed GNN wallet/token embeddings (frozen). For each
    test event (wallet w, token t), the score is the cosine similarity
    between z_w and z_t. Negatives are sampled uniformly at random
    following the same 1 + n_neg protocol as the other baselines.

    Args:
        wallet_embeddings: [n_wallets, d] pre-computed wallet vectors
        token_embeddings:  [n_tokens, d] pre-computed token vectors
        test_events:       DataFrame with src, dst, ts columns
        n_neg:             negatives per positive (default 19)
        batch_size:        temporal batch size
        seed:              random seed for negative sampling

    Returns:
        dict with keys: mrr, hits@1, hits@3, hits@10, ap
    """
    n_tokens = token_embeddings.shape[0]

    # L2-normalise embeddings for cosine similarity via dot product
    w_norms = np.linalg.norm(wallet_embeddings, axis=1, keepdims=True)
    w_norms = np.maximum(w_norms, 1e-8)
    w_normed = wallet_embeddings / w_norms

    t_norms = np.linalg.norm(token_embeddings, axis=1, keepdims=True)
    t_norms = np.maximum(t_norms, 1e-8)
    t_normed = token_embeddings / t_norms

    rng = np.random.RandomState(seed)
    all_pos_scores = []
    all_neg_scores = []

    for batch in build_temporal_batches(test_events, batch_size):
        src = batch["src"]
        dst_pos = batch["dst"]
        n_pos = len(src)

        neg_dst = rng.randint(0, n_tokens, size=(n_pos, n_neg))

        # Cosine similarity = dot product of L2-normalised vectors
        # Positive scores: [n_pos]
        w_vecs = w_normed[src]           # [n_pos, d]
        t_pos_vecs = t_normed[dst_pos]   # [n_pos, d]
        pos_scores = np.sum(w_vecs * t_pos_vecs, axis=1).astype(np.float32)

        # Negative scores: [n_pos, n_neg]
        t_neg_vecs = t_normed[neg_dst]   # [n_pos, n_neg, d]
        neg_scores = np.einsum("pd,pnd->pn", w_vecs, t_neg_vecs).astype(np.float32)

        all_pos_scores.append(pos_scores)
        all_neg_scores.append(neg_scores)

    pos_all = np.concatenate(all_pos_scores)
    neg_all = np.concatenate(all_neg_scores, axis=0)

    metrics = {
        "mrr": compute_mrr(pos_all, neg_all),
        "hits@1": compute_hits_at_k(pos_all, neg_all, k=1),
        "hits@3": compute_hits_at_k(pos_all, neg_all, k=3),
        "hits@10": compute_hits_at_k(pos_all, neg_all, k=10),
        "ap": compute_link_ap(pos_all, neg_all),
    }
    log.info("CosineSim:  MRR=%.4f  H@1=%.4f  H@3=%.4f  H@10=%.4f  AP=%.4f",
             metrics["mrr"], metrics["hits@1"], metrics["hits@3"],
             metrics["hits@10"], metrics["ap"])
    return metrics

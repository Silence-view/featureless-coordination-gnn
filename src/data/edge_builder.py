"""Edge construction for the wallet-token heterogeneous graph.

Three edge types computed here:
  - wallet -> token  (direct trading relationship)
  - wallet <-> wallet  (same on-chain transaction)
  - wallet <-> wallet  (temporal co-trading, Jaccard-like with decay)

The temporal co-trading formula:
    e_ab = sum_{i in T_a & T_b} exp(-lambda * |t_a(i) - t_b(i)|)

Uses scipy.sparse throughout to keep memory manageable.
"""

import logging

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

log = logging.getLogger(__name__)


def compute_wallet_token_edges(
    tx: pd.DataFrame,
    active_wallets: set,
    w2i: dict[str, int],
    t2i: dict[str, int],
) -> np.ndarray:
    """Build wallet->token edge index from trading pairs.

    Returns ndarray of shape [2, E].
    """
    mask = tx["wallet"].isin(active_wallets) & tx["mint"].isin(t2i)
    pairs = tx[mask].groupby(["wallet", "mint"]).size().reset_index()[["wallet", "mint"]]

    wi = pairs["wallet"].map(w2i).values.astype(np.int64)
    ti = pairs["mint"].map(t2i).values.astype(np.int64)

    edge_index = np.stack([wi, ti])
    log.info("wallet->token edges: %d", edge_index.shape[1])
    return edge_index


def compute_same_tx_edges(
    tx: pd.DataFrame,
    active_wallets: set,
    w2i: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Wallet pairs that shared an on-chain transaction.

    Returns (edge_index [2, E], weights [E]) -- weights are all 1.0.
    """
    txf = tx[tx["wallet"].isin(active_wallets)]

    sig_wallets = txf.groupby("signature")["wallet"].apply(lambda s: list(s.unique()))
    multi = sig_wallets[sig_wallets.apply(len) > 1]
    log.info("Multi-wallet signatures: %d", len(multi))

    rows, cols = [], []
    seen = set()
    for wlist in multi:
        ws = set(wlist)
        for a in ws:
            for b in ws:
                if a != b:
                    pair = (w2i[a], w2i[b])
                    if pair not in seen:
                        rows.append(pair[0])
                        cols.append(pair[1])
                        seen.add(pair)

    ei = np.array([rows, cols], dtype=np.int64)
    ew = np.ones(len(rows), dtype=np.float32)
    log.info("same_tx edges: %d directed pairs", len(rows))
    return ei, ew


def compute_temporal_cotrade_edges(
    tx: pd.DataFrame,
    active_wallets: set,
    w2i: dict[str, int],
    lambda_decay: float = 1.0 / 3600,
    target_mean_degree: float = 15.0,
    max_wallets_per_token: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Temporal co-trading edges with exponential decay.

    For each token traded by both wallets a and b, contributes
    exp(-lambda * |t_a - t_b|) to the edge weight. Thresholded
    via binary search to hit target_mean_degree.

    Returns (edge_index [2, E], weights [E]).
    """
    n = len(w2i)

    tmp = tx.copy()
    tmp["ts_sec"] = tmp["block_timestamp"].astype(np.int64) // 10**6  # datetime64[us] -> seconds

    # first trade per (wallet, token) then keep only active
    ft = (
        tmp.groupby(["wallet", "mint"])["ts_sec"].min()
        .reset_index().rename(columns={"ts_sec": "first_ts"})
    )
    ft = ft[ft["wallet"].isin(active_wallets)]
    ft["widx"] = ft["wallet"].map(w2i)

    log.info("Computing temporal co-trade edges (lambda=%.6f) ...", lambda_decay)

    all_r, all_c, all_v = [], [], []
    done = 0

    for mint, grp in ft.groupby("mint"):
        k = len(grp)
        if k < 2:
            continue
        # cap to avoid quadratic blowup on popular tokens
        if k > max_wallets_per_token:
            grp = grp.sample(max_wallets_per_token, random_state=42)
            k = max_wallets_per_token

        widx = grp["widx"].values.astype(np.int64)
        times = grp["first_ts"].values.astype(np.float64)

        # pairwise time diffs - upper triangle only
        dt = np.abs(times[:, None] - times[None, :])
        wts = np.exp(-lambda_decay * dt)

        ri, ci = np.triu_indices(k, k=1)
        vals = wts[ri, ci]

        # drop negligible weights
        keep = vals > 0.01
        if keep.any():
            all_r.append(widx[ri[keep]])
            all_c.append(widx[ci[keep]])
            all_v.append(vals[keep].astype(np.float32))

        done += 1
        if done % 500 == 0:
            log.info("  %d / %d tokens", done, ft["mint"].nunique())

    if not all_r:
        log.warning("No co-trade edges found")
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    all_r = np.concatenate(all_r)
    all_c = np.concatenate(all_c)
    all_v = np.concatenate(all_v)

    log.info("Raw upper-triangle entries: %s", f"{len(all_v):,}")

    # aggregate duplicate pairs across tokens
    sp = coo_matrix((all_v, (all_r, all_c)), shape=(n, n)).tocsr()
    sp = sp + sp.T  # symmetrise

    if sp.nnz == 0:
        log.warning("Empty co-trade matrix")
        return np.zeros((2, 0), dtype=np.int64), np.zeros(0, dtype=np.float32)

    # threshold to hit target degree
    tau = _find_threshold(sp, n, target_mean_degree)
    log.info("Threshold tau=%.4f for target degree %.1f", tau, target_mean_degree)

    sp.data[sp.data < tau] = 0
    sp.eliminate_zeros()

    coo = sp.tocoo()
    edge_index = np.stack([coo.row, coo.col]).astype(np.int64)
    edge_weight = coo.data.astype(np.float32)

    deg = len(coo.data) / max(n, 1)
    log.info("Co-trade edges after threshold: %s (mean deg %.1f)", f"{len(coo.data):,}", deg)

    return edge_index, edge_weight


def _find_threshold(sparse_mat: csr_matrix, n_nodes: int, target_deg: float) -> float:
    """Binary search for weight threshold that gives target mean degree."""
    data = sparse_mat.data.copy()
    if len(data) == 0:
        return 0.0

    lo, hi = float(data.min()), float(data.max())
    target_edges = target_deg * n_nodes

    # TODO: could do this analytically with sorted cumsum but binary search is fine
    for _ in range(50):
        mid = (lo + hi) / 2
        if np.sum(data >= mid) > target_edges:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-6:
            break

    return (lo + hi) / 2

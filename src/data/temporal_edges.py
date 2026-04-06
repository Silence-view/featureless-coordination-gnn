"""Temporal edge extraction for link prediction.

Each interaction event is (wallet_idx, token_idx, timestamp_sec).
Events are sorted chronologically so we can do proper temporal batching
without any future-leakage — the model only ever sees edges from the past.
"""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def extract_temporal_events(
    tx: pd.DataFrame,
    active_wallets: set,
    w2i: dict[str, int],
    t2i: dict[str, int],
) -> pd.DataFrame:
    """Extract chronologically-sorted interaction events.

    Each row is a unique (wallet, token, timestamp) trading event.
    Duplicate (wallet, token) pairs at different times are kept — they
    represent repeated interactions, which is exactly what we want for
    temporal link prediction.

    Returns DataFrame with columns [src, dst, ts] where:
        src = wallet index (into w2i)
        dst = token index  (into t2i)
        ts  = unix timestamp in seconds
    Sorted by ts ascending.
    """
    mask = tx["wallet"].isin(active_wallets) & tx["mint"].isin(t2i)
    sub = tx.loc[mask, ["wallet", "mint", "block_timestamp"]].copy()

    sub["src"] = sub["wallet"].map(w2i)
    sub["dst"] = sub["mint"].map(t2i)
    sub["ts"] = sub["block_timestamp"].astype(np.int64) // 10**6  # datetime64[us] -> seconds

    events = (
        sub[["src", "dst", "ts"]]
        .dropna()
        .astype({"src": np.int64, "dst": np.int64, "ts": np.int64})
        .sort_values("ts")
        .reset_index(drop=True)
    )

    log.info("Extracted %s temporal events (%d wallets, %d tokens)",
             f"{len(events):,}", events["src"].nunique(), events["dst"].nunique())
    return events


def temporal_train_val_test_split(
    events: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split events chronologically (NOT randomly).

    This is critical for temporal link prediction: the model must never
    see future edges during training. We simply cut by sorted position.

    Returns (train_events, val_events, test_events).
    """
    n = len(events)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = events.iloc[:train_end].reset_index(drop=True)
    val_df = events.iloc[train_end:val_end].reset_index(drop=True)
    test_df = events.iloc[val_end:].reset_index(drop=True)

    log.info("Temporal split: train=%d  val=%d  test=%d",
             len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def build_temporal_batches(events: pd.DataFrame, batch_size: int = 1024):
    """Yield chronological batches of edges for training/evaluation.

    Each batch is a dict with keys:
        src:   np.ndarray [B]  wallet indices
        dst:   np.ndarray [B]  token indices
        ts:    np.ndarray [B]  timestamps (seconds)
        label: np.ndarray [B]  all ones (positive edges)
    """
    n = len(events)
    for start in range(0, n, batch_size):
        chunk = events.iloc[start : start + batch_size]
        yield {
            "src": chunk["src"].values,
            "dst": chunk["dst"].values,
            "ts": chunk["ts"].values,
            "label": np.ones(len(chunk), dtype=np.float32),
        }


def sample_negatives_temporal(
    batch: dict,
    n_tokens: int,
    n_neg: int = 49,
    seed: int | None = None,
) -> dict:
    """For each positive edge, sample n_neg random token destinations.

    We sample uniformly from all tokens rather than restricting to the
    same time window — at our graph density the collision rate with true
    positives is negligible, and uniform negatives are standard practice
    in TGB / DyGLib benchmarks.

    Returns augmented batch where each array is expanded by n_neg per
    positive, with label=0 for negatives.
    """
    rng = np.random.RandomState(seed)
    n_pos = len(batch["src"])

    neg_dst = rng.randint(0, n_tokens, size=(n_pos, n_neg))

    # tile source and timestamp to match negatives
    src_rep = np.repeat(batch["src"], n_neg)
    ts_rep = np.repeat(batch["ts"], n_neg)
    neg_dst_flat = neg_dst.reshape(-1)

    # concat positives + negatives
    out = {
        "src": np.concatenate([batch["src"], src_rep]),
        "dst": np.concatenate([batch["dst"], neg_dst_flat]),
        "ts": np.concatenate([batch["ts"], ts_rep]),
        "label": np.concatenate([
            batch["label"],
            np.zeros(n_pos * n_neg, dtype=np.float32),
        ]),
        # keep the per-positive negative matrix for ranking metrics
        "neg_dst": neg_dst,
    }
    return out

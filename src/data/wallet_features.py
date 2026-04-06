"""Wallet behavioural feature engineering.

Builds 15 features per wallet from raw Solana tx data. Keeps only
wallets that traded >= min_tokens distinct tokens (default 20).
"""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def filter_active_wallets(tx: pd.DataFrame, min_tokens: int = 20) -> set:
    """Wallets that traded at least min_tokens distinct tokens."""
    counts = tx.groupby("wallet")["mint"].nunique()
    active = set(counts[counts >= min_tokens].index)
    log.info("Active wallets (>=%d tokens): %d / %d", min_tokens, len(active), len(counts))
    return active


def compute_wallet_features(tx: pd.DataFrame, min_tokens: int = 20) -> pd.DataFrame:
    """Compute 15 behavioural features for each active wallet.

    Returns DataFrame indexed by wallet address with 15 columns.
    """
    active = filter_active_wallets(tx, min_tokens)

    tx = tx.copy()
    tx["ts_sec"] = tx["block_timestamp"].astype(np.int64) // 10**6  # datetime64[us] -> seconds

    # entry rank computed over ALL wallets, not just active ones
    first_trade = (
        tx.groupby(["wallet", "mint"])["ts_sec"].min()
        .reset_index().rename(columns={"ts_sec": "first_ts"})
    )
    first_trade["entry_rank"] = first_trade.groupby("mint")["first_ts"].rank(method="min")
    first_trade["is_sniper"] = first_trade["entry_rank"] <= 5

    ft = first_trade[first_trade["wallet"].isin(active)]

    log.info("Computing features for %d wallets ...", len(active))

    # filter tx to active wallets for the rest
    txa = tx[tx["wallet"].isin(active)].copy()
    txa["abs_change"] = txa["token_change"].abs()
    txa["date"] = txa["block_timestamp"].dt.date

    feats = {}

    # 1-2: basic counts
    feats["n_unique_tokens"] = txa.groupby("wallet")["mint"].nunique()
    feats["n_transactions"] = txa.groupby("wallet").size()

    # 3-4: volume
    buys = txa.loc[txa["token_change"] > 0]
    feats["total_buy_volume"] = buys.groupby("wallet")["token_change"].sum()

    sells = txa.loc[txa["token_change"] < 0].copy()
    sells["abs_sell"] = sells["token_change"].abs()
    feats["total_sell_volume"] = sells.groupby("wallet")["abs_sell"].sum()

    # 6: entry rank (averaged across tokens)
    feats["avg_entry_rank"] = ft.groupby("wallet")["entry_rank"].mean()

    # 7: sniper score = fraction of tokens where wallet was top-5
    sniper_n = ft.groupby("wallet")["is_sniper"].sum()
    sniper_d = ft.groupby("wallet").size()
    feats["sniper_score"] = sniper_n / sniper_d

    # 8: average hold duration per token
    spans = txa.groupby(["wallet", "mint"])["ts_sec"].agg(["min", "max"])
    spans["dur"] = spans["max"] - spans["min"]
    feats["avg_time_in_token"] = spans.reset_index().groupby("wallet")["dur"].mean()

    # 9: co-transaction partners
    feats["n_same_tx_partners"] = _same_tx_partners(txa)

    # 10: fees
    feats["avg_fee"] = txa.groupby("wallet")["fee_sol"].mean()

    # 11: temporal spread (std of trade timestamps)
    feats["temporal_spread"] = txa.groupby("wallet")["ts_sec"].std().fillna(0)

    # 12: busiest day
    tpd = txa.groupby(["wallet", "date"])["mint"].nunique().reset_index()
    feats["max_tokens_per_day"] = tpd.groupby("wallet")["mint"].max()

    # 13: avg trade size
    feats["avg_token_change"] = txa.groupby("wallet")["abs_change"].mean()

    # 14: quick flippers
    feats["sell_within_1h_ratio"] = _sell_within_1h(txa)

    # 15: how many days they showed up
    feats["unique_days_active"] = txa.groupby("wallet")["date"].nunique()

    # assemble
    df = pd.DataFrame(feats)
    df.index.name = "wallet"

    # 5: buy/sell ratio (needs both volumes)
    total = df["total_buy_volume"].fillna(0) + df["total_sell_volume"].fillna(0)
    df["buy_sell_ratio"] = (
        df["total_buy_volume"].fillna(0) / total.replace(0, np.nan)
    ).fillna(0.5)

    df = df.fillna(0)

    # canonical column order
    df = df[[
        "n_unique_tokens", "n_transactions",
        "total_buy_volume", "total_sell_volume", "buy_sell_ratio",
        "avg_entry_rank", "sniper_score", "avg_time_in_token",
        "n_same_tx_partners", "avg_fee", "temporal_spread",
        "max_tokens_per_day", "avg_token_change",
        "sell_within_1h_ratio", "unique_days_active",
    ]]

    log.info("Wallet features: %s", df.shape)
    return df


# ---- helpers ----

def _same_tx_partners(txa: pd.DataFrame) -> pd.Series:
    """Count unique wallets that appeared in the same on-chain tx."""
    sig_n = txa.groupby("signature")["wallet"].nunique()
    multi_sigs = sig_n[sig_n > 1].index

    if len(multi_sigs) == 0:
        return pd.Series(dtype=np.float64)

    # this is the bottleneck for large datasets
    tx_multi = txa[txa["signature"].isin(multi_sigs)][["signature", "wallet"]].drop_duplicates()

    partners: dict[str, set] = defaultdict(set)
    for _, grp in tx_multi.groupby("signature"):
        ws = set(grp["wallet"].tolist())
        if len(ws) > 1:
            for w in ws:
                partners[w].update(ws - {w})

    return pd.Series({w: len(p) for w, p in partners.items()}, dtype=np.float64)


def _sell_within_1h(txa: pd.DataFrame) -> pd.Series:
    """Fraction of tokens where first sell was within 1h of first buy."""
    buys = txa.loc[txa["token_change"] > 0]
    sells = txa.loc[txa["token_change"] < 0]

    first_buy = (
        buys.groupby(["wallet", "mint"])["ts_sec"].min()
        .reset_index().rename(columns={"ts_sec": "buy_ts"})
    )
    first_sell = (
        sells.groupby(["wallet", "mint"])["ts_sec"].min()
        .reset_index().rename(columns={"ts_sec": "sell_ts"})
    )

    m = first_buy.merge(first_sell, on=["wallet", "mint"], how="left")
    m["fast_sell"] = ((m["sell_ts"] - m["buy_ts"]) <= 3600).fillna(False)

    return m.groupby("wallet")["fast_sell"].mean()


def zscore_normalise(train: pd.DataFrame, *others, clip=10.0):
    """Z-score normalisation fitted on train set."""
    mu = train.mean()
    sigma = train.std().replace(0, 1).clip(lower=1e-6)
    out = [((train - mu) / sigma).clip(-clip, clip)]
    for df in others:
        out.append(((df - mu) / sigma).clip(-clip, clip))
    return tuple(out)

"""Assemble PyG HeteroData graphs from features and edges.

Node types: token, wallet
Edge types:
  (wallet, trades, token)      -- direct trading
  (token, traded_by, wallet)   -- reverse of above
  (wallet, co_trades, wallet)  -- temporal co-trading
  (wallet, same_tx, wallet)    -- shared on-chain transaction
"""

import gc
import logging
import time

import numpy as np
import torch
from torch_geometric.data import HeteroData

from src.data.loader import load_transactions
from src.data.features import (
    load_feat_label, extract_token_features,
    zscore_normalise as zscore_tok,
)
from src.data.wallet_features import (
    compute_wallet_features,
    zscore_normalise as zscore_wal,
)
from src.data.edge_builder import (
    compute_wallet_token_edges,
    compute_same_tx_edges,
    compute_temporal_cotrade_edges,
)

log = logging.getLogger(__name__)


def build_graph(
    split: str,
    tx,
    feat_label_df,
    min_tokens: int = 20,
    lambda_decay: float = 1 / 3600,
    target_degree: float = 15.0,
    wallet_features_normed=None,
    token_features_normed=None,
    labels=None,
) -> HeteroData:
    """Build a single HeteroData graph for one split.

    If normed features are passed (fitted on train), uses those directly.
    Otherwise computes raw features -- useful for the train split itself.
    """
    t0 = time.time()
    log.info("--- Building graph: %s ---", split)

    # wallet features
    if wallet_features_normed is None:
        wf = compute_wallet_features(tx, min_tokens)
    else:
        wf = wallet_features_normed

    active = set(wf.index)
    wallets = sorted(active)
    w2i = {w: i for i, w in enumerate(wallets)}

    # token features
    tx_mints = set(tx["mint"].unique())
    fsub = feat_label_df[feat_label_df["mint_address"].isin(tx_mints)].reset_index(drop=True)

    if token_features_normed is not None and labels is not None:
        tok_X, tok_y, tok_mints = token_features_normed, labels, fsub["mint_address"]
    else:
        tok_X, tok_y, tok_mints = extract_token_features(fsub)

    tokens = tok_mints.tolist()
    t2i = {m: i for i, m in enumerate(tokens)}

    log.info("Nodes: %d tokens, %d wallets", len(tokens), len(wallets))

    # edges
    wt = compute_wallet_token_edges(tx, active, w2i, t2i)
    tw = np.stack([wt[1], wt[0]])

    ct_ei, ct_ew = compute_temporal_cotrade_edges(
        tx, active, w2i, lambda_decay=lambda_decay,
        target_mean_degree=target_degree,
    )
    st_ei, st_ew = compute_same_tx_edges(tx, active, w2i)

    # assemble HeteroData
    data = HeteroData()

    data["token"].x = torch.tensor(tok_X.values, dtype=torch.float32)
    data["wallet"].x = torch.tensor(wf.values, dtype=torch.float32)
    data["token"].y = torch.tensor(tok_y.values, dtype=torch.float32)

    data["token"].mint_address = tokens
    data["wallet"].wallet_address = wallets

    data["wallet", "trades", "token"].edge_index = torch.tensor(wt, dtype=torch.long)
    data["token", "traded_by", "wallet"].edge_index = torch.tensor(tw, dtype=torch.long)

    _set_ww_edges(data, "co_trades", ct_ei, ct_ew, weighted=True)
    _set_ww_edges(data, "same_tx", st_ei, st_ew, weighted=False)

    log.info("HeteroData built in %.1fs: %s", time.time() - t0, data)
    return data


def _set_ww_edges(data, rel, ei, ew, weighted=False):
    """Helper to set wallet-wallet edges, handling empty case."""
    if ei.shape[1] > 0:
        data["wallet", rel, "wallet"].edge_index = torch.tensor(ei, dtype=torch.long)
        if weighted:
            data["wallet", rel, "wallet"].edge_attr = (
                torch.tensor(ew, dtype=torch.float32).unsqueeze(-1)
            )
    else:
        data["wallet", rel, "wallet"].edge_index = torch.zeros((2, 0), dtype=torch.long)
        if weighted:
            data["wallet", rel, "wallet"].edge_attr = torch.zeros((0, 1), dtype=torch.float32)


def build_all_graphs(config: dict) -> dict[str, HeteroData]:
    """Build train/val/test graphs with normalisation fitted on train.

    Expects config dict with keys matching params.yaml structure:
      config["data"]["raw_dir"], config["data"]["feat_label"],
      config["data"]["splits"], config["data"]["min_tokens"],
      config["graph"]["lambda_decay"], config["graph"]["target_mean_degree"]
    """
    import os

    dc = config["data"]
    gc_cfg = config["graph"]

    raw_dir = dc["raw_dir"]
    feat_label_df = load_feat_label(dc["feat_label"])
    min_tok = dc.get("min_tokens", 20)
    lam = gc_cfg.get("lambda_decay", 1 / 3600)
    deg = gc_cfg.get("target_mean_degree", 15.0)

    # phase 1: compute raw features + edges per split
    cache = {}
    for split, subdir in dc["splits"].items():
        log.info("====== Processing %s ======", split)
        data_dir = os.path.join(raw_dir, subdir)
        tx = load_transactions(data_dir)

        wf = compute_wallet_features(tx, min_tok)
        active = set(wf.index)
        wallet_list = sorted(active)
        w2i = {w: i for i, w in enumerate(wallet_list)}

        tx_mints = set(tx["mint"].unique())
        fsub = feat_label_df[feat_label_df["mint_address"].isin(tx_mints)].reset_index(drop=True)
        tok_X, tok_y, tok_mints = extract_token_features(fsub)
        token_list = tok_mints.tolist()
        t2i = {m: i for i, m in enumerate(token_list)}

        wt = compute_wallet_token_edges(tx, active, w2i, t2i)
        tw = np.stack([wt[1], wt[0]])
        ct_ei, ct_ew = compute_temporal_cotrade_edges(tx, active, w2i, lam, deg)
        st_ei, st_ew = compute_same_tx_edges(tx, active, w2i)

        cache[split] = dict(
            wf=wf, tok_X=tok_X, tok_y=tok_y,
            token_list=token_list, wallet_list=wallet_list,
            wt=wt, tw=tw, ct_ei=ct_ei, ct_ew=ct_ew, st_ei=st_ei, st_ew=st_ew,
        )

        del tx
        gc.collect()

    # phase 2: normalise (fit on train)
    log.info("====== Normalisation ======")
    splits = list(dc["splits"].keys())  # [train, val, test]

    wf_normed = dict(zip(splits, zscore_wal(
        cache[splits[0]]["wf"],
        *[cache[s]["wf"] for s in splits[1:]]
    )))
    tok_normed = dict(zip(splits, zscore_tok(
        cache[splits[0]]["tok_X"],
        *[cache[s]["tok_X"] for s in splits[1:]]
    )))

    # phase 3: assemble
    graphs = {}
    for split in splits:
        log.info("====== Assembling: %s ======", split)
        c = cache[split]
        data = HeteroData()

        data["token"].x = torch.tensor(tok_normed[split].values, dtype=torch.float32)
        data["wallet"].x = torch.tensor(wf_normed[split].values, dtype=torch.float32)
        data["token"].y = torch.tensor(c["tok_y"].values, dtype=torch.float32)
        data["token"].mint_address = c["token_list"]
        data["wallet"].wallet_address = c["wallet_list"]

        data["wallet", "trades", "token"].edge_index = torch.tensor(c["wt"], dtype=torch.long)
        data["token", "traded_by", "wallet"].edge_index = torch.tensor(c["tw"], dtype=torch.long)

        _set_ww_edges(data, "co_trades", c["ct_ei"], c["ct_ew"], weighted=True)
        _set_ww_edges(data, "same_tx", c["st_ei"], c["st_ew"], weighted=False)

        graphs[split] = data
        log.info("  %s: %s", split, data)

    # also extract temporal events per split for link prediction
    from src.data.temporal_edges import extract_temporal_events

    temporal_events = {}
    for split in splits:
        c = cache[split]
        w2i = {w: i for i, w in enumerate(c["wallet_list"])}
        t2i = {m: i for i, m in enumerate(c["token_list"])}

        # reload tx for this split (we deleted it earlier to save memory)
        data_dir = os.path.join(raw_dir, dc["splits"][split])
        tx = load_transactions(data_dir)
        events = extract_temporal_events(tx, set(c["wallet_list"]), w2i, t2i)
        temporal_events[split] = events
        del tx
        gc.collect()

    return graphs, temporal_events

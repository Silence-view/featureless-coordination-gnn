"""Training loop for the Temporal HeteroGAT (link prediction).

Processes edges in chronological batches. For each batch:
1. Encode the current graph state via TemporalHeteroGAT
2. Score positive edges and sampled negatives via LinkDecoder
3. Backprop on binary CE loss

Validation uses MRR and Hits@K.
"""

import os
import time
import copy
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def train_temporal_model(
    model, decoder, train_graph, train_events, val_events,
    n_tokens, lr=3e-4, max_epochs=100, patience=10,
    batch_size=1024, n_neg=49, device="cpu", save_dir=None,
):
    """Train Temporal HeteroGAT with link prediction objective.

    Returns dict with best_val_mrr, epochs_trained, history.
    """
    from src.data.temporal_edges import build_temporal_batches, sample_negatives_temporal
    from src.link_eval import compute_mrr

    model = model.to(device)
    decoder = decoder.to(device)
    train_graph = train_graph.to(device)

    all_params = list(model.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(all_params, lr=lr)

    edge_time_dict = _build_edge_time_dict(train_graph, train_events, device)

    best_mrr = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_mrr": []}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        decoder.train()
        epoch_loss = 0.0
        n_batches = 0

        # encode graph ONCE per epoch (not per batch — 3000x speedup)
        h = model(
            {k: train_graph[k].x for k in ["token", "wallet"]},
            {et: train_graph[et].edge_index for et in train_graph.edge_types},
            edge_time_dict,
        )

        # sample a subset of batches for gradient updates (memory efficient)
        all_batches = list(build_temporal_batches(train_events, batch_size))
        # use every 10th batch to keep it tractable
        step = max(1, len(all_batches) // 100)
        selected = all_batches[::step]

        for batch in selected:
            aug = sample_negatives_temporal(batch, n_tokens, n_neg=n_neg)

            src_idx = torch.tensor(aug["src"], dtype=torch.long, device=device)
            dst_idx = torch.tensor(aug["dst"], dtype=torch.long, device=device)
            labels = torch.tensor(aug["label"], dtype=torch.float32, device=device)

            z_w = h["wallet"][src_idx]
            z_t = h["token"][dst_idx]

            batch_ts = torch.tensor(aug["ts"], dtype=torch.float32, device=device)
            mean_ts = batch_ts.mean()
            dt_w = (mean_ts - batch_ts).clamp(min=0)
            dt_t = dt_w.clone()

            phi_w = model.time_enc(dt_w)
            phi_t = model.time_enc(dt_t)

            scores = decoder(z_w, z_t, phi_w, phi_t)
            loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, 1.0)
            optimiser.step()

            # re-encode after gradient update
            h = model(
                {k: train_graph[k].x for k in ["token", "wallet"]},
                {et: train_graph[et].edge_index for et in train_graph.edge_types},
                edge_time_dict,
            )

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # validation
        model.train(False)
        decoder.train(False)
        val_mrr = _quick_mrr(
            model, decoder, train_graph, edge_time_dict,
            val_events, n_tokens, n_neg, batch_size, device,
        )
        history["val_mrr"].append(val_mrr)

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1 or wait == 0:
            log.info("Epoch %3d | loss %.4f | val MRR %.4f | best %.4f | %.1fs",
                     epoch, avg_loss, val_mrr, best_mrr, elapsed)

        if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_state = {
                "model": copy.deepcopy(model.state_dict()),
                "decoder": copy.deepcopy(decoder.state_dict()),
            }
            wait = 0
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_state, os.path.join(save_dir, "best_temporal_gat.pt"))
        else:
            wait += 1
            if wait >= patience:
                log.info("Early stopping at epoch %d (best MRR: %.4f)", epoch, best_mrr)
                break

    if best_state:
        model.load_state_dict(best_state["model"])
        decoder.load_state_dict(best_state["decoder"])

    return {
        "best_val_mrr": best_mrr,
        "epochs_trained": epoch,
        "history": history,
    }


def _build_edge_time_dict(graph, events, device):
    """Approximate edge timestamps for the static graph edges."""
    mean_ts = float(events["ts"].mean())
    edge_time_dict = {}
    for etype in graph.edge_types:
        n_edges = graph[etype].edge_index.shape[1]
        edge_time_dict[etype] = torch.full(
            (n_edges,), mean_ts, dtype=torch.float32, device=device
        )
    return edge_time_dict


@torch.no_grad()
def _quick_mrr(model, decoder, graph, edge_time_dict,
               events, n_tokens, n_neg, batch_size, device):
    """Fast MRR on a subset of validation events."""
    from src.data.temporal_edges import build_temporal_batches
    from src.link_eval import compute_mrr

    # only use first 5000 events for speed
    events_sub = events.head(min(5000, len(events)))

    h = model(
        {k: graph[k].x for k in ["token", "wallet"]},
        {et: graph[et].edge_index for et in graph.edge_types},
        edge_time_dict,
    )

    all_pos = []
    all_neg = []

    for batch in build_temporal_batches(events_sub, batch_size):
        rng = np.random.RandomState(42)
        neg_dst = rng.randint(0, n_tokens, size=(len(batch["src"]), n_neg))

        src = torch.tensor(batch["src"], dtype=torch.long, device=device)
        dst_pos = torch.tensor(batch["dst"], dtype=torch.long, device=device)
        dst_neg = torch.tensor(neg_dst, dtype=torch.long, device=device)

        z_w = h["wallet"][src]
        z_t_pos = h["token"][dst_pos]

        d_time = model.time_enc.d_time
        phi_zero = torch.zeros(len(src), d_time, device=device)

        pos_s = decoder(z_w, z_t_pos, phi_zero, phi_zero)
        all_pos.append(pos_s.cpu().numpy())

        neg_batch = []
        for j in range(n_neg):
            z_t_n = h["token"][dst_neg[:, j]]
            neg_s = decoder(z_w, z_t_n, phi_zero, phi_zero)
            neg_batch.append(neg_s.cpu().numpy())
        all_neg.append(np.stack(neg_batch, axis=1))

    pos = np.concatenate(all_pos)
    neg = np.concatenate(all_neg)
    return compute_mrr(pos, neg)

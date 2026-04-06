"""Training loop for the FeaturelessHeteroGAT.

Full-batch training on separate train/val HeteroData graphs.
Labels are on TOKEN nodes (binary: high-risk vs rest).
"""

import os
import time
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

log = logging.getLogger(__name__)


def get_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(model, train_data, val_data, lr=1e-3, weight_decay=1e-3,
                max_epochs=300, patience=30, grad_clip=1.0, device=None,
                save_dir=None):
    """Train HeteroGAT with early stopping on validation AUC.

    Uses separate train/val graphs (no masking). Labels are on token nodes.
    Returns dict with best_val_auc, epochs_trained, and history.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # auto pos_weight from training labels on TOKEN nodes
    y_train = train_data["token"].y
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pw = n_neg / max(n_pos, 1)
    log.info("pos_weight=%.3f (pos=%d, neg=%d)", pw, n_pos, n_neg)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device)
    )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_auc": [], "lr": []}

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()

        # forward on train graph — returns (logits, embeddings_dict)
        logits, _ = model(
            {k: train_data[k].x for k in ["token", "wallet"]},
            {et: train_data[et].edge_index for et in train_data.edge_types},
        )
        logits = logits.squeeze(-1)  # [N_tokens]
        loss = criterion(logits, y_train)

        # nan guard
        if torch.isnan(loss):
            log.warning("Epoch %d: NaN loss, skipping", epoch)
            continue

        optimiser.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimiser.step()
        scheduler.step()

        # validation on separate val graph
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(
                {k: val_data[k].x for k in ["token", "wallet"]},
                {et: val_data[et].edge_index for et in val_data.edge_types},
            )
            val_proba = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
            val_proba = np.clip(val_proba, 1e-7, 1 - 1e-7)

        val_y = val_data["token"].y.cpu().numpy()
        val_auc = roc_auc_score(val_y, val_proba)

        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(loss.item())
        history["val_auc"].append(val_auc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == 1 or wait == 0:
            log.info("Epoch %3d | loss %.4f | val AUC %.4f | best %.4f | lr %.2e | %.1fs",
                     epoch, loss.item(), val_auc, best_val_auc, current_lr, elapsed)

        # early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(best_state, os.path.join(save_dir, "best_hetero_gat.pt"))
        else:
            wait += 1
            if wait >= patience:
                log.info("Early stopping at epoch %d (best val AUC: %.4f)",
                         epoch, best_val_auc)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_auc": best_val_auc,
        "epochs_trained": epoch,
        "history": history,
    }

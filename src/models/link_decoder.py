"""Link prediction decoder and training utilities.

Given node embeddings from TemporalHeteroGAT, predicts whether a
(wallet, token) interaction occurs at query time t. Temporal context
is injected via Bochner encodings of "time since last activity" for
both the wallet and the token.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LinkDecoder(nn.Module):
    """MLP decoder for temporal link prediction.

    score = sigma(MLP([z_w || z_tau || Phi(dt_w) || Phi(dt_tau)]))

    where dt_w = t_query - t_last_wallet, dt_tau = t_query - t_last_token.
    The temporal encodings let the decoder distinguish "wallet active 5 min ago"
    from "wallet dormant for 3 days" without baking that into the GNN.
    """

    def __init__(self, emb_dim: int = 128, time_dim: int = 64, hidden: int = 128):
        super().__init__()
        input_dim = 2 * emb_dim + 2 * time_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_w: Tensor, z_tau: Tensor,
                phi_w: Tensor, phi_tau: Tensor) -> Tensor:
        """
        Args:
            z_w:    [B, emb_dim]  — wallet embeddings
            z_tau:  [B, emb_dim]  — token embeddings
            phi_w:  [B, time_dim] — Bochner encoding of wallet recency
            phi_tau:[B, time_dim] — Bochner encoding of token recency
        Returns:
            logits: [B] — raw scores (apply sigmoid for probabilities)
        """
        x = torch.cat([z_w, z_tau, phi_w, phi_tau], dim=-1)
        return self.mlp(x).squeeze(-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(pos_src: Tensor, pos_dst: Tensor,
                     n_nodes_dst: int, n_neg: int = 1,
                     rng: torch.Generator = None) -> Tensor:
    """Random negative destination sampling.

    For each positive edge (src, dst), we corrupt the destination by
    sampling uniformly from all destination nodes. No filtering of
    false negatives — at this graph density the collision rate is tiny
    and the noise is actually beneficial (see Yang et al. 2020).

    Args:
        pos_src: [B] source node indices (not used directly, just for shape)
        pos_dst: [B] positive destination indices
        n_nodes_dst: total number of destination nodes to sample from
        n_neg: number of negatives per positive edge
        rng: optional torch.Generator for reproducibility
    Returns:
        neg_dst: [B * n_neg] corrupted destination indices
    """
    n_pos = pos_src.size(0)
    neg_dst = torch.randint(
        0, n_nodes_dst, (n_pos * n_neg,),
        device=pos_src.device, generator=rng,
    )
    return neg_dst


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class LinkPredictionLoss(nn.Module):
    """Binary cross-entropy over positive and negative edge scores.

    Nothing fancy — just concat positives (label=1) and negatives (label=0)
    and compute BCE. Works well enough; margin-based losses didn't help
    in our ablations.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pos_scores: Tensor, neg_scores: Tensor) -> Tensor:
        """
        Args:
            pos_scores: [B] logits for positive edges
            neg_scores: [B * n_neg] logits for negative edges
        Returns:
            scalar loss
        """
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores),
        ], dim=0)
        return self.bce(scores, labels)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    emb_dim, time_dim = 128, 64
    batch = 256

    decoder = LinkDecoder(emb_dim=emb_dim, time_dim=time_dim)
    print(f"Decoder parameters: {decoder.count_parameters():,}")

    # fake embeddings
    z_w = torch.randn(batch, emb_dim)
    z_tau = torch.randn(batch, emb_dim)
    phi_w = torch.randn(batch, time_dim)
    phi_tau = torch.randn(batch, time_dim)

    pos_scores = decoder(z_w, z_tau, phi_w, phi_tau)
    print(f"Positive scores shape: {pos_scores.shape}")

    # negative sampling
    pos_src = torch.arange(batch)
    pos_dst = torch.randint(0, 100, (batch,))
    neg_dst = sample_negatives(pos_src, pos_dst, n_nodes_dst=100, n_neg=3)
    print(f"Negative dst shape: {neg_dst.shape}  (expected {batch * 3})")

    # expand embeddings for negatives
    z_w_neg = z_w.repeat_interleave(3, dim=0)
    phi_w_neg = phi_w.repeat_interleave(3, dim=0)
    z_tau_neg = torch.randn(batch * 3, emb_dim)  # would look up from neg_dst
    phi_tau_neg = torch.randn(batch * 3, time_dim)

    neg_scores = decoder(z_w_neg, z_tau_neg, phi_w_neg, phi_tau_neg)

    criterion = LinkPredictionLoss()
    loss = criterion(pos_scores, neg_scores)
    print(f"Loss: {loss.item():.4f}")

    # quick gradient check
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in decoder.parameters() if p.grad is not None)
    print(f"Total grad norm: {grad_norm:.4f}")
    print("Smoke test passed.")

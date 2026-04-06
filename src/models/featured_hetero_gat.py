"""Featured Heterogeneous GAT for Solana wallet classification.

Uses MLP projections of raw node features instead of learnable type embeddings.
Same graph architecture as FeaturelessHeteroGAT (~86K parameters).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, LayerNorm


class FeaturedHeteroGAT(nn.Module):
    """Graph neural network that classifies wallets using MLP-projected features.

    Each node type's raw features are projected through a 2-layer MLP into a
    shared embedding space; message passing and prediction follow the same
    architecture as FeaturelessHeteroGAT. ~86K parameters.
    """

    def __init__(self, token_feat_dim=116, wallet_feat_dim=15, embed_dim=64,
                 hidden_dim=64, gat_heads=4, gat_head_dim=16, metadata=None,
                 dropout=0.3):
        super().__init__()
        assert metadata is not None, "need (node_types, edge_types) tuple"

        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.dropout = dropout

        # MLP projections — map raw features to shared embedding space
        # Named tok_proj/wal_proj to match existing checkpoint keys
        self.tok_proj = nn.Sequential(
            nn.Linear(token_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.wal_proj = nn.Sequential(
            nn.Linear(wallet_feat_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # --- Layer 1: SAGEConv with sum aggregation ---
        # the key insight: with identical initial embeddings, GAT attention
        # normalises to 1/|N| which is useless. So we use sum aggregation
        # first to differentiate nodes by degree, then apply attention.
        conv1_dict = {}
        for etype in self.edge_types:
            conv1_dict[etype] = SAGEConv(
                (embed_dim, embed_dim), hidden_dim, aggr="sum"
            )
        self.conv1 = HeteroConv(conv1_dict, aggr="sum")
        self.norm1 = nn.ModuleDict({
            ntype: LayerNorm(hidden_dim) for ntype in self.node_types
        })

        # --- Layer 2: GATConv with multi-head attention ---
        out_dim = gat_heads * gat_head_dim  # 4 * 16 = 64
        conv2_dict = {}
        for etype in self.edge_types:
            conv2_dict[etype] = GATConv(
                (hidden_dim, hidden_dim), gat_head_dim,
                heads=gat_heads, concat=True, dropout=dropout,
                add_self_loops=False,  # required for bipartite edges (wallet->token)
            )
        self.conv2 = HeteroConv(conv2_dict, aggr="sum")

        # prediction head (wallet nodes only)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def _project_features(self, x_dict):
        """Project raw node features through per-type MLPs."""
        return {
            "token": self.tok_proj(x_dict["token"]),
            "wallet": self.wal_proj(x_dict["wallet"]),
        }

    def forward(self, x_dict, edge_index_dict):
        """Returns (logits, embeddings_dict) for TOKEN nodes."""
        h = self._project_features(x_dict)

        # layer 1 — sum aggregation to break symmetry
        h = self.conv1(h, edge_index_dict)
        h = {k: F.elu(self.norm1[k](v)) for k, v in h.items()}

        # layer 2 — attention on now-differentiated nodes
        h = self.conv2(h, edge_index_dict)
        h = {k: F.elu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training)
             for k, v in h.items()}

        # predict on TOKEN nodes (high-risk classification)
        token_emb = h["token"]
        logits = self.head(token_emb)

        return logits, h

    def predict_proba(self, x_dict, edge_index_dict):
        """Return sigmoid probabilities for token nodes."""
        with torch.no_grad():
            logits, _ = self.forward(x_dict, edge_index_dict)
        return torch.sigmoid(logits.squeeze(-1))

    def count_parameters(self):
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick smoke test
    node_types = ["token", "wallet"]
    edge_types = [
        ("wallet", "transfers_to", "wallet"),
        ("wallet", "holds", "token"),
        ("token", "held_by", "wallet"),
        ("wallet", "swaps_on", "wallet"),
    ]
    metadata = (node_types, edge_types)

    model = FeaturedHeteroGAT(metadata=metadata)
    print(f"Parameters: {model.count_parameters():,}")

    # dummy graph with actual feature dimensions
    x_dict = {
        "token": torch.randn(50, 116),
        "wallet": torch.randn(200, 15),
    }
    edge_index_dict = {
        ("wallet", "transfers_to", "wallet"): torch.randint(0, 200, (2, 500)),
        ("wallet", "holds", "token"): torch.stack([
            torch.randint(0, 200, (300,)), torch.randint(0, 50, (300,))
        ]),
        ("token", "held_by", "wallet"): torch.stack([
            torch.randint(0, 50, (300,)), torch.randint(0, 200, (300,))
        ]),
        ("wallet", "swaps_on", "wallet"): torch.randint(0, 200, (2, 400)),
    }

    logits, emb = model(x_dict, edge_index_dict)
    print(f"Logits shape: {logits.shape}")
    print(f"Wallet embeddings: {emb['wallet'].shape}")

    proba = model.predict_proba(x_dict, edge_index_dict)
    print(f"Probabilities: min={proba.min():.3f}, max={proba.max():.3f}")
    print("Smoke test passed.")

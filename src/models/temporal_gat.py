"""Temporal Heterogeneous GAT for link prediction on Solana wallet-token graphs.

Architecture overview:
    Type Embeddings (2 x 128-dim, learnable)
        |
    Layer 1: HeteroConv(SAGEConv, aggr=sum) -> 128-d  [breaks symmetry via degree]
        | LayerNorm + ELU
    Layer 2: TemporalHeteroGATLayer -> 128-d  [Bochner-augmented attention]
        | ELU + dropout
    -> node embeddings z_w, z_tau for the decoder

The temporal attention in Layer 2 is the main contribution: attention weights
incorporate a Bochner encoding of the edge time delta, so the model learns
to weight recent vs. old interactions differently for each edge type.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, SAGEConv, LayerNorm
# use PyTorch native scatter_reduce (works on MPS, CPU, CUDA)
# torch_scatter only works on CPU/CUDA
def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """Drop-in replacement for torch_scatter.scatter using native PyTorch."""
    if dim_size is None:
        dim_size = int(index.max()) + 1
    if src.dim() == 1:
        # 1D case: use simple index_add / scatter_reduce
        if reduce == "sum":
            out = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
            out.scatter_add_(0, index, src)
            return out
        elif reduce == "max":
            out = torch.full((dim_size,), float('-inf'), dtype=src.dtype, device=src.device)
            out.scatter_reduce_(0, index, src, reduce="amax", include_self=False)
            return out
    else:
        # multi-dim case
        idx_expanded = index.unsqueeze(-1).expand_as(src)
        if reduce == "sum":
            out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
            out.scatter_add_(0, idx_expanded, src)
            return out
        elif reduce == "max":
            out = torch.full((dim_size, src.size(1)), float('-inf'), dtype=src.dtype, device=src.device)
            out.scatter_reduce_(0, idx_expanded, src, reduce="amax", include_self=False)
            return out
    raise ValueError(f"Unknown reduce: {reduce}")

from .bochner import BochnerTimeEncoding


# ---------------------------------------------------------------------------
# Single edge-type temporal GAT layer
# ---------------------------------------------------------------------------

class TemporalGATLayer(nn.Module):
    """GAT-style attention for one edge type, with Bochner time encoding
    injected into the attention logits.

    attention = softmax( LeakyReLU(a^T [W_s h_s || W_d h_d || Phi(dt)]) / sqrt(D) )
    """

    def __init__(self, in_dim: int, out_dim: int, d_time: int, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.d_time = d_time

        # separate projections for source / destination
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)

        # attention vector over [projected_src || projected_dst || phi(dt)]
        attn_input_dim = 2 * out_dim + d_time
        self.attn = nn.Linear(attn_input_dim, 1, bias=False)
        self.scale = math.sqrt(attn_input_dim)

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_src: Tensor, h_dst: Tensor, edge_index: Tensor,
                phi_dt: Tensor) -> Tensor:
        """
        Args:
            h_src: [N_src, in_dim]
            h_dst: [N_dst, in_dim]
            edge_index: [2, E]
            phi_dt: [E, d_time] — Bochner encoding of edge time deltas
        Returns:
            out: [N_dst, out_dim] aggregated messages
        """
        src_idx, dst_idx = edge_index

        # project
        z_s = self.W_src(h_src[src_idx])   # [E, out_dim]
        z_d = self.W_dst(h_dst[dst_idx])   # [E, out_dim]

        # attention logits with time encoding
        attn_input = torch.cat([z_s, z_d, phi_dt], dim=-1)  # [E, 2*out + d_time]
        e = self.leaky(self.attn(attn_input)).squeeze(-1)    # [E]
        e = e / self.scale

        # sparse softmax per destination node
        alpha = self._sparse_softmax(e, dst_idx, num_nodes=h_dst.size(0))
        alpha = self.dropout(alpha)

        # aggregate: weighted sum of projected source embeddings
        msg = alpha.unsqueeze(-1) * z_s  # [E, out_dim]
        out = scatter(msg, dst_idx, dim=0, dim_size=h_dst.size(0), reduce="sum")
        return out

    @staticmethod
    def _sparse_softmax(logits: Tensor, index: Tensor, num_nodes: int) -> Tensor:
        """Numerically stable softmax over sparse neighbourhoods."""
        logits_max = scatter(logits, index, dim=0, dim_size=num_nodes, reduce="max")
        logits = logits - logits_max[index]
        exp = logits.exp()
        exp_sum = scatter(exp, index, dim=0, dim_size=num_nodes, reduce="sum")
        return exp / (exp_sum[index] + 1e-12)


# ---------------------------------------------------------------------------
# Heterogeneous wrapper — one TemporalGATLayer per edge type
# ---------------------------------------------------------------------------

class TemporalHeteroGATLayer(nn.Module):
    """Applies a separate TemporalGATLayer for each edge type, then sums
    contributions at each destination node (same as HeteroConv aggr=sum)."""

    def __init__(self, in_dim: int, out_dim: int, d_time: int,
                 edge_types: list, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleDict()
        for etype in edge_types:
            key = "__".join(etype)  # e.g. "wallet__trades__token"
            self.convs[key] = TemporalGATLayer(in_dim, out_dim, d_time, dropout)
        self.edge_types = edge_types

    def forward(self, x_dict: dict, edge_index_dict: dict,
                phi_dt_dict: dict) -> dict:
        """
        Args:
            x_dict: {node_type: Tensor[N, in_dim]}
            edge_index_dict: {(src_type, rel, dst_type): [2, E]}
            phi_dt_dict: {(src_type, rel, dst_type): [E, d_time]} Bochner encodings
        Returns:
            out_dict: {node_type: Tensor[N, out_dim]}
        """
        # accumulate messages per destination node type
        out_dict = {}
        for etype in self.edge_types:
            src_type, rel, dst_type = etype
            key = "__".join(etype)

            h_src = x_dict[src_type]
            h_dst = x_dict[dst_type]
            edge_index = edge_index_dict[etype]
            phi_dt = phi_dt_dict[etype]

            msg = self.convs[key](h_src, h_dst, edge_index, phi_dt)

            if dst_type in out_dict:
                out_dict[dst_type] = out_dict[dst_type] + msg
            else:
                out_dict[dst_type] = msg

        return out_dict


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TemporalHeteroGAT(nn.Module):
    """Two-layer hetero GNN for temporal link prediction on Solana graphs.

    Layer 1: SAGEConv(sum) to break symmetry from identical type embeddings.
    Layer 2: Temporal GAT with Bochner-encoded time deltas in attention.

    Expected edge types:
        (wallet, trades, token), (token, traded_by, wallet),
        (wallet, co_trades, wallet), (wallet, same_tx, wallet)
    """

    def __init__(self, metadata, embed_dim=128, d_time=64, dropout=0.2):
        """
        Args:
            metadata: (node_types, edge_types) as PyG convention
            embed_dim: dimension of type embeddings and hidden layers
            d_time: dimension of Bochner time encoding
            dropout: dropout rate for attention + final layer
        """
        super().__init__()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.embed_dim = embed_dim

        # --- type embeddings (no input features) ---
        self.type_embeddings = nn.ModuleDict({
            ntype: nn.Embedding(1, embed_dim) for ntype in self.node_types
        })

        # --- time encoder (shared across all layers) ---
        self.time_enc = BochnerTimeEncoding(d_time)

        # --- Layer 1: SAGEConv sum aggregation ---
        conv1_dict = {}
        for etype in self.edge_types:
            conv1_dict[etype] = SAGEConv(
                (embed_dim, embed_dim), embed_dim, aggr="sum"
            )
        self.conv1 = HeteroConv(conv1_dict, aggr="sum")
        self.norm1 = nn.ModuleDict({
            ntype: LayerNorm(embed_dim) for ntype in self.node_types
        })

        # --- Layer 2: Temporal heterogeneous GAT ---
        self.conv2 = TemporalHeteroGATLayer(
            in_dim=embed_dim, out_dim=embed_dim, d_time=d_time,
            edge_types=self.edge_types, dropout=dropout,
        )

        self.dropout = dropout

    def _get_type_embeddings(self, x_dict):
        """Broadcast the single learnable embedding to every node of that type."""
        out = {}
        for ntype, x in x_dict.items():
            n = x.size(0)
            idx = torch.zeros(n, dtype=torch.long, device=x.device)
            out[ntype] = self.type_embeddings[ntype](idx)
        return out

    def forward(self, x_dict, edge_index_dict, edge_time_dict, t_query=None):
        """
        Args:
            x_dict: {node_type: Tensor[N, *]} — shape used to count nodes only
            edge_index_dict: {edge_type: [2, E]}
            edge_time_dict: {edge_type: Tensor[E]} — edge timestamps (seconds)
            t_query: float or Tensor — query time for computing dt = t_query - t_edge.
                     If None, uses max timestamp + 1 (handy for training).
        Returns:
            node_embs: {node_type: Tensor[N, embed_dim]}
        """
        # infer query time
        if t_query is None:
            all_times = torch.cat([t.float() for t in edge_time_dict.values()])
            t_query = all_times.max() + 1.0

        # compute Bochner encodings for each edge type
        phi_dt_dict = {}
        for etype, timestamps in edge_time_dict.items():
            dt = (t_query - timestamps.float()).clamp(min=0)
            phi_dt_dict[etype] = self.time_enc(dt)

        # initial embeddings
        h = self._get_type_embeddings(x_dict)

        # layer 1 — SAGEConv sum (degree-aware, breaks symmetry)
        h = self.conv1(h, edge_index_dict)
        h = {k: F.elu(self.norm1[k](v)) for k, v in h.items()}

        # layer 2 — temporal GAT
        h = self.conv2(h, edge_index_dict, phi_dt_dict)
        h = {k: F.elu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training)
             for k, v in h.items()}

        return h

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    node_types = ["wallet", "token"]
    edge_types = [
        ("wallet", "trades", "token"),
        ("token", "traded_by", "wallet"),
        ("wallet", "co_trades", "wallet"),
        ("wallet", "same_tx", "wallet"),
    ]
    metadata = (node_types, edge_types)

    model = TemporalHeteroGAT(metadata=metadata, embed_dim=128, d_time=64)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}")

    # dummy graph
    n_wallets, n_tokens = 500, 100
    x_dict = {
        "wallet": torch.randn(n_wallets, 1),
        "token": torch.randn(n_tokens, 1),
    }
    edge_index_dict = {
        ("wallet", "trades", "token"): torch.stack([
            torch.randint(0, n_wallets, (2000,)),
            torch.randint(0, n_tokens, (2000,)),
        ]),
        ("token", "traded_by", "wallet"): torch.stack([
            torch.randint(0, n_tokens, (2000,)),
            torch.randint(0, n_wallets, (2000,)),
        ]),
        ("wallet", "co_trades", "wallet"): torch.randint(0, n_wallets, (2, 1500)),
        ("wallet", "same_tx", "wallet"): torch.randint(0, n_wallets, (2, 1000)),
    }
    # fake timestamps (unix-ish, spread over a few days)
    edge_time_dict = {}
    base_t = 1_700_000_000.0
    for etype, ei in edge_index_dict.items():
        n_edges = ei.size(1)
        edge_time_dict[etype] = torch.rand(n_edges) * 86400 * 7 + base_t

    embs = model(x_dict, edge_index_dict, edge_time_dict)
    print(f"Wallet embeddings: {embs['wallet'].shape}")
    print(f"Token embeddings:  {embs['token'].shape}")
    print("Smoke test passed.")

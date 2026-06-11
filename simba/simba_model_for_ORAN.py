"""simba_model_for_ORAN.py — Simba adapted for cross-topology O-RAN stress detection.

WHY A NEW MODEL FILE
--------------------
The original Simba (simba_model.py) was built for a SINGLE fixed graph:

  1. `nn.Embedding(num_nodes, ...)` in the graph learner ties the learned graph to
     a fixed node count *and* fixed node identities. A model trained on a 3-node
     graph cannot run on a 4-node graph, and node "0" must always mean the same BS.
  2. The classifier flattens ALL nodes into one vector
     (`nn.Linear(combined_dim * num_nodes, ...)`) and emits ONE system-wide label.
     That hard-codes num_nodes and gives no per-entity output.

Neither survives cross-topology leave-one-out (LOO), where topologies have
different node counts (cu0_du0du1 = 1 CU + 2 DU, cu1_du2 = 1 CU + 1 DU,
cu2_du3du4du5 = 1 CU + 3 DU) and the test topology is never seen in training.

THREE ADAPTATIONS (all required for cross-topology generalization)
------------------------------------------------------------------
  A. Type-shared input projections. CU has cu_dim features, DU has du_dim
     features (different). Two Linear projections map each entity TYPE into a
     shared d_model space, after which CU and DU are processed by the SAME
     backbone weights. This mirrors TopoAR's type-shared design — the model
     learns "what stress looks like" per entity type, not per node id.

  B. Feature-derived graph learning. The MTGNN-style asymmetric adjacency is
     kept, but the two node embeddings are produced by a Linear on each node's
     temporal embedding instead of an `nn.Embedding` table indexed by node id.
     The graph is now a function of *what the node is doing*, so it works for any
     node count and any unseen topology.

  C. Per-node output. The classifier is applied to every node independently and
     emits a logit vector PER node, so the output shape adapts to the node count
     and we get per-entity (CU, each DU) predictions — exactly what TopoAR reports.

The temporal Transformer and the mix-hop graph-convolution propagation are kept
faithful to the original Simba (Figures 3, 7, 8 of the paper).

INPUT / OUTPUT CONTRACT
-----------------------
  forward(cu, du):
    cu : (B, L, cu_dim)        — CU window  (CU is a single entity per topology)
    du : (B, L, N, du_dim)     — DU window  (N DUs, varies per topology)
  returns:
    logits : (B, M, num_classes)   where M = N + 1, node 0 = CU, nodes 1..N = DUs
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding over the time axis."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, d_model)
        return x + self.pe[:, : x.size(1)]


class TemporalEncoder(nn.Module):
    """Shared Transformer over the time axis, applied independently per node.

    Collapses each node's L-step window into one d_model embedding by mean-pooling
    the encoder output over time (Simba Fig. 7 temporal branch).
    """

    def __init__(self, d_model, heads, layers, hidden, dropout=0.1):
        super().__init__()
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=hidden,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B*M, L, d_model)
        x = self.pos(x)
        out = self.encoder(x)
        return out.mean(dim=1)  # (B*M, d_model)


class FeatureGraphLearner(nn.Module):
    """MTGNN-style asymmetric adjacency, but node embeddings come from features.

    Original Simba used nn.Embedding(num_nodes); here the two embeddings are
    Linear projections of each node's temporal embedding, so the learned graph
    is topology-size-agnostic and works on unseen node counts.
    """

    def __init__(self, in_dim, emb_dim, top_k=None, alpha=3.0):
        super().__init__()
        self.theta1 = nn.Linear(in_dim, emb_dim)
        self.theta2 = nn.Linear(in_dim, emb_dim)
        self.alpha = alpha
        self.top_k = top_k

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: (B, M, in_dim)
        m1 = torch.tanh(self.alpha * self.theta1(h))
        m2 = torch.tanh(self.alpha * self.theta2(h))
        adj = F.relu(torch.tanh(
            self.alpha * (torch.bmm(m1, m2.transpose(1, 2)) - torch.bmm(m2, m1.transpose(1, 2)))
        ))
        # top_k sparsification is optional and pointless for the tiny O-RAN graphs
        # (2-4 nodes); left in for parity with the paper when M is large.
        if self.top_k is not None and 0 < self.top_k < adj.size(-1):
            topv, _ = torch.topk(adj, self.top_k, dim=-1)
            mask = adj >= topv[..., -1:]
            adj = adj * mask.float()
        return adj  # (B, M, M)


class MixHopPropagation(nn.Module):
    """Mix-hop propagation (Simba Fig. 8b): sum of per-hop feature selectors."""

    def __init__(self, in_dim, out_dim, hops):
        super().__init__()
        self.hops = hops
        self.selectors = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(hops + 1)])

    def forward(self, x, adj):  # x: (B, M, in_dim)  adj: (B, M, M)
        cur = x
        acc = self.selectors[0](cur)
        for i in range(1, self.hops + 1):
            cur = torch.bmm(adj, cur)
            acc = acc + self.selectors[i](cur)
        return acc  # (B, M, out_dim)


class GCLayer(nn.Module):
    """Graph convolution with in-flow + out-flow propagation (Simba Fig. 8a)."""

    def __init__(self, in_dim, out_dim, hops):
        super().__init__()
        self.out_flow = MixHopPropagation(in_dim, out_dim, hops)
        self.in_flow = MixHopPropagation(in_dim, out_dim, hops)

    def forward(self, x, adj):
        return self.out_flow(x, adj) + self.in_flow(x, adj.transpose(1, 2))


class SimbaORAN(nn.Module):
    """Per-node Simba for cross-topology O-RAN stress detection.

    Args:
        cu_dim, du_dim : feature dims of CU and DU streams (after preprocessing).
        num_classes    : 2 for binary (normal vs stress); >2 if multi-fault.
        d_model        : shared hidden dim for both entity types.
        gc_channels    : graph-conv output channels.
        gc_hops        : mix-hop order.
        tf_heads/layers/hidden : temporal Transformer config.
        gl_emb_dim, top_k      : graph learner config (top_k=None → dense graph).
    """

    def __init__(self, cu_dim, du_dim, num_classes=2, d_model=64,
                 gl_emb_dim=16, top_k=None, gc_hops=2, gc_channels=32,
                 tf_heads=4, tf_layers=2, tf_hidden=128, dropout=0.1):
        super().__init__()
        # (A) type-shared input projections
        self.cu_proj = nn.Linear(cu_dim, d_model)
        self.du_proj = nn.Linear(du_dim, d_model)
        # temporal branch (shared across all nodes)
        self.temporal = TemporalEncoder(d_model, tf_heads, tf_layers, tf_hidden, dropout)
        # (B) spatial branch — feature-derived graph + mix-hop GC
        self.graph = FeatureGraphLearner(d_model, gl_emb_dim, top_k)
        self.gc = GCLayer(d_model, gc_channels, gc_hops)
        # (C) per-node classifier
        self.head = nn.Sequential(
            nn.Linear(d_model + gc_channels, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, cu: torch.Tensor, du: torch.Tensor) -> torch.Tensor:
        # cu: (B, L, cu_dim)   du: (B, L, N, du_dim)
        B, L, N, _ = du.shape
        cu_h = self.cu_proj(cu)                              # (B, L, d_model)
        du_h = self.du_proj(du)                              # (B, L, N, d_model)

        # Stack CU (node 0) + DUs (nodes 1..N) into a unified node axis.
        nodes = torch.cat([cu_h.unsqueeze(2), du_h], dim=2)  # (B, L, M, d_model)
        M = N + 1
        d = nodes.size(-1)

        # Temporal: one embedding per node (shared Transformer over time).
        seq = nodes.permute(0, 2, 1, 3).reshape(B * M, L, d)  # (B*M, L, d_model)
        temp = self.temporal(seq).reshape(B, M, d)            # (B, M, d_model)

        # Spatial: learn graph from temporal embeddings, propagate.
        adj = self.graph(temp)                                # (B, M, M)
        spat = self.gc(temp, adj)                             # (B, M, gc_channels)

        fused = torch.cat([temp, spat], dim=-1)               # (B, M, d_model+gc_channels)
        return self.head(fused)                               # (B, M, num_classes)

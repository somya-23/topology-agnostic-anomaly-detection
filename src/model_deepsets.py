"""DeepSets-style TopoAR: mean-pooling over DU embeddings for N-invariant aggregation.

Architecture change vs TopoAR (model.py):

  Original attention aggregation:
      q      = Q(h)                                          (B, d)
      scores = [q·K_cu, q·K_du_0, ..., q·K_du_{N-1}] / √d  (B, 1+N)
      α      = softmax(scores)                               (B, 1+N)
      s      = α_0·V_cu + Σ_i α_i·V_du_i                   (B, d)
      h, c   = LSTM(s, h, c)

  DeepSets replacement:
      cu_ctx = V_CU(cu_tok)                                  (B, d)
      du_agg = mean_i V_DU(du_tok_i)                        (B, d)  ← N-invariant
      s      = concat(cu_ctx, du_agg)                        (B, 2d)
      h, c   = LSTM(s, h, c)                                 LSTM input = 2d

Why mean-pooling is topology-agnostic (DeepSets, Zaheer et al. 2017):
  Softmax attention over 1+N keys is a weighted combination where α sums to 1.
  Adding a DU rescales all weights → the attended context s changes magnitude
  with N, making the model's predictions N-dependent.

  Mean-pooling computes the AVERAGE per-DU value embedding regardless of N.
  The DU contribution to the LSTM context is always "what does a typical DU
  look like right now?" — topology count is irrelevant. This is a strict
  permutation-invariant, cardinality-invariant set aggregation.

API compatibility with model.py:
  - init_state(batch, device) → same signature.
  - project_tokens(cu, du)    → same signature.
  - step(cu_tok, du_tok, h, c) → returns (cu_hat, du_hat, h, c, None);
    the 5th value is None (no attention weights) to stay compatible with
    callers that unpack 5 values from step().
  - forward(cu_seq, du_seq)   → same signature, same return shapes.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DeepSetsTopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim  = cu_dim
        self.du_dim  = du_dim
        self.d       = embed_dim

        # Input projections + type bias + LayerNorm (shared across entities of same type).
        self.W_CU  = nn.Linear(cu_dim, embed_dim, bias=False)
        self.W_DU  = nn.Linear(du_dim, embed_dim, bias=False)
        self.e_CU  = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU  = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Value projections (no Q/K needed — attention replaced by mean-pool).
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)

        # LSTM input = concat(cu_ctx, du_agg) → 2 * embed_dim.
        # Hidden state is still embed_dim.
        self.lstm = nn.LSTMCell(2 * embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)

        # Per-entity decoders: [h_norm ; entity_token] → entity features.
        self.D_CU = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, cu_dim),
        )
        self.D_DU = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, du_dim),
        )

    def init_state(
        self, batch_size: int, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.d, device=device)
        c = torch.zeros(batch_size, self.d, device=device)
        return h, c

    def project_tokens(
        self, cu: torch.Tensor, du: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cu (B, cu_dim) → cu_tok (B, d);  du (B, N, du_dim) → du_tok (B, N, d)."""
        cu_tok = self.LN_CU(self.W_CU(cu) + self.e_CU)    # (B, d)
        du_tok = self.LN_DU(self.W_DU(du) + self.e_DU)    # (B, N, d)
        return cu_tok, du_tok

    def step(
        self,
        cu_tok: torch.Tensor,   # (B, d)
        du_tok: torch.Tensor,   # (B, N, d)
        h: torch.Tensor,        # (B, d)
        c: torch.Tensor,        # (B, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """One autoregressive step.

        Returns (cu_hat, du_hat, h_new, c_new, None).
        The trailing None keeps the return signature compatible with TopoAR.step()
        callers that unpack a 5-tuple (the 5th element was attention weights).
        """
        B, N, d = du_tok.shape

        # ── DeepSets aggregation ──────────────────────────────────────────────
        cu_ctx = self.V_CU(cu_tok)                          # (B, d)
        du_agg = self.V_DU(du_tok).mean(dim=1)             # (B, d)  — mean over N
        s      = torch.cat([cu_ctx, du_agg], dim=-1)       # (B, 2d)

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)

        # ── Decoders (identical to TopoAR) ───────────────────────────────────
        cu_hat = self.D_CU(torch.cat([h_norm, cu_tok], dim=-1))             # (B, cu_dim)
        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)
        du_hat = self.D_DU(torch.cat([h_norm_b, du_tok], dim=-1))          # (B, N, du_dim)

        return cu_hat, du_hat, h_new, c_new, None

    def forward(
        self,
        cu_seq: torch.Tensor,   # (B, T, cu_dim)
        du_seq: torch.Tensor,   # (B, T, N, du_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full sequence forward — predictions at step t are for time t+1.

        Returns:
            cu_hat: (B, T, cu_dim)
            du_hat: (B, T, N, du_dim)
        """
        B, T, _ = cu_seq.shape
        device   = cu_seq.device

        h, c = self.init_state(B, device)
        cu_hats, du_hats = [], []

        for t in range(T):
            cu_tok, du_tok = self.project_tokens(cu_seq[:, t], du_seq[:, t])
            cu_hat, du_hat, h, c, _ = self.step(cu_tok, du_tok, h, c)
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)

        return torch.stack(cu_hats, dim=1), torch.stack(du_hats, dim=1)

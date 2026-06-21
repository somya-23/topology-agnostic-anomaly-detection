"""model_sharedec.py — TopoAR with type-separate encoders but ONE shared decoder.

Motivation. In the v0 CU_NET cross-topology failure (test=cu0_du0du1), TopoAR's
*type-private* CU decoder D_CU reconstructs the unseen topology's net_ratio
~2500x worse than the fully-shared baselines' single decoder (normal recon error
77.1 vs 0.03), which blows the max-pool lift past the p99.9 threshold on ~half
the normal rows (402 FPs, precision 0.22). D_CU is trained only on CU rows from
two topologies, so it overfits the near-constant train net_ratio and extrapolates
badly. The fully-shared baselines avoid this (their decoder sees CU + every DU
row, so it is far more regularized) but pay for it elsewhere (they collapse to
F1~0.0 on the largest 6-DU topology).

This model is the best-of-both hybrid: keep TopoAR's type-SEPARATE encoders and
per-type query->entities attention (the part that keeps cu2_du3du4du5 working),
but replace the two type-private decoders D_CU/D_DU with ONE shared decoder D
(the part that regularizes net_ratio reconstruction and should fix cu0).

The ONLY change from TopoAR (model.py):
  - D_CU, D_DU  ->  a single shared D that outputs the unified du_dim layout.
    For a DU we use all du_dim outputs; for the CU we slice back the 7 real CU
    positions (cu_idx), so the loss/score still touch only real CU features.

Everything else — W_CU/W_DU, e_CU/e_DU, LN_CU/LN_DU, per-type K/V, Q, LSTMCell,
LN_h — is identical to TopoAR. Drop-in: same __init__(cu_dim, du_dim, embed_dim)
and forward(cu_seq, du_seq) -> (cu_hat, du_hat), so the v0 pipeline runs it via a
model swap, exactly like the set baselines.

Unified du_dim layout (same convention as model_setbaselines):
  0 cpu, 1 mem_pct, 2 mem_bytes, 3 fs_writes, 4 net_tx, 5 net_rx,
  6..27 PCI(22), 28 net_diff, 29 net_ratio
CU order:  cpu, mem_pct, mem_bytes, net_tx, net_rx, net_diff, net_ratio
  -> cu_idx maps CU's 7 features onto [0, 1, 2, 4, 5, du_dim-2, du_dim-1].
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def cu_to_uni_index(du_dim: int) -> list:
    # cpu->0, mem_pct->1, mem_bytes->2, net_tx->4, net_rx->5, net_diff->28, net_ratio->29
    return [0, 1, 2, 4, 5, du_dim - 2, du_dim - 1]


class SharedDecoderTopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 32):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d = embed_dim
        self.register_buffer(
            "cu_idx", torch.tensor(cu_to_uni_index(du_dim), dtype=torch.long)
        )

        # Type-separate input projections + type embeddings + LayerNorms (unchanged from TopoAR).
        self.W_CU = nn.Linear(cu_dim, embed_dim, bias=False)
        self.W_DU = nn.Linear(du_dim, embed_dim, bias=False)
        self.e_CU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Per-type K and V; single Q on the recurrent hidden state (unchanged from TopoAR).
        self.K_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)

        # THE ONLY ARCHITECTURAL CHANGE: one shared decoder over the unified du_dim
        # output. Sees [h_norm ; entity_token] for both CU and DU.
        self.D = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, du_dim),
        )

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.d, device=device)
        c = torch.zeros(batch_size, self.d, device=device)
        return h, c

    def project_tokens(
        self, cu: torch.Tensor, du: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_tok = self.LN_CU(self.W_CU(cu) + self.e_CU)              # (B, d)
        du_tok = self.LN_DU(self.W_DU(du) + self.e_DU)             # (B, N, d)
        return cu_tok, du_tok

    def step(self, cu_tok, du_tok, h, c):
        B, N, d = du_tok.shape
        scale = 1.0 / math.sqrt(d)

        q = self.Q(h)                                              # (B, d)
        K_cu = self.K_CU(cu_tok)                                   # (B, d)
        K_du = self.K_DU(du_tok)                                   # (B, N, d)
        V_cu = self.V_CU(cu_tok)                                   # (B, d)
        V_du = self.V_DU(du_tok)                                   # (B, N, d)

        score_cu = (q * K_cu).sum(dim=-1, keepdim=True) * scale    # (B, 1)
        score_du = torch.einsum("bd,bnd->bn", q, K_du) * scale     # (B, N)
        scores = torch.cat([score_cu, score_du], dim=1)           # (B, 1+N)
        alpha = torch.softmax(scores, dim=-1)                     # (B, 1+N)

        s_cu = alpha[:, 0:1] * V_cu                                # (B, d)
        s_du = (alpha[:, 1:].unsqueeze(-1) * V_du).sum(dim=1)      # (B, d)
        s = s_cu + s_du                                            # (B, d)

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)                                  # (B, d)

        # Shared decoder: same D for CU and DU. CU output is the 7 real slots.
        cu_dec = self.D(torch.cat([h_norm, cu_tok], dim=-1))      # (B, du_dim)
        cu_hat = cu_dec[:, self.cu_idx]                           # (B, cu_dim)

        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)          # (B, N, d)
        du_hat = self.D(torch.cat([h_norm_b, du_tok], dim=-1))    # (B, N, du_dim)

        return cu_hat, du_hat, h_new, c_new, alpha

    def forward(self, cu_seq, du_seq, return_alpha: bool = False):
        B, T, _ = cu_seq.shape
        h, c = self.init_state(B, cu_seq.device)
        cu_hats, du_hats, alphas = [], [], []
        for t in range(T):
            cu_tok, du_tok = self.project_tokens(cu_seq[:, t, :], du_seq[:, t, :, :])
            cu_hat, du_hat, h, c, alpha = self.step(cu_tok, du_tok, h, c)
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)
            if return_alpha:
                alphas.append(alpha)
        cu_hat_seq = torch.stack(cu_hats, dim=1)                  # (B, T, cu_dim)
        du_hat_seq = torch.stack(du_hats, dim=1)                  # (B, T, N, du_dim)
        if return_alpha:
            return cu_hat_seq, du_hat_seq, torch.stack(alphas, dim=1)
        return cu_hat_seq, du_hat_seq

"""Ablation A2 (revised): fully-shared encoder/decoder, no CU-DU type separation.

This replaces the earlier per-instance "no type sharing" variant. The earlier
variant gave each DU slot its own weights (max_n), which is not topology-agnostic
by construction, so it could not fairly test whether splitting CU and DU weights
matters. This variant is the fair test: it keeps Topoz's exact query->(1+N)
attention, LSTM, and hidden LayerNorm, and changes ONLY the type separation.

Differences from full Topoz (model.py):
  * ONE shared encoder W for the CU and every DU (instead of W_CU and W_DU).
  * ONE shared K, V (instead of K_CU/K_DU, V_CU/V_DU).
  * ONE shared decoder D (instead of D_CU, D_DU).
  * The CU's cu_dim features are scattered into the DU du_dim layout (the DU-only
    slots are zero-padded) so a single shared encoder can process both. The shared
    decoder predicts du_dim per entity; for the CU the real cu_dim positions are
    sliced back, so the loss and score touch only real features.

Topology-agnostic by construction: one shared weight set, no per-instance
parameters, attention over whatever entities are present, so it runs on an unseen
topology with a different N. This is the fully-shared design (the same family as
the DeepSets / Set Transformer baselines, but with Topoz's query attention).

CU 6-feature -> unified 29-slot index map (net_ratio dropped):
  DU layout: 0 cpu, 1 mem_pct, 2 mem_bytes, 3 fs_writes, 4 net_tx, 5 net_rx,
             6..27 PCI(22), 28 net_diff
  CU order:  cpu, mem_pct, mem_bytes, net_tx, net_rx, net_diff
  -> CU maps onto [0, 1, 2, 4, 5, du_dim-1]
"""
import math
from typing import Tuple

import torch
import torch.nn as nn


def _cu_idx_noratio(du_dim: int) -> torch.Tensor:
    # cpu->0, mem_pct->1, mem_bytes->2, net_tx->4, net_rx->5, net_diff->du_dim-1
    return torch.tensor([0, 1, 2, 4, 5, du_dim - 1], dtype=torch.long)


class FullySharedTopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d = embed_dim
        self.register_buffer("cu_idx", _cu_idx_noratio(du_dim))

        # ONE shared encoder for every entity (no type split, no type bias split).
        self.W = nn.Linear(du_dim, embed_dim, bias=False)
        self.e = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_in = nn.LayerNorm(embed_dim)

        # ONE shared K and V; single Q on the recurrent hidden state.
        self.K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)

        # ONE shared decoder: [h_norm ; entity_token] -> next-step (unified du_dim).
        self.D = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, du_dim),
        )

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.d, device=device)
        c = torch.zeros(batch_size, self.d, device=device)
        return h, c

    def _unify_cu(self, cu_t: torch.Tensor) -> torch.Tensor:
        """(B, cu_dim) -> (B, du_dim): CU features scattered, DU-only slots zero."""
        uni = cu_t.new_zeros(cu_t.shape[0], self.du_dim)
        uni[:, self.cu_idx] = cu_t
        return uni

    def step(self, cu_t, du_t, h, c):
        """One autoregressive step. cu_t (B, cu_dim), du_t (B, N, du_dim)."""
        B, N, _ = du_t.shape
        scale = 1.0 / math.sqrt(self.d)

        ent = torch.cat([self._unify_cu(cu_t).unsqueeze(1), du_t], dim=1)   # (B, 1+N, du_dim)
        tok = self.LN_in(self.W(ent) + self.e)                             # (B, 1+N, d)

        q = self.Q(h)                                                      # (B, d)
        K = self.K(tok)                                                    # (B, 1+N, d)
        V = self.V(tok)                                                    # (B, 1+N, d)
        scores = torch.einsum("bd,bmd->bm", q, K) * scale                 # (B, 1+N)
        alpha = torch.softmax(scores, dim=-1)                             # (B, 1+N)
        s = (alpha.unsqueeze(-1) * V).sum(dim=1)                          # (B, d)

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)
        h_exp = h_norm.unsqueeze(1).expand(-1, 1 + N, -1)                 # (B, 1+N, d)
        dec = self.D(torch.cat([h_exp, tok], dim=-1))                     # (B, 1+N, du_dim)
        cu_hat = dec[:, 0, :][:, self.cu_idx]                             # (B, cu_dim)
        du_hat = dec[:, 1:, :]                                            # (B, N, du_dim)
        return cu_hat, du_hat, h_new, c_new, alpha

    def project_tokens(self, cu, du):
        """Kept for API parity with TopoAR (closed-loop callers). Returns raw inputs."""
        return cu, du

    def forward(self, cu_seq, du_seq, return_alpha: bool = False):
        B, T, _ = cu_seq.shape
        h, c = self.init_state(B, cu_seq.device)
        cu_hats, du_hats, alphas = [], [], []
        for t in range(T):
            cu_hat, du_hat, h, c, alpha = self.step(
                cu_seq[:, t, :], du_seq[:, t, :, :], h, c
            )
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)
            if return_alpha:
                alphas.append(alpha)
        cu_hat_seq = torch.stack(cu_hats, dim=1)
        du_hat_seq = torch.stack(du_hats, dim=1)
        if return_alpha:
            return cu_hat_seq, du_hat_seq, torch.stack(alphas, dim=1)
        return cu_hat_seq, du_hat_seq

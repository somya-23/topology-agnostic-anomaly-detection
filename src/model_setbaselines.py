"""model_setbaselines.py — topology-agnostic, TYPE-BLIND set baselines for TopoAR.

Drop-in replacements for CalibratedTopoAR (same __init__(cu_dim, du_dim,
embed_dim) and forward(cu_seq, du_seq) -> (cu_hat, du_hat)), so the existing v0
pipeline runs them unchanged via a model swap. The ONLY differences from
TopoAR (everything else — preprocessing, next-step LSTM objective, lift scoring,
per-type p99.9 thresholds, LOO protocol — is held fixed by the pipeline):

  1. NO CU-DU type separation. TopoAR has two type-separate encoders
     (W_CU, W_DU) and decoders (D_CU, D_DU). These baselines use ONE shared
     encoder and ONE shared decoder for every entity.

  2. To feed CU (cu_dim=7) and DU (du_dim=30) through one shared encoder, every
     entity is unified to du_dim: the CU's 7 features are scattered into the DU
     30-slot layout and the DU-only slots (fs_writes + 22 PCI) are ZERO-PADDED.
     The shared decoder outputs du_dim per entity; for the CU we slice back the
     7 real positions, so the loss/score touch only real features (no masking
     machinery needed downstream).

  3. Entity aggregation replaces TopoAR's query->entities attention:
       DeepSetsBaseline      : mean-pool over entity value tokens (Zaheer 2017)
       SetTransformerBaseline: SAB self-attention across entities + PMA pooling
                               (Lee 2019)

Both are cardinality-invariant and topology-agnostic by construction: one shared
set of weights, no per-instance parameters, pooling/attention over whatever
entities are present, so they run on an unseen topology with a different N.

CU 7-feature -> unified 30-slot index map (DU layout:
  0 cpu, 1 mem_pct, 2 mem_bytes, 3 fs_writes, 4 net_tx, 5 net_rx,
  6..27 PCI(22), 28 net_diff, 29 net_ratio ;
 CU order: cpu, mem_pct, mem_bytes, net_tx, net_rx, net_diff, net_ratio).
"""
from typing import Tuple

import torch
import torch.nn as nn


def cu_to_uni_index(du_dim: int) -> list:
    # cpu->0, mem_pct->1, mem_bytes->2, net_tx->4, net_rx->5, net_diff->28, net_ratio->29
    return [0, 1, 2, 4, 5, du_dim - 2, du_dim - 1]


class _SetBaseline(nn.Module):
    AGG = None  # "deepsets" | "settransformer"

    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 32, n_heads: int = 4):
        super().__init__()
        self.cu_dim, self.du_dim, self.d = cu_dim, du_dim, embed_dim
        self.uni_dim = du_dim
        self.register_buffer("cu_idx", torch.tensor(cu_to_uni_index(du_dim), dtype=torch.long))

        # (1) ONE shared encoder for every entity — no type bias, no type split.
        self.W = nn.Linear(du_dim, embed_dim, bias=False)
        self.LN_in = nn.LayerNorm(embed_dim)

        # (3) aggregation-specific blocks
        if self.AGG == "settransformer":
            self.sab_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            self.LN_a = nn.LayerNorm(embed_dim)
            self.sab_ff = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim))
            self.LN_f = nn.LayerNorm(embed_dim)
            self.pma_seed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.pma_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        else:  # deepsets
            self.V = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)
        # ONE shared decoder: [h_norm ; entity_token] -> next-step (unified du_dim)
        self.D = nn.Sequential(nn.Linear(2 * embed_dim, embed_dim), nn.GELU(),
                               nn.Linear(embed_dim, du_dim))

    def init_state(self, B: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(B, self.d, device=device),
                torch.zeros(B, self.d, device=device))

    def _unify_cu(self, cu_t: torch.Tensor) -> torch.Tensor:
        """(B, cu_dim) -> (B, du_dim), CU features scattered, rest zero-padded."""
        uni = cu_t.new_zeros(cu_t.shape[0], self.uni_dim)
        uni[:, self.cu_idx] = cu_t
        return uni

    def step(self, cu_t, du_t, h, c):
        B, N, _ = du_t.shape
        ent = torch.cat([self._unify_cu(cu_t).unsqueeze(1), du_t], dim=1)   # (B, 1+N, du_dim)
        tok = self.LN_in(self.W(ent))                                       # (B, 1+N, d) shared encoder

        if self.AGG == "settransformer":
            a, _ = self.sab_attn(tok, tok, tok)              # self-attention across entities
            tok = self.LN_a(tok + a)
            tok = self.LN_f(tok + self.sab_ff(tok))          # SAB output (per-entity, context-mixed)
            seed = self.pma_seed.expand(B, -1, -1)           # (B, 1, d)
            s, _ = self.pma_attn(seed, tok, tok)             # PMA pooling -> (B, 1, d)
            s = s.squeeze(1)
        else:  # deepsets: shared value projection then mean-pool
            s = self.V(tok).mean(dim=1)                       # (B, d)

        h, c = self.lstm(s, (h, c))
        h_norm = self.LN_h(h)
        h_exp = h_norm.unsqueeze(1).expand(-1, 1 + N, -1)    # (B, 1+N, d)
        dec = self.D(torch.cat([h_exp, tok], dim=-1))        # (B, 1+N, du_dim)
        cu_hat = dec[:, 0, :][:, self.cu_idx]                 # (B, cu_dim) real CU features
        du_hat = dec[:, 1:, :]                                # (B, N, du_dim)
        return cu_hat, du_hat, h, c

    def forward(self, cu_seq, du_seq, return_alpha: bool = False):
        B, T, _ = cu_seq.shape
        h, c = self.init_state(B, cu_seq.device)
        cu_hats, du_hats = [], []
        for t in range(T):
            cu_hat, du_hat, h, c = self.step(cu_seq[:, t, :], du_seq[:, t, :, :], h, c)
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)
        cu_hat_seq = torch.stack(cu_hats, dim=1)             # (B, T, cu_dim)
        du_hat_seq = torch.stack(du_hats, dim=1)             # (B, T, N, du_dim)
        if return_alpha:
            return cu_hat_seq, du_hat_seq, None
        return cu_hat_seq, du_hat_seq


class DeepSetsBaseline(_SetBaseline):
    """Shared encoder + mean-pool over entities (DeepSets, Zaheer et al. 2017)."""
    AGG = "deepsets"


class SetTransformerBaseline(_SetBaseline):
    """Shared encoder + SAB self-attention across entities + PMA pooling
    (Set Transformer, Lee et al. 2019)."""
    AGG = "settransformer"

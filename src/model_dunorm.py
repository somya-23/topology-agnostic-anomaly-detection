"""DUNormTopoAR — learned data-driven /N substitute via attention-fenced extra DU dim.

Replaces the hand-coded /N_DU on CU net_tx / net_rx with a model-learned per-DU
weight, while keeping the attention mechanism unchanged.

Architecture delta vs TopoAR (src/model.py):

  W_DU widened: nn.Linear(du_dim, embed_dim + 1)
    - First `d` dims  → after type bias + LN, fed into K_DU / V_DU / attention.
    - Last  1  dim    → softplus → strictly positive per-DU scalar `extra_i`.
                        Summed across DUs → sum_extra = Σ_i softplus(extra_i)  (B,).
                        Used to normalize the CU's raw net_tx / net_rx INSIDE the
                        model, producing two additional input features for W_CU:
                            net_tx_norm = net_tx / sum_extra
                            net_rx_norm = net_rx / sum_extra

  W_CU widened on the INPUT side: nn.Linear(cu_dim + 2, embed_dim)
    Receives the raw CU features concatenated with the two sum_extra-normalized
    copies of net_tx, net_rx. Embedding dimension `d` is unchanged.

The decoder D_CU still outputs `cu_dim` (the preprocessed input space — 6 for
the dunorm variant: cpu, mem_pct, mem_bytes, net_tx (raw), net_rx (raw),
net_ratio). The 2 normalized features are derived inside the model from those
6 + sum_extra, so they need no separate prediction target.

Topology-agnostic guarantees:
  * The extra dim is type-shared (one column of W_DU); not per-instance.
  * sum_extra is permutation-invariant in DUs (softplus + sum).
  * Attention sees only the first d dims of du_tok by construction — the +1
    dim is never indexed by K_DU / V_DU / Q.
  * Softplus + clamp(min=1e-6) on the denominator guards against division blow-up.

API matches TopoAR:
  init_state(B, device) → (h, c)
  project_tokens(cu, du) → (cu_tok, du_tok)
  step(cu_tok, du_tok, h, c) → (cu_hat, du_hat, h_new, c_new, alpha)
  forward(cu_seq, du_seq) → (cu_hat, du_hat)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DUNormTopoAR(nn.Module):
    # CU input column indices (after slice + derived) — used to pick raw net_tx/rx
    # for the sum_extra-normalized copies. Must match slice_features ordering:
    #   0=cpu  1=mem_pct  2=mem_bytes  3=net_tx  4=net_rx  5=net_ratio
    NET_TX_IDX = 3
    NET_RX_IDX = 4

    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d      = embed_dim

        # W_CU receives raw CU + 2 normalized (net_tx/sum_extra, net_rx/sum_extra).
        self.W_CU = nn.Linear(cu_dim + 2, embed_dim, bias=False)
        # W_DU widened by 1 — first d dims attend; last 1 dim is the extra scalar.
        self.W_DU = nn.Linear(du_dim, embed_dim + 1, bias=False)

        self.e_CU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Attention operates entirely in embed_dim — the +1 dim is fenced off.
        self.K_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q    = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)

        # Decoders predict the preprocessed input space (cu_dim / du_dim).
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
        """cu (B, cu_dim) → cu_tok (B, d);  du (B, N, du_dim) → du_tok (B, N, d).

        Order (DU must be projected first because sum_extra modifies the CU input):
          1. du_proj = W_DU(du) → (B, N, d+1).
             Split: du_attn (B, N, d) and du_extra_raw (B, N).
             Softplus on the extra → positive du_extra (B, N).
             Sum across DUs → sum_extra (B,). Clamp ≥ 1e-6 for safe division.
          2. Build cu_aug = concat(cu, net_tx/sum_extra, net_rx/sum_extra)  (B, cu_dim+2).
          3. cu_tok = LN_CU(W_CU(cu_aug) + e_CU).
          4. du_tok = LN_DU(du_attn + e_DU)  — only the d-dim slice gets the
                                                 type bias + LN. The +1 dim was
                                                 used only to form sum_extra.
        """
        # 1. DU projection — produces attention features AND the extra scalar.
        du_proj      = self.W_DU(du)                         # (B, N, d+1)
        du_attn      = du_proj[..., :self.d]                 # (B, N, d)
        du_extra_raw = du_proj[..., self.d]                  # (B, N)
        du_extra     = F.softplus(du_extra_raw)              # (B, N) — strictly > 0
        sum_extra    = du_extra.sum(dim=1)                   # (B,)
        denom        = sum_extra.clamp(min=1e-6).unsqueeze(-1)   # (B, 1)

        # 2. CU input augmentation — sum_extra-normalized copies of net_tx, net_rx.
        net_tx = cu[:, self.NET_TX_IDX:self.NET_TX_IDX + 1]  # (B, 1)
        net_rx = cu[:, self.NET_RX_IDX:self.NET_RX_IDX + 1]  # (B, 1)
        cu_aug = torch.cat([cu, net_tx / denom, net_rx / denom], dim=-1)  # (B, cu_dim+2)

        # 3 & 4. Standard type-bias + LN.
        cu_tok = self.LN_CU(self.W_CU(cu_aug) + self.e_CU)   # (B, d)
        du_tok = self.LN_DU(du_attn + self.e_DU)             # (B, N, d)
        return cu_tok, du_tok

    def step(
        self,
        cu_tok: torch.Tensor,   # (B, d)
        du_tok: torch.Tensor,   # (B, N, d)
        h: torch.Tensor,        # (B, d)
        c: torch.Tensor,        # (B, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One autoregressive step. Identical to TopoAR.step() — the dunorm
        modification lives entirely in project_tokens(); attention/LSTM/decoders
        are unchanged."""
        B, N, d = du_tok.shape
        scale = 1.0 / math.sqrt(d)

        q     = self.Q(h)
        K_cu  = self.K_CU(cu_tok)
        K_du  = self.K_DU(du_tok)
        V_cu  = self.V_CU(cu_tok)
        V_du  = self.V_DU(du_tok)

        score_cu = (q * K_cu).sum(dim=-1, keepdim=True) * scale         # (B, 1)
        score_du = torch.einsum("bd,bnd->bn", q, K_du) * scale          # (B, N)
        alpha    = torch.softmax(torch.cat([score_cu, score_du], dim=1), dim=-1)

        s_cu = alpha[:, 0:1] * V_cu
        s_du = (alpha[:, 1:].unsqueeze(-1) * V_du).sum(dim=1)
        s    = s_cu + s_du

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)

        cu_hat   = self.D_CU(torch.cat([h_norm, cu_tok], dim=-1))
        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)
        du_hat   = self.D_DU(torch.cat([h_norm_b, du_tok], dim=-1))

        return cu_hat, du_hat, h_new, c_new, alpha

    def forward(
        self,
        cu_seq: torch.Tensor,   # (B, T, cu_dim)
        du_seq: torch.Tensor,   # (B, T, N, du_dim)
        return_alpha: bool = False,
    ):
        B, T, _ = cu_seq.shape
        device   = cu_seq.device

        h, c = self.init_state(B, device)
        cu_hats, du_hats, alphas = [], [], []
        for t in range(T):
            cu_tok, du_tok = self.project_tokens(cu_seq[:, t], du_seq[:, t])
            cu_hat, du_hat, h, c, alpha = self.step(cu_tok, du_tok, h, c)
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)
            if return_alpha:
                alphas.append(alpha)

        cu_hat_seq = torch.stack(cu_hats, dim=1)
        du_hat_seq = torch.stack(du_hats, dim=1)
        if return_alpha:
            return cu_hat_seq, du_hat_seq, torch.stack(alphas, dim=1)
        return cu_hat_seq, du_hat_seq

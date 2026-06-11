"""A5: No hidden-state LayerNorm — remove self.LN_h, use raw h_new in decoders.

Only change from TopoAR: LN_h removed; step() uses h_new directly instead of
LN_h(h_new). Everything else (projections, attention, LSTM, decoders) is identical.

LN_h was justified as "bounds composition drift across N regimes" — removing it
tests whether the LayerNorm is actually needed for topology-agnostic stability.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class NoHiddenLNTopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d = embed_dim

        # Identical to TopoAR except LN_h is absent
        self.W_CU = nn.Linear(cu_dim, embed_dim, bias=False)
        self.W_DU = nn.Linear(du_dim, embed_dim, bias=False)
        self.e_CU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        self.K_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q   = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        # LN_h intentionally absent

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

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, self.d, device=device),
                torch.zeros(batch_size, self.d, device=device))

    def project_tokens(
        self, cu: torch.Tensor, du: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_tok = self.LN_CU(self.W_CU(cu) + self.e_CU)
        du_tok = self.LN_DU(self.W_DU(du) + self.e_DU)
        return cu_tok, du_tok

    def step(
        self,
        cu_tok: torch.Tensor,
        du_tok: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, d = du_tok.shape
        scale = 1.0 / math.sqrt(d)

        q    = self.Q(h)
        K_cu = self.K_CU(cu_tok)
        K_du = self.K_DU(du_tok)
        V_cu = self.V_CU(cu_tok)
        V_du = self.V_DU(du_tok)

        score_cu = (q * K_cu).sum(dim=-1, keepdim=True) * scale
        score_du = torch.einsum("bd,bnd->bn", q, K_du) * scale
        alpha    = torch.softmax(torch.cat([score_cu, score_du], dim=1), dim=-1)

        s_cu = alpha[:, 0:1] * V_cu
        s_du = (alpha[:, 1:].unsqueeze(-1) * V_du).sum(dim=1)
        s    = s_cu + s_du

        h_new, c_new = self.lstm(s, (h, c))
        # ABLATION: use h_new directly — LN_h removed
        h_norm = h_new

        cu_hat = self.D_CU(torch.cat([h_norm, cu_tok], dim=-1))
        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)
        du_hat = self.D_DU(torch.cat([h_norm_b, du_tok], dim=-1))

        return cu_hat, du_hat, h_new, c_new, alpha

    def forward(
        self,
        cu_seq: torch.Tensor,
        du_seq: torch.Tensor,
        return_alpha: bool = False,
    ):
        B, T, _ = cu_seq.shape
        N = du_seq.shape[2]
        h, c = self.init_state(B, cu_seq.device)
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


AblationModel = NoHiddenLNTopoAR

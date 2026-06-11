"""A2: No type-sharing — per-instance W_DU, K_DU, V_DU, D_DU via ModuleList.

Only change from TopoAR: W_DU, K_DU, V_DU, D_DU are per-DU-instance ModuleLists
instead of shared across all DUs. All other modules (W_CU, K_CU, V_CU, Q, D_CU,
LSTMCell, LN_CU, LN_DU, LN_h, e_CU, e_DU) remain shared.

With max_n=3, DU indices 0,1,2 each have their own projection weights.
When tested on a topology with a DU index never seen during training (e.g., DU_2
when only DU_0 and DU_1 were in train), those weights are at random init →
catastrophic performance, proving type-sharing is essential for zero-shot transfer.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

MAX_N = 3  # max DU count across all topologies in the experiment


class NoTypeSharingTopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64, max_n: int = MAX_N):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d = embed_dim
        self.max_n = max_n

        # Shared CU projections (unchanged from TopoAR)
        self.W_CU = nn.Linear(cu_dim, embed_dim, bias=False)
        self.e_CU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)

        # Shared type embedding + LN for DU (only projections are per-instance)
        self.e_DU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Per-instance DU modules — the ONLY change from TopoAR
        self.W_DU_list = nn.ModuleList(
            [nn.Linear(du_dim, embed_dim, bias=False) for _ in range(max_n)]
        )
        self.K_DU_list = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(max_n)]
        )
        self.V_DU_list = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(max_n)]
        )
        self.D_DU_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, du_dim),
            )
            for _ in range(max_n)
        ])

        # Shared CU attention + recurrent modules (unchanged from TopoAR)
        self.K_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)
        self.D_CU = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, cu_dim),
        )

    def init_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, self.d, device=device),
                torch.zeros(batch_size, self.d, device=device))

    def project_tokens(
        self, cu: torch.Tensor, du: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cu_tok = self.LN_CU(self.W_CU(cu) + self.e_CU)           # (B, d)
        N = du.shape[1]
        du_toks = [
            self.LN_DU(self.W_DU_list[i](du[:, i, :]) + self.e_DU)
            for i in range(N)
        ]
        du_tok = torch.stack(du_toks, dim=1)                       # (B, N, d)
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
        K_du = torch.stack([self.K_DU_list[i](du_tok[:, i, :]) for i in range(N)], dim=1)
        V_cu = self.V_CU(cu_tok)
        V_du = torch.stack([self.V_DU_list[i](du_tok[:, i, :]) for i in range(N)], dim=1)

        score_cu = (q * K_cu).sum(dim=-1, keepdim=True) * scale
        score_du = torch.einsum("bd,bnd->bn", q, K_du) * scale
        alpha    = torch.softmax(torch.cat([score_cu, score_du], dim=1), dim=-1)

        s_cu = alpha[:, 0:1] * V_cu
        s_du = (alpha[:, 1:].unsqueeze(-1) * V_du).sum(dim=1)
        s    = s_cu + s_du

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)

        cu_hat = self.D_CU(torch.cat([h_norm, cu_tok], dim=-1))

        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)
        du_hat = torch.stack(
            [self.D_DU_list[i](torch.cat([h_norm, du_tok[:, i, :]], dim=-1)) for i in range(N)],
            dim=1,
        )
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


AblationModel = NoTypeSharingTopoAR

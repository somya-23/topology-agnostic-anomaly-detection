"""A3: Mean pooling instead of learned attention.

Only change from TopoAR: Q, K_CU, K_DU removed; aggregation replaced with
uniform mean over all entity value-tokens (1/(1+N) weight each).
LSTM, decoders, token projections, LN_h all unchanged.
"""

from typing import Tuple

import torch
import torch.nn as nn


class MeanPoolTopoAR(nn.Module):
    """TopoAR with softmax attention replaced by uniform mean pooling.

    Interface identical to TopoAR: project_tokens / step / forward / init_state.
    """

    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d = embed_dim

        # Input projections (identical to TopoAR)
        self.W_CU = nn.Linear(cu_dim, embed_dim, bias=False)
        self.W_DU = nn.Linear(du_dim, embed_dim, bias=False)
        self.e_CU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Value projections (Q, K_CU, K_DU removed — not needed for mean pooling)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)

        # Recurrent + decoder (identical to TopoAR)
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)
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
        cu_tok = self.LN_CU(self.W_CU(cu) + self.e_CU)   # (B, d)
        du_tok = self.LN_DU(self.W_DU(du) + self.e_DU)   # (B, N, d)
        return cu_tok, du_tok

    def step(
        self,
        cu_tok: torch.Tensor,
        du_tok: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        B, N, d = du_tok.shape

        V_cu = self.V_CU(cu_tok)    # (B, d)
        V_du = self.V_DU(du_tok)    # (B, N, d)

        # Uniform mean: weight 1/(1+N) for each entity
        all_vals = torch.cat([V_cu.unsqueeze(1), V_du], dim=1)  # (B, 1+N, d)
        s = all_vals.mean(dim=1)                                  # (B, d)

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)

        cu_hat = self.D_CU(torch.cat([h_norm, cu_tok], dim=-1))

        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)
        du_hat = self.D_DU(torch.cat([h_norm_b, du_tok], dim=-1))

        return cu_hat, du_hat, h_new, c_new, None  # alpha=None (no attention)

    def forward(
        self,
        cu_seq: torch.Tensor,
        du_seq: torch.Tensor,
        return_alpha: bool = False,
    ):
        B, T, _ = cu_seq.shape
        N = du_seq.shape[2]
        h, c = self.init_state(B, cu_seq.device)
        cu_hats, du_hats = [], []
        for t in range(T):
            cu_tok, du_tok = self.project_tokens(cu_seq[:, t], du_seq[:, t])
            cu_hat, du_hat, h, c, _ = self.step(cu_tok, du_tok, h, c)
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)
        return torch.stack(cu_hats, dim=1), torch.stack(du_hats, dim=1)


AblationModel = MeanPoolTopoAR

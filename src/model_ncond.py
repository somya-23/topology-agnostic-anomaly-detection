"""N-conditioned TopoAR: linear FiLM conditioning on N_DU.

Motivation
----------
Without any preprocessing (no /N_DU on net_tx/rx), the CU's net features scale
linearly with N_DU because the CU forwards traffic for all N DUs. A model trained
on N=2 and N=3 sees data in one absolute scale range; at inference on N=1 the
input lands outside the training distribution.

FiLM conditioning (Perez et al., 2018) gives the model an explicit channel to
receive and act on N_DU. For each token embedding, the model applies:

    cu_tok = LN_CU(W_CU · cu + e_CU) * (1 + γ_cu(N)) + β_cu(N)
    du_tok = LN_DU(W_DU · du + e_DU) * (1 + γ_du(N)) + β_du(N)

where (γ, β) are learned LINEAR functions of N (no nonlinearity):

    [γ_cu, β_cu, γ_du, β_du] = W_film · N + b_film    shape (B, 4d)

Using a purely linear map in N is critical for generalization to unseen N:
  - Trained on N ∈ {2, 3}, two points uniquely determine a line.
  - At N=1 the model extrapolates on the same line — exactly as baseline(N) does
    for mean subtraction, but here the model learns ALL per-feature scale and
    shift parameters directly from reconstruction loss.
  - Any nonlinearity (Tanh, ReLU) would break this extrapolation guarantee.

The rest of the architecture is identical to TopoAR (model.py): same attention
mechanism, same LSTM, same decoders. Only the token embeddings are FiLM-modulated.

API compatibility with model.py:
  - project_tokens(cu, du): N_DU inferred from du.shape[1] — same call signature.
  - step(cu_tok, du_tok, h, c): returns (cu_hat, du_hat, h, c, alpha).
  - forward(cu_seq, du_seq): same signature and return shapes.
  - init_state(batch, device): identical.
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn


class NCondTopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d      = embed_dim

        # FiLM parameter generator: scalar N_DU → (4 * embed_dim,) params.
        # Purely linear — no activation — guarantees linear extrapolation to unseen N.
        # Output layout: [γ_cu (d), β_cu (d), γ_du (d), β_du (d)].
        self.n_film = nn.Linear(1, 4 * embed_dim)

        # Input projections (same as TopoAR).
        self.W_CU  = nn.Linear(cu_dim, embed_dim, bias=False)
        self.W_DU  = nn.Linear(du_dim, embed_dim, bias=False)
        self.e_CU  = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU  = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Attention (same as TopoAR).
        self.K_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q    = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)

        # Decoders (same as TopoAR).
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

    def _film_params(
        self, n_du: int, B: int, device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-feature FiLM (γ, β) for CU and DU from scalar N_DU.

        Returns: gamma_cu, beta_cu, gamma_du, beta_du — each shape (B, d).
        The linear map ensures the model extrapolates correctly to unseen N.
        """
        n_t   = torch.full((B, 1), float(n_du), device=device)
        params = self.n_film(n_t)                                   # (B, 4d)
        gamma_cu, beta_cu, gamma_du, beta_du = params.chunk(4, dim=-1)
        return gamma_cu, beta_cu, gamma_du, beta_du

    def init_state(
        self, batch_size: int, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.d, device=device)
        c = torch.zeros(batch_size, self.d, device=device)
        return h, c

    def project_tokens(
        self, cu: torch.Tensor, du: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cu (B, cu_dim), du (B, N, du_dim) → cu_tok (B, d), du_tok (B, N, d).

        N_DU is inferred from du.shape[1] — call signature identical to TopoAR.
        FiLM is applied after LayerNorm so the base token is topology-independent;
        only the scale/shift are N-dependent.
        """
        B, N, _ = du.shape
        gamma_cu, beta_cu, gamma_du, beta_du = self._film_params(N, B, cu.device)

        # CU token: project → LN → FiLM
        cu_base = self.LN_CU(self.W_CU(cu) + self.e_CU)            # (B, d)
        cu_tok  = cu_base * (1.0 + gamma_cu) + beta_cu              # (B, d)

        # DU token: project → LN → FiLM (γ/β broadcast over N)
        du_base = self.LN_DU(self.W_DU(du) + self.e_DU)            # (B, N, d)
        gamma_du_b = gamma_du.unsqueeze(1)                          # (B, 1, d)
        beta_du_b  = beta_du.unsqueeze(1)                           # (B, 1, d)
        du_tok = du_base * (1.0 + gamma_du_b) + beta_du_b          # (B, N, d)

        return cu_tok, du_tok

    def step(
        self,
        cu_tok: torch.Tensor,   # (B, d)  — already FiLM-modulated
        du_tok: torch.Tensor,   # (B, N, d)
        h: torch.Tensor,        # (B, d)
        c: torch.Tensor,        # (B, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One autoregressive step — identical to TopoAR.step() post-projection.

        Returns (cu_hat, du_hat, h_new, c_new, alpha).
        """
        B, N, d = du_tok.shape
        scale   = 1.0 / math.sqrt(d)

        q     = self.Q(h)                                               # (B, d)
        K_cu  = self.K_CU(cu_tok)                                       # (B, d)
        K_du  = self.K_DU(du_tok)                                       # (B, N, d)
        V_cu  = self.V_CU(cu_tok)                                       # (B, d)
        V_du  = self.V_DU(du_tok)                                       # (B, N, d)

        score_cu = (q * K_cu).sum(dim=-1, keepdim=True) * scale        # (B, 1)
        score_du = torch.einsum("bd,bnd->bn", q, K_du) * scale         # (B, N)
        alpha    = torch.softmax(
            torch.cat([score_cu, score_du], dim=1), dim=-1
        )                                                               # (B, 1+N)

        s_cu  = alpha[:, 0:1] * V_cu
        s_du  = (alpha[:, 1:].unsqueeze(-1) * V_du).sum(dim=1)
        s     = s_cu + s_du                                             # (B, d)

        h_new, c_new = self.lstm(s, (h, c))
        h_norm = self.LN_h(h_new)

        cu_hat   = self.D_CU(torch.cat([h_norm, cu_tok], dim=-1))              # (B, cu_dim)
        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)
        du_hat   = self.D_DU(torch.cat([h_norm_b, du_tok], dim=-1))           # (B, N, du_dim)

        return cu_hat, du_hat, h_new, c_new, alpha

    def forward(
        self,
        cu_seq: torch.Tensor,   # (B, T, cu_dim)
        du_seq: torch.Tensor,   # (B, T, N, du_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full sequence forward. N_DU inferred from du_seq.shape[2] per step."""
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

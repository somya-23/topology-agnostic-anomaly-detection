"""TopoAR baseline: type-shared autoregressive multi-entity attention.

Forward at step t given:
    cu  shape (B, cu_dim)         the CU's feature vector at time t
    du  shape (B, N, du_dim)      the N DU feature vectors at time t
    h   shape (B, d)              previous LSTM hidden state
    c   shape (B, d)              previous LSTM cell state

Computation (multi-key softmax attention over 1+N keys, then LSTM update):

    cu_tok = LN_CU(W_CU · cu + e_CU)                              (B, d)
    du_tok = LN_DU(W_DU · du[:,i,:] + e_DU) for each i             (B, N, d)
    q      = Q · h                                                (B, d)
    K_cu   = K_CU(cu_tok)                                         (B, d)
    K_du   = K_DU(du_tok)                                         (B, N, d)
    V_cu   = V_CU(cu_tok)                                         (B, d)
    V_du   = V_DU(du_tok)                                         (B, N, d)
    scores = stack([q·K_cu, q·K_du[:,0], ..., q·K_du[:,N-1]]) /√d (B, 1+N)
    α      = softmax(scores, dim=-1)                              (B, 1+N)
    s      = α[:,0]·V_cu + Σ_i α[:,1+i]·V_du[:,i]                 (B, d)
    h, c   = LSTMCell(s, (h, c))                                  (B, d), (B, d)
    h_norm = LayerNorm(h)
    cu_hat   = D_CU([h_norm ; cu_tok])                            (B, cu_dim)
    du_hat_i = D_DU([h_norm ; du_tok[:,i,:]])                     (B, N, du_dim)

Topology agnosticism:
    * No per-instance parameters anywhere — W_CU/W_DU/K_CU/K_DU/V_CU/V_DU/Q,
      LSTMCell, LN_CU/LN_DU/LN_h, D_CU/D_DU are all type-shared.
    * Softmax is a convex combination → ‖s‖ ≤ max ‖V_j‖ regardless of N.
    * LayerNorm on h_t bounds composition drift across N regimes.

The full sequence forward is a Python loop over T (autoregressive); we cannot
parallelize across t because h_t depends on h_{t-1}. Per step the batch is
fully vectorized (matmuls do all B and N work at once).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class TopoAR(nn.Module):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 64):
        super().__init__()
        self.cu_dim = cu_dim
        self.du_dim = du_dim
        self.d = embed_dim

        # Type-shared input projections + type embeddings + LayerNorms.
        self.W_CU = nn.Linear(cu_dim, embed_dim, bias=False)
        self.W_DU = nn.Linear(du_dim, embed_dim, bias=False)
        self.e_CU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.e_DU = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.LN_CU = nn.LayerNorm(embed_dim)
        self.LN_DU = nn.LayerNorm(embed_dim)

        # Per-type K and V; single Q (acts on the recurrent hidden state).
        self.K_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_CU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_DU = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Q   = nn.Linear(embed_dim, embed_dim, bias=False)

        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.LN_h = nn.LayerNorm(embed_dim)

        # Per-entity decoders see [h_norm ; entity_token].
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
        h = torch.zeros(batch_size, self.d, device=device)
        c = torch.zeros(batch_size, self.d, device=device)
        return h, c

    def project_tokens(
        self, cu: torch.Tensor, du: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """cu -> (B, d), du -> (B, N, d), with type bias + LayerNorm applied."""
        cu_tok = self.LN_CU(self.W_CU(cu) + self.e_CU)              # (B, d)
        du_tok = self.LN_DU(self.W_DU(du) + self.e_DU)              # (B, N, d)
        return cu_tok, du_tok

    def step(
        self,
        cu_tok: torch.Tensor,                                       # (B, d)
        du_tok: torch.Tensor,                                       # (B, N, d)
        h: torch.Tensor,                                            # (B, d)
        c: torch.Tensor,                                            # (B, d)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One autoregressive step. Returns (cu_hat, du_hat, h_new, c_new, alpha).

        cu_hat:  (B, cu_dim)            next-step CU prediction
        du_hat:  (B, N, du_dim)         next-step DU predictions (per-entity)
        h_new:   (B, d)                 updated LSTM hidden state
        c_new:   (B, d)                 updated LSTM cell state
        alpha:   (B, 1+N)               attention weights (key 0 = CU, 1..N = DUs)
        """
        B, N, d = du_tok.shape
        scale = 1.0 / math.sqrt(d)

        q = self.Q(h)                                               # (B, d)
        K_cu = self.K_CU(cu_tok)                                    # (B, d)
        K_du = self.K_DU(du_tok)                                    # (B, N, d)
        V_cu = self.V_CU(cu_tok)                                    # (B, d)
        V_du = self.V_DU(du_tok)                                    # (B, N, d)

        # Multi-key scores: q · K_j for each key j ∈ {cu, du_0, ..., du_{N-1}}.
        score_cu = (q * K_cu).sum(dim=-1, keepdim=True) * scale     # (B, 1)
        score_du = torch.einsum("bd,bnd->bn", q, K_du) * scale      # (B, N)
        scores = torch.cat([score_cu, score_du], dim=1)             # (B, 1+N)
        alpha = torch.softmax(scores, dim=-1)                       # (B, 1+N)

        # s = Σ α_j · V_j
        s_cu = alpha[:, 0:1] * V_cu                                 # (B, d)
        s_du = (alpha[:, 1:].unsqueeze(-1) * V_du).sum(dim=1)       # (B, d)
        s = s_cu + s_du                                             # (B, d)

        h_new, c_new = self.lstm(s, (h, c))                         # (B, d), (B, d)
        h_norm = self.LN_h(h_new)                                   # (B, d)

        # Per-entity decoders see [h_norm ; entity_token].
        cu_in = torch.cat([h_norm, cu_tok], dim=-1)                 # (B, 2d)
        cu_hat = self.D_CU(cu_in)                                   # (B, cu_dim)

        h_norm_b = h_norm.unsqueeze(1).expand(-1, N, -1)            # (B, N, d)
        du_in = torch.cat([h_norm_b, du_tok], dim=-1)               # (B, N, 2d)
        du_hat = self.D_DU(du_in)                                   # (B, N, du_dim)

        return cu_hat, du_hat, h_new, c_new, alpha

    def forward(
        self,
        cu_seq: torch.Tensor,                                       # (B, T, cu_dim)
        du_seq: torch.Tensor,                                       # (B, T, N, du_dim)
        return_alpha: bool = False,
    ):
        """Run the full sequence. Predictions at step t are for time t+1.

        Returns:
            cu_hat:  (B, T, cu_dim)         predictions x̂_{t+1, CU}
            du_hat:  (B, T, N, du_dim)      predictions x̂_{t+1, DU_i}
            (alpha): (B, T, 1+N) if return_alpha=True
        """
        B, T, _ = cu_seq.shape
        N = du_seq.shape[2]
        device = cu_seq.device

        h, c = self.init_state(B, device)
        cu_hats, du_hats, alphas = [], [], []

        for t in range(T):
            cu_t = cu_seq[:, t, :]
            du_t = du_seq[:, t, :, :]
            cu_tok, du_tok = self.project_tokens(cu_t, du_t)
            cu_hat, du_hat, h, c, alpha = self.step(cu_tok, du_tok, h, c)
            cu_hats.append(cu_hat)
            du_hats.append(du_hat)
            if return_alpha:
                alphas.append(alpha)

        cu_hat_seq = torch.stack(cu_hats, dim=1)                    # (B, T, cu_dim)
        du_hat_seq = torch.stack(du_hats, dim=1)                    # (B, T, N, du_dim)
        if return_alpha:
            return cu_hat_seq, du_hat_seq, torch.stack(alphas, dim=1)
        return cu_hat_seq, du_hat_seq

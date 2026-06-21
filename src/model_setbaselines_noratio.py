"""model_setbaselines_noratio.py — set baselines for the net_ratio-DROPPED feature set.

Identical to model_setbaselines.DeepSetsBaseline / SetTransformerBaseline, except
the CU->unified index map is rebuilt for the case where the net_ratio derived
feature has been removed from both CU and DU.

After dropping net_ratio:
  CU order (6): cpu, mem_pct, mem_bytes, net_tx, net_rx, net_diff
  unified DU layout (29): 0 cpu, 1 mem_pct, 2 mem_bytes, 3 fs_writes, 4 net_tx,
                          5 net_rx, 6..27 PCI(22), 28 net_diff
  -> CU maps onto [0, 1, 2, 4, 5, du_dim-1]   (net_diff is the last slot now)

The parent classes build cu_idx for the 7-feature layout in __init__; here we
overwrite that buffer with the 6-feature mapping.
"""
import torch

from model_setbaselines import DeepSetsBaseline, SetTransformerBaseline


def _cu_idx_noratio(du_dim: int) -> torch.Tensor:
    # cpu->0, mem_pct->1, mem_bytes->2, net_tx->4, net_rx->5, net_diff->28(=du_dim-1)
    return torch.tensor([0, 1, 2, 4, 5, du_dim - 1], dtype=torch.long)


class DeepSetsBaselineNoRatio(DeepSetsBaseline):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 32, n_heads: int = 4):
        super().__init__(cu_dim, du_dim, embed_dim, n_heads)
        self.cu_idx = _cu_idx_noratio(du_dim)


class SetTransformerBaselineNoRatio(SetTransformerBaseline):
    def __init__(self, cu_dim: int, du_dim: int, embed_dim: int = 32, n_heads: int = 4):
        super().__init__(cu_dim, du_dim, embed_dim, n_heads)
        self.cu_idx = _cu_idx_noratio(du_dim)

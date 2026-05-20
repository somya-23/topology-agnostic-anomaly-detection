"""Sequence dataset + multi-topology batch sampler for TopoAR.

The sequence dataset slices each topology stream into fixed-length windows of
WINDOW_LEN timesteps. Windows respect block_id boundaries — a window is
emitted only if all WINDOW_LEN consecutive rows belong to the same block.
That way the LSTM never has to bridge a stress-state change inside a window,
which keeps next-step targets coherent.

The multi-topology sampler enforces the user-confirmed Exp C rule: each batch
is drawn entirely from one topology, so all rows in the batch share N. The
order of batches is shuffled across topologies per epoch — the optimizer sees
N=1 batches and N=2 batches alternating, and learns weights robust to either.
We never feed a topology-id to the model.
"""

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class TopologySequenceDataset(Dataset):
    """Fixed-length window dataset for a single topology.

    Args:
        cu_s:        (T, cu_dim) scaled CU stream
        du_s:        (T, N, du_dim) scaled DU stream
        block_id:    (T,) int — sliding windows must stay inside one block
        window_len:  number of timesteps per training window
        stride:      step size between window starts (default 1 = max overlap)

    Each item is a window of WINDOW_LEN timesteps. The training loss applies
    teacher forcing — at step t the model sees real x_t and predicts x_{t+1};
    therefore each window provides WINDOW_LEN-1 effective prediction targets.
    """

    def __init__(
        self,
        cu_s: np.ndarray,
        du_s: np.ndarray,
        block_id: np.ndarray,
        window_len: int = 64,
        stride: int = 1,
    ):
        assert cu_s.shape[0] == du_s.shape[0] == block_id.shape[0]
        self.cu = cu_s.astype(np.float32)
        self.du = du_s.astype(np.float32)
        self.block_id = block_id
        self.window_len = window_len
        self.N = du_s.shape[1]
        self.cu_dim = cu_s.shape[1]
        self.du_dim = du_s.shape[2]

        T = len(cu_s)
        # Valid window start: a window [start, start+L) where all block_ids match.
        starts = []
        s = 0
        while s + window_len <= T:
            seg_block = block_id[s : s + window_len]
            if (seg_block == seg_block[0]).all():
                starts.append(s)
                s += stride
            else:
                # advance to the next block boundary so we don't waste steps
                first_change = np.argmax(seg_block != seg_block[0])
                s = s + first_change
        self.starts = np.array(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = int(self.starts[idx])
        e = s + self.window_len
        return {
            "cu": torch.from_numpy(self.cu[s:e]),                   # (L, cu_dim)
            "du": torch.from_numpy(self.du[s:e]),                   # (L, N, du_dim)
        }


class MultiTopologyBatchSampler(Sampler):
    """Yields batches whose indices all live inside one topology dataset.

    Used with a ConcatDataset of per-topology TopologySequenceDataset instances.
    Each batch is a list of indices into the concatenated dataset, but every
    index in the batch comes from the same underlying topology — so when the
    DataLoader collates them, the resulting tensor has a uniform N axis.
    """

    def __init__(
        self,
        per_topology_lengths: List[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.lengths = per_topology_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        # Cumulative offsets into the ConcatDataset.
        self.offsets = [0]
        for L in per_topology_lengths:
            self.offsets.append(self.offsets[-1] + L)
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        all_batches: List[List[int]] = []
        for topo_idx, L in enumerate(self.lengths):
            order = list(range(L))
            if self.shuffle:
                rng.shuffle(order)
            base = self.offsets[topo_idx]
            for i in range(0, L, self.batch_size):
                chunk = order[i : i + self.batch_size]
                if not chunk:
                    continue
                all_batches.append([base + j for j in chunk])
        if self.shuffle:
            rng.shuffle(all_batches)
        for b in all_batches:
            yield b

    def __len__(self) -> int:
        return sum((L + self.batch_size - 1) // self.batch_size for L in self.lengths)


def collate_windows(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of (L, cu_dim) / (L, N, du_dim) windows into a batch.

    Caller guarantees all items in `items` came from the same topology, so the
    N axis is uniform. We just torch.stack along a new batch dim 0.
    """
    cu = torch.stack([it["cu"] for it in items], dim=0)             # (B, L, cu_dim)
    du = torch.stack([it["du"] for it in items], dim=0)             # (B, L, N, du_dim)
    return {"cu": cu, "du": du}


def split_windows(n_windows: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Random window-level train/val split. Returns (train_idx, val_idx)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_windows)
    n_val = max(1, int(round(val_frac * n_windows)))
    return perm[n_val:], perm[:n_val]

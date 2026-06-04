import torch
from torch.utils.data import Dataset
import numpy as np

class EEGWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        eeg,                 # [N_trials, T, C]
        window_size,
        hop_size,
        normalize=True,
        stats=None,
        dtype=torch.float32
    ):
        self.window_size = window_size
        self.hop_size = hop_size
        self.normalize = normalize
        self.dtype = dtype

        if eeg.ndim != 3:
            raise ValueError("EEG must have shape [N_trials, T, C]")

        self.eeg = eeg
        self.n_trials, self.T, self.C = eeg.shape

        # ----------------------------
        # Window index map (per trial)
        # ----------------------------
        self.index_map = []
        for trial in range(self.n_trials):
            max_start = self.T - window_size
            for start in range(0, max_start + 1, hop_size):
                self.index_map.append((trial, start))

        # ----------------------------
        # Optional normalization
        # ----------------------------
        if self.normalize:
            if stats is None:
                self.mean, self.std = self._compute_stats()
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]

            self.std = np.maximum(self.std, 1e-6)

    def _compute_stats(self):
        # Compute over all trials but per channel
        flat = self.eeg.reshape(-1, self.C)
        mean = flat.mean(axis=0)
        std = flat.std(axis=0)
        return mean, std

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        trial, start = self.index_map[idx]

        window = self.eeg[
            trial,
            start:start + self.window_size,
            :
        ]

        if self.normalize:
            window = (window - self.mean) / self.std

        return torch.tensor(window, dtype=self.dtype)

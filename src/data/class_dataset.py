import torch
from torch.utils.data import Dataset
import numpy as np

class EEGClassDataset(Dataset):
    """
    Windowed EEG Dataset with class labels for classification tasks.

    Returns:
        X: torch.Tensor of shape [T, C]
        y: torch.Tensor scalar (long)
    """

    def __init__(
        self,
        eeg_list,
        labels,
        window_size,
        hop_size,
        normalize=True,
        stats=None,
        dtype=torch.float32
    ):
        """
        Parameters
        ----------
        eeg_list : list of np.ndarray
            Each element corresponds to one class.
            Each array shape:
                [T, C] or [N_trials, T, C]

        labels : list or np.ndarray
            Integer labels corresponding to eeg_list

        window_size : int
        hop_size : int
        normalize : bool
        stats : dict or None
            {'mean': np.ndarray[C], 'std': np.ndarray[C]}
        dtype : torch.dtype
        """

        assert len(eeg_list) == len(labels), "Mismatch eeg_list and labels"

        self.window_size = window_size
        self.hop_size = hop_size
        self.normalize = normalize
        self.dtype = dtype

        self.data = []
        self.labels = []

        # Standardize to [N_trials, T, C]
        for eeg, label in zip(eeg_list, labels):
            if eeg.ndim == 2:
                eeg = eeg[None, ...]
            elif eeg.ndim != 3:
                raise ValueError("EEG must have shape [T, C] or [N, T, C]")

            self.data.append(eeg)
            self.labels.append(label)

        # Precompute index map
        self.index_map = []
        for class_idx, eeg in enumerate(self.data):
            n_trials, T, _ = eeg.shape

            for trial in range(n_trials):
                max_start = T - window_size
                for start in range(0, max_start + 1, hop_size):
                    self.index_map.append((class_idx, trial, start))

        # Normalization
        if self.normalize:
            if stats is None:
                self.mean, self.std = self._compute_stats()
            else:
                self.mean = stats["mean"]
                self.std = stats["std"]

            self.std = np.maximum(self.std, 1e-6)

    def _compute_stats(self):
        """
        Compute global per-channel mean/std across ALL classes.
        """
        all_data = []

        for eeg in self.data:
            reshaped = eeg.reshape(-1, eeg.shape[-1])  # [*, C]
            all_data.append(reshaped)

        all_data = np.concatenate(all_data, axis=0)

        mean = all_data.mean(axis=0)
        std = all_data.std(axis=0)

        return mean, std

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        class_idx, trial, start = self.index_map[idx]

        eeg = self.data[class_idx]
        label = self.labels[class_idx]

        window = eeg[
            trial,
            start : start + self.window_size,
            :
        ]

        if self.normalize:
            window = (window - self.mean) / self.std

        X = torch.tensor(window, dtype=self.dtype)
        y = torch.tensor(label, dtype=torch.long)

        return X, y



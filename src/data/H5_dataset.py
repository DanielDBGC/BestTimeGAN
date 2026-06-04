import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGH5Dataset(Dataset):

    def __init__(self, h5_path):

        self.h5_path = h5_path
        self.h5 = None

        with h5py.File(h5_path, "r") as f:
            self.length = len(f["trials"])

    def _ensure_open(self):

        if self.h5 is None:

            self.h5 = h5py.File(
                self.h5_path,
                "r",
            )

            self.x = self.h5["trials"]
            self.y = self.h5["labels"]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        self._ensure_open()

        x = torch.from_numpy(
            self.x[idx]
        ).float()

        y = torch.tensor(
            self.y[idx],
            dtype=torch.long,
        )

        return x, y
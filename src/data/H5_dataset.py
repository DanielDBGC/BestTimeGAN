import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EEGH5Dataset(Dataset):
    """
    Lazy-loading EEG dataset backed by an HDF5 file.

    The file is expected to contain two datasets:
        ``trials`` : float array of shape [N, T, C]
        ``labels`` : int   array of shape [N]

    Parameters
    ----------
    h5_path : str
        Path to the .h5 file written by ``save_dataset_h5``.
    keep_classes : list of int or None
        When provided, only samples whose label is in this list are
        returned.  Labels are returned **as-is** (no remapping).

        Example — train the GAN on only two classes::

            dataset = EEGH5Dataset("data/processed/eeg_train.h5",
                                   keep_classes=[0, 1])

        Leave as ``None`` (default) to use all classes unchanged.
    """

    def __init__(self, h5_path: str, keep_classes: list[int] | None = None):
        self.h5_path = h5_path
        self.h5      = None          # opened lazily in workers

        # ------------------------------------------------------------------
        # Read all labels once at construction time to build the index map.
        # The file stays closed afterwards; actual data is read lazily.
        # ------------------------------------------------------------------
        with h5py.File(h5_path, "r") as f:
            all_labels = f["labels"][:]   # [N]  int ndarray, fully in RAM

        if keep_classes is not None:
            keep_set = set(keep_classes)

            # Indices of samples belonging to the requested classes
            mask = np.isin(all_labels, list(keep_set))
            self._indices = np.where(mask)[0]          # flat indices into the H5

            self._labels = all_labels[self._indices].astype(np.int64)
        else:
            # No filter: identity mapping over all samples
            self._indices = np.arange(len(all_labels), dtype=np.int64)
            self._labels  = all_labels.astype(np.int64)

    # ------------------------------------------------------------------
    # Lazy file handle (safe for DataLoader multiprocessing)
    # ------------------------------------------------------------------
    def _ensure_open(self):
        if self.h5 is None:
            self.h5     = h5py.File(self.h5_path, "r")
            self.trials = self.h5["trials"]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        self._ensure_open()

        h5_idx = int(self._indices[idx])

        x = torch.from_numpy(self.trials[h5_idx].astype(np.float32))
        y = torch.tensor(self._labels[idx], dtype=torch.long)

        return x, y
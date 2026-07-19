"""
preprocess.py — Build train / val / test .h5 datasets from raw EEG.

Usage
-----
Run from the project root:

    python preprocess.py

The script calls `load_all_freq` with an explicit `stim_freqs` list.
To train on only 2 SSVEP classes (e.g. 8 Hz and 12 Hz), edit
`STIM_FREQS` below and re-run — no other file needs to change.

Output files
------------
    data/processed/eeg_train.h5
    data/processed/eeg_val.h5
    data/processed/eeg_test.h5
    data/stats/stats_{train,val,test}.pkl
"""

import pickle
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch

from src.utils.config import WANTED_CHANNELS, WINDOW_SIZE, WINDOW_STRIDE
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.preprocessing import load_all_freq, save_dataset_h5, robust_clip_normalize

# ---------------------------------------------------------------------------
# Configuration — edit these to change which classes / subjects are used
# ---------------------------------------------------------------------------

# SSVEP stimulus frequencies to include.
# Set to None to load all defaults (4, 8, 12, …, 60 Hz).
# Example: [8.0, 12.0] trains on only 2 classes.
STIM_FREQS = None  # <- change to e.g. [8.0, 12.0] for a 2-class run

BLOCKS = list(range(1, 13))

SUBJ_TRAIN = list(range(1, 10))   # subjects 1–9
SUBJ_VAL   = list(range(10, 11))  # subject 10
SUBJ_TEST  = list(range(11, 12))  # subject 11

# H5 output paths
PATH_TRAIN = "data/processed/eeg_train_8.h5"
PATH_VAL   = "data/processed/eeg_val_8.h5"
PATH_TEST  = "data/processed/eeg_test_8.h5"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
set_seed(42)
logger = get_logger("preprocess")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if STIM_FREQS is not None:
    logger.info(f"Class filter active — loading only {len(STIM_FREQS)} class(es): {STIM_FREQS}")
else:
    logger.info("No class filter — loading all default stimulus frequencies.")


# ---------------------------------------------------------------------------
# Helper: window a loaded (data, labels) tuple and write to .h5
# ---------------------------------------------------------------------------
def make_split(
    subject_ids: list[int],
    h5_path: str,
    stats_path: str,
    split_name: str,
) -> None:
    """
    Load raw EEG for the given subjects, window it, normalize, and save to .h5.

    Parameters
    ----------
    subject_ids : subjects to include in this split
    h5_path     : destination .h5 file
    stats_path  : destination .pkl file for per-subject normalization stats
    split_name  : label used in log messages ('train', 'val', 'test')
    """
    logger.info(f"─── Building '{split_name}' split (subjects {subject_ids}) ───")

    # 1. Load all frequencies / classes for this split
    data, labels, subj_stats = load_all_freq(
        subject_ids=subject_ids,
        block_ids=BLOCKS,
        base_path="data/raw",
        logger=logger,
        stim_freqs=STIM_FREQS,
        duration_sec=5,
        picks=WANTED_CHANNELS,
    )
    # data:   [N_trials, T, C]
    # labels: list of float stimulus frequencies, length N_trials

    # 2. Compute windowing dimensions
    num_trials, total_len, num_features = data.shape
    num_windows = (total_len - WINDOW_SIZE) // WINDOW_STRIDE + 1
    logger.info(f"  Trials: {num_trials} | Time length: {total_len} | Windows/trial: {num_windows}")

    # 3. Truncate to exact multiple of window grid (avoids partial trailing window)
    truncated_len  = (num_windows - 1) * WINDOW_STRIDE + WINDOW_SIZE
    data_truncated = data[:, :truncated_len, :]

    # 4. Zero-copy sliding windows via stride tricks
    s_trial, s_time, s_feat = data_truncated.strides
    windowed = as_strided(
        data_truncated,
        shape=(num_trials, num_windows, WINDOW_SIZE, num_features),
        strides=(s_trial, s_time * WINDOW_STRIDE, s_time, s_feat),
    )

    # 5. Flatten trial × window → flat sample dimension
    X = windowed.reshape(-1, WINDOW_SIZE, num_features)

    # 6. Repeat per-trial label for every window, then remap to 0-based integers
    windowed_labels = np.repeat(labels, num_windows)
    _, y = np.unique(windowed_labels, return_inverse=True)

    logger.info(f"  X shape: {X.shape}  |  y shape: {y.shape}")
    logger.info(f"  Unique classes: {np.unique(y)}")
    logger.info(f"  Raw stats   — mean: {X.mean():.4f}  std: {X.std():.4f}  "
                f"min: {X.min():.4f}  max: {X.max():.4f}")

    # 7. Robust clip + re-normalize
    X = robust_clip_normalize(X)
    logger.info(f"  Clipped stats — mean: {X.mean():.4f}  std: {X.std():.4f}  "
                f"min: {X.min():.4f}  max: {X.max():.4f}")

    # 8. Save to disk
    save_dataset_h5(h5_path, X, y)
    with open(stats_path, "wb") as f:
        pickle.dump(subj_stats, f)
    logger.info(f"  Saved → {h5_path}")
    logger.info(f"  Saved → {stats_path}")


# ---------------------------------------------------------------------------
# Build all three splits
# ---------------------------------------------------------------------------
make_split(SUBJ_TRAIN, PATH_TRAIN, "data/stats/stats_train.pkl", "train")
make_split(SUBJ_VAL,   PATH_VAL,   "data/stats/stats_val.pkl",   "val")
make_split(SUBJ_TEST,  PATH_TEST,  "data/stats/stats_test.pkl",  "test")

logger.info("Done — all splits written.")

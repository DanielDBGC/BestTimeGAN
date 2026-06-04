import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided
from torch.utils.data import DataLoader
import pickle
from src.utils.config import (
    WANTED_CHANNELS,
    WINDOW_SIZE,
    WINDOW_STRIDE
)

from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.preprocessing import load_all_freq, save_dataset_h5, robust_clip_normalize

set_seed(42)
logger = get_logger("preprocess")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
path_train = "data/processed/eeg_train.h5"
path_val = "data/processed/eeg_val.h5"
path_test = "data/processed/eeg_test.h5"

subj = list(range(1,10))
blocks = list(range(1,13))

eeg = load_all_freq(subj, blocks, "data/raw", logger=logger, duration_sec=5, l_freq=0.5, h_freq=50.0, picks=WANTED_CHANNELS)

data = eeg[0]
labels = eeg[1]

# 1. Calculate shape parameters
num_trials, total_len, num_features = data.shape
num_windows = ((total_len - WINDOW_SIZE) // WINDOW_STRIDE) + 1
logger.info(f"Number of windows: {num_windows}")

# 2. Slice the data to drop the remaining trailing data points
# This ensures perfect math for striding (1800, 5120, 9)
truncated_len = (num_windows - 1) * WINDOW_STRIDE + WINDOW_SIZE
data_truncated = data[:, :truncated_len, :]

# 3. Use stride_tricks to create windows without copying data
shape = (num_trials, num_windows, WINDOW_SIZE, num_features)

# Calculate original strides in bytes
s_trial, s_time, s_feat = data_truncated.strides
# New strides: 
# Moving 1 window forward means moving `WINDOW_STRIDE` time steps
strides = (s_trial, s_time * WINDOW_STRIDE, s_time, s_feat)

windowed_data = as_strided(data_truncated, shape=shape, strides=strides)

# 4. Repeat the labels for each window
# Each of the 1800 trials now has `num_windows` (37) chunks
windowed_labels = np.repeat(labels, num_windows)

# 5. Flatten the trial and window dimensions if needed for ML training
# Final Shape: (1800 * 37, 512, 9) -> (66600, 512, 9)
X = windowed_data.reshape(-1, WINDOW_SIZE, num_features)
_, y = np.unique(windowed_labels, return_inverse=True)

logger.info(f"Shape of X: {X.shape}")
logger.info(f"Shape of y: {y.shape}")
logger.info(f"Unique labels: {np.unique(y)}")
logger.info(f"Data stats: {X.mean():.4f}, {X.std():.4f}, {X.min():.4f}, {X.max():.4f}")
X = robust_clip_normalize(X)
logger.info(f"Data stats after clipping: {X.mean():.4f}, {X.std():.4f}, {X.min():.4f}, {X.max():.4f}")

save_dataset_h5(path_train, X, y)

with open("data/stats/stats_train.pkl", 'wb') as f:
    pickle.dump(eeg[2], f)

subj = list(range(10,11))

eeg = load_all_freq(subj, blocks, "data/raw", logger=logger, duration_sec=5, l_freq=0.5, h_freq=50.0, picks=WANTED_CHANNELS)

data = eeg[0]
labels = eeg[1]

# 1. Calculate shape parameters
num_trials, total_len, num_features = data.shape
num_windows = ((total_len - WINDOW_SIZE) // WINDOW_STRIDE) + 1
logger.info(f"Number of windows: {num_windows}")

# 2. Slice the data to drop the remaining trailing data points
# This ensures perfect math for striding (1800, 5120, 9)
truncated_len = (num_windows - 1) * WINDOW_STRIDE + WINDOW_SIZE
data_truncated = data[:, :truncated_len, :]

# 3. Use stride_tricks to create windows without copying data
shape = (num_trials, num_windows, WINDOW_SIZE, num_features)

# Calculate original strides in bytes
s_trial, s_time, s_feat = data_truncated.strides
# New strides: 
# Moving 1 window forward means moving `WINDOW_STRIDE` time steps
strides = (s_trial, s_time * WINDOW_STRIDE, s_time, s_feat)

windowed_data = as_strided(data_truncated, shape=shape, strides=strides)

# 4. Repeat the labels for each window
# Each of the 1800 trials now has `num_windows` (37) chunks
windowed_labels = np.repeat(labels, num_windows)

# 5. Flatten the trial and window dimensions if needed for ML training
# Final Shape: (1800 * 37, 512, 9) -> (66600, 512, 9)
X = windowed_data.reshape(-1, WINDOW_SIZE, num_features)
_, y = np.unique(windowed_labels, return_inverse=True)

logger.info(f"Shape of X: {X.shape}")
logger.info(f"Shape of y: {y.shape}")
logger.info(f"Unique labels: {np.unique(y)}")
logger.info(f"Data stats: {X.mean():.4f}, {X.std():.4f}, {X.min():.4f}, {X.max():.4f}")
X = robust_clip_normalize(X)
logger.info(f"Data stats after clipping: {X.mean():.4f}, {X.std():.4f}, {X.min():.4f}, {X.max():.4f}")

save_dataset_h5(path_val, X, y)
with open("data/stats/stats_val.pkl", 'wb') as f:
    pickle.dump(eeg[2], f)

subj = list(range(11,12))


eeg = load_all_freq(subj, blocks, "data/raw", logger=logger, duration_sec=5, l_freq=0.5, h_freq=50.0, picks=WANTED_CHANNELS)

data = eeg[0]
labels = eeg[1]

# 1. Calculate shape parameters
num_trials, total_len, num_features = data.shape
num_windows = ((total_len - WINDOW_SIZE) // WINDOW_STRIDE) + 1
logger.info(f"Number of windows: {num_windows}")

# 2. Slice the data to drop the remaining trailing data points
truncated_len = (num_windows - 1) * WINDOW_STRIDE + WINDOW_SIZE
data_truncated = data[:, :truncated_len, :]

# 3. Use stride_tricks to create windows without copying data
shape = (num_trials, num_windows, WINDOW_SIZE, num_features)

s_trial, s_time, s_feat = data_truncated.strides
strides = (s_trial, s_time * WINDOW_STRIDE, s_time, s_feat)

windowed_data = as_strided(data_truncated, shape=shape, strides=strides)

# 4. Repeat the labels for each window
windowed_labels = np.repeat(labels, num_windows)

# 5. Flatten the trial and window dimensions if needed for ML training
X = windowed_data.reshape(-1, WINDOW_SIZE, num_features)
_, y = np.unique(windowed_labels, return_inverse=True)

logger.info(f"Final shape of X: {X.shape}")
logger.info(f"Final shape of y: {y.shape}")
logger.info(f"Unique labels: {np.unique(y)}")

save_dataset_h5(path_test, X, y)

with open("data/stats/stats_test.pkl", 'wb') as f:
    pickle.dump(eeg[2], f)

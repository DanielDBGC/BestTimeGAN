import mne
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import h5py
import logging
import torch

PROJECT_ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.normalization import (
    compute_channel_stats,
    apply_channel_norm,
    compute_global_minmax,
    apply_global_minmax
)

def save_dataset_h5(
    path: str,
    trials: np.ndarray,
    labels: np.ndarray
):
    with h5py.File(path, "w") as f:

        f.create_dataset(
            "trials",
            data=trials,
        )

        f.create_dataset(
            "labels",
            data=labels,
        )

def load_multiple_subjects(
    subject_ids: list[int],
    block_ids: list[int],
    stim_freq: float,
    base_path: str,
    logger,
    **kwargs,
) -> tuple[np.ndarray, dict, dict]:
    """
    Returns:
        np.ndarray of shape [N_trials, T, C] (bounded to [0,1])
        subject_stats: dict mapping subj_id -> {'mean': ..., 'std': ...}
        global_stats: dict with {'min': ..., 'max': ...}
    """
    all_trials = []
    subject_stats = {}

    for subj in subject_ids:

        subject_segments = []

        # ----------------------------
        # 1) Load all segments
        # ----------------------------
        for block_id in block_ids:
            edf_path = f"{base_path}/sub-{subj:03d}_ses-04_block-{block_id:03d}_task-ssvep_eeg.edf"
            tsv_path = f"{base_path}/sub-{subj:03d}_ses-04_block-{block_id:03d}_task-ssvep_events.tsv"

            segments = extract_frequency_segments(
                edf_path=edf_path,
                tsv_path=tsv_path,
                stim_freq=stim_freq,
                **kwargs,
            )

            if len(segments) > 0:
                subject_segments.extend(segments)  # keep as list of (T, C)

        if len(subject_segments) == 0:
            continue

        # ----------------------------
        # 2) Normalize per subject (Z-score)
        # ----------------------------
        subject_concat = np.concatenate(subject_segments, axis=0)
        stats = compute_channel_stats(subject_concat)
        subject_stats[subj] = stats

        subject_segments = [
            apply_channel_norm(seg, stats) for seg in subject_segments
        ]

        logger.info(f"Subject: {subj}")
        logger.info(f"Mean: {stats['mean']}")
        logger.info(f"Std: {stats['std']}")

        # ----------------------------
        # 3) Store trials
        # ----------------------------
        all_trials.extend(subject_segments)

    if len(all_trials) == 0:
        raise RuntimeError("No data loaded from any subject.")

    # IMPORTANT: requires equal-length trials
    trials_arr = np.stack(all_trials, axis=0)  # [N_trials, T, C]


    return trials_arr, subject_stats

def load_all_freq(
    subject_ids: list[int],
    block_ids: list[int],
    base_path: str,
    logger,
    stim_freqs: list[float] | None = None,
    normalize: bool = True,
    **kwargs,
) -> tuple[np.ndarray, list, dict]:
    """
    Load SSVEP EEG data for multiple subjects across all requested stimulus frequencies.

    Each subject is Z-score normalised per channel before being pooled with others.

    Parameters
    ----------
    subject_ids : list of int
        Subject IDs to load (e.g. [1, 2, 3]).
    block_ids : list of int
        Recording block IDs to include per subject.
    base_path : str
        Root directory containing the raw EDF and TSV files.
    logger : logging.Logger
        Logger instance for progress messages.
    stim_freqs : list of float or None
        SSVEP stimulus frequencies (Hz) to include.
        **Pass a subset here to restrict training to fewer classes.**
        For example, ``stim_freqs=[8.0, 12.0]`` yields only two-class data.
        Defaults to ``[4, 8, 12, …, 60]`` when None.
    **kwargs
        Forwarded to :func:`extract_frequency_segments`
        (``duration_sec``, ``l_freq``, ``h_freq``, ``picks``).

    Returns
    -------
    trials_arr : np.ndarray of shape [N_trials, T, C]
        Stacked, normalised EEG windows.  All trials must have equal length T.
    all_labels : list of float
        Raw stimulus frequency for each trial (before any integer remapping).
        Length matches ``trials_arr.shape[0]``.
    subject_stats : dict
        Maps each loaded subject ID to its per-channel ``{'mean', 'std'}`` dict.
    """


    # Default: 4, 8, 12, ..., 60
    if stim_freqs is None:
        stim_freqs = list(range(4, 61, 4))

    all_trials = []
    all_labels = []

    subject_stats = {}

    for subj in subject_ids:

        subject_segments = []
        subject_labels = []

        # -------------------------------------------------
        # 1) Load all segments for all blocks/frequencies
        # -------------------------------------------------
        for block_id in block_ids:

            edf_path = (
                f"{base_path}/sub-{subj:03d}_ses-04_block-{block_id:03d}_task-ssvep_eeg.edf"
            )

            tsv_path = (
                f"{base_path}/sub-{subj:03d}_ses-04_block-{block_id:03d}_task-ssvep_events.tsv"
            )
            
            try:
                data = mne.io.read_raw_edf(edf_path, preload=True)
            except Exception as e:
                logger.warning(f"Could not load {edf_path}: {e}")
                continue
                
            # ----------------------------
            # Broadband filter 1-100 Hz to preserve all SSVEP peaks
            # ----------------------------
            data.filter(
                l_freq=1.0,
                h_freq=100.0,
                fir_design="firwin",
                phase="zero",
            )
            
            sfreq = data.info["sfreq"]
            events_df = pd.read_csv(tsv_path, sep="\t")

            for stim_freq in stim_freqs:
                stim_events = events_df[
                    (events_df["stim_frequency"] == stim_freq)
                    & (events_df["value"] % 2 == 0)
                ]

                segments = []
                for _, row in stim_events.iterrows():
                    onset = row["onset"]          # seconds
                    duration_samples = (kwargs.get('duration_sec', 5) * sfreq) + 124

                    tmin = onset
                    tmax = onset + duration_samples

                    seg = data.copy().crop(
                        tmin=tmin / sfreq, 
                        tmax=tmax / sfreq
                    )

                    final_data = seg.get_data(picks=kwargs.get('picks')).T  # (T, C)
                    segments.append(final_data)

                if len(segments) > 0:
                    subject_segments.extend(segments)
                    subject_labels.extend(
                        [stim_freq] * len(segments)
                    )

        if len(subject_segments) == 0:
            continue

        # -------------------------------------------------
        # 2) Per-subject normalization
        # -------------------------------------------------
        if normalize:
            subject_concat = np.concatenate(subject_segments, axis=0)

            stats = compute_channel_stats(subject_concat)

            subject_stats[subj] = stats

            subject_segments = [
                apply_channel_norm(seg, stats)
                for seg in subject_segments
            ]

        logger.info(f"Subject: {subj}")
        normalized_concat = np.concatenate(subject_segments, axis=0)
        logger.info(f"Mean: {normalized_concat.mean(axis=0)}")
        logger.info(f"Std: {normalized_concat.std(axis=0)}")

        # -------------------------------------------------
        # 3) Store
        # -------------------------------------------------
        all_trials.extend(subject_segments)
        all_labels.extend(subject_labels)

    if len(all_trials) == 0:
        raise RuntimeError("No data loaded from any subject.")

    # Requires equal-length trials
    trials_arr = np.stack(all_trials, axis=0)

    return trials_arr, all_labels, subject_stats


def extract_frequency_segments(
    edf_path: str,
    tsv_path: str,
    stim_freq: float,
    duration_sec: float,
    picks: list[str],
    ) -> list[np.ndarray]:
    """
    Load one EDF block and extract all segments for a given stimulation frequency.

    Returns:
        list of arrays, each shape (T, C)
    """
    data = mne.io.read_raw_edf(edf_path, preload=True)
    
    # ----------------------------
    # Broadband filter 1-100 Hz
    # ----------------------------
    data.filter(
        l_freq=1.0,
        h_freq=100.0,
        fir_design="firwin",
        phase="zero",
    )

    sfreq = data.info["sfreq"]

    events_df = pd.read_csv(tsv_path, sep="\t")

    stim_events = events_df[
        (events_df["stim_frequency"] == stim_freq)
        & (events_df["value"] % 2 == 0)
    ]

    segments = []

    for _, row in stim_events.iterrows():
        onset = row["onset"]          # seconds
        duration_samples = (duration_sec * sfreq) + 124

        tmin = onset
        tmax = onset + duration_samples

        seg = data.copy().crop(
            tmin=tmin / sfreq, 
            tmax=tmax / sfreq
        )

        final_data = seg.get_data(picks=picks).T  # (T, C)
        segments.append(final_data)

    return segments

def load_subject_frequency(
    block_ids: list[int],
    stim_freq: float,
    base_path: str,
    **kwargs,
    ) -> np.ndarray:
    """
    Load and concatenate all EEG data for one subject and one frequency.
    """
    all_segments = []

    for block_id in block_ids:
        edf_path = f"{base_path}/sub-001_ses-04_block-{block_id:03d}_task-ssvep_eeg.edf"
        tsv_path = f"{base_path}/sub-001_ses-04_block-{block_id:03d}_task-ssvep_events.tsv"

        segments = extract_frequency_segments(
            edf_path=edf_path,
            tsv_path=tsv_path,
            stim_freq=stim_freq,
            **kwargs,
        )

        if len(segments) > 0:
            block_data = np.concatenate(segments, axis=0)
            all_segments.append(block_data)

    if len(all_segments) == 0:
        raise RuntimeError("No segments found")

    return np.concatenate(all_segments, axis=0)


def robust_clip_normalize(X: np.ndarray, clip_sigma: float = 4.0) -> np.ndarray:
    """
    Robustly clip extreme outliers using Median and IQR, preventing 
    outliers from skewing the normalization metrics.
    """
    # 1. Calculate robust statistics (Median and IQR)
    median = np.median(X, axis=(0, 1), keepdims=True)
    
    q75, q25 = np.percentile(X, [75, 25], axis=(0, 1), keepdims=True)
    iqr = q75 - q25
    
    # 2. Approximate standard deviation from IQR (for normal distributions std ≈ IQR / 1.349)
    pseudo_std = iqr / 1.349
    
    # 3. Z-score using robust metrics
    X_norm = (X - median) / (pseudo_std + 1e-8)
    
    # 4. Clip (or you could use np.tanh(X_norm / clip_sigma) * clip_sigma for soft clipping)
    X_clip = np.tanh(X_norm / 2.5) * 2.5
    # 5. Final re-center to ensure mean 0, std 1
    m2 = np.mean(X_clip, axis=(0, 1), keepdims=True)
    s2 = np.std(X_clip, axis=(0, 1), keepdims=True)
    
    return (X_clip - m2) / (s2 + 1e-8)
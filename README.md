# BestTimeGAN

A conditional TimeGAN framework for synthesising **SSVEP EEG time series**. The model learns to generate realistic, stimulus-frequency-conditioned multichannel EEG segments and is evaluated with a TSTR/TRTR classifier benchmark.

---

## Overview

BestTimeGAN adapts the [TimeGAN](https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) architecture to the domain of Steady-State Visual Evoked Potential (SSVEP) EEG data. Key extensions over vanilla TimeGAN include:

- **Conditional generation** — label embeddings are injected into the Embedder, Supervisor, and Generator so every synthetic segment is tied to a specific stimulus frequency class (15 classes, 4–60 Hz in 4 Hz steps).
- **TCN Discriminator** — a Temporal Convolutional Network (TCN) replaces the recurrent discriminator in the joint training stage, providing stable gradients over long sequences.
- **Geometry loss** — a cosine-similarity regulariser on the latent space prevents representation collapse in the Embedder.
- **Spectral loss** — a Log-Spectral Distance (LSD) term in joint training keeps the frequency profile of synthetic signals consistent with real data.
- **Best-checkpoint + early stopping** — training scripts monitor validation loss and automatically save the best model.

---

## Architecture

The pipeline follows the four-component TimeGAN framework:

```
Raw EEG (B, T, C)
      │
      ▼
 ┌──────────┐       ┌──────────┐
 │ Embedder │──────▶│ Recovery │  (Autoencoder stage)
 └──────────┘       └──────────┘
      │  latent h
      ▼
 ┌──────────┐
 │Supervisor│  (predicts next latent step from current)
 └──────────┘
      │
      ▼
 ┌───────────┐   noise z   ┌───────────────┐
 │ Generator │◀────────────│ TCNDiscriminator│  (joint GAN stage)
 └───────────┘             └───────────────┘
      │  synthetic h_hat
      ▼
 ┌──────────┐
 │ Recovery │  (decodes synthetic h_hat → EEG)
 └──────────┘
```

| Component | Architecture | Input → Output |
|---|---|---|
| **Embedder** (`cEmbedder`) | GRU + label embedding + LayerNorm | `(B, T, 9)` → `(B, T, 12)` |
| **Recovery** (`cRecovery`) | GRU + label embedding | `(B, T, 12)` → `(B, T, 9)` |
| **Supervisor** | GRU + label embedding + TBPTT | `(B, T, 12)` → `(B, T, 12)` |
| **Generator** | GRU + LayerNorm + Linear | `(B, T, z_dim)` → `(B, T, 12)` |
| **Discriminator** | TCN (dilated Conv1D) | `(B, T, 12)` → `(B, 1)` |

---

## Repository Structure

```
BestTimeGAN/
├── Autoencoder.py          # Entry point: train Embedder + Recovery
├── Supervisor.py           # Entry point: train Supervisor
├── GAN.py                  # Entry point: joint GAN training
├── preprocess.py           # EEG loading, windowing & HDF5 export
├── preprocess_test.py      # Sanity-check script for preprocessing
│
├── src/
│   ├── data/
│   │   ├── H5_dataset.py       # PyTorch Dataset for HDF5 EEG files
│   │   ├── preprocessing.py    # MNE-based EEG loader & normalisation
│   │   └── window_dataset.py   # Sliding-window Dataset
│   ├── models/
│   │   ├── embedder.py         # Embedder & cEmbedder (conditional)
│   │   ├── recovery.py         # Recovery & cRecovery (conditional)
│   │   ├── supervisor.py       # Supervisor (GRU + TBPTT)
│   │   ├── generator.py        # Generator (GRU)
│   │   ├── discriminator.py    # GRU Discriminator & TCNDiscriminator
│   │   └── geometry.py         # Cosine-similarity geometry loss
│   ├── losses/
│   │   └── losses.py           # All loss functions (reconstruction, supervised, GAN, spectral)
│   ├── training/
│   │   ├── train_embedder.py   # Autoencoder training loop
│   │   ├── train_supervisor.py # Supervisor training loop
│   │   ├── train_timegan.py    # Joint GAN training loop
│   │   └── train_classifier.py # Downstream TSTR/TRTR classifier training
│   └── utils/
│       ├── config.py           # All hyperparameters
│       ├── logging.py          # Logger factory
│       └── seed.py             # Global seed setter
│
├── evaluation/
│   ├── evaluation.py           # Metrics: geometry, kNN, PSD, ACF, SSVEP SNR
│   ├── evaluateAutoencoder.ipynb
│   ├── evaluateSupervisor.ipynb
│   ├── evaluateGAN.ipynb
│   ├── TSTR.ipynb              # Train-on-Synthetic Test-on-Real benchmark
│   └── TRTR.ipynb              # Train-on-Real Test-on-Real baseline
│
├── data/
│   ├── raw/                    # Original EEG recordings (not tracked by git)
│   ├── processed/              # HDF5 train/val/test splits
│   └── stats/                  # Normalisation statistics (.pkl)
│
├── checkpoints/                # Saved model weights
├── logs/                       # Training logs
└── notebooks/                  # Exploratory notebooks
```

---

## Hyperparameters

All key hyperparameters live in [`src/utils/config.py`](src/utils/config.py):

| Parameter | Value | Description |
|---|---|---|
| `WANTED_CHANNELS` | `['PZ','PO3','PO4','PO5','PO6','POZ','OZ','O1','O2']` | Occipital EEG channels |
| `NUM_CHANNELS` | 9 | Number of EEG channels |
| `WINDOW_SIZE` | 512 | Samples per window (≈ 512 ms at 1 kHz) |
| `WINDOW_STRIDE` | 128 | Hop size between windows |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `LATENT_DIM` | 12 | Latent space dimensionality |
| `EPOCHS_EMBEDDER` | 150 | Autoencoder training epochs |
| `EPOCHS_SUPERVISOR` | 50 | Supervisor training epochs |
| `EPOCHS_JOINT` | 250 | Joint GAN training epochs |
| `LAMBDA_SUP` | 85 | Weight for supervisor loss in joint stage |
| `LAMBDA_MOM` | 15 | Weight for moment-matching loss |
| `LAMBDA_SPEC` | 0.2 | Weight for spectral (LSD) loss |
| `DISCRIMINATOR_THRESHOLD` | 0.1 | Skip discriminator update if loss below threshold |

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch ≥ 2.0 (CUDA recommended)
- [MNE-Python](https://mne.tools/) for EEG loading
- `h5py`, `numpy`, `scipy`

```bash
pip install torch mne h5py numpy scipy
```

### Data Preparation

Place raw EEG recordings (`.fif` or equivalent format expected by MNE) inside `data/raw/`, organised by subject and block:

```
data/raw/
└── subject_<N>/
    └── block_<M>/
        └── <recording>.<ext>
```

Then run the preprocessing script to generate windowed HDF5 datasets:

```bash
python preprocess.py
```

This creates:
- `data/processed/eeg_train.h5` — subjects 1–9
- `data/processed/eeg_val.h5`   — subject 10
- `data/processed/eeg_test.h5`  — subject 11
- `data/stats/stats_*.pkl`       — per-split normalisation statistics

### Training

Training follows a **three-stage pipeline**:

#### Stage 1 — Autoencoder (Embedder + Recovery)

```bash
python Autoencoder.py
```

Trains the conditional Embedder and Recovery for `EPOCHS_EMBEDDER` epochs. Saves best weights to `checkpoints/embedder_*.pt` and `checkpoints/recovery_*.pt`.

#### Stage 2 — Supervisor

```bash
python Supervisor.py
```

Loads the frozen Embedder from Stage 1 and trains the Supervisor to predict the next latent step. Saves best checkpoint to `checkpoints/supervisor_*.pt`.

#### Stage 3 — Joint GAN

```bash
python GAN.py
```

Loads pre-trained Embedder, Supervisor, and Recovery; jointly trains the Generator and TCN Discriminator. Saves Generator and Discriminator to `checkpoints/`.

---

## Evaluation

### Autoencoder Diagnostics

Computed automatically at the end of `Autoencoder.py`, or interactively in [`evaluation/evaluateAutoencoder.ipynb`](evaluation/evaluateAutoencoder.ipynb):

| Metric | Description |
|---|---|
| **Per-channel MSE** | Reconstruction error for each of the 9 EEG channels |
| **Per-class MSE** | Reconstruction error grouped by stimulus frequency |
| **SSVEP SNR (dB)** | Signal-to-noise ratio at stimulus frequency via Welch PSD; > 3 dB indicates detectable response |
| **Latent std per class** | Healthy range: 0.1–0.4; below 0.05 → collapse, above 0.8 → explosion |
| **Distance correlation** | Pearson correlation between pairwise distances in input and latent space |
| **kNN preservation** | Fraction of k-nearest neighbours preserved after encoding |
| **Spectral (LSD)** | Log-spectral distance between real and reconstructed signals |
| **ACF error** | Autocorrelation function MSE between real and reconstructed signals |

### TSTR / TRTR Benchmark

Train a downstream EEG classifier on **synthetic** data, then evaluate on **real** test data (TSTR), and compare against training on real data (TRTR).

Run interactively in:
- [`evaluation/TSTR.ipynb`](evaluation/TSTR.ipynb)
- [`evaluation/TRTR.ipynb`](evaluation/TRTR.ipynb)

---

## License

This repository is part of a thesis project at [ITAM](https://www.itam.mx/). Please contact the author before reusing the code.

"""
HyperparameterTuningGAN.py
==========================
Optuna study to optimise the loss-weighting lambdas used in the
TimeGAN joint training loop:

    lambda_sup  – supervised (temporal-prediction) loss weight
    lambda_mom  – moment-matching loss weight
    lambda_spec – spectral (correlation) loss weight
    lambda_adv  – adversarial loss weight

Strategy
--------
Each Optuna trial instantiates fresh Generator + Discriminator weights (E and R
are loaded from the saved checkpoints and kept frozen; S is loaded and
*unfrozen* so it adapts jointly with G — matching production).  A short
version of train_timegan (TUNE_EPOCHS epochs) is run, and the trial returns
the LDA classification accuracy on a Train-on-Synthetic / Test-on-Real (TSTR)
evaluation using Welch PSD power features at the stimulus harmonics.

Because the generator is randomly initialised each trial, results are noisy.
A MedianPruner is used so that clearly bad trials are stopped early.

Usage
-----
    python HyperparameterTuningGAN.py

Results are persisted in an SQLite database (optuna_timegan.db) so that a
study can be resumed after interruption:

    python HyperparameterTuningGAN.py   # resumes automatically

After the study finishes, the best hyperparameters are printed and written
to best_lambdas.json.
"""

import json
import logging
import os

import numpy as np
import torch
import optuna
from optuna.pruners import MedianPruner
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.data.H5_dataset import EEGH5Dataset
from src.models.embedder import cEmbedder
from src.models.recovery import cRecovery
from src.models.supervisor import Supervisor
from src.models.generator import cGenerator
from src.models.discriminator import cTCNDiscriminator
from src.models.freq_conditioning import build_freq_basis
from src.losses.losses import (
    generator_adv_loss,
    discriminator_loss,
    ssvep_corr_loss,
    r1_penalty,
)
from src.utils.config import (
    LATENT_DIM,
    NOISE_DIM,
    NUM_CHANNELS,
    BATCH_SIZE,
    HIDDEN_DIM_GENERATOR,
    HIDDEN_DIM_DISCRIMINATOR,
    NUM_LAYERS_GENERATOR,
    NUM_LAYERS_SUPERVISOR,
    LR_GENERATOR,
    LR_DISCRIMINATOR,
    ALL_STIM_FREQS,
    SSVEP_FS,
    FREQ_DIM,
    FREQ_N_HARMONICS,
)
from src.utils.seed import set_seed
from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# ── Study configuration ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Classes to include in the study (must match GAN.py)
GAN_CLASSES = [3, 6]

# Short training so each trial finishes quickly; increase for better signal.
TUNE_EPOCHS = 30

# Batch size can be smaller than production to speed up trials.
TUNE_BATCH_SIZE = 32

# How many Optuna trials to run in total.
N_TRIALS = 40

# Path to the pre-trained E / S / R checkpoints.
CKPT_DIR = "checkpoints"

# Optuna storage — SQLite lets us resume after a crash.
STORAGE = "sqlite:///optuna_timegan.db"
STUDY_NAME = "timegan_lambda_search"

# TSTR evaluation: number of synthetic samples to generate per class.
N_SYNTH_PER_CLASS = 200

# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger("HyperparamTuning")


def _build_models(device: torch.device):
    """Instantiate and return fresh G and D; load E, S, R from disk.

    S is *unfrozen* (matching production train_timegan.py) so it adapts
    jointly with G.  E and R remain frozen.
    """
    # ── Pre-trained components (E uses label embeddings, R uses label embeddings) ──
    E = cEmbedder(x_dim=NUM_CHANNELS, h_dim=LATENT_DIM).to(device)
    R = cRecovery(h_dim=LATENT_DIM, x_dim=NUM_CHANNELS).to(device)

    # ── S now uses sinusoidal freq conditioning (freq_dim) ──
    S = Supervisor(
        h_dim=LATENT_DIM,
        num_layers=NUM_LAYERS_SUPERVISOR,
        freq_dim=FREQ_DIM,
    ).to(device)

    # ── Fresh G and D (freq-conditioned) ──
    G = cGenerator(
        z_dim=NOISE_DIM,
        h_dim=HIDDEN_DIM_GENERATOR,
        num_layers=NUM_LAYERS_GENERATOR,
        out_dim=LATENT_DIM,
        freq_dim=FREQ_DIM,
    ).to(device)

    D = cTCNDiscriminator(
        in_channels=LATENT_DIM,
        hidden_channels=HIDDEN_DIM_DISCRIMINATOR,
        freq_dim=FREQ_DIM,
    ).to(device)

    # Load pre-trained frozen components
    E.load_state_dict(torch.load(f"{CKPT_DIR}/embedder_24_500.pt",  weights_only=True))
    S.load_state_dict(torch.load(f"{CKPT_DIR}/supervisor_24_50.pt", weights_only=True))
    R.load_state_dict(torch.load(f"{CKPT_DIR}/recovery_24_500.pt",  weights_only=True))

    # E is frozen permanently (always used inside torch.no_grad())
    E.eval()
    for p in E.parameters():
        p.requires_grad_(False)

    # S is UNFROZEN — it adapts jointly with G (matching production)
    S.requires_grad_(True)
    # Zero out dropout on S and R so they behave deterministically even in train() mode
    if hasattr(S, "rnn"):
        S.rnn.dropout = 0.0

    # R is frozen
    R.requires_grad_(False)
    if hasattr(R, "rnn"):
        R.rnn.dropout = 0.0

    return E, G, S, R, D


# ---------------------------------------------------------------------------
# ── LDA-on-Welch-power TSTR evaluation ──────────────────────────────────
# ---------------------------------------------------------------------------

def _extract_welch_features(
    signals: np.ndarray,
    target_freqs_hz: list[float],
    fs: float,
    n_harmonics: int = 2,
) -> np.ndarray:
    """Extract Welch PSD power at each target frequency × harmonic.

    Parameters
    ----------
    signals       : (N, T) or (N, T, C) — if multi-channel, averaged first.
    target_freqs_hz : list of stimulus frequencies for the classes being evaluated.
    fs            : sampling frequency in Hz.
    n_harmonics   : number of harmonics to include (1 = fundamental only).

    Returns
    -------
    features : (N, len(target_freqs_hz) * n_harmonics) power features.
    """
    if signals.ndim == 3:
        # Average across channels → (N, T)
        signals = signals.mean(axis=-1)

    feats = []
    for sig in signals:
        f, pxx = welch(sig, fs=fs, nperseg=min(len(sig), int(fs * 2)))
        row = []
        for target in target_freqs_hz:
            for h in range(1, n_harmonics + 1):
                idx = np.argmin(np.abs(f - target * h))
                row.append(pxx[idx])
        feats.append(row)
    return np.array(feats)


@torch.no_grad()
def _lda_tstr_accuracy(
    G, S, R,
    val_loader: DataLoader,
    device: torch.device,
    orig_labels_map: torch.Tensor,
    gan_stim_freqs: list[float],
    fs: float,
    n_harmonics: int,
    n_synth_per_class: int = 200,
) -> float:
    """Train-on-Synthetic / Test-on-Real accuracy using LDA on Welch power.

    1. Generate n_synth_per_class synthetic signals per class.
    2. Extract Welch PSD power features at each stimulus harmonic.
    3. Fit LDA on synthetic features.
    4. Score on real validation features.
    5. Return accuracy (higher is better).
    """
    G.eval(); S.eval(); R.eval()

    n_classes = len(orig_labels_map)
    T_synth = None  # will be inferred from first val batch

    # ── Determine sequence length from validation data ──
    for vb in val_loader:
        x_v, _ = vb
        T_synth = x_v.shape[1]
        break

    if T_synth is None:
        return 0.0

    # ── Build label-index maps ──
    _inv_map = {int(v): i for i, v in enumerate(orig_labels_map.tolist())}

    # ── Generate synthetic data ──
    synth_signals = []
    synth_labels = []

    for local_idx in range(n_classes):
        orig_label = int(orig_labels_map[local_idx])
        remaining = n_synth_per_class
        while remaining > 0:
            B = min(remaining, TUNE_BATCH_SIZE)
            z = torch.randn(B, T_synth, NOISE_DIM, device=device)

            labels_local = torch.full((B,), local_idx, dtype=torch.long, device=device)
            orig_labels = torch.full((B,), orig_label, dtype=torch.long, device=device)

            freq_basis_local = build_freq_basis(labels_local, T_synth, gan_stim_freqs, fs, n_harmonics).to(device)
            freq_basis_orig = build_freq_basis(orig_labels, T_synth, ALL_STIM_FREQS, fs, n_harmonics).to(device)

            h_fake = G(z, freq_basis_local)
            h_fake_sup = S(h_fake, freq_basis_orig)
            x_fake = R(h_fake_sup, orig_labels).float()

            synth_signals.append(x_fake.cpu().numpy())
            synth_labels.extend([local_idx] * B)
            remaining -= B

    synth_signals = np.concatenate(synth_signals, axis=0)  # (N_synth, T, C)
    synth_labels = np.array(synth_labels)

    # ── Collect real validation data ──
    real_signals = []
    real_labels = []
    for vb in val_loader:
        x_v, v_raw_labels = vb
        # Map raw labels to local indices
        v_local = np.array([_inv_map[int(l)] for l in v_raw_labels.tolist()])
        real_signals.append(x_v.numpy())
        real_labels.extend(v_local.tolist())

    real_signals = np.concatenate(real_signals, axis=0)  # (N_real, T, C)
    real_labels = np.array(real_labels)

    # ── Extract features ──
    # Use all stimulus frequencies across GAN classes as the target set
    target_freqs = gan_stim_freqs

    X_train = _extract_welch_features(synth_signals, target_freqs, fs, n_harmonics=2)
    y_train = synth_labels

    X_val = _extract_welch_features(real_signals, target_freqs, fs, n_harmonics=2)
    y_val = real_labels

    # ── Fit LDA and score ──
    try:
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
        acc = clf.score(X_val, y_val)
    except Exception:
        # LDA can fail if features are degenerate (e.g., all zeros early in training)
        acc = 0.0

    G.train(); S.train(); R.train()
    return acc


# ---------------------------------------------------------------------------
# ── Simplified training loop ─────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _run_trial_training(
    G, D, S, R, E,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    trial: optuna.Trial,
    lambda_sup: float,
    lambda_mom: float,
    lambda_spec: float,
    lambda_adv: float,
    warmup_epochs: int,
    orig_labels_map: torch.Tensor,
    gan_stim_freqs: list[float],
    fs: float,
    n_harmonics: int,
) -> float:
    """
    Training loop for Optuna trials, mirroring production train_timegan.py.

    Key features matching production:
    - Sinusoidal freq-basis conditioning via build_freq_basis + FiLM
    - S unfrozen and included in the generator optimizer
    - R1 penalty on discriminator (instead of gradient penalty)
    - AMP (autocast + GradScaler) for speed
    - D:G ratio of 5:1
    - CosineAnnealingLR scheduler
    - LDA TSTR accuracy as the evaluation metric (higher is better)
    """
    opt_gs = torch.optim.Adam(
        list(G.parameters()) + list(S.parameters()),
        lr=LR_GENERATOR, betas=(0.0, 0.9),
    )
    opt_d = torch.optim.Adam(
        D.parameters(), lr=LR_DISCRIMINATOR, betas=(0.0, 0.9),
    )

    scaler_gs = GradScaler(device.type, enabled=device.type == "cuda")
    scaler_d  = GradScaler(device.type, enabled=device.type == "cuda")

    scheduler_gs = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gs, T_max=TUNE_EPOCHS)
    scheduler_d  = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d,  T_max=TUNE_EPOCHS)

    is_cond_E = isinstance(E, cEmbedder)
    is_cond_G = isinstance(G, cGenerator)
    is_cond_D = isinstance(D, cTCNDiscriminator)
    is_cond_R = isinstance(R, cRecovery)

    # ── Label mapping ──
    if orig_labels_map is not None:
        _inv_map = {int(v): i for i, v in enumerate(orig_labels_map.tolist())}
        def _to_local(raw_labels: torch.Tensor) -> torch.Tensor:
            return torch.tensor(
                [_inv_map[int(l)] for l in raw_labels.tolist()],
                dtype=torch.long, device=raw_labels.device,
            )
    else:
        def _to_local(raw_labels):
            return raw_labels

    # ── Warm-up Loop (Generator + S only, no adversarial) ──────────────────
    if warmup_epochs > 0:
        logger.info(f"  Trial {trial.number} | Starting {warmup_epochs} warmup epochs")
        for epoch in range(warmup_epochs):
            G.train(); S.train(); R.train()
            for batch_idx, batch in enumerate(train_loader):
                x, raw_labels = batch
                x = x.to(device)
                raw_labels = raw_labels.to(device)

                labels_local = _to_local(raw_labels)
                orig_labels = orig_labels_map[labels_local] if orig_labels_map is not None else raw_labels

                B, T, _ = x.shape
                z = torch.randn(B, T, NOISE_DIM, device=device)

                # Build freq bases
                freq_basis_local = build_freq_basis(labels_local, T, gan_stim_freqs, fs, n_harmonics).to(device)
                freq_basis_orig = build_freq_basis(orig_labels, T, ALL_STIM_FREQS, fs, n_harmonics).to(device)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h_fake = G(z, freq_basis_local)
                    h_fake_sup = S(h_fake, freq_basis_orig)

                    # Supervised loss
                    g_sup = torch.mean((h_fake_sup[:, :-1, :] - h_fake[:, 1:, :]) ** 2)

                    # Moment matching in data space
                    x_fake = R(h_fake_sup, orig_labels) if is_cond_R else R(h_fake_sup)
                    x_fake = x_fake.float()

                    mean_loss = torch.mean((x.mean(0) - x_fake.mean(0)) ** 2)
                    var_loss  = torch.mean((x.var(0, unbiased=False) - x_fake.var(0, unbiased=False)) ** 2)
                    g_mom = mean_loss + var_loss

                    # Spectral (Correlation) loss
                    physical_freqs = torch.tensor(ALL_STIM_FREQS, device=device, dtype=torch.float32)[orig_labels]
                    spec_loss = ssvep_corr_loss(x_fake, physical_freqs, fs)

                    g_loss = lambda_spec * spec_loss + lambda_mom * g_mom + lambda_sup * g_sup

                opt_gs.zero_grad(set_to_none=True)
                scaler_gs.scale(g_loss).backward()
                scaler_gs.unscale_(opt_gs)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
                scaler_gs.step(opt_gs)
                scaler_gs.update()

    # ── Adversarial Loop ────────────────────────────────────────────────────
    val_score = 0.0
    for epoch in range(TUNE_EPOCHS):
        G.train(); S.train(); R.train(); D.train()

        for batch_idx, batch in enumerate(train_loader):
            x, raw_labels = batch
            x = x.to(device)
            raw_labels = raw_labels.to(device)

            labels_local = _to_local(raw_labels)
            orig_labels = orig_labels_map[labels_local] if orig_labels_map is not None else raw_labels

            B, T, _ = x.shape

            # Build freq bases for this batch
            freq_basis_local = build_freq_basis(labels_local, T, gan_stim_freqs, fs, n_harmonics).to(device)
            freq_basis_orig = build_freq_basis(orig_labels, T, ALL_STIM_FREQS, fs, n_harmonics).to(device)

            # Frozen embedder → real latents
            with torch.no_grad():
                if is_cond_E:
                    h_real, _ = E(x, orig_labels)
                else:
                    h_real = E(x)

            # ── Discriminator step ────────────────────────────────────────
            with torch.no_grad():
                z_d = torch.randn(B, T, NOISE_DIM, device=device)
                h_fake_d = G(z_d, freq_basis_local)
                h_fake_d = S(h_fake_d, freq_basis_orig)

            # Requires grad on real latents for R1 penalty
            h_real.requires_grad_(True)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                d_real = D(h_real, freq_basis_local)
                d_fake = D(h_fake_d, freq_basis_local)
                d_loss_val = discriminator_loss(d_real, d_fake)

            with torch.autocast(device_type=device.type, enabled=False):
                r1 = r1_penalty(d_real, h_real)
                d_loss = d_loss_val + 10.0 * (10.0 / 2.0) * r1  # Gamma/2 * R1

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ── Generator + Supervisor step (every 5 batches, D:G = 5:1) ──
            if batch_idx % 5 == 0:
                z = torch.randn(B, T, NOISE_DIM, device=device)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h_fake = G(z, freq_basis_local)
                    h_fake_sup = S(h_fake, freq_basis_orig)

                    # Adversarial
                    d_fake_g = D(h_fake_sup, freq_basis_local)
                    g_adv = generator_adv_loss(d_fake_g)

                    # Supervised prediction loss
                    g_sup_fake = torch.mean(
                        (h_fake_sup[:, :-1, :] - h_fake[:, 1:, :]) ** 2
                    )
                    h_real_slice = h_real[:, :-1, :].detach()
                    freq_basis_orig_slice = freq_basis_orig[:, :-1, :]
                    h_real_pred = S(h_real_slice, freq_basis_orig_slice)
                    g_sup_real = torch.mean(
                        (h_real_pred - h_real[:, 1:, :].detach()) ** 2
                    )
                    g_sup = g_sup_fake + g_sup_real

                    # Moment matching in data space
                    x_fake = R(h_fake_sup, orig_labels) if is_cond_R else R(h_fake_sup)
                    x_fake = x_fake.float()
                    mean_loss = torch.mean((x.mean(0) - x_fake.mean(0)) ** 2)
                    var_loss  = torch.mean((x.var(0, unbiased=False) - x_fake.var(0, unbiased=False)) ** 2)
                    g_mom = mean_loss + var_loss

                    # Spectral (Correlation) loss
                    physical_freqs = torch.tensor(ALL_STIM_FREQS, device=device, dtype=torch.float32)[orig_labels]
                    spec_loss = ssvep_corr_loss(x_fake, physical_freqs, fs)

                    # Total generator loss
                    g_loss = (
                        lambda_adv * g_adv
                        + lambda_spec * spec_loss
                        + lambda_mom * g_mom
                        + lambda_sup * g_sup
                    )

                opt_gs.zero_grad(set_to_none=True)
                scaler_gs.scale(g_loss).backward()
                scaler_gs.unscale_(opt_gs)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
                scaler_gs.step(opt_gs)
                scaler_gs.update()

        # ── Step LR schedulers ──
        scheduler_gs.step()
        scheduler_d.step()

        # ── Intermediate report to Optuna (enables pruning) ───────────────
        # Evaluate every 5 epochs to keep it fast; always evaluate the last epoch.
        if epoch % 5 == 0 or epoch == TUNE_EPOCHS - 1:
            val_score = _lda_tstr_accuracy(
                G, S, R, val_loader, device,
                orig_labels_map, gan_stim_freqs, fs, n_harmonics,
                n_synth_per_class=N_SYNTH_PER_CLASS,
            )
            trial.report(val_score, step=epoch)

            logger.info(
                f"  Trial {trial.number} | Epoch {epoch:02d}/{TUNE_EPOCHS-1} "
                f"| TSTR_acc={val_score:.4f} "
                f"| λ_sup={lambda_sup:.2f} λ_mom={lambda_mom:.2f} "
                f"λ_spec={lambda_spec:.3f} λ_adv={lambda_adv:.2f}"
            )

            if trial.should_prune():
                logger.info(f"  Trial {trial.number} pruned at epoch {epoch}.")
                raise optuna.TrialPruned()

    return val_score


# ---------------------------------------------------------------------------
# ── Optuna objective ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def objective(
    trial: optuna.Trial,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    orig_labels_map: torch.Tensor,
    gan_stim_freqs: list[float],
) -> float:
    """
    Suggest lambda values, run the short training loop, return TSTR accuracy.

    Search space
    ~~~~~~~~~~~~
    lambda_sup  : log-uniform in [1, 200]
    lambda_mom  : log-uniform in [1, 50]
    lambda_spec : log-uniform in [0.01, 5]
    lambda_adv  : log-uniform in [0.1, 10]
    warmup_epochs : int in [3, 15]
    """
    lambda_sup  = trial.suggest_float("lambda_sup",  0,   1.0, log=False)
    lambda_mom  = trial.suggest_float("lambda_mom",  0,    1.0, log=False)
    lambda_spec = trial.suggest_float("lambda_spec", 0,    1.0, log=False)
    lambda_adv  = trial.suggest_float("lambda_adv",  0,    1.0, log=False)
    warmup_epochs = trial.suggest_int("warmup_epochs", 10, 20)

    logger.info(
        f"Trial {trial.number} started | "
        f"λ_sup={lambda_sup:.3f}  λ_mom={lambda_mom:.3f}  "
        f"λ_spec={lambda_spec:.4f}  λ_adv={lambda_adv:.3f}  "
        f"warmup={warmup_epochs}"
    )

    set_seed(trial.number)  # reproducible but different per trial
    E, G, S, R, D = _build_models(device)

    val_score = _run_trial_training(
        G, D, S, R, E,
        train_loader, val_loader,
        device, trial,
        lambda_sup, lambda_mom, lambda_spec, lambda_adv,
        warmup_epochs,
        orig_labels_map, gan_stim_freqs,
        fs=SSVEP_FS, n_harmonics=FREQ_N_HARMONICS,
    )

    return val_score


# ---------------------------------------------------------------------------
# ── Main ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"GAN classes: {GAN_CLASSES}")
    logger.info(f"Tune epochs per trial: {TUNE_EPOCHS}  |  N_TRIALS: {N_TRIALS}")

    # ── Build stim_freqs for the GAN subset (local label → frequency) ──
    if GAN_CLASSES is not None:
        gan_stim_freqs = [ALL_STIM_FREQS[c] for c in GAN_CLASSES]
    else:
        gan_stim_freqs = list(ALL_STIM_FREQS)

    orig_labels_map = (
        torch.tensor(GAN_CLASSES, dtype=torch.long, device=device)
        if GAN_CLASSES is not None else None
    )

    # ── Data loaders ──────────────────────────────────────────────────────
    train_dataset = EEGH5Dataset("data/processed/eeg_train.h5", keep_classes=GAN_CLASSES)
    val_dataset   = EEGH5Dataset("data/processed/eeg_val.h5",   keep_classes=GAN_CLASSES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TUNE_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # ── Optuna study ──────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",   # TSTR accuracy — higher is better
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info(f"Study '{STUDY_NAME}' loaded/created. Running {N_TRIALS} trials …")

    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader, device,
            orig_labels_map, gan_stim_freqs,
        ),
        n_trials=N_TRIALS,
        gc_after_trial=True,
    )

    # ── Report results ────────────────────────────────────────────────────
    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial: #{best.number}  |  TSTR_acc = {best.value:.6f}")
    logger.info(f"  lambda_sup  = {best.params['lambda_sup']:.4f}")
    logger.info(f"  lambda_mom  = {best.params['lambda_mom']:.4f}")
    logger.info(f"  lambda_spec = {best.params['lambda_spec']:.4f}")
    logger.info(f"  lambda_adv  = {best.params['lambda_adv']:.4f}")
    logger.info(f"  warmup_epochs = {best.params['warmup_epochs']}")
    logger.info("=" * 60)

    # Save to JSON so GAN.py / config.py can pick them up easily
    out = {
        "lambda_sup":     best.params["lambda_sup"],
        "lambda_mom":     best.params["lambda_mom"],
        "lambda_spec":    best.params["lambda_spec"],
        "lambda_adv":     best.params["lambda_adv"],
        "warmup_epochs":  best.params["warmup_epochs"],
        "tstr_accuracy":  best.value,
        "trial":          best.number,
    }
    with open("best_lambdas.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Best lambdas saved to best_lambdas.json")
    print("\nBest lambdas:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

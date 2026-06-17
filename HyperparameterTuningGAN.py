"""
HyperparameterTuning.py
=======================
Optuna study to optimise the three loss-weighting lambdas used in the
TimeGAN joint training loop:

    lambda_sup  – supervised (temporal-prediction) loss weight
    lambda_mom  – moment-matching loss weight
    lambda_spec – spectral (LSD) loss weight

Strategy
--------
Each Optuna trial instantiates fresh Generator + Discriminator weights (E, S, R
are loaded from the saved checkpoints and kept frozen for speed), runs a
*short* version of train_timegan (TUNE_EPOCHS epochs), and returns the
combined validation metric  acf_error + lsd  evaluated on the val split.

Because the generator is randomly initialised each trial, results are noisy.
A MedianPruner is used so that clearly bad trials are stopped early.

Usage
-----
    python HyperparameterTuning.py

Results are persisted in an SQLite database (optuna_timegan.db) so that a
study can be resumed after interruption:

    python HyperparameterTuning.py   # resumes automatically

After the study finishes, the best hyperparameters are printed and written
to best_lambdas.json.
"""

import json
import logging
import os

import torch
import optuna
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader

from src.data.H5_dataset import EEGH5Dataset
from src.models.embedder import cEmbedder
from src.models.recovery import cRecovery
from src.models.supervisor import Supervisor
from src.models.generator import cGenerator
from src.models.discriminator import cTCNDiscriminator
from src.losses.losses import (
    generator_adv_loss,
    discriminator_loss,
    supervised_loss,
    acf_error,
    ssvep_spectral_loss,
    gradient_penalty,
)
from src.utils.config import (
    LATENT_DIM,
    NOISE_DIM,
    NUM_CHANNELS,
    BATCH_SIZE,
    HIDDEN_DIM_GENERATOR,
    HIDDEN_DIM_DISCRIMINATOR,
    NUM_LAYERS_GENERATOR,
    LABEL_EMB_DIM,
    LR_GENERATOR,
    LR_DISCRIMINATOR,
    SUP_LOSS_WINDOW,
)
from src.utils.seed import set_seed
from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# ── Study configuration ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Classes to include in the study (must match GAN.py)
GAN_CLASSES = [2, 3]

# Short training so each trial finishes quickly; increase for better signal.
TUNE_EPOCHS = 25

# Batch size can be smaller than production to speed up trials.
TUNE_BATCH_SIZE = 32

# How many Optuna trials to run in total.
N_TRIALS = 50

# Path to the pre-trained E / S / R checkpoints.
CKPT_DIR = r"c:\Users\danie_13ucdo4\OneDrive\Desktop\ITAM\Tesis\Prueba\BestTimeGAN\checkpoints"

# Optuna storage — SQLite lets us resume after a crash.
STORAGE = "sqlite:///optuna_timegan.db"
STUDY_NAME = "timegan_lambda_search"

# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger("HyperparamTuning")


def _build_models(device: torch.device, n_classes: int):
    """Instantiate and return fresh G and D; load frozen E, S, R from disk."""
    E = cEmbedder(x_dim=NUM_CHANNELS, h_dim=LATENT_DIM).to(device)
    S = Supervisor(h_dim=LATENT_DIM, num_layers=1, out_steps=SUP_LOSS_WINDOW).to(device)
    R = cRecovery(h_dim=LATENT_DIM, x_dim=NUM_CHANNELS).to(device)

    G = cGenerator(
        z_dim=NOISE_DIM,
        h_dim=HIDDEN_DIM_GENERATOR,
        num_layers=NUM_LAYERS_GENERATOR,
        out_dim=LATENT_DIM,
        num_classes=n_classes,
        label_emb_dim=LABEL_EMB_DIM,
    ).to(device)

    D = cTCNDiscriminator(
        in_channels=LATENT_DIM,
        hidden_channels=HIDDEN_DIM_DISCRIMINATOR,
        num_classes=n_classes,
        label_emb_dim=LABEL_EMB_DIM,
    ).to(device)

    # Load pre-trained frozen components
    E.load_state_dict(torch.load(f"{CKPT_DIR}\\embedder_12.0.pt",  weights_only=True))
    S.load_state_dict(torch.load(f"{CKPT_DIR}\\supervisor_12.0.pt", weights_only=True))
    R.load_state_dict(torch.load(f"{CKPT_DIR}\\recovery_12.0.pt",  weights_only=True))

    # E is only ever called inside torch.no_grad(), so eval() is fine for it.
    # S and R participate in the generator backward graph (g_sup, g_mom, spec_loss),
    # so they MUST stay in train() mode — cuDNN RNN backward requires training mode.
    E.eval()
    for p in list(E.parameters()) + list(S.parameters()) + list(R.parameters()):
        p.requires_grad_(False)

    return E, G, S, R, D


def _val_score(
    G, S, R, E,
    val_loader: DataLoader,
    device: torch.device,
    orig_labels_map,
) -> float:
    """
    Evaluate the generator on the validation set.

    Returns the mean of  acf_error + lsd  (lower is better).
    """
    G.eval(); S.eval(); R.eval()
    total = 0.0
    n = 0

    try:
        with torch.no_grad():
            for batch in val_loader:
                x_v, v_labels = batch
                x_v      = x_v.to(device)
                v_labels = v_labels.to(device)
                orig_v   = orig_labels_map[v_labels] if orig_labels_map is not None else v_labels

                B, T, _ = x_v.shape
                z = torch.randn(B, T, NOISE_DIM, device=device)

                h_fake  = G(z, v_labels)
                h_fakes = S(h_fake, orig_v)
                x_fake  = R(h_fakes, orig_v).float()

                total += (acf_error(x_v, x_fake) + ssvep_spectral_loss(x_v, x_fake)).item()
                n += 1
    finally:
        # Always restore training mode — cuDNN RNN backward requires it.
        G.train(); S.train(); R.train()

    return total / max(1, n)


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
    orig_labels_map,
) -> float:
    """
    Stripped-down version of train_timegan suited for Optuna trials.

    Key simplifications vs. production loop:
    - No AMP / GradScaler (keeps code short; acceptable for short runs).
    - No per-metric checkpointing.
    - Reports intermediate values to Optuna so MedianPruner can kill bad trials.
    - Returns the final validation score.
    """
    opt_gs = torch.optim.Adam(
        list(G.parameters()) + list(S.parameters()),
        lr=LR_GENERATOR, betas=(0.0, 0.9),
    )
    opt_d = torch.optim.Adam(D.parameters(), lr=LR_DISCRIMINATOR)

    sup_window = SUP_LOSS_WINDOW

    for epoch in range(TUNE_EPOCHS):
        # Explicitly set training mode at the start of every epoch to guard
        # against any accidental eval() left over from validation.
        # R must also be train() — it is in the backward graph via g_mom / spec_loss.
        G.train(); S.train(); R.train(); D.train()

        for batch_idx, batch in enumerate(train_loader):
            x, labels = batch
            x      = x.to(device)
            labels = labels.to(device)

            orig_labels = orig_labels_map[labels] if orig_labels_map is not None else labels

            B, T, _ = x.shape

            # Frozen embedder → real latents
            with torch.no_grad():
                h_real, _ = E(x, orig_labels)

            # ── Discriminator step ────────────────────────────────────────
            with torch.no_grad():
                z_d     = torch.randn(B, T, NOISE_DIM, device=device)
                h_fake_d = G(z_d, labels)
                h_fake_d = S(h_fake_d, orig_labels)

            d_real = D(h_real, labels)
            d_fake = D(h_fake_d, labels)
            gp     = gradient_penalty(D, h_real, h_fake_d, device=device, labels=labels)
            d_loss = discriminator_loss(d_real, d_fake) + 10.0 * gp

            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()

            # ── Generator + Supervisor step  (every 5 batches) ───────────
            if batch_idx % 5 == 0:
                z      = torch.randn(B, T, NOISE_DIM, device=device)
                h_fake = G(z, labels)
                h_fakes = S(h_fake, orig_labels)

                # Adversarial
                g_adv = generator_adv_loss(D(h_fakes, labels))

                # Supervised prediction loss
                g_sup_fake = torch.mean(
                    (h_fakes[:, :-sup_window, :] - h_fake[:, sup_window:, :]) ** 2
                )
                h_real_slice = h_real[:, :-sup_window, :].detach()
                h_real_pred  = S(h_real_slice, orig_labels)
                g_sup_real   = torch.mean(
                    (h_real_pred - h_real[:, sup_window:, :]) ** 2
                )
                g_sup = g_sup_fake + g_sup_real

                # Moment matching in data space
                x_fake = R(h_fakes, orig_labels).float()
                mean_loss = torch.mean((x.mean(0)           - x_fake.mean(0))           ** 2)
                var_loss  = torch.mean((x.var(0, unbiased=False) - x_fake.var(0, unbiased=False)) ** 2)
                idx = torch.randperm(T, device=device)[:64]
                cov_real = torch.cov(x[:, idx, :].reshape(-1, x.shape[-1]).T)
                cov_fake = torch.cov(x_fake[:, idx, :].reshape(-1, x_fake.shape[-1]).T)
                g_mom = mean_loss + var_loss + torch.mean((cov_real - cov_fake) ** 2)

                # Spectral (LSD)
                spec_loss = ssvep_spectral_loss(x, x_fake)

                g_loss = g_adv + lambda_spec * spec_loss + lambda_mom * g_mom + lambda_sup * g_sup

                opt_gs.zero_grad(set_to_none=True)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(G.parameters()) + list(S.parameters()), max_norm=5.0
                )
                opt_gs.step()

        # ── Intermediate report to Optuna (enables pruning) ───────────────
        val_score = _val_score(G, S, R, E, val_loader, device, orig_labels_map)
        trial.report(val_score, step=epoch)

        logger.info(
            f"  Trial {trial.number} | Epoch {epoch:02d}/{TUNE_EPOCHS-1} "
            f"| val_score={val_score:.5f} "
            f"| lambda_sup={lambda_sup:.1f} lambda_mom={lambda_mom:.1f} lambda_spec={lambda_spec:.3f}"
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
    orig_labels_map,
    n_classes: int,
) -> float:
    """
    Suggest lambda values, run the short training loop, return the val score.

    Search space
    ~~~~~~~~~~~~
    lambda_sup  : log-uniform in [1, 200]   (was 85 in production)
    lambda_mom  : log-uniform in [1, 100]   (was 15 in production)
    lambda_spec : log-uniform in [0.01, 5]  (was 0.2 in production)
    """
    lambda_sup  = trial.suggest_float("lambda_sup",  50.0,  200.0, log=True)
    lambda_mom  = trial.suggest_float("lambda_mom",  1.0,  50.0, log=True)
    lambda_spec = trial.suggest_float("lambda_spec", 0.01,   5.0, log=True)

    logger.info(
        f"Trial {trial.number} started | "
        f"lambda_sup={lambda_sup:.3f}  lambda_mom={lambda_mom:.3f}  lambda_spec={lambda_spec:.4f}"
    )

    set_seed(trial.number)  # reproducible but different per trial
    E, G, S, R, D = _build_models(device, n_classes)

    val_score = _run_trial_training(
        G, D, S, R, E,
        train_loader, val_loader,
        device, trial,
        lambda_sup, lambda_mom, lambda_spec,
        orig_labels_map,
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

    n_classes = len(GAN_CLASSES) if GAN_CLASSES is not None else 15

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

    orig_labels_map = (
        torch.tensor(GAN_CLASSES, dtype=torch.long, device=device)
        if GAN_CLASSES is not None else None
    )

    # ── Optuna study ──────────────────────────────────────────────────────
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,   # resume if the study already exists in the DB
    )

    logger.info(f"Study '{STUDY_NAME}' loaded/created. Running {N_TRIALS} trials …")

    study.optimize(
        lambda trial: objective(
            trial, train_loader, val_loader, device, orig_labels_map, n_classes
        ),
        n_trials=N_TRIALS,
        gc_after_trial=True,   # free GPU memory between trials
    )

    # ── Report results ────────────────────────────────────────────────────
    best = study.best_trial
    logger.info("=" * 60)
    logger.info(f"Best trial: #{best.number}  |  val_score = {best.value:.6f}")
    logger.info(f"  lambda_sup  = {best.params['lambda_sup']:.4f}")
    logger.info(f"  lambda_mom  = {best.params['lambda_mom']:.4f}")
    logger.info(f"  lambda_spec = {best.params['lambda_spec']:.4f}")
    logger.info("=" * 60)

    # Save to JSON so GAN.py / config.py can pick them up easily
    out = {
        "lambda_sup":  best.params["lambda_sup"],
        "lambda_mom":  best.params["lambda_mom"],
        "lambda_spec": best.params["lambda_spec"],
        "val_score":   best.value,
        "trial":       best.number,
    }
    with open("best_lambdas.json", "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Best lambdas saved to best_lambdas.json")
    print("\nBest lambdas:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

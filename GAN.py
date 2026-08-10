import torch
from torch.utils.data import DataLoader

from src.utils.config import (
    LATENT_DIM,
    LR_GENERATOR,
    LR_DISCRIMINATOR,
    EPOCHS_JOINT,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    BATCH_SIZE,
    NUM_CHANNELS,
    WANTED_CHANNELS,
    NUM_LAYERS_EMBEDDER,
    NUM_LAYERS_SUPERVISOR,
    NUM_LAYERS_GENERATOR,
    NUM_LAYERS_DISCRIMINATOR,
    NUM_LAYERS_RECOVERY,
    HIDDEN_DIM_DISCRIMINATOR,
    HIDDEN_DIM_GENERATOR,
    WARMUP_EPOCHS,
    LAMBDA_SUP,
    LAMBDA_MOM,
    LAMBDA_SPEC,
    LAMBDA_ADV,
    NUM_CLASSES,
    LABEL_EMB_DIM,
    NOISE_DIM,
    FREQ_DIM,
    FREQ_N_HARMONICS,
    SSVEP_FS,
    ALL_STIM_FREQS,
)

from src.utils.seed import set_seed
from src.utils.logging import get_logger
import logging

# ---------------------------------------------------------------------------
# Class filter — set to None to train on all classes in the .h5 files,
# or pass a list of 0-based label integers to restrict the GAN to a subset.
# Example: [0, 1] trains on the first two classes only.
# The autoencoder / supervisor .h5 files are unchanged.
# ---------------------------------------------------------------------------
GAN_CLASSES = [3, 6]  # <- edit here; None = all classes
from src.data.H5_dataset import EEGH5Dataset
from src.models.embedder import cEmbedder
from src.models.recovery import cRecovery
from src.models.supervisor import Supervisor
from src.models.generator import cGenerator
from src.models.discriminator import cTCNDiscriminator

from src.training.train_timegan import train_timegan


def main():
    set_seed(42)
    logger = get_logger("TimeGAN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Data — EEGH5Dataset yields (x, label) pairs.
    # Pass keep_classes to restrict the GAN to a subset of the full
    # label set that was used for the autoencoder / supervisor.
    # ------------------------------------------------------------------
    n_gan_classes = len(GAN_CLASSES) if GAN_CLASSES is not None else NUM_CLASSES
    logger.info(
        f"GAN class filter: {GAN_CLASSES}  ({n_gan_classes} class(es))"
    )

    train_dataset = EEGH5Dataset("data/processed/eeg_train_8.h5", keep_classes=GAN_CLASSES)
    val_dataset   = EEGH5Dataset("data/processed/eeg_val_8.h5",   keep_classes=GAN_CLASSES)

    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )

    logger.info(f"Dataset windows: {len(train_dataset)}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    E = cEmbedder(
        x_dim=NUM_CHANNELS,
        h_dim=LATENT_DIM
    ).to(device)

    S = Supervisor(
        h_dim=LATENT_DIM,
        num_layers=NUM_LAYERS_SUPERVISOR,
        freq_dim=FREQ_DIM,
    ).to(device)

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

    R = cRecovery(
        h_dim=LATENT_DIM,
        x_dim=NUM_CHANNELS,
    ).to(device)

    # Load pre-trained autoencoder weights (E, S, R)
    ckpt_dir = "checkpoints"
    E.load_state_dict(torch.load(f"{ckpt_dir}/embedder_24_500.pt",  weights_only=True))
    S.load_state_dict(torch.load(f"{ckpt_dir}/supervisor_24_50.pt", weights_only=True))
    R.load_state_dict(torch.load(f"{ckpt_dir}/recovery_24_500.pt",  weights_only=True))
    logger.info("Loaded pre-trained E, S, R weights.")

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    optimizer_gs = torch.optim.Adam(
        list(G.parameters()) + list(S.parameters()),   # S is no longer frozen!
        lr=LR_GENERATOR,
        betas=(0.0, 0.9),
    )
    optimizer_d = torch.optim.Adam(
        D.parameters(),
        lr=LR_DISCRIMINATOR,
        betas=(0.0, 0.9),
    )

    # ------------------------------------------------------------------
    # Train TimeGAN
    # ------------------------------------------------------------------
    # Build the stim_freqs list for the GAN subset (local label -> frequency)
    if GAN_CLASSES is not None:
        gan_stim_freqs = [ALL_STIM_FREQS[c] for c in GAN_CLASSES]
    else:
        gan_stim_freqs = list(ALL_STIM_FREQS)

    orig_labels_map = torch.tensor(GAN_CLASSES, dtype=torch.long).to(device) if GAN_CLASSES is not None else None

    train_timegan(
        E,
        G,
        S,
        R,
        D,
        dataloader,
        val_dataloader,
        optimizer_gs,
        optimizer_d,
        device,
        epochs=EPOCHS_JOINT,
        logger=logger,
        lambda_sup=LAMBDA_SUP,
        lambda_mom=LAMBDA_MOM,
        lambda_spec=LAMBDA_SPEC,
        lambda_adv=LAMBDA_ADV,
        z_dim=NOISE_DIM,
        warmup_epochs=WARMUP_EPOCHS,
        orig_labels_map=orig_labels_map,
        stim_freqs=gan_stim_freqs,
        fs=SSVEP_FS,
        n_harmonics=FREQ_N_HARMONICS,
    )

    # Final explicit saves (best-per-metric are saved inside train_timegan)
    torch.save(G.state_dict(), "checkpoints/generator_250_test.pt")
    torch.save(D.state_dict(), "checkpoints/discriminator_250_test.pt")
    logger.info("Final models saved.")


if __name__ == "__main__":
    main()

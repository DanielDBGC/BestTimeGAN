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
    LAMBDA_SUP,
    LAMBDA_MOM,
    DISCRIMINATOR_THRESHOLD,
    LAMBDA_SPEC,
    SUP_LOSS_WINDOW,
    NUM_CLASSES,
    LABEL_EMB_DIM,
    NOISE_DIM,
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
GAN_CLASSES = [0, 1]  # <- edit here; None = all classes
from src.data.H5_dataset import EEGH5Dataset
from src.models.embedder import cEmbedder
from src.models.recovery import Recovery
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

    train_dataset = EEGH5Dataset("data/processed/eeg_train.h5", keep_classes=GAN_CLASSES)
    val_dataset   = EEGH5Dataset("data/processed/eeg_val.h5",   keep_classes=GAN_CLASSES)

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

    logger.info(f"Dataset windows: {len(train_dataset)}")

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    E = cEmbedder(
        x_dim=NUM_CHANNELS,
        h_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        label_emb_dim=LABEL_EMB_DIM,
        num_layers=NUM_LAYERS_EMBEDDER,
    ).to(device)

    S = Supervisor(
        h_dim=LATENT_DIM,
        num_layers=NUM_LAYERS_SUPERVISOR,
        num_classes=NUM_CLASSES,
        label_emb_dim=LABEL_EMB_DIM,
    ).to(device)

    G = cGenerator(
        z_dim=NOISE_DIM,              # = LATENT_DIM (no +4 workaround)
        h_dim=HIDDEN_DIM_GENERATOR,
        num_layers=NUM_LAYERS_GENERATOR,
        out_dim=LATENT_DIM,
        num_classes=n_gan_classes,    # sized to the GAN subset
        label_emb_dim=LABEL_EMB_DIM,
    ).to(device)

    D = cTCNDiscriminator(
        in_channels=LATENT_DIM,
        hidden_channels=HIDDEN_DIM_DISCRIMINATOR,
        num_classes=n_gan_classes,    # sized to the GAN subset
        label_emb_dim=LABEL_EMB_DIM,
    ).to(device)

    R = Recovery(
        h_dim=LATENT_DIM,
        x_dim=NUM_CHANNELS,
        num_layers=NUM_LAYERS_RECOVERY,
    ).to(device)

    # Load pre-trained autoencoder weights (E, S, R)
    ckpt_dir = "c:\\Users\\danie_13ucdo4\\OneDrive\\Desktop\\ITAM\\Tesis\\Prueba\\BestTimeGAN\\checkpoints"
    E.load_state_dict(torch.load(f"{ckpt_dir}\\embedder_12.0.pt",  weights_only=True))
    S.load_state_dict(torch.load(f"{ckpt_dir}\\supervisor_12.0.pt", weights_only=True))
    R.load_state_dict(torch.load(f"{ckpt_dir}\\recovery_12.0.pt",  weights_only=True))
    logger.info("Loaded pre-trained E, S, R weights.")

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    optimizer_gs = torch.optim.Adam(
        list(G.parameters()) + list(S.parameters()),
        lr=LR_GENERATOR,
        betas=(0.0, 0.9),
    )
    optimizer_d = torch.optim.Adam(
        D.parameters(),
        lr=LR_DISCRIMINATOR,
    )

    # ------------------------------------------------------------------
    # Train TimeGAN
    # ------------------------------------------------------------------
    orig_labels_map = torch.tensor(GAN_CLASSES, dtype=torch.long, device=device) if GAN_CLASSES is not None else None

    train_timegan(
        E,
        G,
        S,
        R,
        D,
        dataloader,
        optimizer_gs,
        optimizer_d,
        device,
        epochs=EPOCHS_JOINT,
        logger=logger,
        threshold=DISCRIMINATOR_THRESHOLD,
        lambda_sup=LAMBDA_SUP,
        lambda_mom=LAMBDA_MOM,
        lambda_spec=LAMBDA_SPEC,
        z_dim=NOISE_DIM,
        sup_loss_window=SUP_LOSS_WINDOW,
        orig_labels_map=orig_labels_map,
    )

    # Final explicit saves (best-per-metric are saved inside train_timegan)
    torch.save(G.state_dict(), "checkpoints/generator_cond.pt")
    torch.save(D.state_dict(), "checkpoints/discriminator_cond.pt")
    logger.info("Final models saved.")


if __name__ == "__main__":
    main()

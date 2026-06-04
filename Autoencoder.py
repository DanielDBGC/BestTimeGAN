# train_autoencoder.py
import torch
from torch.utils.data import DataLoader, random_split

from src.utils.config import (
    NUM_CHANNELS,
    LATENT_DIM,
    LR_EMBEDDER,
    LR_RECOVERY,
    EPOCHS_EMBEDDER,
    WINDOW_SIZE,
    WINDOW_STRIDE,
    BATCH_SIZE, 
    WANTED_CHANNELS,
    NUM_LAYERS_EMBEDDER,
    NUM_LAYERS_RECOVERY
)

from evaluation.evaluation import run_diagnostics
import json
from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.H5_dataset import EEGH5Dataset
from src.models.embedder import cEmbedder
from src.models.recovery import cRecovery
from src.models.geometry import GeometryLoss
from src.losses.losses import reconstruction_loss
from src.training.train_embedder import train_autoencoder


def main():
    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    set_seed(42)
    logger = get_logger("autoencoder")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    
    STIMULUS_FREQS = list(range(4, 61, 4))  # 6–20 Hz


    train_dataset = EEGH5Dataset("data/processed/eeg.h5")
    val_dataset = EEGH5Dataset("data/processed/eeg_val.h5")


    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    logger.info(f"Dataset windows: {len(train_dataset)}")
    logger.info(f"Val dataset windows: {len(val_dataset)}")

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    E = cEmbedder(
        x_dim=NUM_CHANNELS,
        h_dim=LATENT_DIM
    ).to(device)

    R = cRecovery(
        h_dim=LATENT_DIM,
        x_dim=NUM_CHANNELS
    ).to(device)

    logger.info("Embedder and Recovery initialized")


    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    history = train_autoencoder(
        E,
        R,
        dataloader,
        val_dataloader,
        lr_embedder=LR_EMBEDDER,
        lr_recovery=LR_RECOVERY,
        device=device,
        epochs=EPOCHS_EMBEDDER,
        logger=logger
    )

    run_diagnostics(
        E, R, val_dataloader,
        device=device,
        fs=1000,
        stimulus_freqs=STIMULUS_FREQS[:5],
        num_classes=15,
        logger=logger
    )


    # --------------------------------------------------
    # Save checkpoints
    # --------------------------------------------------
    torch.save(E.state_dict(), "checkpoints/embedder_12.0.pt")
    torch.save(R.state_dict(), "checkpoints/recovery_12.0.pt")

    with open("checkpoints/history.json", "w") as f:
        json.dump(history, f, indent=4)

    logger.info("Autoencoder training complete")


if __name__ == "__main__":
    main()

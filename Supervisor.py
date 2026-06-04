import torch
from torch.utils.data import DataLoader

from src.utils.config import (
    LATENT_DIM,
    LR_SUPERVISOR,
    EPOCHS_SUPERVISOR,
    WINDOW_SIZE,
    WINDOW_STRIDE, 
    BATCH_SIZE,
    WANTED_CHANNELS,
    NUM_LAYERS_EMBEDDER,
    NUM_LAYERS_SUPERVISOR,
    NUM_CHANNELS
)

from src.utils.seed import set_seed
from src.utils.logging import get_logger
from src.data.H5_dataset import EEGH5Dataset
from src.models.embedder import cEmbedder
from src.models.supervisor import Supervisor

from src.training.train_supervisor import train_supervisor

def main():
    set_seed(42)
    logger = get_logger("supervisor")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_dataset = EEGH5Dataset("data/processed/eeg_train.h5")
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

    S = Supervisor(
        h_dim=LATENT_DIM,
        num_layers=NUM_LAYERS_SUPERVISOR
    ).to(device)

    logger.info("Embedder and Supervisor initialized")

    E.load_state_dict(torch.load("c:\\Users\\danie_13ucdo4\\OneDrive\\Desktop\\ITAM\\Tesis\\Prueba\\BestTimeGAN\\checkpoints\\embedder_12.0.pt", weights_only=True))

    logger.info("Embedder loaded")
    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    optimizer = torch.optim.AdamW(S.parameters(), lr=LR_SUPERVISOR, weight_decay=1e-4)
    train_supervisor(
        E,
        S,
        dataloader,
        optimizer,
        device,
        epochs=EPOCHS_SUPERVISOR,
        logger=logger,
        val_loader=val_dataloader
    )

    torch.save(S.state_dict(), "checkpoints/supervisor_12.0.pt")
    logger.info("Supervisor saved")

if __name__ == "__main__": 
    main()
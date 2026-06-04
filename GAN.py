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
    SUP_LOSS_WINDOW
)

from src.utils.seed import set_seed
from src.utils.logging import get_logger
import logging
from src.data.preprocessing import load_multiple_subjects
from src.data.window_dataset import EEGWindowDataset
from src.models.embedder import Embedder
from src.models.recovery import Recovery
from src.models.supervisor import Supervisor
from src.models.generator import Generator
from src.models.discriminator import Discriminator, TCNDiscriminator

from src.training.train_timegan import train_timegan

def main():
    set_seed(42)
    logger = get_logger("TimeGAN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    subj = list(range(1,11))
    blocks = list(range(1,13))

    # eeg = load_subject_frequency(blocks, 12.0, "data/raw", duration_sec=5, l_freq=10.0, h_freq=40.0, picks=WANTED_CHANNELS)
    # eeg = load_multiple_subjects(subj, blocks, 16.0, "data/raw", duration_sec=5, l_freq=10.0, h_freq=40.0, picks=WANTED_CHANNELS)
    eeg = load_multiple_subjects(subj, blocks, 12.0, "data/raw", duration_sec=5, l_freq=10.0, h_freq=40.0, picks=WANTED_CHANNELS)
    logger.info(f"EEG shape: {eeg[0].shape}")
    logger.info(f"Mean after norm: {eeg[0].mean(axis=2)}")
    logger.info(f"Std after norm: {eeg[0].std(axis=2)}")


    dataset = EEGWindowDataset(
        eeg=eeg[0],
        window_size=WINDOW_SIZE,
        hop_size=WINDOW_STRIDE,
        normalize=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    logger.info(f"Dataset windows: {len(dataset)}")

    # Initialize models
    E = Embedder(
        x_dim=NUM_CHANNELS, 
        h_dim=LATENT_DIM, 
        num_layers=NUM_LAYERS_EMBEDDER
    ).to(device)

    S = Supervisor(
        h_dim=LATENT_DIM, 
        num_layers=NUM_LAYERS_SUPERVISOR
    ).to(device)

    G = Generator(
        z_dim=LATENT_DIM+4, 
        h_dim=HIDDEN_DIM_GENERATOR, 
        num_layers=NUM_LAYERS_GENERATOR
    ).to(device)

    D = TCNDiscriminator(
        in_channels=LATENT_DIM,
        hidden_channels=HIDDEN_DIM_DISCRIMINATOR
    ).to(device)

    R = Recovery(
        h_dim=LATENT_DIM, 
        x_dim=NUM_CHANNELS,
        num_layers=NUM_LAYERS_RECOVERY
    ).to(device)

    E.load_state_dict(torch.load("c:\\Users\\danie_13ucdo4\\OneDrive\\Desktop\\ITAM\\Tesis\\Prueba\\BestTimeGAN\\checkpoints\\embedder_12.0.pt", weights_only=True))
    S.load_state_dict(torch.load("c:\\Users\\danie_13ucdo4\\OneDrive\\Desktop\\ITAM\\Tesis\\Prueba\\BestTimeGAN\\checkpoints\\supervisor_12.0.pt", weights_only=True))
    R.load_state_dict(torch.load("c:\\Users\\danie_13ucdo4\\OneDrive\\Desktop\\ITAM\\Tesis\\Prueba\\BestTimeGAN\\checkpoints\\recovery_12.0.pt", weights_only=True))

    # Initialize optimizers
    optimizer_gs = torch.optim.Adam(list(G.parameters()) + list(S.parameters()), lr=LR_GENERATOR, betas=[0.0, 0.9])
    optimizer_d = torch.optim.Adam(D.parameters(), lr=LR_DISCRIMINATOR)

    # Train TimeGAN
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
        z_dim=LATENT_DIM,
        sup_loss_window=SUP_LOSS_WINDOW
    )
    torch.save(G.state_dict(), "checkpoints/generator_12.4.pt")
    torch.save(D.state_dict(), "checkpoints/discriminator_12.4.pt")
    logger.info("Models saved")

if __name__ == "__main__":
    main()

import torch
from torch.amp import autocast, GradScaler
from src.models.embedder import Embedder, cEmbedder
from src.models.generator import Generator, cGenerator
from src.models.supervisor import Supervisor
from src.models.discriminator import Discriminator, cTCNDiscriminator
from src.models.recovery import Recovery, cRecovery
from src.losses.losses import (
    generator_adv_loss,
    discriminator_loss,
    supervised_loss,
    acf_error,
    psd_error,
    lsd,
    gradient_penalty,
)
import logging


def train_timegan(
    E, G, S, R, D,
    dataloader,
    val_dataloader,
    opt_gs,
    opt_d,
    device,
    epochs: int,
    logger,
    threshold: float,
    lambda_sup: float,
    lambda_mom: float,
    lambda_spec: float,
    z_dim: int,
    sup_loss_window: int = 1,
    orig_labels_map: torch.Tensor = None,
):
    """
    Joint GAN training loop for TimeGAN.

    Supports both:
    - Class-conditional models (cGenerator / cTCNDiscriminator) — dataloader
      must yield (x, labels) batches.
    - Legacy unconditional models — dataloader may yield x only.

    Improved checkpointing: tracks best ACF, best PSD, and best combined
    score separately and saves a checkpoint for each.
    """
    logger.info("Starting TimeGAN joint training...")

    E.eval()  # frozen permanently

    scaler_gs = GradScaler('cuda')
    scaler_d  = GradScaler('cuda')

    is_cond_E = isinstance(E, cEmbedder)
    is_cond_G = isinstance(G, cGenerator)
    is_cond_D = isinstance(D, cTCNDiscriminator)
    is_cond_R = isinstance(R, cRecovery)

    # ------------------------------------------------------------------
    # Validation data will be evaluated using val_dataloader
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-metric bests + patience
    # ------------------------------------------------------------------
    best_acf    = float("inf")
    best_psd    = float("inf")
    best_total  = float("inf")

    patience        = 20
    patience_counter = 0
    stop_training   = False

    def _save(tag, state):
        torch.save(state, f"checkpoints/best_timegan_{tag}.pt")
        logger.info(f"Checkpoint saved: best_timegan_{tag}.pt")

    checkpoint_state = lambda: {
        "E": E.state_dict(),
        "G": G.state_dict(),
        "S": S.state_dict(),
        "R": R.state_dict(),
        "D": D.state_dict(),
    }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Unpack labels when the dataset is conditional
            if isinstance(batch, (tuple, list)):
                x, labels = batch
                labels = labels.to(device)
            else:
                x      = batch
                labels = None

            # Get the original labels for pre-trained models E and S
            if labels is not None and orig_labels_map is not None:
                orig_labels = orig_labels_map[labels]
            else:
                orig_labels = labels

            x = x.to(device)
            B, T, _ = x.shape

            with torch.no_grad():
                if is_cond_E:
                    h_real, _ = E(x, orig_labels)
                else:
                    h_real = E(x)

            # ===========================================================
            # (1) Discriminator update  — 1 step per batch
            # ===========================================================
            
            with torch.no_grad():
                z_d    = torch.randn(B, T, z_dim, device=device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h_fake_d = G(z_d, labels) if is_cond_G else G(z_d)
                    h_fake_d = S(h_fake_d, orig_labels) if orig_labels is not None else S(h_fake_d)

            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                d_real = D(h_real, labels) if is_cond_D else D(h_real)
                d_fake = D(h_fake_d, labels) if is_cond_D else D(h_fake_d)
                d_loss_val = discriminator_loss(d_real, d_fake)

            gp = gradient_penalty(D, h_real, h_fake_d, device=device, labels=labels)
            d_loss = d_loss_val + 10.0 * gp

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ===========================================================
            # (2) Generator + Supervisor update — every 5 batches
            # ===========================================================
            if batch_idx % 5 == 0:
                z = torch.randn(B, T, z_dim, device=device)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    # Fake latent trajectory
                    h_fake     = G(z, labels) if is_cond_G else G(z)          # [B, T, H]
                    h_fake_sup = S(h_fake, orig_labels) if orig_labels is not None else S(h_fake)  # [B, T, H]

                    # Adversarial loss
                    d_fake_g = D(h_fake_sup, labels) if is_cond_D else D(h_fake_sup)
                    g_adv    = generator_adv_loss(d_fake_g)

                    # -----------------------------------------------------------
                    # Supervised loss
                    # -----------------------------------------------------------
                    g_sup_fake = torch.mean(
                        (h_fake_sup[:, :-sup_loss_window, :] - h_fake[:, sup_loss_window:, :]) ** 2
                    )

                    h_real_slice = h_real[:, :-sup_loss_window, :].detach()
                    h_real_pred  = (
                        S(h_real_slice, orig_labels) if orig_labels is not None else S(h_real_slice)
                    )
                    g_sup_real = torch.mean(
                        (h_real_pred - h_real[:, sup_loss_window:, :]) ** 2
                    )

                    g_sup = g_sup_fake + g_sup_real

                    # -----------------------------------------------------------
                    # Moment matching in data space
                    # -----------------------------------------------------------
                    x_fake = R(h_fake_sup, orig_labels) if is_cond_R else R(h_fake_sup)
                    x_fake = x_fake.float()  # ensure float32 for stable statistics/spectral ops

                    mean_real = torch.mean(x,      dim=0)
                    mean_fake = torch.mean(x_fake, dim=0)

                    var_real  = torch.var(x,      dim=0, unbiased=False)
                    var_fake  = torch.var(x_fake, dim=0, unbiased=False)

                    # Subsample T dimension for covariance
                    idx = torch.randperm(T, device=device)[:64]
                    x_sub      = x[:, idx, :].reshape(-1, x.shape[-1])
                    x_fake_sub = x_fake[:, idx, :].reshape(-1, x_fake.shape[-1])
                    cov_real   = torch.cov(x_sub.T)
                    cov_fake   = torch.cov(x_fake_sub.T)

                    mean_loss = torch.mean((mean_real - mean_fake) ** 2)
                    var_loss  = torch.mean((var_real  - var_fake)  ** 2)
                    cov_loss  = torch.mean((cov_real  - cov_fake)  ** 2)

                    g_mom = mean_loss + var_loss + cov_loss

                    # -----------------------------------------------------------
                    # Spectral loss (Log-Spectral Distance)
                    # -----------------------------------------------------------
                    spectral_loss = lsd(x, x_fake)

                    # -----------------------------------------------------------
                    # Total generator loss
                    # -----------------------------------------------------------
                    g_loss = g_adv + lambda_sup * g_sup + lambda_mom * g_mom + lambda_spec * spectral_loss

                opt_gs.zero_grad(set_to_none=True)
                scaler_gs.scale(g_loss).backward()
                scaler_gs.step(opt_gs)
                scaler_gs.update()

        # ---------------------------------------------------------------
        # Logging + per-metric checkpoint saving (every 5 epochs)
        # ---------------------------------------------------------------
        if epoch % 5 == 0:
            with torch.no_grad():
                G.eval(); S.eval(); R.eval()

                acf_val_accum = 0.0
                psd_val_accum = 0.0
                n_val_batches = 0

                for val_batch in val_dataloader:
                    if isinstance(val_batch, (tuple, list)):
                        v_batch, v_labels = val_batch
                    else:
                        v_batch, v_labels = val_batch, None
                    
                    v_batch = v_batch.to(device)
                    v_labels = v_labels.to(device) if v_labels is not None else None

                    B_v, T_v, _ = v_batch.shape
                    z_val = torch.randn(B_v, T_v, z_dim, device=device)

                    orig_v_labels = orig_labels_map[v_labels] if (v_labels is not None and orig_labels_map is not None) else v_labels

                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        h_fv    = G(z_val, v_labels) if is_cond_G else G(z_val)
                        h_fv_s  = S(h_fv, orig_v_labels) if orig_v_labels is not None else S(h_fv)
                        x_fv    = R(h_fv_s, orig_v_labels) if is_cond_R else R(h_fv_s)
                    x_fv = x_fv.float()

                    acf_val_accum += acf_error(v_batch, x_fv)
                    psd_val_accum += lsd(v_batch, x_fv)
                    n_val_batches += 1

                acf_val = acf_val_accum / max(1, n_val_batches)
                psd_val = psd_val_accum / max(1, n_val_batches)
                total_val = acf_val + psd_val

                G.train(); S.train(); R.train()

                improved_any = False

                if acf_val < best_acf:
                    best_acf = acf_val
                    _save("acf",   checkpoint_state())
                    improved_any = True

                if psd_val < best_psd:
                    best_psd = psd_val
                    _save("psd",   checkpoint_state())
                    improved_any = True

                if total_val < best_total:
                    best_total = total_val
                    _save("total", checkpoint_state())
                    improved_any = True

                if improved_any:
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    stop_training = True

                logger.info(
                    f"[Joint] Epoch {epoch:03d} | "
                    f"D: {d_loss.item():.4f} | "
                    f"G_adv: {g_adv.item():.4f} | "
                    f"G_sup: {g_sup.item():.4f} | "
                    f"G_mom: {g_mom.item():.4f} | "
                    f"PSD_err: {spectral_loss.item():.4f} | "
                    f"Val ACF: {acf_val.item():.4f} | "
                    f"Val PSD: {psd_val.item():.4f} | "
                    f"Val total: {total_val.item():.4f} | "
                    f"Patience: {patience_counter}/{patience}"
                )

        if stop_training:
            break

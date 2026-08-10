import torch
from torch.amp import autocast, GradScaler
from src.models.embedder import Embedder, cEmbedder
from src.models.generator import cGenerator
from src.models.supervisor import Supervisor
from src.models.discriminator import Discriminator, cTCNDiscriminator
from src.models.recovery import Recovery, cRecovery
from src.models.freq_conditioning import build_freq_basis
from src.losses.losses import (
    generator_adv_loss,
    discriminator_loss,
    acf_error,
    ssvep_corr_loss,
    r1_penalty,
)
from src.utils.config import ALL_STIM_FREQS
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
    lambda_sup: float,
    lambda_mom: float,
    lambda_spec: float,
    lambda_adv: float,
    z_dim: int,
    warmup_epochs: int = 10,
    orig_labels_map: torch.Tensor = None,
    stim_freqs: list = None,
    fs: float = 1000.0,
    n_harmonics: int = 3,
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

    E.eval()  # frozen permanently — always used inside torch.no_grad(), eval() is safe

    # S is NO LONGER frozen! We unfreeze it so it adapts with the generator.
    # R remains frozen because it maps latents to data.
    S.requires_grad_(True)
    R.requires_grad_(False)

    # However, since they are in train() mode, Dropout is still active! 
    if hasattr(S, 'rnn'):
        S.rnn.dropout = 0.0
    if hasattr(R, 'rnn'):
        R.rnn.dropout = 0.0

    scaler_gs = GradScaler(device.type, enabled=device.type == "cuda")
    scaler_d  = GradScaler(device.type, enabled=device.type == "cuda")
    
    scheduler_gs = torch.optim.lr_scheduler.CosineAnnealingLR(opt_gs, T_max=epochs)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs)

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

    patience        = 50
    patience_counter = 0
    stop_training   = False

    def _save(tag, state):
        torch.save(state, f"checkpoints/best_timegan_250_test_{tag}.pt")
        logger.info(f"Checkpoint saved: best_timegan_250_test_{tag}.pt")

    checkpoint_state = lambda: {
        "E": E.state_dict(),
        "G": G.state_dict(),
        "S": S.state_dict(),
        "R": R.state_dict(),
        "D": D.state_dict(),
    }

    # ------------------------------------------------------------------
    # Build a reverse map: original label value -> local 0-based index
    # used by G and D (which have num_classes = n_gan_classes, not NUM_CLASSES).
    # E.g. if orig_labels_map = [2, 3], then raw label 2 -> 0, raw label 3 -> 1.
    # ------------------------------------------------------------------
    if orig_labels_map is not None:
        _inv_map = {int(v): i for i, v in enumerate(orig_labels_map.tolist())}
        def _to_local(raw_labels: torch.Tensor) -> torch.Tensor:
            """Remap raw dataset labels to 0-based local indices for G/D."""
            return torch.tensor(
                [_inv_map[int(l)] for l in raw_labels.tolist()],
                dtype=torch.long,
                device=raw_labels.device,
            )
    else:
        def _to_local(raw_labels):
            return raw_labels

    # ------------------------------------------------------------------
    # Supervised Warm-Up Loop (Generator only)
    # ------------------------------------------------------------------
    if warmup_epochs > 0:
        logger.info(f"Starting {warmup_epochs} epochs of supervised warm-up for Generator...")
        for epoch in range(warmup_epochs):
            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (tuple, list)):
                    x, raw_labels = batch
                    raw_labels = raw_labels.to(device)
                else:
                    x          = batch
                    raw_labels = None

                if raw_labels is not None:
                    labels_local = _to_local(raw_labels)
                    labels_local = labels_local.to(device)
                    
                    orig_labels  = orig_labels_map[labels_local] if orig_labels_map is not None else raw_labels

                else:
                    labels_local = None
                    orig_labels  = None

                x = x.to(device)
                B, T, _ = x.shape
                z = torch.randn(B, T, z_dim, device=device)

                # Build freq basis for G/D (local labels) and S (orig labels)
                freq_basis_local = build_freq_basis(labels_local, T, stim_freqs, fs, n_harmonics).to(device) if labels_local is not None else None
                freq_basis_orig = build_freq_basis(orig_labels, T, ALL_STIM_FREQS, fs, n_harmonics).to(device) if orig_labels is not None else None

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    h_fake     = G(z, freq_basis_local) if is_cond_G else G(z)
                    h_fake_sup = S(h_fake, freq_basis_orig) if freq_basis_orig is not None else S(h_fake)

                    # Supervised loss
                    g_sup = torch.mean((h_fake_sup[:, :-1, :] - h_fake[:, 1:, :]) ** 2)

                    # Moment matching
                    x_fake = R(h_fake_sup, orig_labels) if is_cond_R else R(h_fake_sup)
                    x_fake = x_fake.float()

                    mean_real = torch.mean(x, dim=0)
                    mean_fake = torch.mean(x_fake, dim=0)
                    var_real  = torch.var(x, dim=0, unbiased=False)
                    var_fake  = torch.var(x_fake, dim=0, unbiased=False)

                    mean_loss = torch.mean((mean_real - mean_fake) ** 2)
                    var_loss  = torch.mean((var_real  - var_fake)  ** 2)
                    g_mom = mean_loss + var_loss

                    # Spectral loss (Correlation)
                    physical_freqs = torch.tensor(ALL_STIM_FREQS, device=device, dtype=torch.float32)[orig_labels]
                    spec_loss = ssvep_corr_loss(x_fake, physical_freqs, fs)

                    g_loss = lambda_spec * spec_loss + lambda_mom * g_mom + lambda_sup * g_sup 

                opt_gs.zero_grad(set_to_none=True)
                scaler_gs.scale(g_loss).backward()
                scaler_gs.unscale_(opt_gs)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
                scaler_gs.step(opt_gs)
                scaler_gs.update()

            logger.info(
                f"[Warm-up] Epoch {epoch:03d}/{warmup_epochs-1} | "
                f"G_sup: {g_sup.item():.4f} | "
                f"G_mom: {g_mom.item():.4f} | "
                f"SNR_err: {spec_loss.item():.4f}"
            )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Unpack labels when the dataset is conditional
            if isinstance(batch, (tuple, list)):
                x, raw_labels = batch
                raw_labels = raw_labels.to(device)
            else:
                x          = batch
                raw_labels = None

            # labels_local: 0-based indices for G and D embedding tables
            # orig_labels:  original class values for pre-trained E, S, R
            if raw_labels is not None:
                labels_local = _to_local(raw_labels)                           # e.g. {2,3} -> {0,1}
                orig_labels  = orig_labels_map[labels_local] if orig_labels_map is not None else raw_labels
            else:
                labels_local = None
                orig_labels  = None

            x = x.to(device)
            B, T, _ = x.shape

            # Build freq basis for this batch
            freq_basis_local = build_freq_basis(labels_local, T, stim_freqs, fs, n_harmonics) if labels_local is not None else None
            freq_basis_orig = build_freq_basis(orig_labels, T, ALL_STIM_FREQS, fs, n_harmonics) if orig_labels is not None else None

            with torch.no_grad():
                if is_cond_E:
                    h_real, _ = E(x, orig_labels)
                else:
                    h_real = E(x)

            # ===========================================================
            # (1) Discriminator update
            # ===========================================================
            with torch.no_grad():
                z_d    = torch.randn(B, T, z_dim, device=device)
                h_fake_d = G(z_d, freq_basis_local) if is_cond_G else G(z_d)
                h_fake_d = S(h_fake_d, freq_basis_orig) if freq_basis_orig is not None else S(h_fake_d)

            # Requires grad on real latents for R1 penalty
            h_real.requires_grad_(True)
            
            with torch.autocast(device_type="cuda"):
                d_real = D(h_real, freq_basis_local) if is_cond_D else D(h_real)
                d_fake = D(h_fake_d, freq_basis_local) if is_cond_D else D(h_fake_d)
                d_loss_val = discriminator_loss(d_real, d_fake)
            
            with torch.autocast(device_type="cuda", enabled=False):
                r1 = r1_penalty(d_real, h_real)
                d_loss = d_loss_val + 10.0 * (10.0 / 2.0) * r1  # Gamma/2 * R1

            opt_d.zero_grad(set_to_none=True)
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            # ===========================================================
            # (2) Generator + Supervisor update — every 5 batches (D:G = 5:1)
            # ===========================================================
            if batch_idx % 5 == 0:
                z = torch.randn(B, T, z_dim, device=device)

                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    # Fake latent trajectory
                    h_fake     = G(z, freq_basis_local) if is_cond_G else G(z)    # [B, T, H]
                    h_fake_sup = S(h_fake, freq_basis_orig) if freq_basis_orig is not None else S(h_fake)  # [B, T, H]

                    # Adversarial loss on supervisor-smoothed output
                    d_fake_g = D(h_fake_sup, freq_basis_local) if is_cond_D else D(h_fake_sup)
                    g_adv    = generator_adv_loss(d_fake_g)

                    g_adv_total = generator_adv_loss(d_fake_g)

                    # -----------------------------------------------------------
                    # Supervised loss
                    # Since S is unfrozen, we also want it to predict real trajectories well.
                    # -----------------------------------------------------------
                    g_sup_fake = torch.mean((h_fake_sup[:, :-1, :] - h_fake[:, 1:, :]) ** 2)
                    
                    h_real_slice = h_real[:, :-1, :].detach()
                    # Slice freq_basis to match the shortened sequence for S
                    freq_basis_orig_slice = freq_basis_orig[:, :-1, :] if freq_basis_orig is not None else None
                    h_real_pred = S(h_real_slice, freq_basis_orig_slice) if freq_basis_orig_slice is not None else S(h_real_slice)
                    g_sup_real = torch.mean((h_real_pred - h_real[:, 1:, :].detach()) ** 2)
                    
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

                    mean_loss = torch.mean((mean_real - mean_fake) ** 2)
                    var_loss  = torch.mean((var_real  - var_fake)  ** 2)

                    g_mom = mean_loss + var_loss

                    # -----------------------------------------------------------
                    # Spectral loss (Correlation)
                    # -----------------------------------------------------------
                    physical_freqs = torch.tensor(ALL_STIM_FREQS, device=device, dtype=torch.float32)[orig_labels]
                    spec_loss = ssvep_corr_loss(x_fake, physical_freqs, fs)

                    # -----------------------------------------------------------
                    # Total generator loss
                    # -----------------------------------------------------------
                    g_loss = lambda_adv * g_adv_total + lambda_spec * spec_loss + lambda_mom * g_mom + lambda_sup * g_sup  

                opt_gs.zero_grad(set_to_none=True)
                scaler_gs.scale(g_loss).backward()
                scaler_gs.unscale_(opt_gs)
                torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=5.0)
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
                        v_batch, v_raw_labels = val_batch
                    else:
                        v_batch, v_raw_labels = val_batch, None
                    
                    v_batch = v_batch.to(device)
                    v_raw_labels = v_raw_labels.to(device) if v_raw_labels is not None else None

                    B_v, T_v, _ = v_batch.shape
                    z_val = torch.randn(B_v, T_v, z_dim, device=device)

                    if v_raw_labels is not None:
                        v_labels_local = _to_local(v_raw_labels)
                        orig_v_labels  = orig_labels_map[v_labels_local] if orig_labels_map is not None else v_raw_labels
                    else:
                        v_labels_local = None
                        orig_v_labels  = None

                    # Build freq basis for validation batch
                    v_freq_basis_local = build_freq_basis(v_labels_local, T_v, stim_freqs, fs, n_harmonics) if v_labels_local is not None else None
                    v_freq_basis_orig = build_freq_basis(orig_v_labels, T_v, ALL_STIM_FREQS, fs, n_harmonics) if orig_v_labels is not None else None

                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        h_fv    = G(z_val, v_freq_basis_local) if is_cond_G else G(z_val)
                        h_fv_s  = S(h_fv, v_freq_basis_orig) if v_freq_basis_orig is not None else S(h_fv)
                        x_fv    = R(h_fv_s, orig_v_labels) if is_cond_R else R(h_fv_s)
                    x_fv = x_fv.float()

                    acf_val_accum += acf_error(v_batch, x_fv)
                    v_physical_freqs = torch.tensor(ALL_STIM_FREQS, device=device, dtype=torch.float32)[orig_v_labels]
                    psd_val_accum += ssvep_corr_loss(x_fv, v_physical_freqs, fs)
                    n_val_batches += 1

                acf_val = acf_val_accum / max(1, n_val_batches)
                psd_val = psd_val_accum / max(1, n_val_batches)
                # Scale psd_val (SNR error in dB) down so it doesn't overpower ACF error
                total_val = acf_val + 0.001 * psd_val

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
                    f"G_adv: {g_adv_total.item():.4f} | "
                    f"G_sup: {g_sup.item():.4f} | "
                    f"G_mom: {g_mom.item():.4f} | "
                    f"SNR_err: {spec_loss.item():.4f} | "
                    f"Val ACF: {acf_val.item():.4f} | "
                    f"Val SNR: {psd_val.item():.4f} | "
                    f"Val total: {total_val.item():.4f} | "
                    f"Patience: {patience_counter}/{patience}"
                )

        scheduler_gs.step()
        scheduler_d.step()

        if stop_training:
            break

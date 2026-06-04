import torch
import torch.nn.functional as F
from src.models.embedder import cEmbedder
from src.models.recovery import cRecovery
from torch.optim import Adam
from src.losses.losses import reconstruction_loss, spectral_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from evaluation.evaluation import (
    per_channel_mse, per_class_mse, ssvep_snr,
    latent_std_per_class, log_gradient_norms
)

def adaptive_lambda(base_lambda: float,
                    aux_loss: torch.Tensor,
                    recon_loss: torch.Tensor,
                    max_fraction: float = 0.20) -> float:
    """
    Scales base_lambda down if aux_loss * base_lambda would exceed
    max_fraction of recon_loss. Prevents any auxiliary objective from
    overwhelming reconstruction, regardless of their absolute magnitudes.
    """
    with torch.no_grad():
        if aux_loss.item() < 1e-8:
            return base_lambda
        max_contrib = max_fraction * recon_loss.item()
        natural_contrib = base_lambda * aux_loss.item()
        if natural_contrib > max_contrib:
            return max_contrib / aux_loss.item()
    return base_lambda



def train_autoencoder(
    E: cEmbedder,
    R: cRecovery,
    train_loader,
    val_loader,
    logger,
    *,
    epochs:          int   = 200,
    lr_embedder:     float = 5e-4,
    lr_recovery:     float = 1e-3,
    device:          torch.device = torch.device("cpu"),
    fs:              float = 1000.0,
    # Phase 1: reconstruction-only warmup (fraction of total epochs)
    warmup_fraction: float = 0.2,
    # Final lambda values for auxiliary losses
    lambda_spec_max: float = 0.25,
    lambda_cls_max:  float = 0.08,
    num_classes:     int   = 15,
    log_every:       int   = 10,
    grad_log_every:  int   = 50,
) -> dict:
    """
    Two-phase training strategy
    ---------------------------
    Phase 1 (first warmup_fraction of epochs):
        Only reconstruction loss. Lets the autoencoder learn a stable
        mapping before auxiliary losses add competing objectives.
 
    Phase 2 (remaining epochs):
        All losses active, with lambda values linearly warmed up from 0
        to their max values over the second phase.
 
    Separate optimisers
    -------------------
    cEmbedder typically benefits from a lower LR than cRecovery because
    the embedding must remain stable for the downstream TimeGAN components.
    """
    logger.info("Starting training for Autoencoder")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"LR Embedder: {lr_embedder}")
    logger.info(f"LR Recovery: {lr_recovery}")
    logger.info(f"Device: {device}")
    logger.info(f"FS: {fs}")
    logger.info(f"Warmup fraction: {warmup_fraction}")
    logger.info(f"Lambda spec max: {lambda_spec_max}")
    logger.info(f"Lambda cls max: {lambda_cls_max}")
    logger.info(f"Num classes: {num_classes}")
    logger.info(f"Log every: {log_every}")
    logger.info(f"Grad log every: {grad_log_every}")

    warmup_end = int(epochs * warmup_fraction)
    best_val = float('inf')
    patience_counter = 0
    PATIENCE = 20  # stop if val doesn't improve for 20 log intervals

    # Separate optimisers with independent learning rates
    opt_E = Adam(E.parameters(), lr=lr_embedder, weight_decay=1e-5)
    opt_R = Adam(R.parameters(), lr=lr_recovery, weight_decay=1e-5)

    sched_E = CosineAnnealingLR(opt_E, T_max=epochs, eta_min=5e-5)
    sched_R = CosineAnnealingLR(opt_R, T_max=epochs, eta_min=5e-5)

    E.to(device)
    R.to(device)

    history = {
        "train_loss": [],
        "val_loss":   [],
        "recon":      [],
        "spec":       [],
        "cls":        [],
        "h_std":      [],
        "lr_E":       [],
        "lr_R":       [],
    }

    for epoch in range(epochs):

        # ── Lambda schedule ──────────────────────────────────────────────
        # Phase 1: all auxiliary lambdas are 0
        # Phase 2: linearly ramp from 0 to max over the remaining epochs
        if epoch < warmup_end:
            lambda_spec = 0.0
            lambda_cls  = 0.0
            phase = "warmup"
        else:
            progress    = (epoch - warmup_end) / max(1, epochs - warmup_end)
            lambda_spec = lambda_spec_max * progress
            lambda_cls  = lambda_cls_max  * progress
            phase = "full"

        # Reset LR at the beginning of Phase 2 for a fresh start
        if epoch == warmup_end:
            for pg in opt_E.param_groups: pg['lr'] = lr_embedder
            for pg in opt_R.param_groups: pg['lr'] = lr_recovery

        # ── Training loop ─────────────────────────────────────────────────
        E.train()
        R.train()

        epoch_loss  = 0.0
        epoch_recon = 0.0
        epoch_spec  = 0.0
        epoch_cls   = 0.0
        h_std_accum = 0.0
        n_batches   = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)   # [B, T, C]
            y = y.to(device)   # [B]

            # cEmbedder returns (h, logits)
            h, logits = E(x, y)   # h: [B, T, H], logits: [B, num_classes]
            x_tilde   = R(h, y)   # [B, T, C]

            # ── Compute individual losses ──────────────────────────────
            loss_recon = reconstruction_loss(x, x_tilde)
            loss_spec  = spectral_loss(x, x_tilde)
            loss_cls   = F.cross_entropy(logits, y)

            lam_spec = adaptive_lambda(lambda_spec, loss_spec, loss_recon)

            loss = (
                loss_recon
                + lam_spec * loss_spec
                + lambda_cls * loss_cls
            )

            # ── NaN/Inf guard ──────────────────────────────────────────
            if not torch.isfinite(loss):
                logger.error(
                    f"[Epoch {epoch:03d} Batch {batch_idx}] "
                    f"Non-finite loss detected: {loss.item():.6f}. "
                    f"Recon={loss_recon.item():.6f} "
                    f"Spec={loss_spec.item():.6f} "
                    f"Cls={loss_cls.item():.6f}. Aborting."
                )
                return history

            # ── Backward & optimiser step (inside batch loop) ──────────
            opt_E.zero_grad()
            opt_R.zero_grad()
            loss.backward()

            # Log gradient norms periodically for debugging
            if epoch % grad_log_every == 0 and batch_idx == 0:
                log_gradient_norms(E, R)

            torch.nn.utils.clip_grad_norm_(
                list(E.parameters()) + list(R.parameters()),
                max_norm=1.0,
            )

            opt_E.step()
            opt_R.step()

            epoch_loss  += loss.item()
            epoch_recon += loss_recon.item()
            epoch_spec  += loss_spec.item()
            epoch_cls   += loss_cls.item()
            h_std_accum += h.std().item()
            n_batches   += 1

        # Step schedulers once per epoch (after all batches)
        sched_E.step()
        sched_R.step()

        # ── Per-epoch averages ─────────────────────────────────────────
        avg = lambda t: t / max(1, n_batches)
        epoch_loss  = avg(epoch_loss)
        epoch_recon = avg(epoch_recon)
        epoch_spec  = avg(epoch_spec)
        epoch_cls   = avg(epoch_cls)
        h_std_mean  = avg(h_std_accum)

        history["train_loss"].append(epoch_loss)
        history["recon"].append(epoch_recon)
        history["spec"].append(epoch_spec)
        history["cls"].append(epoch_cls)
        history["h_std"].append(h_std_mean)
        history["lr_E"].append(opt_E.param_groups[0]["lr"])
        history["lr_R"].append(opt_R.param_groups[0]["lr"])

        # ── Validation ────────────────────────────────────────────────
        if val_loader is not None and epoch % log_every == 0:
            E.eval()
            R.eval()
            val_loss_accum = 0.0
            with torch.no_grad():
                for x_v, y_v in val_loader:
                    x_v, y_v = x_v.to(device), y_v.to(device)
                    h_v, _   = E(x_v, y_v)
                    x_hat_v  = R(h_v, y_v)
                    val_loss_accum += reconstruction_loss(x_v, x_hat_v).item()
            val_loss = val_loss_accum / max(1, len(val_loader))
            history["val_loss"].append(val_loss)

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save(E.state_dict(), "best_embedder.pt")
                logger.info("Saved best_embedder.pt (new best val loss)")
            else:
                patience_counter += 1
                logger.info(
                    f"Epoch {epoch}: val_loss did not improve. "
                    f"Patience: {patience_counter}/{PATIENCE}"
                )
                if patience_counter >= PATIENCE:
                    logger.info("Early stopping triggered.")
                    break
        else:
            val_loss = float("nan")

        # ── Logging ───────────────────────────────────────────────────
        if epoch % log_every == 0:
            logger.info(
                f"[{phase}] Epoch {epoch:03d}/{epochs} | "
                f"Train: {epoch_loss:.6f} | "
                f"Val: {val_loss:.6f} | "
                f"Recon: {epoch_recon:.6f} | "
                f"Spec({lambda_spec:.3f}): {epoch_spec:.6f} | "
                f"Cls({lambda_cls:.3f}): {epoch_cls:.6f} | "
                f"h.std: {h_std_mean:.4f} | "
                f"LR_E: {opt_E.param_groups[0]['lr']:.2e} | "
                f"LR_R: {opt_R.param_groups[0]['lr']:.2e}"
            )

    return history

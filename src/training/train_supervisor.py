import torch
from src.models.embedder import cEmbedder
from src.models.supervisor import Supervisor
from src.models.freq_conditioning import build_freq_basis
from src.losses.losses import supervisor_loss
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging

def make_nan_hook(name):
    def hook(grad):
        if grad is not None and torch.isnan(grad).any():
            print(f"NaN gradient in: {name} | shape: {grad.shape}")
    return hook

def train_supervisor(
    E,
    S,
    dataloader,
    optimizer,
    device,
    epochs: int,
    logger,
    val_loader=None,
    log_every: int = 5,
    patience: int = 20,
    checkpoint_path: str = "best_supervisor_24_50.pt",
    stim_freqs: list = None,
    fs: float = 1000.0,
    n_harmonics: int = 3,
):
    """
    Train the Supervisor network with optional validation loop and best-
    checkpoint saving.

    Parameters
    ----------
    E               : frozen cEmbedder – used only for inference
    S               : Supervisor to be trained (single-step predictor)
    dataloader      : training DataLoader  (x, y) batches
    optimizer       : optimiser for S
    device          : torch.device
    epochs          : total training epochs
    logger          : logging.Logger instance
    val_loader      : optional validation DataLoader; enables val-loss tracking,
                      best-checkpoint saving, and early stopping when provided
    log_every       : log (and validate) every this many epochs   (default 5)
    patience        : early-stopping patience in *log intervals*  (default 20)
    checkpoint_path : path to save the best-val-loss checkpoint   (default
                      "best_supervisor.pt")
    """
    logger.info("Starting training for Supervisor")
    logger.info(f"Epochs: {epochs} | log_every: {log_every} | patience: {patience}")
    logger.info(f"Device: {device} | Checkpoint: {checkpoint_path}")

    E.eval()    # frozen

    # --------------------------------------------------
    # Hook for NaN detection
    # --------------------------------------------------
    for name, param in S.named_parameters():
        if param.requires_grad:
            param.register_hook(make_nan_hook(name))

    steps_per_epoch = len(dataloader)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
    )

    best_val        = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_mse":  [],
        "train_cons": [],
        "val_loss":   [],
        "val_mse":    [],
    }

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        S.train()

        epoch_loss = 0.0
        epoch_mse  = 0.0
        epoch_cons = 0.0

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                h, _ = E(x, y)                    # (B, T, H)

            B, T, H = h.shape

            # Build freq basis for this batch
            freq_basis = build_freq_basis(y, T, stim_freqs, fs, n_harmonics)

            # Single-step prediction: S(h_{0:T-2}) → predict h_{1:T-1}
            h_input  = h[:, :-1, :]               # (B, T-1, H)
            h_target = h[:, 1:, :]                 # (B, T-1, H)
            fb_input = freq_basis[:, :-1, :]       # (B, T-1, freq_dim)
            h_hat    = S(h_input, fb_input)        # (B, T-1, H)

            # 2-step consistency: S(h_hat_{0:T-3}) should ≈ S(h_{1:T-2})
            fb_2step = fb_input[:, :-1, :]         # (B, T-2, freq_dim)
            h_hat_2step     = S(h_hat[:, :-1, :], fb_2step)  # (B, T-2, H)
            h_hat_from_next = h_hat[:, 1:, :].detach()       # (B, T-2, H)

            loss, loss_mse, loss_cons = supervisor_loss(
                h_hat, h_target, h_hat_2step, h_hat_from_next,
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(S.parameters(), 5.0)

            assert not torch.isnan(grad_norm), "Still NaN"
            if grad_norm > 50.0:
                logger.warning(f"Suspiciously large norm: {grad_norm:.2f}")

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_mse  += loss_mse.item()
            epoch_cons += loss_cons.item()

        n_batches   = max(1, len(dataloader))
        epoch_loss /= n_batches
        epoch_mse  /= n_batches
        epoch_cons /= n_batches

        history["train_loss"].append(epoch_loss)
        history["train_mse"].append(epoch_mse)
        history["train_cons"].append(epoch_cons)

        # ── Validation ────────────────────────────────────────────────────
        if epoch % log_every == 0:
            if val_loader is not None:
                S.eval()
                val_loss_accum = 0.0
                val_mse_accum  = 0.0

                with torch.no_grad():
                    for x_v, y_v in val_loader:
                        x_v, y_v = x_v.to(device), y_v.to(device)
                        h_v, _   = E(x_v, y_v)

                        # Build freq basis for validation batch
                        fb_v = build_freq_basis(y_v, h_v.shape[1], stim_freqs, fs, n_harmonics)

                        h_input_v  = h_v[:, :-1, :]
                        h_target_v = h_v[:, 1:, :]
                        fb_v_input = fb_v[:, :-1, :]
                        h_hat_v    = S(h_input_v, fb_v_input)

                        fb_v_2step = fb_v_input[:, :-1, :]
                        h_hat_2step_v     = S(h_hat_v[:, :-1, :], fb_v_2step)
                        h_hat_from_next_v = h_hat_v[:, 1:, :].detach()

                        v_loss, v_mse, _ = supervisor_loss(
                            h_hat_v, h_target_v,
                            h_hat_2step_v, h_hat_from_next_v,
                        )
                        val_loss_accum += v_loss.item()
                        val_mse_accum  += v_mse.item()

                n_val    = max(1, len(val_loader))
                val_loss = val_loss_accum / n_val
                val_mse  = val_mse_accum  / n_val

                history["val_loss"].append(val_loss)
                history["val_mse"].append(val_mse)

                # ── Best checkpoint & early stopping ──────────────────────
                if val_loss < best_val:
                    best_val         = val_loss
                    patience_counter = 0
                    torch.save(S.state_dict(), checkpoint_path)
                    logger.info(
                        f"[Supervisor] Epoch {epoch:03d} | "
                        f"New best val loss: {val_loss:.6f} — "
                        f"saved {checkpoint_path}"
                    )
                else:
                    patience_counter += 1
                    logger.info(
                        f"[Supervisor] Epoch {epoch:03d} | "
                        f"Val did not improve ({val_loss:.6f} >= {best_val:.6f}). "
                        f"Patience: {patience_counter}/{patience}"
                    )
                    if patience_counter >= patience:
                        logger.info(
                            f"[Supervisor] Early stopping triggered at epoch {epoch}."
                        )
                        break

                logger.info(
                    f"[Supervisor] Epoch {epoch:03d} | "
                    f"Train Loss: {epoch_loss:.6f} | Train MSE: {epoch_mse:.6f} | "
                    f"Train Cons: {epoch_cons:.6f} | "
                    f"Val Loss: {val_loss:.6f} | Val MSE: {val_mse:.6f}"
                )
            else:
                # No val_loader: log training metrics only
                logger.info(
                    f"[Supervisor] Epoch {epoch:03d} | "
                    f"Loss: {epoch_loss:.6f} | "
                    f"Loss MSE: {epoch_mse:.6f} | "
                    f"Loss Cons: {epoch_cons:.6f}"
                )

    E.train()
    return history

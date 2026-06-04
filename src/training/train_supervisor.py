import torch
from src.models.embedder import cEmbedder
from src.models.supervisor import Supervisor
from src.losses.losses import supervised_loss
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
    log_every: int = 10,
    patience: int = 20,
    checkpoint_path: str = "best_supervisor.pt",
):
    """
    Train the Supervisor network with optional validation loop and best-
    checkpoint saving.

    Parameters
    ----------
    E               : frozen cEmbedder – used only for inference
    S               : Supervisor to be trained
    dataloader      : training DataLoader  (x, y) batches
    optimizer       : optimiser for S
    device          : torch.device
    epochs          : total training epochs
    logger          : logging.Logger instance
    val_loader      : optional validation DataLoader; enables val-loss tracking,
                      best-checkpoint saving, and early stopping when provided
    log_every       : log (and validate) every this many epochs   (default 10)
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

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val        = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_mse":  [],
        "train_tcl":  [],
        "val_loss":   [],
        "val_mse":    [],
    }

    for epoch in range(epochs):
        # ── Training ──────────────────────────────────────────────────────
        S.train()

        epoch_loss = 0.0
        epoch_mse  = 0.0
        epoch_tcl  = 0.0
        lambda_tcl = min(0.05, 0.05 * (epoch / max(1, epochs * 0.5)))

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                h, _ = E(x, y)

            h_hat = S(h[:, :-1, :], y)

            loss, loss_mse, loss_tcl = supervised_loss(
                h[:, 1:, :], h_hat, lambda_tcl
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(S.parameters(), 5.0)

            assert not torch.isnan(grad_norm), "Still NaN"
            assert grad_norm < 50.0, f"Suspiciously large norm: {grad_norm:.2f}"

            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse  += loss_mse.item()
            epoch_tcl  += loss_tcl.item()

        n_batches   = max(1, len(dataloader))
        epoch_loss /= n_batches
        epoch_mse  /= n_batches
        epoch_tcl  /= n_batches

        scheduler.step()

        history["train_loss"].append(epoch_loss)
        history["train_mse"].append(epoch_mse)
        history["train_tcl"].append(epoch_tcl)

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
                        h_hat_v  = S(h_v[:, :-1, :], y_v)
                        v_loss, v_mse, _ = supervised_loss(
                            h_v[:, 1:, :], h_hat_v, lambda_tcl
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
                    f"Train TCL: {epoch_tcl:.6f} | "
                    f"Val Loss: {val_loss:.6f} | Val MSE: {val_mse:.6f}"
                )
            else:
                # No val_loader: log training metrics only
                logger.info(
                    f"[Supervisor] Epoch {epoch:03d} | "
                    f"Loss: {epoch_loss:.6f} | "
                    f"Loss MSE: {epoch_mse:.6f} | "
                    f"Loss TCL: {epoch_tcl:.6f}"
                )

    E.train()
    return history

import torch
from src.models.embedder import Embedder
from src.models.generator import Generator
from src.models.supervisor import Supervisor
from src.models.discriminator import Discriminator
from src.losses.losses import (
    generator_adv_loss, 
    discriminator_loss, 
    supervised_loss, 
    acf_error, 
    psd_error,
    lsd,
    gradient_penalty)
import logging

def train_timegan(
    E, G, S, R, D,
    dataloader,
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
    sup_loss_window: int = 1
):
    logger.info("Starting TimeGAN joint training...")

    E.eval()  # frozen permanently

    # -----------------------------------------
    # Validation batch (fixed)
    # -----------------------------------------
    val_batch = next(iter(dataloader)).to(device)

    best_score = float("inf")
    patience = 20
    patience_counter = 0
    stop_training = False


    for epoch in range(epochs):
        for x in dataloader:
            x = x.to(device)
            B, T, _ = x.shape

            # =====================================================
            # (1) Discriminator update (5 steps per G step)
            # =====================================================
            for _ in range(5):
                with torch.no_grad():
                    h_real = E(x)
                    z_d = torch.randn(B, T, z_dim+4, device=device)
                    h_fake = S(G(z_d))

                d_real = D(h_real)
                d_fake = D(h_fake)

                d_loss_val = discriminator_loss(d_real, d_fake)
                if epoch % 3 ==0:
                    gp = gradient_penalty(D, h_real, h_fake, device=device)
                else:
                    gp = 0.0
                
                # lambda_gp = 10.0
                d_loss = d_loss_val + 10.0 * gp

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            # =====================================================
            # (2) Generator + Supervisor update
            # =====================================================
            # -----------------------------------------
            # Sample noise
            # -----------------------------------------
            z = torch.randn(B, T, z_dim+4, device=device)

            with torch.no_grad():
                h_real = E(x)  # (B, T, H)

            # ----- Fake latent trajectory -----
            h_fake = G(z)               # (B, T, H)
            h_fake_sup = S(h_fake)      # (B, T, H)

            # ----- Adversarial loss -----
            d_fake_g = D(h_fake_sup)
            g_adv = generator_adv_loss(d_fake_g)

            # =================================================
            # Supervised loss
            # =================================================
            g_sup_fake = torch.mean((h_fake_sup[:, :-sup_loss_window, :] - h_fake[:, sup_loss_window:, :]) ** 2)

            h_real_pred = S(h_real[:, :-sup_loss_window, :])
            g_sup_real = torch.mean((h_real_pred - h_real[:, sup_loss_window:, :]) ** 2)

            g_sup = g_sup_fake + g_sup_real

            # =================================================
            # Moment matching in DATA space
            # =================================================
            x_fake = R(h_fake_sup)

            mean_real = torch.mean(x, dim=0)
            mean_fake = torch.mean(x_fake, dim=0)

            var_real = torch.var(x, dim=0, unbiased=False)
            var_fake = torch.var(x_fake, dim=0, unbiased=False)

            x_flat = x.reshape(-1, x.shape[-1])
            x_fake_flat = x_fake.reshape(-1, x_fake.shape[-1])

            cov_real = torch.cov(x_flat.T)
            cov_fake = torch.cov(x_fake_flat.T)

            mean_loss = torch.mean((mean_real - mean_fake) ** 2)
            var_loss = torch.mean((var_real - var_fake) ** 2)
            cov_loss = torch.mean((cov_real - cov_fake) ** 2)

            g_mom = mean_loss + var_loss + cov_loss

            # =================================================
            # Spectral loss
            # =================================================
            spectral_loss = lsd(x, x_fake)

            # =================================================
            # Total generator loss
            # =================================================
            g_loss = g_adv + lambda_sup * g_sup + lambda_mom * g_mom + lambda_spec * spectral_loss

            opt_gs.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_gs.step()

        #if epoch > 0 and epoch % 100 == 0:
        #    lambda_sup = max(lambda_sup - 5, 65)

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        if epoch % 5 == 0:
            
            with torch.no_grad():
                G.eval()
                S.eval()
                R.eval()

                B, T, _ = val_batch.shape
                z_val = torch.randn(B, T, z_dim+4, device=device)

                h_fake = G(z_val)
                h_fake_sup = S(h_fake)
                x_fake = R(h_fake_sup)

                acf_val = acf_error(val_batch, x_fake)
                psd_val = lsd(val_batch, x_fake)

                total_val = acf_val + psd_val

                G.train()
                S.train()
                R.train()


                if total_val < best_score:
                    best_score = total_val
                    patience_counter = 0

                    torch.save({
                        "E": E.state_dict(),
                        "G": G.state_dict(),
                        "S": S.state_dict(),
                        "R": R.state_dict(),
                        "D": D.state_dict(),
                    }, "best_timegan_16.3.pt")

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
                    f"Val_err: {total_val.item():.4f}"
                )
        if stop_training:
            break
       

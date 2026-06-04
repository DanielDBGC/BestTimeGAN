import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from src.utils.logging import get_logger
from src.models.embedder import cEmbedder
from src.models.recovery import cRecovery

logger = get_logger(__name__)

def flatten_batch(x):
    # (B, T, C) -> (B, T*C)
    return x.reshape(x.shape[0], -1)


def pairwise_distances(x):
    # x: (B, D)
    # returns (B, B)
    x_norm = (x**2).sum(dim=1, keepdim=True)
    dist = x_norm + x_norm.T - 2 * x @ x.T
    return torch.sqrt(torch.clamp(dist, min=1e-8))

@torch.no_grad()
def geometry_metrics(embedder, dataloader, device="cuda", max_batches=10):
    embedder.eval()

    corrs = []
    stresses = []

    for i, (x,y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        h, logits = embedder(x, y)

        x_flat = flatten_batch(x)
        h_flat = flatten_batch(h)

        Dx = pairwise_distances(x_flat)
        Dh = pairwise_distances(h_flat)

        Dx_vec = Dx.flatten()
        Dh_vec = Dh.flatten()

        # correlation
        corr = torch.corrcoef(torch.stack([Dx_vec, Dh_vec]))[0,1]

        # stress (MDS-style)
        stress = torch.norm(Dx - Dh) / torch.norm(Dx)

        corrs.append(corr.item())
        stresses.append(stress.item())

    return {
        "distance_corr": sum(corrs)/len(corrs),
        "stress": sum(stresses)/len(stresses)
    }

def knn_indices(D, k):
    # D: (B, B)
    return torch.topk(D, k=k+1, largest=False).indices[:, 1:]  # skip self


@torch.no_grad()
def knn_preservation(embedder, dataloader, k=5, device="cuda", max_batches=10):
    embedder.eval()

    overlaps = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        h, logits = embedder(x, y)

        x_flat = flatten_batch(x)
        h_flat = flatten_batch(h)

        Dx = pairwise_distances(x_flat)
        Dh = pairwise_distances(h_flat)

        knn_x = knn_indices(Dx, k)
        knn_h = knn_indices(Dh, k)

        for i in range(x.shape[0]):
            set_x = set(knn_x[i].tolist())
            set_h = set(knn_h[i].tolist())

            overlap = len(set_x & set_h) / k
            overlaps.append(overlap)

    return sum(overlaps) / len(overlaps)

def compute_psd(x):
    # x: (B, T, C)
    Xf = torch.fft.rfft(x, dim=1)
    psd = (Xf.abs() ** 2).mean(dim=2)  # average over channels
    return psd

def lsd(x_real, x_fake):
    eps = 1e-8

    psd_real = compute_psd(x_real) + eps
    psd_fake = compute_psd(x_fake) + eps

    return torch.sqrt(torch.mean((torch.log(psd_real) - torch.log(psd_fake))**2))

@torch.no_grad()
def psd_error(embedder, recovery, dataloader, device="cuda", max_batches=10):
    embedder.eval()
    recovery.eval()

    errors = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        h, logits = embedder(x, y)
        x_hat = recovery(h, y)

        err = lsd(x, x_hat)

        errors.append(err.item())

    return sum(errors) / len(errors)

def autocorrelation(x, max_lag=50):
    # x: (B, T, C)
    B, T, C = x.shape
    x = x - x.mean(dim=1, keepdim=True)

    acfs = []
    for lag in range(1, max_lag):
        num = (x[:, :-lag] * x[:, lag:]).mean(dim=1)
        den = (x**2).mean(dim=1)
        acf = num / (den + 1e-8)
        acfs.append(acf)

    return torch.stack(acfs, dim=1)  # (B, L, C)


@torch.no_grad()
def acf_error(embedder, recovery, dataloader, device="cuda", max_batches=10):
    embedder.eval()
    recovery.eval()

    errors = []

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        h, _ = embedder(x, y)
        x_hat = recovery(h, y)

        acf_real = autocorrelation(x)
        acf_rec = autocorrelation(x_hat)

        err = F.mse_loss(acf_real, acf_rec)
        errors.append(err.item())

    return sum(errors) / len(errors)

@torch.no_grad()
def latent_interpolation(embedder, recovery, x1, x2, steps=5):
    embedder.eval()
    recovery.eval()

    h1, _ = embedder(x1.unsqueeze(0))
    h2, _ = embedder(x2.unsqueeze(0))

    outputs = []

    for alpha in torch.linspace(0, 1, steps):
        h = alpha * h1 + (1 - alpha) * h2
        x_hat = recovery(h)
        outputs.append(x_hat.squeeze(0))

    return torch.stack(outputs)  # (steps, T, C)

def evaluate_autoencoder(embedder, recovery, dataloader, device="cuda"):
    geom = geometry_metrics(embedder, dataloader, device)
    knn = knn_preservation(embedder, dataloader, device=device)
    psd = psd_error(embedder, recovery, dataloader, device)
    acf = acf_error(embedder, recovery, dataloader, device)

    return {
        "distance_corr": geom["distance_corr"],
        "stress": geom["stress"],
        "knn_overlap": knn,
        "psd_error": psd,
        "acf_error": acf
    }

@torch.no_grad()
def per_channel_mse(x: torch.Tensor, x_tilde: torch.Tensor) -> list:
    """Returns MSE for each of the C channels, averaged over B and T."""
    err = (x - x_tilde).pow(2).mean(dim=[0, 1])  # [C]
    return err.cpu().tolist()

@torch.no_grad()
def per_class_mse(
    E: nn.Module,
    R: nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 15,
) -> dict:
    """Computes mean reconstruction MSE for every class label."""
    sums   = torch.zeros(num_classes)
    counts = torch.zeros(num_classes)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h, logits = E(x, y)
        x_hat = R(h, y)
        err = (x - x_hat).pow(2).mean(dim=[1, 2]).cpu()  # [B]
        for label, e in zip(y.cpu().tolist(), err.tolist()):
            sums[label]   += e
            counts[label] += 1
    return {
        cls: (sums[cls] / counts[cls]).item()
        for cls in range(num_classes)
        if counts[cls] > 0
    }
 
 
@torch.no_grad()
def ssvep_snr(
    signal: np.ndarray,
    fs: float,
    target_freq: float,
    signal_bw: float = 0.5,
    noise_bw:  float = 2.0,
) -> float:
    """
    Signal-to-noise ratio at the SSVEP stimulus frequency.
    Returns SNR in dB. A value > 3 dB indicates a detectable response.
    """
    try:
        import scipy.signal as spsig
        f, Pxx = spsig.welch(signal, fs, nperseg=min(len(signal), 256))
        sig_mask   = np.abs(f - target_freq) < signal_bw
        noise_mask = (np.abs(f - target_freq) > signal_bw) & \
                     (np.abs(f - target_freq) < noise_bw)
        if sig_mask.sum() == 0 or noise_mask.sum() == 0:
            return float("nan")
        signal_power = Pxx[sig_mask].mean()
        noise_power  = Pxx[noise_mask].mean()
        if noise_power == 0:
            return float("nan")
        return float(10 * np.log10(signal_power / noise_power))
    except ImportError:
        logger.warning("scipy not available — skipping SNR computation")
        return float("nan")
 
 
@torch.no_grad()
def latent_std_per_class(
    E: nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 15,
) -> dict:
    """
    Computes mean h.std() for each class.
    Healthy range: 0.1 - 0.4. Below 0.05 -> collapse. Above 0.8 -> exploding.
    """
    stds    = {c: [] for c in range(num_classes)}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        h, logits = E(x, y)
        for label in y.unique():
            mask = (y == label)
            stds[label.item()].append(h[mask].std().item())
    return {c: float(np.mean(v)) for c, v in stds.items() if v}
 
 
@torch.no_grad()
def log_gradient_norms(
    E: nn.Module,
    R: nn.Module,
) -> None:
    """Logs the gradient norm for every named parameter. Call after backward()."""
    for name, p in list(E.named_parameters()) + list(R.named_parameters()):
        if p.grad is not None:
            logger.debug(f"  grad [{name}]: {p.grad.norm():.5f}")


@torch.no_grad()
def run_diagnostics(
    E: cEmbedder,
    R: cRecovery,
    val_loader,
    *,
    device:         torch.device,
    fs:             float,
    stimulus_freqs: list,
    num_classes:    int = 15,
    logger
) -> None:
    """
    Runs the full diagnostic suite from the evaluation plan:
      - Per-channel MSE
      - Per-class MSE
      - SSVEP SNR (dB) on a representative batch
      - Latent std per class
    """
    E.eval()
    R.eval()
    logger.info("Starting diagnostics...")
 
    # Grab one batch for sample-level diagnostics
    x_sample, y_sample = next(iter(val_loader))
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)
    h_sample, logits = E(x_sample, y_sample)
    x_hat_sample = R(h_sample, y_sample)
 
    # --- Per-channel MSE ---------------------------------------------------------------------─
    ch_mse = per_channel_mse(x_sample, x_hat_sample)
    logger.info("--- Per-channel reconstruction MSE ---")
    for c, e in enumerate(ch_mse):
        bar = "-" * int(e * 500)
        logger.info(f"  Ch{c:02d}: {e:.5f}  {bar}")
 
    # --- Per-class MSE ------------------------------------------------------------------------─
    logger.info("--- Per-class reconstruction MSE ---")
    class_mse = per_class_mse(E, R, val_loader, device, num_classes)
    for cls, err in sorted(class_mse.items()):
        logger.info(f"  Class {cls:02d}: {err:.5f}")
 
    # --- SSVEP SNR ------------------------------------------------------------------------------─
    logger.info("--- SSVEP SNR at stimulus frequencies (sample batch) ---")
    x_np   = x_sample.cpu().numpy()    # [B, T, C]
    xh_np  = x_hat_sample.cpu().numpy()
    snr_real = []
    snr_recon = []
    for b in range(min(4, x_np.shape[0])):
        for f0 in stimulus_freqs[:3]:  # spot-check first 3 freqs
            # Average SNR over all 9 channels
            snr_r = np.mean([ssvep_snr(x_np[b,:,c],  fs, f0) for c in range(x_np.shape[2])])
            snr_h = np.mean([ssvep_snr(xh_np[b,:,c], fs, f0) for c in range(xh_np.shape[2])])
            snr_real.append(snr_r)
            snr_recon.append(snr_h)
    if snr_real and not all(np.isnan(snr_real)):
        logger.info(
            f"  Mean SNR — Real: {np.nanmean(snr_real):.2f} dB | "
            f"Reconstructed: {np.nanmean(snr_recon):.2f} dB"
        )
    else:
        logger.warning(
            "  SNR could not be computed — check that stimulus_freqs match "
            "your actual paradigm and that window length is sufficient for "
            f"Welch (need T >= 256 for fs={fs:.0f} Hz at 4 Hz resolution)"
        )
 
    # --- Latent std per class ---------------------------------------------------------------
    logger.info("--- Latent h.std per class (healthy: 0.1-0.4) ---")
    h_stds = latent_std_per_class(E, val_loader, device, num_classes)
    for cls, s in sorted(h_stds.items()):
        status = "Good" if 0.05 < s < 0.8 else "Not Good"
        logger.info(f"  Class {cls:02d}: {s:.4f}  {status}")


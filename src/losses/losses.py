import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss functions
mse = nn.MSELoss()

def reconstruction_loss(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    """
    Huber loss (smooth L1) with delta=0.1.
 
    EEG signals contain frequent sharp transients (blinks, muscle artefacts).
    Huber behaves like L2 for small errors and L1 for large ones, preventing
    single outlier time-steps from dominating the gradient.
    """
    return F.huber_loss(x_tilde, x, delta=0.1)


def discriminator_loss(d_real, d_fake):
    # WGAN discriminator loss: minimize fake - real
    return torch.mean(d_fake) - torch.mean(d_real)

def generator_adv_loss(d_fake):
    # WGAN generator loss: minimize -fake
    return -torch.mean(d_fake)

def gradient_penalty(D, h_real, h_fake, device="cuda", labels=None):
    B, T, C = h_real.shape
    alpha = torch.rand(B, 1, 1, device=device)
    alpha = alpha.expand(B, T, C)

    interpolates = alpha * h_real + ((1 - alpha) * h_fake)
    interpolates = interpolates.requires_grad_(True)

    if labels is not None:
        d_interpolates = D(interpolates, labels)
    else:
        d_interpolates = D(interpolates)

    if d_interpolates.ndim > 2:
        d_interpolates = d_interpolates.mean(dim=1)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.reshape(B, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def compute_acf(x, max_lag=100):
    # x: (B, T, C)
    x = x - x.mean(dim=1, keepdim=True)

    acf = []
    for lag in range(max_lag):
        if lag == 0:
            v = (x * x).mean()
        else:
            v = (x[:, :-lag, :] * x[:, lag:, :]).mean()
        acf.append(v)

    return torch.stack(acf)

def acf_error(x_real, x_fake, max_lag=100):
    acf_real = compute_acf(x_real, max_lag)
    acf_fake = compute_acf(x_fake, max_lag)

    return torch.mean((acf_real - acf_fake) ** 2)

def compute_psd(x):
    # x: (B, T, C)
    fft = torch.fft.rfft(x, dim=1)
    psd = torch.mean(torch.abs(fft) ** 2, dim=(0, 2))  # average batch & channels
    return psd

def lsd(x_real, x_fake):
    eps = 1e-8

    psd_real = torch.abs(torch.fft.rfft(x_real, dim=1)) ** 2 + eps
    psd_fake = torch.abs(torch.fft.rfft(x_fake, dim=1)) ** 2 + eps

    return torch.sqrt(torch.mean((torch.log(psd_real) - torch.log(psd_fake))**2))

def psd_error(x_real, x_fake):
    psd_real = compute_psd(x_real)
    psd_fake = compute_psd(x_fake)

    return torch.mean((psd_real - psd_fake) ** 2)

def geometry_loss(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss encouraging the latent trajectory to preserve
    the directional structure of the input signal.
 
    Operates on the mean-pooled representation to avoid conflating temporal
    and channel dimensions.
    """
    x_mean = x.mean(dim=1)                   # [B, C]
    h_mean = h.mean(dim=1)                   # [B, H]
 
    # Project h back to x_dim for a fair directional comparison
    # (only used for the geometry term — no extra parameters needed
    #  because we take cosine similarity after normalising both)
    x_norm = F.normalize(x_mean, dim=-1)
    h_norm = F.normalize(h_mean, dim=-1)

    # cosine_similarity returns [B]; we want the mean dissimilarity
    # Note: only valid when C == H. If not, skip this loss or add a
    # projection layer. Check: assert x.shape[-1] == h.shape[-1]
    if x_norm.shape[-1] != h_norm.shape[-1]:
        return torch.tensor(0.0, device=x.device)
 
    return (1.0 - F.cosine_similarity(x_norm, h_norm)).mean()


def spectral_loss(x: torch.Tensor, x_tilde: torch.Tensor) -> torch.Tensor:
    """
    Log-magnitude spectral loss across all channels.
 
    Using log(1 + |FFT|) weights lower-frequency components (where SSVEP
    stimulus responses live) more strongly than high-frequency noise, which
    is the opposite of raw MSE on FFT coefficients.
    """
    # rfft along the time axis (dim=1)
    X     = torch.fft.rfft(x,      dim=1, norm='ortho').abs()   # [B, T//2+1, C]
    X_hat = torch.fft.rfft(x_tilde, dim=1, norm='ortho').abs()  # [B, T//2+1, C]
    return F.mse_loss(torch.log1p(X_hat), torch.log1p(X))

def ssvep_band_loss(
    x: torch.Tensor,
    x_tilde: torch.Tensor,
    fs: float,
    stimulus_freqs: list,
    bandwidth_hz: float = 2.0,
) -> torch.Tensor:
    """
    Frequency-band loss restricted to SSVEP stimulus frequencies.
 
    Computes MSE on the FFT magnitude only within ±bandwidth_hz of each
    stimulus frequency (and its second harmonic). This directly targets the
    information that matters for SSVEP classification.
 
    Args:
        x, x_tilde: [B, T, C]
        fs:          sampling frequency in Hz
        stimulus_freqs: list of SSVEP stimulus frequencies, e.g. [6,7,8,...,15]
        bandwidth_hz:   half-width of the frequency band around each target
    """
    T = x.shape[1]
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(x.device)  # [T//2+1]
 
    # Build a mask covering all stimulus frequencies and their 2nd harmonics
    mask = torch.zeros(freqs.shape, dtype=torch.bool, device=x.device)
    for f0 in stimulus_freqs:
        for harmonic in [f0, 2 * f0]:
            mask |= (freqs - harmonic).abs() <= bandwidth_hz
 
    X     = torch.fft.rfft(x,       dim=1, norm='ortho').abs()  # [B, T//2+1, C]
    X_hat = torch.fft.rfft(x_tilde, dim=1, norm='ortho').abs()  # [B, T//2+1, C]
 
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
 
    # Index only the masked frequency bins
    return F.mse_loss(X_hat[:, mask, :], X[:, mask, :])


def spectral_convergence_loss(real_mag, fake_mag, eps=1e-8):
    diff = torch.norm(real_mag - fake_mag, p='fro')
    ref = torch.norm(real_mag, p='fro')
    return diff / (ref + eps)

def ssvep_snr_loss(x_real, x_fake, sfreq=1000, 
                   stim_freqs=(16., 24.),
                   n_harmonics=3, sig_bw=0.5, noise_bw=2.0):
    """
    Computes MSE between the SNR (in dB) of real and fake signals at SSVEP frequencies
    and their harmonics.
    """
    T = x_real.shape[1]
    freqs = torch.fft.rfftfreq(T, d=1.0/sfreq).to(x_real.device)  # [F]

    # Calculate power spectra using FFT
    X_real = torch.fft.rfft(x_real, dim=1)   # [B, F, C]
    X_fake = torch.fft.rfft(x_fake, dim=1)
    
    P_real = X_real.abs() ** 2  # [B, F, C]
    P_fake = X_fake.abs() ** 2

    loss = 0.0
    valid_bands = 0
    eps = 1e-8

    for sf in stim_freqs:
        for h in range(1, n_harmonics + 1):
            target = sf * h
            
            sig_mask = (freqs - target).abs() <= sig_bw
            noise_mask = ((freqs - target).abs() > sig_bw) & ((freqs - target).abs() <= noise_bw)
            
            if not sig_mask.any() or not noise_mask.any():
                continue
                
            valid_bands += 1

            sig_power_real = P_real[:, sig_mask, :].mean(dim=1)
            noise_power_real = P_real[:, noise_mask, :].mean(dim=1)
            
            sig_power_fake = P_fake[:, sig_mask, :].mean(dim=1)
            noise_power_fake = P_fake[:, noise_mask, :].mean(dim=1)
            
            snr_real = sig_power_real / (noise_power_real + eps)
            snr_fake = sig_power_fake / (noise_power_fake + eps)
            
            snr_real_db = 10.0 * torch.log10(snr_real + eps)
            snr_fake_db = 10.0 * torch.log10(snr_fake + eps)
            
            loss += F.mse_loss(snr_fake_db, snr_real_db)

    if valid_bands == 0:
        return torch.tensor(0.0, device=x_real.device)
        
    return loss / valid_bands

def supervisor_loss(
    h_hat: torch.Tensor,
    h_target: torch.Tensor,
    h_hat_2step: torch.Tensor,
    h_hat_from_next: torch.Tensor,
    lambda_cons: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-step supervisor loss: plain MSE + 2-step consistency.

    Parameters
    ----------
    h_hat          : S(h_{0:T-2})     – predicted next latent states   (B, T-1, H)
    h_target       : h_{1:T-1}        – ground-truth targets           (B, T-1, H)
    h_hat_2step    : S(h_hat)_{0:T-3} – 2-step rollout predictions     (B, T-2, H)
    h_hat_from_next: S(h_{1:T-2})     – 1-step predictions from t+1   (B, T-2, H)
    lambda_cons    : weight for the consistency term                    (default 1.0)

    Returns
    -------
    total, loss_mse, loss_cons
    """
    loss_mse  = F.mse_loss(h_hat, h_target)
    loss_cons = F.mse_loss(h_hat_2step, h_hat_from_next)
    total = loss_mse + lambda_cons * loss_cons
    return total, loss_mse, loss_cons


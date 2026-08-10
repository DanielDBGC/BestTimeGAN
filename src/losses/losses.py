import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    # Hinge discriminator loss
    return F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()

def generator_adv_loss(d_fake):
    # Hinge generator loss
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

def r1_penalty(d_real, x_real):
    """
    R1 regularization penalty.
    d_real: output of discriminator on real data
    x_real: real data (requires_grad=True)
    """
    grad_real = torch.autograd.grad(
        outputs=d_real.sum(),
        inputs=x_real,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty

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

def ssvep_snr_loss(x_real, x_fake, batch_stim_freqs, sfreq=1000,
                   n_harmonics=3, sig_bw=0.5, noise_bw=4.0):
    B, T, C = x_real.shape
    freq_res = sfreq / T
    
    # Safety check
    assert noise_bw > 2 * freq_res, \
        f"noise_bw={noise_bw} Hz too narrow for freq_res={freq_res:.2f} Hz/bin"

    freqs = torch.fft.rfftfreq(T, d=1.0/sfreq).to(x_real.device)

    # Hann window to reduce spectral leakage
    window = torch.hann_window(T, device=x_real.device).unsqueeze(0).unsqueeze(-1)
    window_power = (window ** 2).mean()

    X_real = torch.fft.rfft(x_real * window, dim=1)
    X_fake = torch.fft.rfft(x_fake * window, dim=1)
    P_real = X_real.abs() ** 2 / (window_power * T)
    P_fake = X_fake.abs() ** 2 / (window_power * T)

    eps = 1e-8
    total_loss = torch.tensor(0.0, device=x_real.device)
    valid_count = 0

    for i in range(B):
        sf = batch_stim_freqs[i].item()
        sample_snrs_real, sample_snrs_fake = [], []

        for h in range(1, n_harmonics + 1):
            target = sf * h
            if target > freqs.max():
                break

            sig_mask = (freqs - target).abs() <= sig_bw
            noise_mask = (
                ((freqs - target).abs() > sig_bw) & 
                ((freqs - target).abs() <= noise_bw) &
                (freqs > freq_res)  # exclude DC
            )

            if not sig_mask.any() or noise_mask.sum() < 2:  # need at least 2 noise bins
                continue

            snr_r = P_real[i, sig_mask, :].mean(0) / (P_real[i, noise_mask, :].mean(0) + eps)
            snr_f = P_fake[i, sig_mask, :].mean(0) / (P_fake[i, noise_mask, :].mean(0) + eps)

            sample_snrs_real.append(10.0 * torch.log10(snr_r + eps))
            sample_snrs_fake.append(10.0 * torch.log10(snr_f + eps))

        if sample_snrs_real:
            snr_r_all = torch.stack(sample_snrs_real)  # [H, C]
            snr_f_all = torch.stack(sample_snrs_fake)
            total_loss = total_loss + F.mse_loss(snr_f_all, snr_r_all)
            valid_count += 1

    return total_loss / max(valid_count, 1)

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

def reference_signals(freq, T, fs, n_harmonics=2):
    """
    freq: (B,) stim frequency per sample, in Hz
    returns: (B, T, 2*n_harmonics)
    """
    device, dtype = freq.device, freq.dtype
    t = torch.arange(T, device=device, dtype=dtype) / fs   # (T,)
    freq = freq.view(-1, 1)                                 # (B, 1)
    t = t.view(1, -1)                                       # (1, T)
    refs = []
    for h in range(1, n_harmonics + 1):
        angle = 2 * math.pi * h * freq * t                  # (B, T) broadcast, per-sample freq
        refs.append(torch.sin(angle))
        refs.append(torch.cos(angle))
    return torch.stack(refs, dim=-1)                         # (B, T, K)


def ssvep_corr_loss(x, freq, fs, n_harmonics=2, eps=1e-6):
    """
    x:    (B, T, C) generated signal
    freq: (B,)      stim frequency per sample, in Hz
    """
    B, T, C = x.shape
    refs = reference_signals(freq, T, fs, n_harmonics)       # (B, T, K)

    x_c = x - x.mean(dim=1, keepdim=True)                    # (B, T, C)
    r_c = refs - refs.mean(dim=1, keepdim=True)               # (B, T, K)  <- mean over TIME, per sample

    cross = torch.einsum('btc,btk->bck', x_c, r_c)            # (B, C, K)
    x_norm = x_c.norm(dim=1)                                  # (B, C)
    r_norm = r_c.norm(dim=1)                                  # (B, K)
    denom = x_norm.unsqueeze(-1) * r_norm.unsqueeze(1) + eps  # (B, C, K)

    corr = cross / denom                                      # Pearson r per (sample, channel, ref)
    return 1 - (corr ** 2).mean()

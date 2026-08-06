"""
Frequency-aware conditioning utilities for SSVEP TimeGAN.

Provides:
- ``build_freq_basis``  — constructs sin/cos features from stimulus labels
- ``FiLMLayer``         — Feature-wise Linear Modulation (γ·x + β)
"""

import torch
import torch.nn as nn
import math
from typing import Sequence


def build_freq_basis(
    labels: torch.Tensor,
    T: int,
    stim_freqs: Sequence[float],
    fs: float = 1000.0,
    n_harmonics: int = 3,
) -> torch.Tensor:
    """
    Build a sinusoidal frequency basis for each sample in the batch.

    For each label, looks up the physical stimulus frequency and computes
    sin(2π·k·f·t) and cos(2π·k·f·t) for k = 1 … n_harmonics.

    Parameters
    ----------
    labels      : [B] integer class labels (indices into stim_freqs)
    T           : number of timesteps
    stim_freqs  : list/tuple of stimulus frequencies in Hz, indexed by label
    fs          : sampling frequency in Hz (default 1000)
    n_harmonics : number of harmonics (1 = fundamental only, 2 = +2nd, etc.)

    Returns
    -------
    freq_basis : [B, T, 2 * n_harmonics]
        Channel ordering: [sin(f), cos(f), sin(2f), cos(2f), ...]
    """
    B = labels.shape[0]
    device = labels.device

    # Time vector: [T]
    t = torch.arange(T, device=device, dtype=torch.float32) / fs  # seconds

    # Look up physical frequencies: [B]
    stim_freqs_t = torch.tensor(stim_freqs, device=device, dtype=torch.float32)
    f = stim_freqs_t[labels]  # [B]

    # Build basis: for each harmonic k, compute sin/cos
    channels = []
    for k in range(1, n_harmonics + 1):
        # phase: [B, T] = 2π · k · f[b] · t[t]
        phase = (2.0 * math.pi * k) * f.unsqueeze(1) * t.unsqueeze(0)  # [B, T]
        channels.append(torch.sin(phase))  # [B, T]
        channels.append(torch.cos(phase))  # [B, T]

    # Stack: [B, T, 2*n_harmonics]
    return torch.stack(channels, dim=-1)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Given a frequency-basis vector, produces per-channel scale (γ) and
    shift (β) that modulate hidden activations:

        out = γ · x + β

    Parameters
    ----------
    freq_dim   : dimensionality of the frequency basis input
    hidden_dim : number of channels to modulate
    """

    def __init__(self, freq_dim: int, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(freq_dim, 2 * hidden_dim)

        # Initialize γ ≈ 1, β ≈ 0 so the layer is near-identity at init
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        # Set the bias for the γ half to 1 (so γ starts at 1)
        with torch.no_grad():
            self.fc.bias[:hidden_dim] = 1.0

    def forward(self, x: torch.Tensor, freq_basis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : [..., hidden_dim]  — hidden activations
        freq_basis : [..., freq_dim]    — frequency basis (same leading dims as x)

        Returns
        -------
        modulated : [..., hidden_dim]
        """
        params = self.fc(freq_basis)  # [..., 2 * hidden_dim]
        gamma, beta = params.chunk(2, dim=-1)  # each [..., hidden_dim]
        return gamma * x + beta

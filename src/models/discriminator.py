import torch
import torch.nn as nn
from .freq_conditioning import FiLMLayer


class Discriminator(nn.Module):
    """Legacy bidirectional GRU discriminator — kept for backward compatibility."""

    def __init__(self, input_dim, h_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        self.ln = nn.LayerNorm(h_dim * 2)
        self.fc = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.LayerNorm(h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(h_dim, 1)
        )

    def forward(self, h):
        d, _ = self.rnn(h)          # [B, T, h_dim*2]
        d = self.ln(d)
        y = self.fc(d)              # [B, T, 1]
        return y


class TCNDiscriminator(torch.nn.Module):
    """Legacy unconditional TCN discriminator — kept for backward compatibility."""

    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()

        self.net = torch.nn.Sequential(
            # (B, C, T)
            torch.nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=4, dilation=4),
            torch.nn.LeakyReLU(0.2),

            torch.nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=8, dilation=8),
            torch.nn.LeakyReLU(0.2),
        )

        self.head = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)

        h = self.net(x)        # (B, H, T)
        h = h.mean(dim=2)      # global average pooling → (B, H)

        out = self.head(h)     # (B, 1)
        return out


# ---------------------------------------------------------------------------
# Helpers for cTCNDiscriminator
# ---------------------------------------------------------------------------

class _ResidualTCNBlock(nn.Module):
    """
    One dilated causal-ish Conv1d block with:
    - Spectral normalization on the main conv
    - LayerNorm on channel dim (applied post-activation, in [B, C, T] layout)
    - Residual skip via 1×1 conv when in/out channels differ
    """

    def __init__(self, in_ch: int, out_ch: int, dilation: int, freq_dim: int):
        super().__init__()
        pad = dilation  # preserves sequence length for kernel_size=3
        self.conv = nn.utils.spectral_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation)
        )
        self.act  = nn.LeakyReLU(0.2, inplace=True)
        # LayerNorm expects [..., C]; we'll permute before/after
        self.norm = nn.LayerNorm(out_ch)
        
        self.film = FiLMLayer(freq_dim, out_ch)

        # Skip connection
        self.skip = (
            nn.utils.spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, freq_basis: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        residual = self.skip(x)               # [B, C_out, T]
        out = self.act(self.conv(x))          # [B, C_out, T]
        # LayerNorm on channel dim: permute to [B, T, C], norm, permute back
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = self.film(out, freq_basis)
        out = out.transpose(1, 2)
        return out + residual                 # [B, C_out, T]


class cTCNDiscriminator(nn.Module):
    """
    Conditional TCN Discriminator with sinusoidal frequency conditioning.

    Receives a latent sequence h [B, T, in_channels] and a precomputed
    frequency basis [B, T, freq_dim], concatenates them along the channel
    axis, then passes through 4 residual dilated-conv blocks (dilations
    1, 2, 4, 8) with spectral normalization.  Per-timestep linear head → scalar.

    Parameters
    ----------
    in_channels     : latent space dimension (LATENT_DIM)
    hidden_channels : internal TCN width
    freq_dim        : dimensionality of the sinusoidal frequency basis
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        freq_dim: int = 6,
    ):
        super().__init__()

        # First conv: maps in_channels → hidden_channels
        self.input_conv = nn.utils.spectral_norm(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        )

        # Residual TCN blocks (dilations: 1, 2, 4, 8)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            _ResidualTCNBlock(hidden_channels, hidden_channels, d, freq_dim)
            for d in dilations
        ])

        # Scalar output head
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, h: torch.Tensor, freq_basis: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h          : [B, T, in_channels]
        freq_basis : [B, T, freq_dim]  — precomputed sinusoidal features

        Returns
        -------
        out    : [B, T, 1]
        """
        B, T, _ = h.shape

        x = h.transpose(1, 2)                              # [B, C, T]
        x = self.input_conv(x)                             # [B, hidden, T]

        for block in self.blocks:
            x = block(x, freq_basis)                       # [B, hidden, T]

        # No global average pooling. We evaluate each timestep.
        x = x.transpose(1, 2)                              # [B, T, hidden]
        out = self.head(x)                                 # [B, T, 1]
        
        # We return the per-timestep scores. Hinge loss will apply ReLU 
        # to each timestep independently.
        return out


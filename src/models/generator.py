import torch
import torch.nn as nn
import torch.nn.functional as F
from .freq_conditioning import FiLMLayer


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation, **kwargs
        )

    def forward(self, x):
        # x is of shape (B, C, T)
        # causal padding was added on both sides, we need to remove the right side
        out = super(CausalConv1d, self).forward(x)
        return out[:, :, :-self.padding[0]]


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, freq_dim):
        super().__init__()
        
        self.conv = CausalConv1d(in_channels, 2 * out_channels, kernel_size, dilation=dilation)
        
        # FiLM modulation replaces the old cond_conv (label embedding projection).
        # Applied to the gated activation output, re-injecting frequency info at
        # every layer so the stimulus signal stays alive through depth.
        self.film = FiLMLayer(freq_dim, out_channels)
        
        # 1x1 convs for residual and skip connection
        self.res_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.skip_conv = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, freq_basis):
        """
        x          : [B, C, T]
        freq_basis : [B, T, freq_dim]
        """
        residual = x
        
        # Dilated causal conv
        out = self.conv(x)
        
        # Gated activation
        # out has 2 * out_channels
        filter_out, gate_out = out.chunk(2, dim=1)
        out = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        
        # FiLM modulation: transpose to [B, T, C], modulate, transpose back
        out_t = out.transpose(1, 2)                  # [B, T, out_channels]
        out_t = self.film(out_t, freq_basis)          # [B, T, out_channels]
        out = out_t.transpose(1, 2)                   # [B, out_channels, T]
        
        skip = self.skip_conv(out)
        res = self.res_conv(out) + residual
        
        return res, skip


class cGenerator(nn.Module):
    """
    Conditional Generator for TimeGAN using Gated Causal TCN (WaveNet-style)
    with sinusoidal frequency conditioning via FiLM modulation.

    Takes noise z [B, T, z_dim] and a precomputed frequency basis
    [B, T, freq_dim], and produces a latent trajectory [B, T, out_dim].

    The frequency basis (sin/cos of f_stim and harmonics) is injected at
    every GatedResidualBlock via FiLM, keeping the stimulus signal alive
    through the full depth of the network.
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_layers: int,
        out_dim: int = 12,
        freq_dim: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,  # Not used in standard WaveNet, but kept for signature
    ):
        super().__init__()

        self.start_conv = nn.Conv1d(z_dim, h_dim, 1)
        
        self.blocks = nn.ModuleList()
        # Typically dilations go 1, 2, 4, 8...
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(
                GatedResidualBlock(
                    in_channels=h_dim,
                    out_channels=h_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    freq_dim=freq_dim,
                )
            )

        # Output layers (post-skip connections)
        self.end_conv1 = nn.Conv1d(h_dim, h_dim, 1)
        self.end_conv2 = nn.Conv1d(h_dim, out_dim, 1)

    def forward(self, z: torch.Tensor, freq_basis: torch.Tensor) -> torch.Tensor:
        """
        z          : [B, T, z_dim]
        freq_basis : [B, T, freq_dim]  — precomputed sinusoidal features
        Returns    : [B, T, out_dim]
        """
        B, T, _ = z.size()
        
        # Prepare inputs (transpose to [B, C, T] for conv1d)
        x = z.transpose(1, 2)  # [B, z_dim, T]
        x = self.start_conv(x) # [B, h_dim, T]

        skip_connections = []
        
        for block in self.blocks:
            x, skip = block(x, freq_basis)
            skip_connections.append(skip)
            
        # Sum skip connections
        out = sum(skip_connections)
        out = F.relu(out)
        out = self.end_conv1(out)
        out = F.relu(out)
        out = self.end_conv2(out)  # [B, out_dim, T]
        
        # Transpose back to [B, T, out_dim]
        return out.transpose(1, 2)
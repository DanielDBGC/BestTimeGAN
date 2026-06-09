import torch
import torch.nn as nn


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

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        pad = dilation  # preserves sequence length for kernel_size=3
        self.conv = nn.utils.spectral_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation)
        )
        self.act  = nn.LeakyReLU(0.2, inplace=True)
        # LayerNorm expects [..., C]; we'll permute before/after
        self.norm = nn.LayerNorm(out_ch)

        # Skip connection
        self.skip = (
            nn.utils.spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size=1))
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, T]
        residual = self.skip(x)               # [B, C_out, T]
        out = self.act(self.conv(x))          # [B, C_out, T]
        # LayerNorm on channel dim: permute to [B, T, C], norm, permute back
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out + residual                 # [B, C_out, T]


class cTCNDiscriminator(nn.Module):
    """
    Conditional TCN Discriminator.

    Receives a latent sequence h [B, T, in_channels] and integer class labels
    [B], concatenates a label embedding along the channel axis, then passes
    through 4 residual dilated-conv blocks (dilations 1, 2, 4, 8) with
    spectral normalization.  Global average pooling + linear head → scalar.

    Parameters
    ----------
    in_channels     : latent space dimension (LATENT_DIM)
    hidden_channels : internal TCN width
    num_classes     : number of class labels
    label_emb_dim   : embedding size for conditioning
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_classes: int = 15,
        label_emb_dim: int = 16,
    ):
        super().__init__()

        # Label conditioning
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # First conv: maps (in_channels + label_emb_dim) → hidden_channels
        cond_in = in_channels + label_emb_dim
        self.input_conv = nn.utils.spectral_norm(
            nn.Conv1d(cond_in, hidden_channels, kernel_size=1)
        )

        # Residual TCN blocks (dilations: 1, 2, 4, 8)
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            _ResidualTCNBlock(hidden_channels, hidden_channels, d)
            for d in dilations
        ])

        # Scalar output head
        self.head = nn.Linear(hidden_channels, 1)

    def forward(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h      : [B, T, in_channels]
        labels : [B]

        Returns
        -------
        out    : [B, 1]
        """
        B, T, _ = h.shape

        # Concat label embedding along channel dim
        emb = self.label_emb(labels)                       # [B, label_emb_dim]
        emb = emb.unsqueeze(1).expand(-1, T, -1)           # [B, T, label_emb_dim]
        x = torch.cat([h, emb], dim=-1)                    # [B, T, in_ch + emb]

        # Transpose to [B, C, T] for Conv1d
        x = x.transpose(1, 2)                              # [B, C, T]
        x = self.input_conv(x)                             # [B, hidden, T]

        for block in self.blocks:
            x = block(x)                                   # [B, hidden, T]

        x = x.mean(dim=2)                                  # global avg pool → [B, hidden]
        return self.head(x)                                 # [B, 1]

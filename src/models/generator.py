import torch
import torch.nn as nn


class Generator(nn.Module):
    """Legacy unconditional Generator — kept for backward compatibility."""
    def __init__(self, z_dim, h_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=z_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.ln = nn.LayerNorm(h_dim)
        self.fc = nn.Linear(h_dim, 12)

    def forward(self, z):
        # z: [B, T, z_dim]
        h_hat, _ = self.rnn(z)
        h_hat = self.ln(h_hat)
        h_hat = self.fc(h_hat)
        return h_hat  # [B, T, h_dim]


class _ResidualGRUBlock(nn.Module):
    """Single GRU layer with residual skip connection and LayerNorm."""

    def __init__(self, h_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=h_dim,
            hidden_size=h_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(h_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden=None):
        # x: [B, T, h_dim]
        out, h = self.gru(x, hidden)      # [B, T, h_dim]
        out = self.drop(out)
        out = self.norm(out + x)          # residual connection
        return out, h


class cGenerator(nn.Module):
    """
    Conditional Generator for TimeGAN.

    Takes noise z [B, T, z_dim] and integer class labels [B], and produces
    a latent trajectory [B, T, out_dim] that can be passed directly to the
    Supervisor (which operates in LATENT_DIM space).

    Architecture
    ------------
    1. Label embedding  : Embedding(num_classes, label_emb_dim)
    2. Input projection : Linear(z_dim + label_emb_dim, h_dim)
    3. Residual GRU stack: num_layers × _ResidualGRUBlock(h_dim)
    4. Output projection: Linear(h_dim, out_dim)   — out_dim = LATENT_DIM
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        num_layers: int,
        out_dim: int = 12,
        num_classes: int = 15,
        label_emb_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()

        # --- conditioning ---
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)

        # --- input projection: merge noise + label embedding ---
        self.input_proj = nn.Linear(z_dim + label_emb_dim, h_dim)
        self.input_norm = nn.LayerNorm(h_dim)

        # --- residual GRU stack ---
        self.blocks = nn.ModuleList(
            [_ResidualGRUBlock(h_dim, dropout=dropout if i < num_layers - 1 else 0.0)
             for i in range(num_layers)]
        )

        # --- output projection to latent space ---
        self.output_proj = nn.Linear(h_dim, out_dim)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z      : [B, T, z_dim]   — noise input
        labels : [B]             — integer class labels (0 … num_classes-1)

        Returns
        -------
        h_hat  : [B, T, out_dim]
        """
        # Expand label embedding across time
        emb = self.label_emb(labels)                        # [B, label_emb_dim]
        emb = emb.unsqueeze(1).expand(-1, z.size(1), -1)   # [B, T, label_emb_dim]

        # Project input
        x = self.input_norm(self.input_proj(torch.cat([z, emb], dim=-1)))  # [B, T, h_dim]

        # Residual GRU pass
        for block in self.blocks:
            x, _ = block(x)

        return self.output_proj(x)  # [B, T, out_dim]
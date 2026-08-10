import torch
import torch.nn as nn
from .freq_conditioning import FiLMLayer


class Supervisor(nn.Module):
    """
    Supervisor with sinusoidal frequency conditioning.

    Frequency information enters at two points:
    1. **Input concat** — freq_basis is concatenated with h before the
       input projection, giving the GRU explicit per-timestep phase info.
    2. **Post-GRU FiLM** — a FiLMLayer modulates the GRU output before
       the gated residual connection, re-injecting frequency information
       after the recurrent pass so it doesn't wash out.
    """

    def __init__(self, h_dim, num_layers, freq_dim=6):
        super().__init__()

        self.input_proj = nn.Linear(h_dim + freq_dim, h_dim)

        # Reduced hidden dimension (no more 5x expansion)
        rnn_hidden_dim = h_dim

        self.rnn = nn.GRU(
            input_size=h_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )

        # FiLM modulation after GRU — keeps frequency alive through depth
        self.film = FiLMLayer(freq_dim, rnn_hidden_dim)

        # Output layers for correction and gate
        self.correction_proj = nn.Linear(rnn_hidden_dim, h_dim)
        self.gate_proj = nn.Linear(rnn_hidden_dim, h_dim)

    def forward(self, h, freq_basis):
        """
        h          : [B, T, h_dim]
        freq_basis : [B, T, freq_dim]  — precomputed sinusoidal features
        """
        x = torch.cat([h, freq_basis], dim=-1)  # [B, T, h_dim + freq_dim]

        x = self.input_proj(x)  # [B, T, h_dim]

        # Forward pass
        rnn_out, _ = self.rnn(x)  # [B, T, rnn_hidden_dim]

        # FiLM modulation — re-inject frequency info post-RNN
        rnn_out = self.film(rnn_out, freq_basis)

        correction = self.correction_proj(rnn_out)
        gate = torch.sigmoid(self.gate_proj(rnn_out))

        # Gated residual connection ensures small gentle corrections
        out = h + gate * correction

        return out
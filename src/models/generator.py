import torch
import torch.nn as nn

class Generator(nn.Module):
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
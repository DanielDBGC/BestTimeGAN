import torch
import torch.nn as nn

class Discriminator(nn.Module):
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

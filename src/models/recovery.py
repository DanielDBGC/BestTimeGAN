import torch
import torch.nn as nn

class Recovery(nn.Module):
    def __init__(self, h_dim, x_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=h_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc1 = nn.Linear(h_dim, x_dim)

    def forward(self, h):
        r, _ = self.rnn(h)
        x_tilde = self.fc1(r)
        return x_tilde  # [B, T, x_dim]

class cRecovery(nn.Module):
    def __init__(self, h_dim, x_dim, num_layers=2, num_classes=15, label_emb_dim=16):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.rnn = nn.GRU(
            input_size=h_dim + label_emb_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, x_dim),
            # No final activation — raw EEG values are unbounded
        )
 
    def forward(self, h, labels):
        # h:      [B, T, h_dim]
        # labels: [B]
        emb = self.label_emb(labels)
        emb = emb.unsqueeze(1).expand(-1, h.size(1), -1)   # [B, T, label_emb_dim]
        h_cond = torch.cat([h, emb], dim=2)                 # [B, T, h_dim + emb]
        r, _ = self.rnn(h_cond)                             # [B, T, h_dim]
        return self.output_proj(r)                          # [B, T, x_dim]


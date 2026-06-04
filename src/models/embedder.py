import torch
import torch.nn as nn

# Models
class Embedder(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=x_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.lim = nn.Sigmoid()

    def forward(self, x):
        # x: [B, T, x_dim]
        h, _ = self.rnn(x)
        return self.lim(h)  # [B, T, h_dim]

class cEmbedder(nn.Module):
    def __init__(self, x_dim, h_dim, num_classes=15, label_emb_dim=16, num_layers=2):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.input_proj = nn.Linear(x_dim + label_emb_dim, h_dim)
        self.rnn = nn.GRU(
            input_size=h_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(h_dim)
        self.classifier = nn.Linear(h_dim, num_classes)
 
    def forward(self, x, labels):
        # x:      [B, T, x_dim]
        # labels: [B]
        emb = self.label_emb(labels)                        # [B, label_emb_dim]
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)   # [B, T, label_emb_dim]
        x = torch.cat([x, emb], dim=2)                     # [B, T, x_dim + emb]
        x = self.input_proj(x)                              # [B, T, h_dim]
        h, _ = self.rnn(x)                                  # [B, T, h_dim]
        logits = self.classifier(h.mean(dim=1))
        return self.norm(h), logits                       # [B, T, h_dim], [B, num_classes]

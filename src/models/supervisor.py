import torch
import torch.nn as nn

class Supervisor(nn.Module):
    def __init__(self, h_dim, num_layers, num_classes=15, label_emb_dim=16):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, label_emb_dim)
        self.input_proj = nn.Linear(h_dim + label_emb_dim, h_dim)        
        self.rnn = nn.GRU(
            input_size=h_dim,
            hidden_size=h_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.7 if num_layers > 1 else 0.0
        )
        self.input_norm = nn.LayerNorm(h_dim)
        self.output_proj = nn.Linear(h_dim, h_dim) 

    def forward(self, h, labels, tbptt_chunk=256):
        emb = self.label_emb(labels).unsqueeze(1).expand(-1, h.size(1), -1)
        h   = self.input_norm(self.input_proj(torch.cat([h, emb], dim=-1)))

        hidden  = None
        outputs = []
        for chunk in h.split(tbptt_chunk, dim=1):
            out, hidden = self.rnn(chunk, hidden)
            hidden = hidden.detach()         
            outputs.append(out)

        s = torch.cat(outputs, dim=1)
        return self.output_proj(s)
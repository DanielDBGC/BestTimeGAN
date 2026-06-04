import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometryLoss(nn.Module):
    """
    Cosine-similarity geometry loss between input signal and latent encoding.

    Enforces that the latent representation h preserves the directional
    structure of the input x, preventing representation collapse and
    ensuring h is meaningful for the downstream TimeGAN generator.

    Args:
        h_dim:      hidden dimension of the embedder output (e.g. 64)
        x_dim:      input channel dimension (e.g. 9)
        pool:       how to collapse the time axis before comparing
                      'mean'  — average over T (global shape)
                      'max'   — max over T (peak structure)
                      'both'  — concatenate mean and max (recommended)
        eps:        numerical stability for cosine similarity
    """

    def __init__(
        self,
        h_dim: int,
        x_dim: int,
        pool: str = "both",
        eps: float = 1e-8,
    ):
        super().__init__()
        self.pool = pool
        self.eps  = eps

        # Projection: h_dim → x_dim
        # No bias — we only want to learn a rotation/scaling, not a shift
        # 'both' pooling doubles the input to the projection
        proj_in = h_dim * 2 if pool == "both" else h_dim
        self.proj = nn.Linear(proj_in, x_dim, bias=False)

        # Initialise with orthogonal weights for a stable starting point
        nn.init.orthogonal_(self.proj.weight)

    def _pool(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, T, D] → [B, D or 2D]"""
        if self.pool == "mean":
            return z.mean(dim=1)
        elif self.pool == "max":
            return z.max(dim=1).values
        else:  # "both"
            return torch.cat([z.mean(dim=1), z.max(dim=1).values], dim=-1)

    def forward(
        self,
        x: torch.Tensor,   # [B, T, x_dim]  — raw input
        h: torch.Tensor,   # [B, T, h_dim]  — embedder output
    ) -> torch.Tensor:
        # Pool both signals over the time axis
        x_pooled = x.mean(dim=1)          # [B, x_dim]  (mean is fine for x)
        h_pooled = self._pool(h)          # [B, h_dim] or [B, 2*h_dim]

        # Project h into x's space
        h_proj = self.proj(h_pooled)      # [B, x_dim]

        # L2-normalise both vectors
        x_norm = F.normalize(x_pooled, dim=-1, eps=self.eps)  # [B, x_dim]
        h_norm = F.normalize(h_proj,   dim=-1, eps=self.eps)  # [B, x_dim]

        # Cosine similarity ∈ [−1, 1]; we want it close to 1
        # (1 − sim) ∈ [0, 2]; mean over batch
        loss = (1.0 - F.cosine_similarity(x_norm, h_norm, dim=-1)).mean()
        return loss
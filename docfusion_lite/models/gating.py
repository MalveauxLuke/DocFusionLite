# src/models/gating.py

import torch
import torch.nn as nn


class DocGate(nn.Module):
    def __init__(self, d_model: int, hidden: int | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden is None:
            hidden = max(64, d_model // 2)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_text: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        if text_mask is None:
            pooled = h_text.mean(dim=1)
        else:
            m = text_mask.to(h_text.dtype).unsqueeze(-1)
            denom = m.sum(dim=1).clamp_min(1.0)
            pooled = (h_text * m).sum(dim=1) / denom

        g = torch.sigmoid(self.mlp(pooled))
        return g.view(-1, 1, 1)

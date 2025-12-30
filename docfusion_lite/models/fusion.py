# src/models/fusion.py

import torch
import torch.nn as nn
from typing import Optional

class FusionStem(nn.Module):
    """
    Initial fusion of text + per-token visual features.

    Used in both 3.1 and 3.3:
      - concat([text, region_proj]) -> MLP -> residual on text
    No cross-attention here.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        dropout: float = 0.1,
        use_gate: bool = False,
        gate_per_dim: bool = False,
    ):
        super().__init__()

        if d_hidden is None:
            d_hidden = 2 * d_model
        

        self.use_gate = use_gate
        self.gate_per_dim = gate_per_dim
        
        self.ln_in = nn.LayerNorm(2 * d_model)

        self.mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

        
        self.gate_layer = nn.Linear(2 * d_model, d_model)

    def forward(
        self,
        h_text: torch.Tensor,      # (B, T, d_model)
        region_proj: torch.Tensor, # (B, T, d_model)
        g_doc: Optional[torch.Tensor] = None, # (B,1,1)
    ) -> torch.Tensor:
        
        x = torch.cat([h_text, region_proj], dim=-1)  # (B,T,2d)
        x = self.ln_in(x)

        delta = self.mlp(x) # (B,T,d)

        if self.use_gate:
            gate_full = torch.sigmoid(self.gate_layer(x))
            if self.gate_per_dim:
                gate = gate_full
            else:
                gate = gate_full.mean(dim=-1, keepdim=True)
            delta = gate * delta

        if g_doc is not None:
            delta = delta * g_doc

        return h_text + delta
    

class FusionLayer(nn.Module):
    """
    Deep fusion layer for 3.3.

    Each layer:
      - (optional) re-fuse text with region_proj via concat+MLP residual
      - cross-attend tokens (queries) to patch_feats (keys/values)
      - residual + norms

    This is only used in 3.3; 3.1 doesn't need this class.
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_region_ffn: bool = True,
        use_gate: bool = False,
        gate_per_dim: bool = False,
        gate_region_with_doc: bool = False,
        gate_cross_with_doc: bool = True,
    ):
        super().__init__()
        if d_hidden is None:
            d_hidden = 2 * d_model

        self.use_region_ffn = use_region_ffn
        self.use_gate = use_gate
        self.gate_per_dim = gate_per_dim
        self.gate_region_with_doc = gate_region_with_doc
        self.gate_cross_with_doc = gate_cross_with_doc

        if use_region_ffn:
            self.ln_region = nn.LayerNorm(2 * d_model)
            self.region_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_model),
                nn.Dropout(dropout),
            )
            # always define gate_layer with full dim
            self.gate_layer = nn.Linear(2 * d_model, d_model)
        
        # Cross-attention to patch features
        self.ln_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        self.dropout_cross = nn.Dropout(dropout)


    def forward(
        self,
        h: torch.Tensor,                 # (B, T, d_model)
        region_proj: torch.Tensor,       # (B, T, d_model)
        patch_feats: torch.Tensor,       # (B, P, d_model)
        g_doc: Optional[torch.Tensor] = None,       # (B,1,1) or None
        text_mask: Optional[torch.Tensor] = None,   # (B, T) or None
        patch_mask: Optional[torch.Tensor] = None,  # (B, P) or None
    ) -> torch.Tensor:
        # 1) Optional region-based FFN fusion
        if self.use_region_ffn:
            x = torch.cat([h, region_proj], dim=-1)  # (B,T,2d)
            x = self.ln_region(x)
            delta = self.region_mlp(x)

            if self.use_gate:
                gate_full = torch.sigmoid(self.gate_layer(x))  # (B,T,d)
                if self.gate_per_dim:
                    gate = gate_full
                else:
                    gate = gate_full.mean(dim=-1, keepdim=True)
                delta = gate * delta

            if (g_doc is not None) and self.gate_region_with_doc: # gating for entire document
                delta = delta * g_doc

            h = h + delta
        # 2) Cross-attention (PRE-LN) + doc-gate on attn residual only    
        if patch_feats is not None:
            key_padding_mask = patch_mask
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.bool()  # True=pad

            q = self.ln_cross(h)
            attn_out, _ = self.cross_attn(
                query=q,
                key=patch_feats,
                value=patch_feats,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            if g_doc is not None:
                attn_out = attn_out * g_doc

            h = h + self.dropout_cross(attn_out)

        if text_mask is not None:
            h = h * text_mask.to(h.dtype).unsqueeze(-1)

        return h
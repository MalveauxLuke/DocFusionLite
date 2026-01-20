from typing import Optional, Tuple

import torch
import torch.nn as nn

from docfusion_lite.utils.geometry import patches_in_token_boxes


class RegionPooler(nn.Module):
    """
    Skeleton for region token pooling.

    Responsibilities (to implement later):
    - Aggregate patch features into one region token per text token
    - Respect token boxes and optionally token masks
    - Optionally project pooled features to `d_region`
    """

    def __init__(self, d_patch: int, d_region: int = 256):
        super().__init__()
        self.d_region = d_region

        self.scorer = nn.Linear(d_patch, 1)

        self.projection = nn.Linear(d_patch, d_region)

    def forward(
        self,
        patch_feats: torch.Tensor,          # (B, P, d_patch)
        token_boxes: torch.Tensor,          # (B, T, 4)  (LayoutLM: long in [0,1000])
        patch_boxes: torch.Tensor,          # (B, P, 4)  (normalized [0,1])
        token_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            region_feats: (B, T, d_patch)
            region_mask:  (B, T)  True where token had >=1 overlapping patch (and not pad if token_mask given)
        """
        tb = token_boxes.to(patch_boxes.dtype)
        if tb.numel() > 0 and tb.max() > 1.5:   # heuristic: detects [0,1000]
            tb = tb / 1000.0

        in_box_mask = patches_in_token_boxes(patch_boxes, tb)  # expects both in [0,1]

        if token_mask is not None:
            token_mask_bool = token_mask.bool()
            in_box_mask = in_box_mask & token_mask_bool.unsqueeze(-1)

        region_mask = in_box_mask.any(dim=-1)  # (B, T)

        scores = self.scorer(patch_feats).squeeze(-1)  # (B, P)

        scores = scores.unsqueeze(1).expand(-1, in_box_mask.size(1), -1)

        scores = scores.masked_fill(~in_box_mask, -1e4)

        attn = torch.softmax(scores, dim=-1)  # (B, T, P)

        region_feats = torch.einsum("btp,bpd->btd", attn, patch_feats)  # (B, T, d_patch)

   
        region_feats = region_feats * region_mask.to(region_feats.dtype).unsqueeze(-1)

        # Optional projection 
         # region_feats = self.projection(region_feats)  # (B, T, d_region)

        return region_feats, region_mask
from typing import Optional, Tuple
from docfusion_lite.utils.geometry import compute_patch_boxes
from docfusion_lite.utils.geometry import patches_in_token_boxes
import torch
import torch.nn as nn


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
		
        # Linear layer for per patch score calculation
		self.scorer = nn.Linear(d_patch, 1)

		# Projection after pooling
		self.projection = nn.Linear(d_patch, d_region)
    

	def forward(
		self,
		patch_feats: torch.Tensor,
		token_boxes: torch.Tensor,
		patch_boxes: torch.Tensor,
		token_mask: Optional[torch.Tensor] = None,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Args:
			patch_feats: (B, P, d_patch)
			token_boxes: (B, T, 4)
			patch_boxes: (B, P, 4)

		Returns:
			region_feats: (B, T, d_patch)
			region_mask:   (B, T)
		"""
		in_box_mask = patches_in_token_boxes(patch_boxes, token_boxes)
		if token_mask is not None:
    		# (B,T) -> (B,T,1) ->  (B,T,P)
			token_mask_bool = token_mask.bool()
			in_box_mask = in_box_mask & token_mask_bool.unsqueeze(-1)
		# Region valid if any patch and token is not padding
		region_mask = in_box_mask.any(dim=-1)

		# patch_feats: (B, P, d_patch)
		scores = self.scorer(patch_feats)
		# scores: (B, P) -> (B, 1, P) -> (B, T, P)
		scores = scores.squeeze(-1) 
		scores = scores.unsqueeze(1).expand(-1, in_box_mask.size(1), -1)
		# mask: (B, T, P) boolean

		scores = scores.masked_fill(~in_box_mask, -1e4)  # invalid patches -> -inf
		
		attn = torch.softmax(scores, dim=-1) # (B, T, P)
		patch_feats_exp = patch_feats.unsqueeze(1)          # (B, 1, P, d_patch)
		attn_exp        = attn.unsqueeze(-1)                # (B, T, P, 1)
		weighted = attn_exp * patch_feats_exp               # (B, T, P, d_patch)
		region_feats = weighted.sum(dim=2)                  # (B, T, d_patch)

		# projection (d_patch -> d_region)
		#region_feats = self.projection(region_feats)  # (B, T, d_region)

	
		region_feats = region_feats * region_mask.unsqueeze(-1).float()  # (B,T,d_patch)
		return region_feats, region_mask


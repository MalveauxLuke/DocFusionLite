"""
Minimal utilities for patch <-> token geometric helpers.

Functions:
- compute_patch_boxes(batch_size, H, W, patch_size) -> (B, P, 4)
- compute_patch_centers(patch_boxes) -> (B, P, 2)
- patches_in_token_boxes(patch_boxes, token_boxes, iou_threshold=0.0) -> (B, T, P) boolean mask

All coordinates are normalized to [0,1].
"""

import torch


def compute_patch_boxes(batch_size: int, H: int, W: int, patch_size: int) -> torch.Tensor:
    """Return normalized patch boxes (B, P, 4) in (x1,y1,x2,y2) format."""
    gh = H // patch_size
    gw = W // patch_size

    rows = torch.arange(gh).unsqueeze(1).expand(gh, gw)
    cols = torch.arange(gw).unsqueeze(0).expand(gh, gw)

    x1 = (cols * patch_size) / W
    y1 = (rows * patch_size) / H
    x2 = ((cols + 1) * patch_size) / W
    y2 = ((rows + 1) * patch_size) / H

    boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(-1, 4)
    return boxes.unsqueeze(0).repeat(batch_size, 1, 1)


def patches_in_token_boxes(
    patch_boxes: torch.Tensor,
    token_boxes: torch.Tensor,
    iou_threshold: float = 0.0,
) -> torch.Tensor:
    """Return boolean mask (B, T, P): True if IoU(patch, token) > iou_threshold.

    patch_boxes: (B, P, 4)  in (x1,y1,x2,y2)
    token_boxes: (B, T, 4)  in (x1,y1,x2,y2)
    iou_threshold: minimum IoU required to consider a patch inside a token box.
                   Use 0.0 for 'any overlap'.
    """
    # Expand dims for broadcasting:
    # patch_boxes -> (B, 1, P, 4)
    # token_boxes -> (B, T, 1, 4)
    pb = patch_boxes.unsqueeze(1)      # (B, 1, P, 4)
    tb = token_boxes.unsqueeze(2)      # (B, T, 1, 4)

    # Extract coords
    px1, py1, px2, py2 = pb[..., 0], pb[..., 1], pb[..., 2], pb[..., 3]
    tx1, ty1, tx2, ty2 = tb[..., 0], tb[..., 1], tb[..., 2], tb[..., 3]

    # Intersection box
    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h  # (B, T, P)

    # Areas
    patch_area = (px2 - px1).clamp(min=0.0) * (py2 - py1).clamp(min=0.0)
    token_area = (tx2 - tx1).clamp(min=0.0) * (ty2 - ty1).clamp(min=0.0)

    union_area = patch_area + token_area - inter_area
    eps = 1e-8
    iou = inter_area / (union_area + eps)  # (B, T, P)

    # Mask: True if IoU above threshold
    in_box_mask = iou > iou_threshold
    return in_box_mask

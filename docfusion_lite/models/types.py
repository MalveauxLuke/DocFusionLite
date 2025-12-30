from dataclasses import dataclass
import torch


@dataclass
class DocBatch:
    """
    Canonical batch passed into the model.forward(batch).

    Shapes:
        input_ids:      (B, T)
        attention_mask: (B, T)
        token_boxes:    (B, T, 4)   # normalized [0,1]
        images:         (B, 3, H, W)
        labels:         optional, e.g. (B,) or (B, T)
    """
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_boxes: torch.Tensor
    images: torch.Tensor
    labels: torch.Tensor | None = None

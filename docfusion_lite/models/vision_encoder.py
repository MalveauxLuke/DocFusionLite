# src/models/vision_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel
from docfusion_lite.utils.geometry import compute_patch_boxes

class VisionEncoder(nn.Module):
    """
    Clean wrapper around a HuggingFace Vision Transformer (DeiT/ViT). This does positional encoding through patching/

    Responsibilities:
    - load model via HF AutoModel
    - optionally freeze backbone
    - expose hidden size (d_model)
    - simple forward() that returns patch embeddings
    - optional: return hidden states
    """

    def __init__(
        self,
        model_name: str = "microsoft/dit-base",
        freeze: bool = True,
        return_hidden_states: bool = False,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=return_hidden_states,
        )

        # patch embedding dim (e.g., 192 for DeiT-tiny)
        self.d_patch = self.model.config.hidden_size

        self.return_hidden_states = return_hidden_states

        if freeze:
            self.freeze_all()

    
    # Freezing utilities
    
    def freeze_all(self):
        """Freeze all backbone parameters."""
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_last_k_layers(self, k: int):
        """
        Unfreeze only the last k transformer layers.
        This assumes ViT/DeiT uses the common `encoder.layer` path.
        """
        layers = self.model.encoder.layer   # works for HF ViTs + DeiT

        for layer in layers[-k:]:
            for p in layer.parameters():
                p.requires_grad = True


    # Forward
    
    def forward(self, pixel_values):
        """
        pixel_values: (B, C, H, W), preprocessed by AutoImageProcessor

        Returns:
            patch_feats: (B, P, d_model)
            patch_boxes: (B, P, 4)
            optionally hidden_states
        """
        B, C, H, W = pixel_values.shape
        outputs = self.model(pixel_values)

        # full sequence: (B, 1 + P, d_model)
        seq = outputs.last_hidden_state

        # drop CLS → keep only patch tokens
        patch_feats = seq[:, 1:, :]   # (B, P, d_model)

        patch_boxes = compute_patch_boxes(B, H, W, self.model.config.patch_size)
        patch_boxes = patch_boxes.to(pixel_values.device)


        if self.return_hidden_states:
            return patch_feats, patch_boxes, outputs.hidden_states
        else:
            return patch_feats, patch_boxes

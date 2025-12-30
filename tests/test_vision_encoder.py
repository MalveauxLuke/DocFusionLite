# tests/test_vision_encoder.py

import torch

from docfusion_lite.models.vision_encoder import VisionEncoder


def test_vision_encoder_forward_shapes():
    encoder = VisionEncoder(
        model_name="microsoft/dit-base",
        freeze=True,
        return_hidden_states=False,
    )

    B = 2
    image_size = encoder.model.config.image_size
    patch_size = encoder.model.config.patch_size

    H = W = image_size
    pixel_values = torch.rand(B, 3, H, W)

    patch_feats, patch_boxes = encoder(pixel_values=pixel_values)

    B_out, P, d_patch = patch_feats.shape
    assert B_out == B
    assert d_patch == encoder.d_patch

    # Expected number of patches
    expected_patches = (H // patch_size) * (W // patch_size)
    assert P == expected_patches

    assert patch_boxes.shape == (B, P, 4)

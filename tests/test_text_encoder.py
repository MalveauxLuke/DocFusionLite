# tests/test_text_encoder.py

import torch

from docfusion_lite.models.text_encoder import TextEncoder


def test_text_encoder_forward_shapes():
    # Instantiate with default DeBERTa backbone
    encoder = TextEncoder(
        model_name="microsoft/deberta-v3-base",
        freeze=True,
        return_hidden_states=False,
    )

    B, T = 2, 8
    vocab_size = encoder.model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    bboxes = torch.rand(B, T, 4)  # [0, 1]

    out = encoder(input_ids=input_ids, attention_mask=attention_mask, bboxes=bboxes)

    assert out.shape == (B, T, encoder.d_model)

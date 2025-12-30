# tests/test_layout_embedding.py

import torch

from docfusion_lite.models.layout_embedding import AbsLayoutEmbedding


def test_layout_embedding_shapes_and_finite():
    B, T = 2, 5
    d_model = 128

    bboxes = torch.rand(B, T, 4)  # already in [0, 1]

    emb = AbsLayoutEmbedding(
        num_buckets=32,
        coord_emb_dim=8,
        d_layout=32,
        d_model=d_model,
        dropout=0.0,
    )

    out = emb(bboxes)  # (B, T, d_model)

    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_layout_embedding_identical_boxes_identical_embeddings():
    B, T = 1, 4
    d_model = 64

    # All boxes identical
    box = torch.tensor([0.3, 0.4, 0.7, 0.8])
    bboxes = box.view(1, 1, 4).repeat(B, T, 1)  # (B, T, 4)

    emb = AbsLayoutEmbedding(
        num_buckets=16,
        coord_emb_dim=4,
        d_layout=16,
        d_model=d_model,
        dropout=0.0,
    )

    out = emb(bboxes)  # (B, T, d_model)

    # All T embeddings should be identical for identical inputs
    diffs = (out[:, 1:, :] - out[:, :1, :]).abs().max()
    assert diffs < 1e-6

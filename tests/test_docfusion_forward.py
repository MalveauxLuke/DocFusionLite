# tests/test_docfusion_forward.py

import torch

from docfusion_lite.models.docfusion_model import DocFusionModel
from docfusion_lite.models.fusion import FusionLayer


class SimpleBatch:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_batch(model: DocFusionModel, batch_size: int = 2, seq_len: int = 16):
    vocab_size = model.text_encoder.model.config.vocab_size
    image_size = model.vision_encoder.model.config.image_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    token_boxes = torch.rand(batch_size, seq_len, 4)
    images = torch.rand(batch_size, 3, image_size, image_size)

    return SimpleBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_boxes=token_boxes,
        images=images,
    )


def _move_batch(batch: SimpleBatch, device: torch.device):
    for k in ("input_ids", "attention_mask", "token_boxes", "images"):
        v = getattr(batch, k)
        setattr(batch, k, v.to(device))
    return batch


def test_docfusion_mode31_forward_shapes():
    torch.manual_seed(0)

    model = DocFusionModel(
        mode="3.1",
        num_fusion_layers=0,
        num_encoder_layers=1,
        num_labels=8,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    )

    batch_size, seq_len = 2, 12
    batch = _make_fake_batch(model, batch_size=batch_size, seq_len=seq_len)

    logits = model(batch)
    assert logits.shape == (batch_size, seq_len, 8)
    assert torch.isfinite(logits).all()


def test_docfusion_mode33_forward_backward():
    torch.manual_seed(0)

    model = DocFusionModel(
        mode="3.3",
        num_fusion_layers=2,
        num_encoder_layers=1,
        num_labels=4,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    )

    batch = _make_fake_batch(model, batch_size=2, seq_len=8)

    logits = model(batch)
    assert logits.shape == (2, 8, 4)
    assert torch.isfinite(logits).all()

    loss = logits.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_padding_masked_tokens_ok():
    torch.manual_seed(0)

    model = DocFusionModel(
        mode="3.3",
        num_fusion_layers=1,
        num_encoder_layers=1,
        num_labels=3,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    )

    B, T = 2, 12
    batch = _make_fake_batch(model, batch_size=B, seq_len=T)

    batch.attention_mask[0, 8:] = 0
    batch.attention_mask[1, 5:] = 0

    logits = model(batch)
    assert logits.shape == (B, T, 3)
    assert torch.isfinite(logits).all()

    loss = logits.mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_fusionlayer_patch_mask_semantics_true_is_pad():
    torch.manual_seed(0)

    B, T, P, d = 2, 7, 11, 64
    layer = FusionLayer(
        d_model=d,
        n_heads=4,
        dropout=0.0,
        use_region_ffn=True,
        use_gate=True,
        gate_per_dim=False,
        gate_region_with_doc=False,
        gate_cross_with_doc=True,
    )

    h = torch.randn(B, T, d, requires_grad=True)
    region_proj = torch.randn(B, T, d, requires_grad=True)
    patch_feats = torch.randn(B, P, d, requires_grad=True)

    patch_mask = torch.zeros(B, P, dtype=torch.bool)
    patch_mask[:, -3:] = True

    g_doc = torch.tensor([[0.0], [1.0]]).view(B, 1, 1)

    out = layer(
        h=h,
        region_proj=region_proj,
        patch_feats=patch_feats,
        g_doc=g_doc,
        text_mask=torch.ones(B, T, dtype=torch.long),
        patch_mask=patch_mask,
    )

    assert out.shape == (B, T, d)
    assert torch.isfinite(out).all()

    out.mean().backward()
    assert h.grad is not None
    assert region_proj.grad is not None
    assert patch_feats.grad is not None


def test_encoders_frozen_exhaustive():
    torch.manual_seed(0)

    model = DocFusionModel(
        mode="3.3",
        num_fusion_layers=1,
        num_encoder_layers=1,
        num_labels=2,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    )

    # 1) Structural: frozen = not trainable
    assert all(not p.requires_grad for p in model.text_encoder.model.parameters())
    assert all(not p.requires_grad for p in model.vision_encoder.model.parameters())


    batch = _make_fake_batch(model, batch_size=2, seq_len=8)

    # 2) Behavioral: ensure no stale grads, then backward should not populate grads
    model.zero_grad(set_to_none=True)

    loss = model(batch).mean()
    loss.backward()

    assert all(p.grad is None for p in model.text_encoder.model.parameters())
    assert all(p.grad is None for p in model.vision_encoder.model.parameters())


def test_doc_gate_changes_outputs_mode33():
    torch.manual_seed(0)

    model = DocFusionModel(
        mode="3.3",
        num_fusion_layers=1,
        num_encoder_layers=1,
        num_labels=5,
        use_doc_gate=True,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    )

    batch = _make_fake_batch(model, batch_size=2, seq_len=10)

    def _gate_zero(h_text, text_mask=None):
        return torch.zeros(h_text.size(0), 1, 1, device=h_text.device)

    def _gate_one(h_text, text_mask=None):
        return torch.ones(h_text.size(0), 1, 1, device=h_text.device)

    orig = model.doc_gate.forward
    model.eval()

    model.doc_gate.forward = _gate_zero
    with torch.no_grad():
        logits0 = model(batch)

    model.doc_gate.forward = _gate_one
    with torch.no_grad():
        logits1 = model(batch)

    model.doc_gate.forward = orig

    assert logits0.shape == logits1.shape
    assert torch.isfinite(logits0).all()
    assert torch.isfinite(logits1).all()
    assert not torch.allclose(logits0, logits1)


def test_forward_cuda_if_available():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)

    device = torch.device("cuda")
    model = DocFusionModel(
        mode="3.1",
        num_fusion_layers=0,
        num_encoder_layers=1,
        num_labels=3,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    ).to(device)

    batch = _make_fake_batch(model, batch_size=2, seq_len=8)
    batch = _move_batch(batch, device)

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, 8, 3)
    assert torch.isfinite(logits).all()

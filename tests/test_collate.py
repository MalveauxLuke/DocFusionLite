# tests/test_collate.py

from pathlib import Path
import tempfile

import torch
from PIL import Image

from docfusion_lite.data.dataset import DocumentDataset
from docfusion_lite.data.dataset import make_collate_fn


def _make_test_image(tmpdir: Path, name: str, size: int = 32) -> str:
    img_path = tmpdir / name
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    img.save(img_path)
    return str(img_path)


def _make_two_samples(tmpdir: Path):
    img1 = _make_test_image(tmpdir, "page1.png", size=32)
    img2 = _make_test_image(tmpdir, "page2.png", size=32)

    sample1 = {
        "words": ["Hello", "world"],
        "bboxes": [[0, 0, 16, 16], [16, 16, 32, 32]],
        "width": 32,
        "height": 32,
        "image_path": img1,
    }

    sample2 = {
        "words": ["Single"],
        "bboxes": [[0, 0, 32, 32]],
        "width": 32,
        "height": 32,
        "image_path": img2,
    }

    return [sample1, sample2]


def test_collate_pads_and_aligns(tmp_path):
    samples = _make_two_samples(tmp_path)

    dataset = DocumentDataset(
        samples=samples,
        model_name="microsoft/deberta-v3-base",
        max_length=16,
        image_root=None,
        image_size=32,
    )

    # Get two items with different lengths after tokenization
    item1 = dataset[0]
    item2 = dataset[1]

    pad_token_id = dataset.tokenizer.pad_token_id
    collate_fn = make_collate_fn(pad_token_id=pad_token_id)

    batch = collate_fn([item1, item2])

    input_ids = batch.input_ids        # (B, T)
    attention_mask = batch.attention_mask
    token_boxes = batch.token_boxes
    images = batch.images

    B, T = input_ids.shape

    # Basic batch shapes
    assert B == 2
    assert attention_mask.shape == (B, T)
    assert token_boxes.shape == (B, T, 4)
    assert images.shape[0] == B  # (B, 3, H, W)

    # Check that padding mask and boxes align (zeros where mask=0)
    for i, item in enumerate([item1, item2]):
        L = item["input_ids"].size(0)

        # First L positions should match original ids and boxes
        assert torch.equal(input_ids[i, :L], item["input_ids"])
        assert torch.equal(attention_mask[i, :L], item["attention_mask"])
        assert torch.allclose(token_boxes[i, :L], item["token_boxes"])

        # After L, we expect mask==0 and boxes==0
        if L < T:
            assert torch.all(attention_mask[i, L:] == 0)
            assert torch.all(token_boxes[i, L:] == 0.0)

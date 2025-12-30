# tests/test_dataset.py

from pathlib import Path
import tempfile

import torch
from PIL import Image

from docfusion_lite.data.dataset import DocumentDataset


def _make_test_image(tmpdir: Path, name: str = "test_page.png", size: int = 32) -> str:
    """Create a small RGB image on disk and return its path."""
    img_path = tmpdir / name
    img = Image.new("RGB", (size, size), color=(128, 128, 128))
    img.save(img_path)
    return str(img_path)


def test_document_dataset_getitem_shapes(tmp_path):
    # Create a tiny 32x32 image
    img_path = _make_test_image(tmp_path, size=32)

    # Minimal fake OCR sample
    sample = {
        "words": ["Hello", "world"],
        "bboxes": [[0, 0, 16, 16], [16, 16, 32, 32]],
        "width": 32,
        "height": 32,
        "image_path": img_path,
    }

    dataset = DocumentDataset(
        samples=[sample],
        model_name="microsoft/deberta-v3-base",
        max_length=32,
        image_root=None,  # we passed absolute path
        image_size=32,
    )

    item = dataset[0]

    # Basic keys exist
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "token_boxes" in item
    assert "image" in item

    input_ids = item["input_ids"]
    attention_mask = item["attention_mask"]
    token_boxes = item["token_boxes"]
    image = item["image"]

    # Shapes
    assert input_ids.ndim == 1  # (T,)
    assert attention_mask.ndim == 1 and attention_mask.shape == input_ids.shape
    assert token_boxes.ndim == 2 and token_boxes.shape[0] == input_ids.shape[0]
    assert token_boxes.shape[1] == 4  # (T, 4)
    assert image.shape[0] == 3  # (3, H, W)

    # BBoxes in [0, 1]
    assert torch.all(token_boxes >= 0.0)
    assert torch.all(token_boxes <= 1.0)

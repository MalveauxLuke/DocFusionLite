from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
from PIL import Image


FUNSD_TYPES = ["question", "answer", "header"]  # "other" -> O


def build_label_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = ["O"]
    for t in FUNSD_TYPES:
        labels += [f"B-{t}", f"I-{t}"]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def _json_paths(root: Path, split: str) -> Tuple[Path, Path]:
    if split == "train":
        return root / "training_data" / "annotations", root / "training_data" / "images"
    if split == "test":
        return root / "testing_data" / "annotations", root / "testing_data" / "images"
    raise ValueError("split must be 'train' or 'test'")


def funsd_to_samples(funsd_root: str, split: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Returns (samples, image_root).

    Each sample matches what your token-classification DocumentDataset expects:
        {
          "words": [...],
          "bboxes": [[x0,y0,x1,y1], ...],          # pixel coords
          "word_labels": ["B-question", ...],      # word-level BIO
          "width": W,
          "height": H,
          "image_path": "xxx.png"
        }
    """
    root = Path(funsd_root)
    ann_dir, img_dir = _json_paths(root, split)

    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        raise FileNotFoundError(f"No FUNSD annotation JSONs found at: {ann_dir}")

    samples: List[Dict[str, Any]] = []

    for jp in ann_files:
        # image shares stem
        ip = img_dir / f"{jp.stem}.png"
        if not ip.exists():
            ip = img_dir / f"{jp.stem}.jpg"
        if not ip.exists():
            continue

        data = json.loads(jp.read_text(encoding="utf-8"))

        img = Image.open(ip).convert("RGB")
        W, H = img.size

        words: List[str] = []
        bboxes: List[List[int]] = []
        word_labels: List[str] = []

        # --- BIO CREATION HAPPENS HERE ---
        for ent in data.get("form", []):
            ent_label = (ent.get("label") or "other").lower()
            ent_words = ent.get("words", [])

            for j, w in enumerate(ent_words):
                txt = (w.get("text") or "").strip()
                if not txt:
                    continue

                x0, y0, x1, y1 = w.get("box", [0, 0, 0, 0])

                # clamp to image bounds
                x0 = int(max(0, min(x0, W)))
                x1 = int(max(0, min(x1, W)))
                y0 = int(max(0, min(y0, H)))
                y1 = int(max(0, min(y1, H)))

                if ent_label == "other":
                    tag = "O"
                else:
                    tag = ("B-" if j == 0 else "I-") + ent_label

                words.append(txt)
                bboxes.append([x0, y0, x1, y1])
                word_labels.append(tag)
        # --- END BIO CREATION ---

        if not words:
            continue

        samples.append(
            {
                "words": words,
                "bboxes": bboxes,
                "word_labels": word_labels,
                "width": W,
                "height": H,
                "image_path": ip.name,
            }
        )

    return samples, str(img_dir)

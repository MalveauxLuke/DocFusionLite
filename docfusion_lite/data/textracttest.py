"""
This converter:
- reads out_detect.json (Textract)
- extracts WORD blocks
- sorts into a stable reading order
- converts Textract normalized boxes -> pixel boxes using the actual image size
- emits 1 sample per page 
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
from PIL import Image

SLICE_FIELDS = ["name", "dob", "id", "phone"]


def build_label_maps(fields: List[str] = SLICE_FIELDS) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Builds BIO label maps for token classification.

    Example labels:
      O
      B-name, I-name
      B-dob,  I-dob
      B-id,   I-id
      B-phone,I-phone
    """
    labels: List[str] = ["O"]
    for f in fields:
        labels.append(f"B-{f}")
        labels.append(f"I-{f}")

    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def num_labels_from_label2id(label2id: Dict[str, int]) -> int:
    return len(label2id)


def _reading_order_key(word: Dict[str, Any]) -> Tuple[int, float, float]:
    x0, y0, x1, y1 = word["box_norm"]
    yc = (y0 + y1) * 0.5
    xc = (x0 + x1) * 0.5
    return (word["page"], yc, xc)


def _extract_textract_words(textract: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts:
      {page:int, text:str, box_norm:(x0,y0,x1,y1), conf:float}
    """
    out: List[Dict[str, Any]] = []
    for b in textract.get("Blocks", []):
        if b.get("BlockType") != "WORD":
            continue
        text = b.get("Text", "")
        if not text:
            continue
        bb = (b.get("Geometry", {}) or {}).get("BoundingBox")
        if not bb:
            continue

        x0 = float(bb["Left"])
        y0 = float(bb["Top"])
        x1 = x0 + float(bb["Width"])
        y1 = y0 + float(bb["Height"])

        # clamp just in case
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        out.append(
            {
                "page": int(b.get("Page", 1)),
                "text": text,
                "box_norm": (x0, y0, x1, y1),
                "conf": float(b.get("Confidence", 0.0)),
            }
        )

    out.sort(key=_reading_order_key)
    return out


def textract_detect_to_page_samples(
    textract_json_path: str | Path,
    image_paths_by_page: Dict[int, str | Path],
    *,
    default_label: str = "O",
    image_path_in_sample: str = "relative",  # "relative" or "absolute"
) -> List[Dict[str, Any]]:
    """
    Convert Textract DetectDocumentText output into a list of samples (1 per page),
    directly consumable by your existing DocumentDataset.
    """
    textract_json_path = Path(textract_json_path)
    with textract_json_path.open("r", encoding="utf-8") as f:
        textract = json.load(f)

    words_all = _extract_textract_words(textract)

    # group by page
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for w in words_all:
        by_page.setdefault(w["page"], []).append(w)

    samples: List[Dict[str, Any]] = []

    for page_num, page_words in sorted(by_page.items(), key=lambda kv: kv[0]):
        if page_num not in image_paths_by_page:
            raise ValueError(
                f"Missing image path for page {page_num}. "
                f"Have pages={sorted(by_page.keys())}, image_paths_by_page={sorted(image_paths_by_page.keys())}"
            )

        img_path = Path(image_paths_by_page[page_num])
        with Image.open(img_path) as img:
            W, H = img.size  # pixel width/height

        words: List[str] = []
        bboxes: List[List[float]] = []

        for w in page_words:
            x0n, y0n, x1n, y1n = w["box_norm"]
            # normalized -> pixel coords
            x0 = x0n * W
            y0 = y0n * H
            x1 = x1n * W
            y1 = y1n * H
            words.append(w["text"])
            bboxes.append([x0, y0, x1, y1])

        # placeholder labels (replace later with real BIO labels)
        word_labels = [default_label] * len(words)

        image_path_field = str(img_path.resolve()) if image_path_in_sample == "absolute" else str(img_path)

        samples.append(
            {
                "words": words,
                "bboxes": bboxes,   # pixel coords
                "width": W,
                "height": H,
                "image_path": image_path_field,
                "word_labels": word_labels,
                "page": page_num,   # extra metadata (ignored by your dataset)
            }
        )

    return samples


def write_jsonl(samples: List[Dict[str, Any]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    samples = textract_detect_to_page_samples(
        textract_json_path="out_detect.json",
        image_paths_by_page={1: "form-wh-380-e-813x1024.jpg"},
        default_label="O",
        image_path_in_sample="relative",
    )

    write_jsonl(samples, "textract_samples.jsonl")

    print("wrote", len(samples), "samples to textract_samples.jsonl")
    print("first sample keys:", samples[0].keys())
    print("num words:", len(samples[0]["words"]))

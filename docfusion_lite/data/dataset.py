from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer

from ..models.types import DocBatch


class DocumentDataset(Dataset):
    """
    Token-level classification dataset.

    Expected sample keys:
        - "words":        List[str] word-level OCR
        - "bboxes":       List[[x0,y0,x1,y1]] pixel coords, same length as words
        - "width":        int/float
        - "height":       int/float
        - "image_path":   str (relative to image_root or absolute)
        - "word_labels":  List[str] word-level BIO tags (e.g., "O", "B-question", ...), same length as words

    __getitem__ returns dict with:
        - "input_ids":      (T,)
        - "attention_mask": (T,)
        - "token_boxes":    (T, 4) float in [0,1]
        - "image":          (3, H, W)
        - "labels":         (T,) long, with -100 on specials (and later padded positions)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        label2id: Dict[str, int],
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 192,
        image_root: Optional[str] = None,
        image_size: int = 384,
    ):
        self.samples = samples
        self.label2id = label2id

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

        self.image_root = Path(image_root) if image_root is not None else None
        self.image_transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _normalize_bboxes(
        bboxes: List[List[float]],
        width: float,
        height: float,
    ) -> List[List[float]]:
        w = max(float(width), 1.0)
        h = max(float(height), 1.0)

        out: List[List[float]] = []
        for x0, y0, x1, y1 in bboxes:
            out.append([float(x0) / w, float(y0) / h, float(x1) / w, float(y1) / h])

        return out

    def _tokenize_align(
        self,
        words: List[str],
        norm_boxes: List[List[float]],
        word_labels: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        word_ids = enc.word_ids(batch_index=0)

        token_boxes: List[List[float]] = []
        token_labels: List[int] = []

        prev_wid = None
        for wid in word_ids:
            if wid is None:
                token_boxes.append([0.0, 0.0, 0.0, 0.0])
                token_labels.append(-100)
                prev_wid = None
                continue

            token_boxes.append(norm_boxes[wid])

            lab = word_labels[wid]
            if prev_wid == wid and lab.startswith("B-"):
                lab = "I-" + lab[2:]

            token_labels.append(self.label2id.get(lab, self.label2id["O"]))
            prev_wid = wid

        input_ids = enc["input_ids"].squeeze(0)              # (T,)
        attention_mask = enc["attention_mask"].squeeze(0)    # (T,)
        token_boxes_t = torch.tensor(token_boxes, dtype=torch.float)     # (T,4)
        labels_t = torch.tensor(token_labels, dtype=torch.long)         # (T,)

        return input_ids, attention_mask, token_boxes_t, labels_t

    def _load_image(self, sample: Dict[str, Any]) -> torch.Tensor:
        image_path = sample["image_path"]
        img_path = self.image_root / image_path if self.image_root is not None else Path(image_path)
        img = Image.open(img_path).convert("RGB")
        return self.image_transform(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        words = sample["words"]
        bboxes = sample["bboxes"]
        width = sample["width"]
        height = sample["height"]
        word_labels = sample["word_labels"]

        if not (len(words) == len(bboxes) == len(word_labels)):
            raise ValueError("words, bboxes, and word_labels must have same length")

        norm_boxes = self._normalize_bboxes(bboxes, width, height)

        input_ids, attention_mask, token_boxes, labels = self._tokenize_align(
            words, norm_boxes, word_labels
        )

        image = self._load_image(sample)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_boxes": token_boxes,
            "image": image,
            "labels": labels,
        }


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def make_collate_fn(pad_token_id: int):
    """
    Pads variable-length token sequences for token classification.

    Output DocBatch:
        - input_ids:      (B,T)
        - attention_mask: (B,T)
        - token_boxes:    (B,T,4)
        - images:         (B,3,H,W)
        - labels:         (B,T) with -100 for padded positions
    """

    def collate(batch: List[Dict[str, Any]]) -> DocBatch:
        bs = len(batch)
        lens = [b["input_ids"].size(0) for b in batch]
        Tm = max(lens)

        input_ids = torch.full((bs, Tm), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bs, Tm), dtype=torch.long)
        token_boxes = torch.zeros((bs, Tm, 4), dtype=torch.float)
        labels = torch.full((bs, Tm), -100, dtype=torch.long)

        images = torch.stack([b["image"] for b in batch], dim=0)

        for i, b in enumerate(batch):
            L = b["input_ids"].size(0)
            input_ids[i, :L] = b["input_ids"]
            attention_mask[i, :L] = b["attention_mask"]
            token_boxes[i, :L] = b["token_boxes"]
            labels[i, :L] = b["labels"]

        return DocBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_boxes=token_boxes,
            images=images,
            labels=labels,
        )

    return collate

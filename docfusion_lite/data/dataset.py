# src/data/dataset.py

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
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
        - "token_boxes":    (T, 4) int in [0,1000]
        - "image":          (3, H, W)
        - "labels":         (T,) long, with -100 on specials (and later padded positions)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        label2id: Dict[str, int],
        model_name: str = "SCUT-DLVCLab/lilt-roberta-en-base",
        max_length: int = 192,
        image_root: Optional[str] = None,
        image_size: int = 224,
    ):
        super().__init__()
        self.samples = samples
        self.label2id = label2id

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length
        print("tokenizer class:", type(self.tokenizer))
        print("name_or_path:", self.tokenizer.name_or_path)


        self.image_root = Path(image_root) if image_root is not None else None
        self.image_size = int(image_size)

        # Letter box
        self.image_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_image_path(self, image_path: str) -> Path:
        if self.image_root is not None:
            return self.image_root / image_path
        return Path(image_path)

    def _letterbox_image_and_boxes(
        self,
        img: Image.Image,
        bboxes: List[List[float]],
    ) -> Tuple[Image.Image, List[List[float]]]:
        """
        Letterbox to (S,S) without distorting aspect ratio.
        Applies the same scale+pad to bboxes (still in pixel coords).

        Returns:
            img_pad: PIL image (S,S)
            bboxes_pad: pixel bboxes in the padded/resized (S,S) coordinate system
        """
        S = int(self.image_size)
        orig_w, orig_h = img.size
        if orig_w <= 0 or orig_h <= 0:
            raise ValueError("Invalid image size")

        # scale so that the long side becomes S
        scale = S / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))

        # resize 
        img_rs = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # pad to SxS (centered)
        pad_x = S - new_w
        pad_y = S - new_h
        pad_left = pad_x // 2
        pad_right = pad_x - pad_left
        pad_top = pad_y // 2
        pad_bottom = pad_y - pad_top

        img_pad = F.pad(img_rs, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)

        # transform bboxes: scale then shift by pad offsets
        out_boxes: List[List[float]] = []
        for x0, y0, x1, y1 in bboxes:
            x0n = float(x0) * scale + pad_left
            y0n = float(y0) * scale + pad_top
            x1n = float(x1) * scale + pad_left
            y1n = float(y1) * scale + pad_top

            if x1n < x0n:
                x0n, x1n = x1n, x0n
            if y1n < y0n:
                y0n, y1n = y1n, y0n

            out_boxes.append([x0n, y0n, x1n, y1n])

        return img_pad, out_boxes

    @staticmethod
    def _pixel_boxes_to_1000(
        bboxes: List[List[float]],
        size: int,
    ) -> List[List[int]]:
        """
        Convert pixel bboxes in an SxS coordinate system into LayoutLM-style integer bboxes in [0,1000].
        """
        S = max(int(size), 1)

        out: List[List[int]] = []
        for x0, y0, x1, y1 in bboxes:
            nx0 = int(round(1000.0 * float(x0) / S))
            ny0 = int(round(1000.0 * float(y0) / S))
            nx1 = int(round(1000.0 * float(x1) / S))
            ny1 = int(round(1000.0 * float(y1) / S))

            # clamp
            nx0 = max(0, min(1000, nx0))
            ny0 = max(0, min(1000, ny0))
            nx1 = max(0, min(1000, nx1))
            ny1 = max(0, min(1000, ny1))

            if nx1 < nx0:
                nx0, nx1 = nx1, nx0
            if ny1 < ny0:
                ny0, ny1 = ny1, ny0

            out.append([nx0, ny0, nx1, ny1])

        return out

    def _tokenize_align(
        self,
        words: List[str],
        bboxes_1000: List[List[int]],
        word_labels: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            words,
            boxes=bboxes_1000,                 
            #is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        word_ids = enc.word_ids(batch_index=0)

        token_labels: List[int] = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                token_labels.append(-100)
                prev_wid = None
                continue

            lab = word_labels[wid]
            if prev_wid == wid and lab.startswith("B-"):
                lab = "I-" + lab[2:]

            token_labels.append(self.label2id.get(lab, self.label2id["O"]))
            prev_wid = wid

        input_ids = enc["input_ids"].squeeze(0)                 # (T,)
        attention_mask = enc["attention_mask"].squeeze(0)       # (T,)

        token_boxes_t = enc["bbox"].squeeze(0).to(torch.long)   # (T,4)

        labels_t = torch.tensor(token_labels, dtype=torch.long) # (T,)

        return input_ids, attention_mask, token_boxes_t, labels_t


    def _load_image_pil(self, sample: Dict[str, Any]) -> Image.Image:
        image_path = sample["image_path"]
        img_path = self._resolve_image_path(image_path)
        return Image.open(img_path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        words = sample["words"]
        bboxes = sample["bboxes"]              
        word_labels = sample["word_labels"]

        if not (len(words) == len(bboxes) == len(word_labels)):
            raise ValueError("words, bboxes, and word_labels must have same length")

        img = self._load_image_pil(sample)

        img_pad, bboxes_pad = self._letterbox_image_and_boxes(img, bboxes)

        bboxes_1000 = self._pixel_boxes_to_1000(bboxes_pad, size=self.image_size)

        input_ids, attention_mask, token_boxes, labels = self._tokenize_align(
            words, bboxes_1000, word_labels
        )

        image = self.image_transform(img_pad)  # (3,S,S)

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
        token_boxes = torch.zeros((bs, Tm, 4), dtype=torch.long)
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

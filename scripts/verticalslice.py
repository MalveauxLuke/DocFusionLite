import os
import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from docfusion_lite.data.textracttest import build_label_maps, textract_detect_to_page_samples
from docfusion_lite.data.dataset import DocumentDataset, make_collate_fn
from docfusion_lite.models.docfusion_model import DocFusionModel


def seed_all(seed: int = 7):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def freeze_module(m: nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def move_batch_to_device(batch, device: torch.device):
    for k, v in vars(batch).items():
        if torch.is_tensor(v):
            setattr(batch, k, v.to(device, non_blocking=True))
    return batch


def overlay_annotations(image_path, words, bboxes, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for word, bbox in zip(words, bboxes):
        x0, y0, x1, y1 = bbox
        draw.rectangle([x0, y0, x1, y1], outline="green", width=2)
        draw.text((x0, max(0, y0 - 10)), word, fill="green", font=font)

    output_path = str(output_path)
    image.save(output_path)
    print(f"Annotated image saved as {output_path}")
    

def tokens_to_word_tags(tokenizer, words, pred_token_ids, *, max_length: int):
    """
    Map token-level predictions -> word-level tags using tokenizer.word_ids().
    """
    enc = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    word_ids = enc.word_ids(batch_index=0)

    wid_to_pred = {}
    for tok_i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in wid_to_pred:
            wid_to_pred[wid] = int(pred_token_ids[tok_i])

    # align to words length (some may be truncated -> default to 0 which is "O")
    out = [wid_to_pred.get(i, 0) for i in range(len(words))]
    return out
def bio_decode_spans(words: List[str], tags: List[str]) -> Dict[str, List[str]]:
    """
    Convert word-level BIO tags into extracted field text.
    Returns {field: [span_text1, span_text2, ...]}.

    Example tags: O, B-name, I-name, B-dob, ...
    """
    out: Dict[str, List[str]] = {}
    cur_field: Optional[str] = None
    cur_tokens: List[str] = []

    def flush():
        nonlocal cur_field, cur_tokens
        if cur_field is not None and cur_tokens:
            out.setdefault(cur_field, []).append(" ".join(cur_tokens))
        cur_field = None
        cur_tokens = []

    for w, tag in zip(words, tags):
        if tag == "O":
            flush()
            continue

        if "-" not in tag:
            flush()
            continue

        pref, field = tag.split("-", 1)

        if pref == "B":
            flush()
            cur_field = field
            cur_tokens = [w]
        elif pref == "I":
            # continue only if same field; otherwise start new
            if cur_field == field:
                cur_tokens.append(w)
            else:
                flush()
                cur_field = field
                cur_tokens = [w]
        else:
            flush()

    flush()
    return out


def write_prediction_json(
    *,
    out_path: str | Path,
    doc_id: str,
    page_num: int,
    words: List[str],
    tags: List[str],
    extracted: Dict[str, List[str]],
) -> None:
    payload = {
        "doc_id": doc_id,
        "page": page_num,
        "extracted": extracted,      # field -> list of strings
        "word_predictions": [
            {"word": w, "tag": t} for w, t in zip(words, tags)
        ],
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote json: {out_path}")


def overlay_predictions(image_path, words, bboxes, word_tags, output_path, *, draw_all: bool = True):
    """
    Draw word boxes; annotate predicted tag above each word.
    Always draws everything by default.
    Annotates ONLY the tag (no word text).
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for bb, tag in zip(bboxes, word_tags):
        x0, y0, x1, y1 = bb

        # box color: red for entity, green for O (optional)
        outline = "red" if tag != "O" else "green"
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=2)

        # tag only (no word)
        draw.text((x0, max(0, y0 - 10)), tag, fill=outline, font=font)

    image.save(str(output_path))
    print(f"Pred overlay saved: {output_path}")


def main():
    seed_all(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # repo root
    ROOT = Path(__file__).resolve().parents[1]

    # paths to textract
    textract_json_path = ROOT / "docfusion_lite" / "data" / "textracttest" / "out_detect_hw.json"
    img_path = ROOT / "docfusion_lite" / "data" / "textracttest" / "form-wh-380-e-813x1024-hw.jpg"

   
    OUT_DIR = Path(__file__).resolve().parent / "textractoutput"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "annotated_page1.jpg"

    # 1 Textract JSON -> samples
    samples = textract_detect_to_page_samples(
        textract_json_path=textract_json_path,
        image_paths_by_page={1: img_path},
        default_label="O",
        image_path_in_sample="absolute",  # makes s0["image_path"] definitely loadable anywhere
    )

    # 2 draw boxes+words
    [s0] = samples  
    overlay_annotations(
        image_path=s0["image_path"],
        words=s0["words"],
        bboxes=s0["bboxes"],
        output_path=out_path,  
    )

    label2id,_ = build_label_maps()

    ds = DocumentDataset(
        samples = samples,
        label2id = label2id,
        image_root = None,
        image_size=224,
        max_length=192,
    )
    print(label2id)
    collate_fn = make_collate_fn(pad_token_id=ds.tokenizer.pad_token_id)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    batch = next(iter(dl))
    batch = move_batch_to_device(batch, device)
    model = DocFusionModel(
        mode="3.1",
        num_fusion_layers=0,
        num_encoder_layers=2,
        num_labels=len(label2id),
        dropout=0.0,
        use_gate_stem=False,
        use_gate_fusion_layer=False,
        use_region_ffn=True,
        text_model_name="microsoft/deberta-v3-base",
        vision_model_name="microsoft/dit-base",
    ).to(device)

    model.text_encoder.freeze_all()
    model.vision_encoder.freeze_all()

    with torch.no_grad():
        logits = model(batch)

    pred_token_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()  # (T,)

    # build id2label from your label maps
    _, id2label = build_label_maps()
    pred_word_ids = tokens_to_word_tags(ds.tokenizer, s0["words"], pred_token_ids, max_length=ds.max_length)
    pred_word_tags = [id2label[i] for i in pred_word_ids]

    out_pred = OUT_DIR / "pred_overlay_page1.jpg"
    overlay_predictions(
        image_path=s0["image_path"],
        words=s0["words"],
        bboxes=s0["bboxes"],
        word_tags=pred_word_tags,
        output_path=out_pred,
        draw_all=False,   # set True to draw every word
    )

    doc_id = Path(s0["image_path"]).stem
    extracted = bio_decode_spans(s0["words"], pred_word_tags)
    write_prediction_json(
        out_path=OUT_DIR / "predictions_page1.json",
        doc_id=doc_id,
        page_num=int(s0.get("page", 1)),
        words=s0["words"],
        tags=pred_word_tags,
        extracted=extracted, )


if __name__ == "__main__":
    main()
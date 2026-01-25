from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

TEXTRACT_JSON_PATH = ROOT / "scripts" / "azuredeid" / "out_detect_hw_deid.json"

IMAGE_PATHS_BY_PAGE: Dict[int, Path] = {
    1: ROOT / "docfusion_lite" / "data" / "textracttest" / "form-wh-380-e-813x1024-hw.jpg",
}

OUT_DIR = Path(__file__).resolve().parent / "azuredeid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALTERED_FLAG_KEY = "deid_altered"


def _bbox_key(block: Dict[str, Any]) -> Tuple[float, float]:
    bb = block["Geometry"]["BoundingBox"]
    return (float(bb["Top"]), float(bb["Left"]))


def _child_ids(block: Dict[str, Any]) -> List[str]:
    for rel in (block.get("Relationships", []) or []):
        if rel.get("Type") == "CHILD":
            ids = rel.get("Ids", []) or []
            return [x for x in ids if isinstance(x, str)]
    return []


def _norm_box_from_block(b: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bb = (b.get("Geometry", {}) or {}).get("BoundingBox")
    if not bb:
        return None
    x0 = float(bb["Left"])
    y0 = float(bb["Top"])
    x1 = x0 + float(bb["Width"])
    y1 = y0 + float(bb["Height"])
    x0 = max(0.0, min(1.0, x0))
    y0 = max(0.0, min(1.0, y0))
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    return (x0, y0, x1, y1)


def _infer_redaction_type(text: str) -> str:
    t = (text or "").strip().lower()
    if not (t.startswith("[") and t.endswith("]")):
        return "phi"
    inner = t[1:-1].strip()

    if inner in {"name", "person", "patient"}:
        return "name"
    if inner in {"date", "dob", "birthdate"}:
        return "dob"
    if inner in {"id", "ssn", "mrn", "memberid"}:
        return "id"
    if inner in {"phone", "phonenumber", "tel"}:
        return "phone"
    return inner or "phi"


def extract_words_in_reading_order(
    textract: Dict[str, Any],
    *,
    altered_flag_key: str,
) -> Dict[int, List[Dict[str, Any]]]:
    blocks = textract.get("Blocks", []) or []
    id2 = {b["Id"]: b for b in blocks if isinstance(b.get("Id"), str)}

    words_by_page: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for b in blocks:
        if b.get("BlockType") != "WORD":
            continue
        wid = b.get("Id")
        if not isinstance(wid, str):
            continue
        box = _norm_box_from_block(b)
        if not box:
            continue

        page = int(b.get("Page", 1))
        txt = (b.get("Text") or "").strip()
        edited = bool(b.get(altered_flag_key, False))

        words_by_page.setdefault(page, {})[wid] = {
            "word_id": wid,
            "page": page,
            "text": txt,
            "box_norm": box,
            "edited": edited,
            "redaction_type": _infer_redaction_type(txt) if edited else "",
        }

    lines = [b for b in blocks if b.get("BlockType") == "LINE"]
    lines.sort(key=_bbox_key)

    out_by_page: Dict[int, List[Dict[str, Any]]] = {}
    seen: set[str] = set()

    for line in lines:
        page = int(line.get("Page", 1))
        for cid in _child_ids(line):
            wb = id2.get(cid)
            if not wb or wb.get("BlockType") != "WORD":
                continue
            wid = wb.get("Id")
            if not isinstance(wid, str) or wid in seen:
                continue
            rec = words_by_page.get(page, {}).get(wid)
            if not rec:
                continue
            seen.add(wid)
            out_by_page.setdefault(page, []).append(rec)

    for page, wmap in words_by_page.items():
        leftovers = [v for k, v in wmap.items() if k not in seen]
        leftovers.sort(
            key=lambda w: (
                (w["box_norm"][1] + w["box_norm"][3]) * 0.5,
                (w["box_norm"][0] + w["box_norm"][2]) * 0.5,
            )
        )
        if leftovers:
            out_by_page.setdefault(page, []).extend(leftovers)

    return out_by_page


def textract_to_samples(
    textract_json_path: Path,
    image_paths_by_page: Dict[int, Path],
    *,
    altered_flag_key: str,
) -> List[Dict[str, Any]]:
    textract = json.loads(textract_json_path.read_text(encoding="utf-8"))
    words_by_page = extract_words_in_reading_order(textract, altered_flag_key=altered_flag_key)

    samples: List[Dict[str, Any]] = []
    for page_num, page_words in sorted(words_by_page.items(), key=lambda kv: kv[0]):
        img_path = image_paths_by_page.get(page_num)
        if img_path is None:
            raise ValueError(f"Missing image path for page {page_num} in IMAGE_PATHS_BY_PAGE")

        with Image.open(img_path) as img:
            W, H = img.size

        words: List[str] = []
        bboxes: List[List[float]] = []
        edited: List[bool] = []
        redaction_types: List[str] = []

        for w in page_words:
            x0n, y0n, x1n, y1n = w["box_norm"]
            words.append(w["text"])
            bboxes.append([x0n * W, y0n * H, x1n * W, y1n * H])
            edited.append(bool(w["edited"]))
            redaction_types.append(w.get("redaction_type", ""))

        samples.append(
            {
                "page": int(page_num),
                "image_path": str(img_path.resolve()),
                "width": W,
                "height": H,
                "words": words,
                "bboxes": bboxes,
                "edited": edited,
                "redaction_types": redaction_types,
            }
        )

    return samples


def write_jsonl(samples: List[Dict[str, Any]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def _clamp_box(bb: List[float], W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bb
    x0i = max(0, min(W, int(x0)))
    y0i = max(0, min(H, int(y0)))
    x1i = max(0, min(W, int(x1)))
    y1i = max(0, min(H, int(y1)))
    if x1i < x0i:
        x0i, x1i = x1i, x0i
    if y1i < y0i:
        y0i, y1i = y1i, y0i
    return x0i, y0i, x1i, y1i


def _short_cat(cat: str) -> str:
    c = (cat or "").strip()
    if not c:
        return "PHI"
    if c.startswith("[") and c.endswith("]"):
        c = c[1:-1]
    return c.upper()


def scrub_sample_to_image(sample: Dict[str, Any], out_path: Path) -> None:
    img = Image.open(sample["image_path"]).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for bb, e, cat in zip(sample["bboxes"], sample["edited"], sample["redaction_types"]):
        if not e:
            continue

        x0, y0, x1, y1 = _clamp_box(bb, W, H)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, 255))

        label = _short_cat(cat)

        # draw label inside box (with tiny padding)
        #tx = x0 + 2
        #ty = y0 + 2

        
        #raw.text((tx, ty), label, fill=(255, 255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_path))

def setup_paths_overlay():
    TEXTRACT_JSON_PATH = ROOT / "scripts" / "azuredeid" / "out_detect_hw_deid.json"

    IMAGE_PATHS_BY_PAGE: Dict[int, Path] = {
        1: ROOT / "docfusion_lite" / "data" / "textracttest" / "form-wh-380-e-813x1024-hw.jpg",
    }

    OUT_DIR = Path(__file__).resolve().parent / "azuredeid"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ALTERED_FLAG_KEY = "deid_altered"
    return TEXTRACT_JSON_PATH, IMAGE_PATHS_BY_PAGE, OUT_DIR, ALTERED_FLAG_KEY
def main() -> None:
    samples = textract_to_samples(
        textract_json_path=TEXTRACT_JSON_PATH,
        image_paths_by_page=IMAGE_PATHS_BY_PAGE,
        altered_flag_key=ALTERED_FLAG_KEY,
    )

    write_jsonl(samples, OUT_DIR / "samples.jsonl")

    for s in samples:
        page = int(s.get("page", 1))
        scrub_sample_to_image(s, OUT_DIR / f"scrubbed_page{page}.png")

    print(f"wrote: {OUT_DIR / 'samples.jsonl'}")
    print(f"wrote scrubbed images to: {OUT_DIR}")


if __name__ == "__main__":
    main()

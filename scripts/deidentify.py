from __future__ import annotations

import os
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

from azure.identity import DefaultAzureCredential
from azure.health.deidentification import DeidentificationClient
from azure.health.deidentification.models import DeidentificationContent, DeidentificationOperationType


def _bbox_key(block: Dict[str, Any]) -> Tuple[float, float]:
    bb = block["Geometry"]["BoundingBox"]
    return (float(bb["Top"]), float(bb["Left"]))


def _child_ids(block: Dict[str, Any]) -> List[str]:
    for rel in block.get("Relationships", []) or []:
        if rel.get("Type") == "CHILD":
            ids = rel.get("Ids", []) or []
            return [x for x in ids if isinstance(x, str)]
    return []


def _group_lines_by_page(textract: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    pages: Dict[int, List[Dict[str, Any]]] = {}
    for b in textract.get("Blocks", []) or []:
        if b.get("BlockType") != "LINE":
            continue
        page = int(b.get("Page", 1))
        pages.setdefault(page, []).append(b)
    for p in pages:
        pages[p].sort(key=_bbox_key)
    return pages


def build_clean_text_and_spans_by_page(textract: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = textract.get("Blocks", []) or []
    id2: Dict[str, Dict[str, Any]] = {b["Id"]: b for b in blocks if isinstance(b.get("Id"), str)}
    pages = _group_lines_by_page(textract)

    out: List[Dict[str, Any]] = []

    for page_num in sorted(pages.keys()):
        chunks: List[str] = []
        cur = 0
        word_spans: Dict[str, Tuple[int, int]] = {}

        for line in pages[page_num]:
            word_ids = [wid for wid in _child_ids(line) if id2.get(wid, {}).get("BlockType") == "WORD"]
            wrote_any = False
            first = True

            for wid in word_ids:
                w = (id2[wid].get("Text") or "").strip()
                if not w:
                    continue

                if not first:
                    chunks.append(" ")
                    cur += 1

                start = cur
                chunks.append(w)
                cur += len(w)
                end = cur

                word_spans[wid] = (start, end)
                first = False
                wrote_any = True

            if wrote_any:
                chunks.append("\n")
                cur += 1

        if chunks and chunks[-1] == "\n":
            chunks.pop()
            cur -= 1

        out.append(
            {
                "page": page_num,
                "text": "".join(chunks),
                "word_spans": word_spans,  # {word_id: [start,end)} in python code points
            }
        )

    return out


def _code_point(x: Any) -> int:
    if x is None:
        return 0
    v = getattr(x, "code_point", None)
    if v is not None:
        return int(v)
    return int(x)


def tag_entities(
    *,
    client: DeidentificationClient,
    text: str,
) -> List[Dict[str, Any]]:
    body = DeidentificationContent(input_text=text, operation_type=DeidentificationOperationType.TAG)
    result = client.deidentify_text(body)

    ents: List[Dict[str, Any]] = []
    tagger = getattr(result, "tagger_result", None)
    for e in (getattr(tagger, "entities", None) or []):
        ents.append(
            {
                "category": str(getattr(e, "category", "")),
                "offset": _code_point(getattr(e, "offset", None)),
                "length": _code_point(getattr(e, "length", None)),
            }
        )
    return ents


def _placeholder_for_category(cat: str) -> str:
    c = (cat or "").strip().lower()

    if "name" in c or c in {"person"}:
        return "[name]"
    if "dob" in c or "birth" in c or "date" in c:
        return "[date]"
    if "phone" in c or "tel" in c:
        return "[phone]"
    if "ssn" in c or "mrn" in c or c == "id" or c.endswith("_id") or "id" == c:
        return "[id]"
    if "address" in c:
        return "[address]"
    if "email" in c:
        return "[email]"

    return f"[{c}]" if c else "[phi]"


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def compute_word_replacements_from_entities(
    *,
    word_spans: Dict[str, Tuple[int, int]],
    entities: List[Dict[str, Any]],
) -> Dict[str, str]:
    # deterministic: left-to-right, longer spans first at same offset
    ents = sorted(entities, key=lambda e: (int(e.get("offset", 0)), -int(e.get("length", 0))))

    word_id_to_new: Dict[str, str] = {}
    for e in ents:
        off = int(e.get("offset", 0))
        ln = int(e.get("length", 0))
        if ln <= 0:
            continue
        end = off + ln
        ph = _placeholder_for_category(str(e.get("category", "")))

        for wid, (w0, w1) in word_spans.items():
            if _overlaps(w0, w1, off, end):
                word_id_to_new[wid] = ph

    return word_id_to_new


def apply_word_replacements(
    *,
    textract: Dict[str, Any],
    word_id_to_new: Dict[str, str],
    altered_flag_key: str = "deid_altered",
) -> Dict[str, Any]:
    out = deepcopy(textract)
    blocks = out.get("Blocks", []) or []
    id2: Dict[str, Dict[str, Any]] = {b["Id"]: b for b in blocks if isinstance(b.get("Id"), str)}

    for b in blocks:
        if b.get("BlockType") != "WORD":
            continue
        wid = b.get("Id")
        orig = (b.get("Text") or "").strip()
        new = word_id_to_new.get(wid, orig)
        b["Text"] = new
        b[altered_flag_key] = (new != orig)

    for b in blocks:
        if b.get("BlockType") != "LINE":
            continue
        words: List[str] = []
        for wid in _child_ids(b):
            wb = id2.get(wid)
            if wb and wb.get("BlockType") == "WORD":
                t = (wb.get("Text") or "").strip()
                if t:
                    words.append(t)
        if words:
            b["Text"] = " ".join(words)

    return out

def setup_paths():
    root = Path(__file__).resolve().parents[1]
    textract_json_path = root / "docfusion_lite" / "data" / "textracttest" / "out_detect_hw.json"
    out_dir = Path(__file__).resolve().parent / "azuredeid"
    out_dir.mkdir(parents=True, exist_ok=True)
    return root, textract_json_path, out_dir

def main() -> Dict[str, Any]:
    endpoint = (
        os.environ.get("AZURE_HEALTH_DEIDENTIFICATION_ENDPOINT")
        or os.environ.get("HEALTHDATAAISERVICES_DEID_SERVICE_ENDPOINT")
        or "https://ama0aehpg9a2bfg8.api.eus001.deid.azure.com"
    )

    root = Path(__file__).resolve().parents[1]
    textract_json_path = root / "docfusion_lite" / "data" / "textracttest" / "out_detect_hw.json"
    print (root)
    out_dir = Path(__file__).resolve().parent / "azuredeid"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(out_dir)
    textract = json.loads(textract_json_path.read_text(encoding="utf-8"))

    pages = build_clean_text_and_spans_by_page(textract)
    (out_dir / "azure_tag_inputs.json").write_text(json.dumps(pages, indent=2), encoding="utf-8")

    outputs_path = out_dir / "azure_tag_outputs.json"
    if outputs_path.exists():
        azure_outputs = json.loads(outputs_path.read_text(encoding="utf-8"))
    else:
        client = DeidentificationClient(endpoint, DefaultAzureCredential())
        azure_outputs = []
        for p in pages:
            entities = tag_entities(client=client, text=p["text"])
            azure_outputs.append({"page": int(p["page"]), "entities": entities})
        outputs_path.write_text(json.dumps(azure_outputs, indent=2), encoding="utf-8")

    page_map = {int(p["page"]): p for p in pages}

    all_word_replacements: Dict[str, str] = {}
    for p in azure_outputs:
        page_num = int(p.get("page", 1))
        entities = p.get("entities", []) or []
        word_spans = (page_map.get(page_num) or {}).get("word_spans", {}) or {}
        all_word_replacements.update(
            compute_word_replacements_from_entities(word_spans=word_spans, entities=entities)
        )

    deid_textract = apply_word_replacements(
        textract=textract,
        word_id_to_new=all_word_replacements,
        altered_flag_key="deid_altered",
    )

    out_textract_path = out_dir / f"{textract_json_path.stem}_deid.json"
    out_textract_path.write_text(json.dumps(deid_textract, indent=2), encoding="utf-8")

    print(f"patched WORDs: {len(all_word_replacements)}")
    print(f"wrote: {out_textract_path}")

    return deid_textract


if __name__ == "__main__":
    main()

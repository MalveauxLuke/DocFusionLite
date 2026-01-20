# DocFusion-Lite

*A custom multimodal document understanding prototype for scanned forms, built primarily as a personal sandbox to implement multimodal attention from scratch and study what actually works.*

DocFusion-Lite is a **Document AI** architecture that fuses:

- **OCR text** via a **pluggable pretrained transformer text encoder**
- **Document images** via a **patch-based vision transformer encoder** (any ViT/DiT-style patch encoder works)
- **Layout information** via **normalized + bucketized bounding-box embeddings**

These streams are combined through a **multi-layer multimodal fusion transformer stack** (text self-attention + text-to-vision cross-attention). The design is **inspired by LayoutLMv3-class models**, but the implementation is **entirely original** and built using **MIT/Apache-friendly components** so it can be used and extended in commercial contexts.

> **Status:** Model + data pipeline are implemented and validated with synthetic forward passes and sanity training runs. This is an engineering/research prototype and is **not production-ready**.

---

## Architecture at a Glance

<p align="center">
  <img src="docs/diagrams/model-architecture.png" width="720" alt="DocFusion-Lite Architecture Diagram"/>
</p>

> *Separate text, layout, and vision backbones feed a fusion stack where tokens can attend to relevant image patches for downstream classification/extraction tasks.*

---

## Why I Built This (Personal R&D Focus)

This repo exists because I wanted to learn the mechanics of modern multimodal document models at the implementation level, not just use them.

My goals were to:

- Implement **cross-attention** and multimodal fusion patterns cleanly in PyTorch
- Build the "data plumbing"  (OCR tokens, bboxes, patch grids, masks) end-to-end
- Create a modular codebase where I can run controlled experiments:
  - freeze vs unfreeze backbones
  - inject vision once vs repeatedly
  - vary fusion depth and attention patterns
  - compare pooling and alignment objectives

Recruiter-facing summary: this is a hands-on multimodal systems project where I built a full document-understanding stack (model + batching + masks + sanity training loop) specifically to explore and validate design choices.

---

## Text Backbone Note: Why I Use LiLT (RoBERTa-Compatible)

In addition to a standard pretrained text encoder, DocFusion-Lite supports **LiLT (Language-Independent Layout Transformer)** as the text backbone.

I added LiLT because I wanted*layout embeddings to enter at the beginning of the language stack, not after fusion.

- When layout is injected late (post-fusion), it can destabilize the representation and "mess up" the token features the rest of the stack expects.
- With LiLT, layout is treated as a first-class input signal early, which makes downstream fusion behavior more predictable.

This also keeps things modular: the text backbone is interchangeable as long as it exposes a standard transformer-style interface.

---

## Motivation (Practical Domain + Constraints)

Scanned documents (especially medical forms) are noisy and visually dense: OCR errors, stamps, checkboxes, tables, tiny fields, and inconsistent layouts. Pure text models miss important signal, but full-scale document foundation models can be expensive to train and awkward to use in commercial pipelines due to licensing constraints.

DocFusion-Lite is targeted at the gap where you want:

- **True multimodality** (text + layout + pixels)
- **Iteration speed** on modest hardware
- **Modularity** to test ideas quickly
- **Commercially safe components** (MIT/Apache-friendly building blocks)

---

## Architecture Overview

DocFusion-Lite follows a simple pattern:

1) **Text stream (OCR tokens + layout)**
- A pretrained transformer encoder produces contextual token embeddings
- Tokens are augmented with **layout embeddings** so the model knows *where* text lives on the page
- Optional: use a layout-aware text backbone (LiLT) to inject layout early

2) **Vision stream (page patches)**
- A patch-based vision transformer encoder embeds image patches
- Captures non-text cues: tables, lines, checkboxes, stamps, spacing, structure
- The vision backbone is intentionally **interchangeable** (any ViT/DiT-style patch encoder works)
- Optional gating to reduce noisy outputs 

3) **Fusion stack (multimodal transformer)**
- Text tokens run **self-attention** over the OCR sequence
- Text tokens run **cross-attention** over image patches to pull in visual context
- Additional encoder capacity can be stacked **on top of fusion** to refine multimodal features before the task head

---

## Current Concerns + Planned Experiments (Honest Engineering Notes)

A key concern I am actively working on:

- **If vision is only injected late, it can be hard for the model to add meaningful visual signal.**
  - If the fusion mechanism is shallow or late, the language pathway can dominate and the vision stream can become weak or noisy.

Possible Fixes:

- **Pretraining matters.** I expect strong alignment objectives to be important so tokens learn to retrieve the right visual context.
  - One planned direction is a **WPA-style objective** (word/patch alignment) so tokens are explicitly trained to associate with relevant patch regions.
- **More encoder capacity above fusion.**
  - Adding **additional encoder layers on top of fusion** so multimodal features can be refined before the task head.
- **Controlled ablations.**
  - Compare inject-once vs multi-layer cross-attention, frozen vs unfrozen vision, fusion depth, and pooling/alignment strategies.


---

## FUNSD Overfit Sanity Check (Result Included)

To validate that the architecture, data pipeline, and optimization loop are wired correctly, this repository includes an overfit test on the FUNSD dataset.

The goal is not benchmarking. It is a standard deep learning sanity check: verifying the model can learn and memorize a small dataset under constrained training settings.

FUNSD overfit was run with `mode=3.1`, **1 fusion layer + 1 post-fusion encoder layer**, and **both backbones frozen** . Training used **AdamW (lr=1e-5, weight_decay=0.01)**, `dropout=0.1`, `grad_clip=1.0`, for **20 epochs**.

**Current run result:**
- **FUNSD F1 (Test dataset): 0.7679**

Notes on interpreting this:
- This is **not** state of the art, and it is not intended to be.
- The multimodal fusion path (cross-attention + layers on top) starts completely randomized, and the run is focused on confirming the end-to-end system can train and overfit as expected.

---

## Vertical Slice Demo (Synthetic End-to-End)

See `demos/verticalslicedemo.ipynb` for a full **vertical slice** of the pipeline using **synthetic inputs** to show how this system could plug into a healthcare workflow (e.g., FMLA-style forms).

It walks through the complete flow:

**OCR output → JSON config (swap labels / rules) → dataset + batching → model inference (untrained) → structured output → visualization overlays**

---

## What’s Here Right Now

- Full **PyTorch model implementation**:
  - pluggable text encoder wrapper (supports early layout injection paths)
  - patch-based vision encoder wrapper
  - layout / bounding-box embedding module
  - multi-layer multimodal fusion transformer
  - optional encoder layers stacked on top of fusion

- Supporting **infrastructure**:
  - data pipeline for OCR tokens, bounding boxes, and images
  - collation logic for batching, padding, and masks
  - synthetic forward-pass demo scripts
  - basic shape and wiring tests to validate end-to-end integration

Large-scale pretraining, fine-tuning on real labeled medical documents, and benchmark metrics are planned once domain data is available.

---

## Quickstart (Dev / Demo)

```bash
git clone https://github.com/<your-username>/DocFusion-Lite.git
cd DocFusion-Lite

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run an overfit test on FUNSD:

```bash
python -m scripts.funsdoverfit
```

---

## Repo Layout (High-Level)

```text
DocFusion-Lite/
├─ README.md
├─ docs/                # diagrams, architecture notes
├─ configs/             # training / debug configs
├─ scripts/             # funsdoverfit.py
├─ src/
│  ├─ data/             # dataset + collation
│  ├─ models/           # encoders, fusion, pooling, heads
│  └─ utils/            # geometry and helpers
└─ tests/               # shape and pipeline tests
```

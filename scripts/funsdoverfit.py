import os
import random
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from docfusion_lite.data.funsd import funsd_to_samples, build_label_maps
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


def token_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    b, t, c = logits.shape
    return nn.functional.cross_entropy(
        logits.reshape(b * t, c),
        labels.reshape(b * t),
        ignore_index=ignore_index,
    )


@torch.no_grad()
def quick_eval_loss(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    ignore_index: int = -100,
    max_batches: int = 20,
) -> float:
    model.eval()

    total = 0.0
    denom = 0

    for i, batch in enumerate(dl):
        if i >= max_batches:
            break

        batch = move_batch_to_device(batch, device)

        labels = getattr(batch, "labels", None)
        if labels is None:
            continue

        logits = model(batch)

        active = (labels != ignore_index)
        n_active = int(active.sum().item())
        if n_active == 0:
            continue

        loss = token_ce_loss(logits, labels, ignore_index=ignore_index)
        total += float(loss.item()) * n_active
        denom += n_active

    return total / max(denom, 1)


# -------------------------
# Debug harness
# -------------------------

def _safe_len(x) -> Optional[int]:
    try:
        return len(x)
    except Exception:
        return None


def _find_list_attr(obj: Any, candidate_paths: Iterable[str]) -> Optional[Tuple[str, Any]]:
    for path in candidate_paths:
        cur = obj
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if not ok:
            continue
        if isinstance(cur, (list, tuple, nn.ModuleList)) or _safe_len(cur) is not None:
            return path, cur
    return None


def print_module_summary(model: nn.Module):
    print("\n[dbg] model:", type(model).__name__)

    if hasattr(model, "text_encoder"):
        print("[dbg] text_encoder:", type(model.text_encoder).__name__)
    else:
        print("[dbg] text_encoder: MISSING")

    if hasattr(model, "vision_encoder"):
        print("[dbg] vision_encoder:", type(model.vision_encoder).__name__)
    else:
        print("[dbg] vision_encoder: MISSING")

    print("[dbg] interesting submodules (fusion/post/encoder/stack):")
    shown = 0
    for name, mod in model.named_modules():
        n = name.lower()
        if name and any(k in n for k in ("post", "fusion", "encoder", "stack")):
            print(" ", name, "->", type(mod).__name__)
            shown += 1
            if shown >= 50:
                print("  ...")
                break


def count_backbone_layers(model: nn.Module):
    print("\n[dbg] layer counts")

    tb = None
    vb = None
    if hasattr(model, "text_encoder"):
        tb = getattr(model.text_encoder, "model", None) or model.text_encoder
    if hasattr(model, "vision_encoder"):
        vb = getattr(model.vision_encoder, "model", None) or model.vision_encoder

    if tb is not None:
        hit = _find_list_attr(tb, ["encoder.layer", "deberta.encoder.layer", "model.encoder.layer", "layers", "layer"])
        if hit is not None:
            path, layers = hit
            print("  text backbone:", path, "len=", len(layers))
        else:
            print("  text backbone: could not find layer list")

    if vb is not None:
        hit = _find_list_attr(vb, ["blocks", "encoder.blocks", "model.blocks", "layers"])
        if hit is not None:
            path, blocks = hit
            print("  vision backbone:", path, "len=", len(blocks))
        else:
            print("  vision backbone: could not find blocks list")

    hit = _find_list_attr(
        model,
        [
            "post_fusion_encoder.layers",
            "post_fusion_encoder.layer",
            "post_fusion_encoder",
            "post_encoder.layers",
            "post_encoder.layer",
            "post_encoder",
            "fusion_encoder.layers",
            "fusion_encoder.layer",
            "fusion_encoder",
        ],
    )
    if hit is not None:
        path, obj = hit
        n = _safe_len(obj)
        print("  post-fusion:", path, "len=", n)
    else:
        print("  post-fusion: could not locate (use print_module_summary output)")


def print_param_freeze_status(model: nn.Module, top_k: int = 30):
    total = 0
    trainable = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print("\n[dbg] params total:", f"{total:,}", "trainable:", f"{trainable:,}")

    print("[dbg] first trainable params:")
    shown = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(" ", name, tuple(p.shape))
            shown += 1
            if shown >= top_k:
                print("  ...")
                break

    if hasattr(model, "text_encoder") and hasattr(model.text_encoder, "model"):
        req = [p.requires_grad for p in model.text_encoder.model.parameters()]
        print("[dbg] text backbone frozen:", (sum(req) == 0))
    if hasattr(model, "vision_encoder") and hasattr(model.vision_encoder, "model"):
        req = [p.requires_grad for p in model.vision_encoder.model.parameters()]
        print("[dbg] vision backbone frozen:", (sum(req) == 0))


def inspect_batch(batch, num_labels: int, ignore_index: int = -100):
    print("\n[dbg] batch check")

    keys = list(vars(batch).keys())
    print("  keys:", keys)

    attn = getattr(batch, "attention_mask", None)
    labels = getattr(batch, "labels", None)

    if torch.is_tensor(attn):
        pad_frac = float((attn == 0).float().mean().item())
        print("  pad(attn==0):", f"{pad_frac:.4f}")

    if torch.is_tensor(labels):
        ignore_frac = float((labels == ignore_index).float().mean().item())
        active = int((labels != ignore_index).sum().item())
        print("  ignore(labels==-100):", f"{ignore_frac:.4f}", "active_tokens:", active)

        if active > 0:
            mn = int(labels[labels != ignore_index].min().item())
            mx = int(labels[labels != ignore_index].max().item())
            print("  label range:", mn, "to", mx, "(num_labels:", num_labels, ")")

            if mx >= num_labels or mn < 0:
                print("  WARNING: labels out of range for num_labels")


def _format_out(x):
    if torch.is_tensor(x):
        return f"{tuple(x.shape)}"
    if isinstance(x, (list, tuple)):
        shapes = []
        for y in x[:4]:
            shapes.append(_format_out(y))
        if len(x) > 4:
            shapes.append("...")
        return f"{type(x).__name__}({', '.join(shapes)})"
    if isinstance(x, dict):
        return f"dict(keys={list(x.keys())[:6]})"
    return type(x).__name__


def _make_hook(name: str):
    def hook_fn(module, inp, out):
        print(f"[hook] {name}: {_format_out(out)}")
    return hook_fn


@torch.no_grad()
def run_one_forward_with_hooks(model: nn.Module, batch, device: torch.device):
    print("\n[dbg] hook forward (one batch)")

    handles = []

    text_mod = getattr(model, "text_encoder", None)
    vision_mod = getattr(model, "vision_encoder", None)

    if text_mod is not None:
        handles.append(text_mod.register_forward_hook(_make_hook("text_encoder")))
    if vision_mod is not None:
        handles.append(vision_mod.register_forward_hook(_make_hook("vision_encoder")))

    post_mod = None
    for cand in ("post_fusion_encoder", "post_encoder", "fusion_encoder"):
        if hasattr(model, cand):
            post_mod = getattr(model, cand)
            break

    if post_mod is not None:
        handles.append(post_mod.register_forward_hook(_make_hook(cand)))

    batch = move_batch_to_device(batch, device)
    _ = model(batch)

    for h in handles:
        h.remove()


def debug_model(model: nn.Module, batch, num_labels: int, device: torch.device, ignore_index: int = -100):
    print("\n====================")
    print("[dbg] debug_model start")
    print_module_summary(model)
    count_backbone_layers(model)
    print_param_freeze_status(model)
    inspect_batch(batch, num_labels=num_labels, ignore_index=ignore_index)
    run_one_forward_with_hooks(model, batch, device=device)
    print("[dbg] debug_model end")
    print("====================\n")


# -------------------------
# Training
# -------------------------

def main():
    seed_all(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    label2id, id2label = build_label_maps()
    print("num_labels:", len(label2id))

    funsd_root = r"C:\Users\luke6\Desktop\DocFusionLite\docfusion_lite\data\dataset"
    print("loading FUNSD:", funsd_root)

    train_samples, train_img_root = funsd_to_samples(funsd_root, "train")
    test_samples, test_img_root = funsd_to_samples(funsd_root, "test")
    print("train_samples:", len(train_samples), "test_samples:", len(test_samples))

    train_ds = DocumentDataset(
        samples=train_samples,
        label2id=label2id,
        image_root=train_img_root,
        image_size=224,
    )

    test_ds = DocumentDataset(
        samples=test_samples,
        label2id=label2id,
        image_root=test_img_root,
        image_size=224,
    )

    collate_fn = make_collate_fn(pad_token_id=train_ds.tokenizer.pad_token_id)
    print("pad_token_id:", train_ds.tokenizer.pad_token_id)

    overfit_n = 100
    idxs = list(range(len(train_ds)))
    random.shuffle(idxs)
    overfit_idxs = idxs[: min(overfit_n, len(idxs))]
    overfit_ds = Subset(train_ds, overfit_idxs)
    print("overfit subset:", len(overfit_ds))

    train_dl = DataLoader(
        overfit_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Overfit eval should be on the same subset (Phase 1 wiring check)
    overfit_eval_dl = DataLoader(
        overfit_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

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

    print("model initialized")

    # Phase 1: freeze backbones for stability
    print("model initialized")

# Phase 1: freeze only the pretrained backbones (clean + explicit)
    model.text_encoder.freeze_all()
    model.vision_encoder.freeze_all()
    print("froze text+vision backbones")


    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable_params))

    opt = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.0)
    print("optimizer: AdamW")

    ignore_index = -100
    grad_clip = 1.0
    epochs = 40
    log_every = 20

    # Where checkpoints go: project-root/checkpoints
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"
    print("ckpt_path:", str(ckpt_path))

    model.train()
    global_step = 0
    running = 0.0

    print("starting training")

    debug_once = True

    for epoch in range(1, epochs + 1):
        print("\nepoch", epoch)

        for batch in train_dl:
            global_step += 1

            batch = move_batch_to_device(batch, device)

            if debug_once:
                debug_model(model, batch, num_labels=len(label2id), device=device, ignore_index=ignore_index)
                debug_once = False

            labels = getattr(batch, "labels", None)
            if labels is None:
                raise RuntimeError("batch has no .labels; collate_fn must return DocBatch(labels=...) padded with -100")

            opt.zero_grad(set_to_none=True)

            logits = model(batch)
            loss = token_ce_loss(logits, labels, ignore_index=ignore_index)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            running += float(loss.item())

            if global_step % log_every == 0:
                avg = running / log_every
                running = 0.0
                print(" step", global_step, "train_loss", f"{avg:.4f}")

        # Phase 1: track overfit-subset loss (main wiring signal)
        ov = quick_eval_loss(model, overfit_eval_dl, device, ignore_index=ignore_index, max_batches=1000)
        print(" overfit_eval_loss", f"{ov:.4f}")

        # Optional: also watch test loss, but don't over-interpret it in Phase 1
        #tv = quick_eval_loss(model, test_dl, device, ignore_index=ignore_index, max_batches=20)
        #print(" test_eval_loss", f"{tv:.4f}")

        # Save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
            },
            str(ckpt_path),
        )
        print(" saved:", ckpt_path.name)

        model.train()

    print("\ndone")


if __name__ == "__main__":
    main()

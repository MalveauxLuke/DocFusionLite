import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from docfusion_lite.data.funsd import funsd_to_samples, build_label_maps
from docfusion_lite.data.dataset import DocumentDataset, make_collate_fn
from docfusion_lite.models.docfusion_model import DocFusionModel

from seqeval.metrics import f1_score, precision_score, recall_score



RESUME_FROM_CKPT = False          # resume from ckpt_last.pt if it exists
SAVE_LAST = True                # keep ckpt_last.pt for resume/debug
BEST_ON = "f1"                  # "f1" (maximize) or "loss" (minimize)
EARLY_STOP_PATIENCE = 10        # epochs with no improvement
MAX_EVAL_BATCHES = None           # set None to eval full test set

UNFREEZE_LAST_K_TEXT_LAYERS = 0 # how many text layers to unfreeze after loading weights

# Eval (seqeval span/entity F1)

@torch.no_grad()
def eval_seqeval_f1(model, dl, device, id2label, ignore_index=-100, max_batches=20):
    model.eval()
    all_true, all_pred = [], []

    for i, batch in enumerate(dl):
        if max_batches is not None and i >= max_batches:
            break

        batch = move_batch_to_device(batch, device)
        labels = batch.labels
        logits = model(batch)
        preds = logits.argmax(dim=-1)

        for y, p in zip(labels, preds):
            true_tags, pred_tags = [], []
            for yt, pt in zip(y.tolist(), p.tolist()):
                if yt == ignore_index:
                    continue
                true_tags.append(id2label[int(yt)])
                pred_tags.append(id2label[int(pt)])

            all_true.append(true_tags)
            all_pred.append(pred_tags)

    return {
        "precision": float(precision_score(all_true, all_pred)),
        "recall": float(recall_score(all_true, all_pred)),
        "f1": float(f1_score(all_true, all_pred)),
    }


# Utilities
def seed_all(seed: int = 7):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
        if max_batches is not None and i >= max_batches:
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


# Checkpoint IO
def load_checkpoint(model: nn.Module, opt: Optional[torch.optim.Optimizer], path: Path, device: torch.device):
    ckpt = torch.load(str(path), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    if opt is not None and "opt" in ckpt and ckpt["opt"] is not None:
        opt.load_state_dict(ckpt["opt"])

    epoch = int(ckpt.get("epoch", 0))
    global_step = int(ckpt.get("global_step", 0))
    best_metric = ckpt.get("best_metric", None)
    bad_epochs = int(ckpt.get("bad_epochs", 0))
    return epoch, global_step, best_metric, bad_epochs


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    opt: Optional[torch.optim.Optimizer],
    epoch: int,
    global_step: int,
    label2id,
    id2label,
    best_metric,
    bad_epochs: int,
):
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "opt": (opt.state_dict() if opt is not None else None),
            "label2id": label2id,
            "id2label": id2label,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
        },
        str(path),
    )


def is_better(metric_value: float, best_value: Optional[float], best_on: str) -> bool:
    if best_value is None:
        return True
    if best_on == "f1":
        return metric_value > best_value
    if best_on == "loss":
        return metric_value < best_value
    raise ValueError("BEST_ON must be 'f1' or 'loss'")


# Training
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

    overfit_n = 50
    idxs = list(range(len(train_ds)))
    random.shuffle(idxs)
    overfit_idxs = idxs[: min(overfit_n, len(idxs))]
    overfit_ds = Subset(train_ds, overfit_idxs)
    print("overfit subset:", len(overfit_ds))

    train_dl = DataLoader(
        train_ds,
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
    overfit_eval_dl = DataLoader(
        overfit_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_last_path = ckpt_dir / "ckpt_last.pt"
    ckpt_best_path = ckpt_dir / "ckpt_best.pt"

    print("ckpt_last:", str(ckpt_last_path))
    print("ckpt_best:", str(ckpt_best_path))

    # model create
    model = DocFusionModel(
        mode="3.1",
        num_fusion_layers=1,
        num_encoder_layers=1,
        num_labels=len(label2id),
        dropout=0.1,
        use_gate_stem=False,
        use_gate_fusion_layer=False,
        use_region_ffn=True,
        text_model_name="SCUT-DLVCLab/lilt-roberta-en-base",
        vision_model_name="microsoft/dit-base",
    ).to(device)
    print("model initialized")

    # resume (LOAD WEIGHTS FIRST) 
    start_epoch = 1
    global_step = 0
    best_metric = None
    bad_epochs = 0

    if RESUME_FROM_CKPT and ckpt_last_path.exists():
        last_epoch, global_step, best_metric, bad_epochs = load_checkpoint(model, None, ckpt_best_path, device=device)
        start_epoch = last_epoch + 1
        print(f"loaded checkpoint weights: {ckpt_best_path} (resume at epoch {start_epoch}, global_step {global_step})")
        if best_metric is not None:
            print(f"resume best_metric={best_metric} bad_epochs={bad_epochs} BEST_ON={BEST_ON}")
    else:
        print("no checkpoint loaded (fresh run)")

    #freeze/unfreeze 
    model.text_encoder.freeze_all()
    model.vision_encoder.freeze_all()
    model.text_encoder.unfreeze_last_k_layers(UNFREEZE_LAST_K_TEXT_LAYERS)

    # optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("trainable params:", sum(p.numel() for p in trainable_params))

    opt = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)
    print("optimizer: AdamW")

    ignore_index = -100
    grad_clip = 1.0
    epochs = 40
    log_every = 20

    model.train()
    running = 0.0
    print("starting training")

    for epoch in range(start_epoch, epochs + 1):
        print("\nepoch", epoch)

        for batch in train_dl:
            global_step += 1
            batch = move_batch_to_device(batch, device)

            labels = getattr(batch, "labels", None)
            if labels is None:
                raise RuntimeError("batch has no .labels")

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

        tv = quick_eval_loss(model, test_dl, device, ignore_index=ignore_index, max_batches=MAX_EVAL_BATCHES)
        metrics = eval_seqeval_f1(model, test_dl, device, id2label, ignore_index=ignore_index, max_batches=MAX_EVAL_BATCHES)

        print(" test_eval_loss", f"{tv:.4f}")
        print(
            " test_f1", f"{metrics['f1']:.4f}",
            "| prec", f"{metrics['precision']:.4f}",
            "| rec", f"{metrics['recall']:.4f}",
        )

        # choose metric for "best"
        cur_metric = float(metrics["f1"]) if BEST_ON == "f1" else float(tv)

        improved = is_better(cur_metric, best_metric, BEST_ON)
        if improved:
            best_metric = cur_metric
            bad_epochs = 0

            save_checkpoint(
                ckpt_best_path,
                model=model,
                opt=opt,
                epoch=epoch,
                global_step=global_step,
                label2id=label2id,
                id2label=id2label,
                best_metric=best_metric,
                bad_epochs=bad_epochs,
            )
            print(f" saved: ckpt_best.pt (BEST_ON={BEST_ON}, best_metric={best_metric:.6f})")
        else:
            bad_epochs += 1
            print(f" no improvement (BEST_ON={BEST_ON}); bad_epochs={bad_epochs}/{EARLY_STOP_PATIENCE}")

        # always save last (optional)
        if SAVE_LAST:
            save_checkpoint(
                ckpt_last_path,
                model=model,
                opt=opt,
                epoch=epoch,
                global_step=global_step,
                label2id=label2id,
                id2label=id2label,
                best_metric=best_metric,
                bad_epochs=bad_epochs,
            )
            print(" saved: ckpt_last.pt")

        # early stop
        if EARLY_STOP_PATIENCE is not None and bad_epochs >= EARLY_STOP_PATIENCE:
            print(f"early stopping: no improvement for {bad_epochs} epochs (BEST_ON={BEST_ON})")
            break

        model.train()

    print("\ndone")


if __name__ == "__main__":
    main()
#python -m scripts.funsdoverfit
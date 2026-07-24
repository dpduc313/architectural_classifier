"""
train.py — Unified fine-tuning script for all 4 models.

Usage:
    .venv\\Scripts\\python.exe execution/training/train.py --model swinv2
    .venv\\Scripts\\python.exe execution/training/train.py --model vit
    .venv\\Scripts\\python.exe execution/training/train.py --model dinov2
    .venv\\Scripts\\python.exe execution/training/train.py --model resnet50

All checkpoints saved to: .tmp/checkpoints/{model_name}_best.pt
All training logs saved to: .tmp/logs/{model_name}_train_log.csv
"""

import argparse
import os
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# ── Local imports ──────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import HeritageDataset, get_class_weights, MODEL_INPUT_SIZES, MODEL_NORM_STATS, CLASSES

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).parent.parent.parent
PROCESSED_DIR  = PROJECT_ROOT / "processed_data"
CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
LOG_DIR        = PROJECT_ROOT / ".tmp" / "logs"

# ── Model registry (timm IDs) ──────────────────────────────────────────────────
MODEL_REGISTRY = {
    "swinv2":   "swinv2_tiny_window8_256.ms_in1k",
    "vit":      "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "dinov2":   "vit_small_patch14_dinov2.lvd142m",
    "resnet50": "resnet50.a1_in1k",
    "effnet":   "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext": "convnext_tiny.fb_in22k_ft_in1k",
}

# ── Hyperparameters ────────────────────────────────────────────────────────────
PHASE1_EPOCHS = 2     # backbone frozen, head only
PHASE2_EPOCHS = 3     # full fine-tune
HEAD_LR       = 1e-3
BACKBONE_LR   = 1e-4
WEIGHT_DECAY  = 0.05
BATCH_SIZE_GPU = 32
BATCH_SIZE_CPU = 8
NUM_WORKERS    = 0    # set 0 on Windows to avoid DataLoader fork issues
NUM_CLASSES    = len(CLASSES)


# ──────────────────────────────────────────────────────────────────────────────
def build_model(model_key: str) -> nn.Module:
    """Load a pretrained timm model and replace its head for NUM_CLASSES."""
    timm_id = MODEL_REGISTRY[model_key]
    print(f"Loading '{timm_id}' from timm (pretrained=True)...")
    # DINOv2 native size is 518; override to 224 for speed on CPU
    extra = {"img_size": 224} if model_key == "dinov2" else {}
    model = timm.create_model(timm_id, pretrained=True, num_classes=NUM_CLASSES, **extra)
    return model


def freeze_backbone(model: nn.Module, model_key: str):
    """Freeze all layers except the classification head."""
    # timm convention: classification head is model.head or model.fc
    head_names = {"head", "fc", "classifier", "head.fc"}
    for name, param in model.named_parameters():
        top = name.split(".")[0]
        param.requires_grad = top in head_names
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase 1 — backbone frozen. Trainable params: {trainable:,}")


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase 2 — full fine-tune. Total trainable params: {total:,}")


def get_optimizer(model: nn.Module, model_key: str, phase: int) -> torch.optim.Optimizer:
    """Return optimizer with separate LR groups for head vs backbone (phase 2)."""
    head_names = {"head", "fc", "classifier"}

    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    else:
        head_params, backbone_params = [], []
        for name, param in model.named_parameters():
            if name.split(".")[0] in head_names:
                head_params.append(param)
            else:
                backbone_params.append(param)
        return torch.optim.AdamW([
            {"params": head_params,     "lr": HEAD_LR},
            {"params": backbone_params, "lr": BACKBONE_LR},
        ], weight_decay=WEIGHT_DECAY)


# ──────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, scaler, is_train: bool):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels, _ in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(images)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(images)
                    loss = criterion(logits, labels)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ──────────────────────────────────────────────────────────────────────────────
def train(model_key: str, max_samples_per_class: int = None):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Model : {model_key}  |  Device: {device}")
    if device.type == "cpu" and max_samples_per_class is None:
        max_samples_per_class = 1000
        print(f"Running on CPU: defaulting max_samples_per_class to {max_samples_per_class} per class for speed.")
    print(f"{'='*60}\n")

    input_size  = MODEL_INPUT_SIZES[model_key]
    norm_mean, norm_std = MODEL_NORM_STATS[model_key]
    batch_size  = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU
    scaler      = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Datasets ──
    train_ds = HeritageDataset(PROCESSED_DIR, "train", input_size, norm_mean, norm_std,
                               max_samples_per_class=max_samples_per_class)
    val_ds   = HeritageDataset(PROCESSED_DIR, "val",   input_size, norm_mean, norm_std,
                               max_samples_per_class=max_samples_per_class)

    class_weights = get_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))

    # ── Model ──
    model = build_model(model_key).to(device)

    checkpoint_path = CHECKPOINT_DIR / f"{model_key}_best.pt"
    log_path        = LOG_DIR / f"{model_key}_train_log.csv"

    best_val_acc = 0.0
    log_rows: list[dict] = []

    # ─────────────────────────────────────────────────
    # PHASE 1: Frozen backbone — train head only
    # ─────────────────────────────────────────────────
    print(f"\n--- Phase 1: Head-only training ({PHASE1_EPOCHS} epochs) ---")
    freeze_backbone(model, model_key)
    optimizer = get_optimizer(model, model_key, phase=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE1_EPOCHS)

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler, is_train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                     device, scaler, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"[P1 E{epoch:02d}/{PHASE1_EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | {elapsed:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"epoch": epoch, "phase": 1,
                        "model_state": model.state_dict(),
                        "val_acc": va_acc}, checkpoint_path)
            print(f"  [BEST] New best val_acc={va_acc:.4f} - checkpoint saved.")

        log_rows.append({"phase": 1, "epoch": epoch, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})

    # ─────────────────────────────────────────────────
    # PHASE 2: Full fine-tuning
    # ─────────────────────────────────────────────────
    print(f"\n--- Phase 2: Full fine-tune ({PHASE2_EPOCHS} epochs) ---")
    unfreeze_all(model)
    optimizer = get_optimizer(model, model_key, phase=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE2_EPOCHS)

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer,
                                     device, scaler, is_train=True)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer,
                                     device, scaler, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"[P2 E{epoch:02d}/{PHASE2_EPOCHS}] "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | {elapsed:.1f}s")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"epoch": epoch, "phase": 2,
                        "model_state": model.state_dict(),
                        "val_acc": va_acc}, checkpoint_path)
            print(f"  [BEST] New best val_acc={va_acc:.4f} - checkpoint saved.")

        log_rows.append({"phase": 2, "epoch": epoch, "train_loss": tr_loss,
                         "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})

    # ─────────────────────────────────────────────────
    # Save log
    # ─────────────────────────────────────────────────
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print(f"\nTraining complete. Best val_acc: {best_val_acc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Log:        {log_path}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a heritage classification model.")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        required=True,
        help="Which model to train: swinv2 | vit | dinov2 | resnet50",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Maximum patches per class for training/val (useful for faster CPU execution).",
    )
    args = parser.parse_args()
    train(args.model, max_samples_per_class=args.max_samples_per_class)

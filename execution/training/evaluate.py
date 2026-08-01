"""
evaluate.py — Load best checkpoints and evaluate all models on the test set.

Usage:
    .venv\\Scripts\\python.exe execution/training/evaluate.py

Outputs (saved to .tmp/results/):
    {model_name}_metrics.json      — accuracy, precision, recall, F1 per class + weighted/macro
    {model_name}_confusion.csv     — confusion matrix
    {model_name}_misclassified.csv — top misclassified examples per class
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import HeritageDataset, MODEL_INPUT_SIZES, MODEL_NORM_STATS, CLASSES
from train import MODEL_REGISTRY, CHECKPOINT_DIR, NUM_WORKERS, BATCH_SIZE_CPU, BATCH_SIZE_GPU

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
RESULTS_DIR   = PROJECT_ROOT / ".tmp" / "results"
NUM_CLASSES   = len(CLASSES)


def load_model(model_key: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    timm_id = MODEL_REGISTRY[model_key]
    extra = {"img_size": 224} if model_key == "dinov2" else {}
    model = timm.create_model(timm_id, pretrained=False, num_classes=NUM_CLASSES, **extra)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded '{model_key}' from epoch {ckpt['epoch']} "
          f"(phase {ckpt['phase']}, val_acc={ckpt['val_acc']:.4f})")
    return model


def evaluate_model(model_key: str, device: torch.device, max_samples_per_class: int = None, data_dir: str = None, checkpoint_dir: str = None, results_dir: str = None):
    # Resolve directory paths
    data_dir_path = Path(data_dir) if data_dir else PROCESSED_DIR
    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    results_dir_path = Path(results_dir) if results_dir else RESULTS_DIR

    checkpoint_path = checkpoint_dir_path / f"{model_key}_best.pt"
    if not checkpoint_path.exists():
        print(f"[SKIP] No checkpoint for '{model_key}'. Run train.py first.")
        return None

    if device.type == "cpu" and max_samples_per_class is None:
        max_samples_per_class = 500
        print(f"Running evaluation on CPU: using max {max_samples_per_class} test patches per class for speed.")

    input_size  = MODEL_INPUT_SIZES[model_key]
    norm_mean, norm_std = MODEL_NORM_STATS[model_key]
    batch_size  = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    test_ds = HeritageDataset(data_dir_path, "test", input_size, norm_mean, norm_std,
                             max_samples_per_class=max_samples_per_class)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS)

    ref_ds = HeritageDataset(data_dir_path, "reference_test", input_size, norm_mean, norm_std,
                            max_samples_per_class=max_samples_per_class)
    ref_loader = DataLoader(ref_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS)

    model = load_model(model_key, checkpoint_path, device)

    def run_eval(loader):
        all_preds, all_labels, all_paths = [], [], []
        all_arch_preds, all_arch_labels = [], []
        all_probs = []

        t0 = time.time()
        with torch.no_grad():
            for images, labels, arch_labels, paths in loader:
                images = images.to(device)
                logits = model(images)
                probs  = torch.softmax(logits, dim=1).cpu()
                preds  = logits.argmax(dim=1).cpu()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_arch_labels.extend(arch_labels.tolist())
                all_paths.extend(paths)
                all_probs.extend(probs.tolist())
        inference_time = time.time() - t0
        return all_preds, all_labels, all_arch_labels, all_paths, all_probs, inference_time

    all_preds, all_labels, all_arch_labels, all_paths, all_probs, inference_time = run_eval(test_loader)
    ms_per_image = (inference_time / max(1, len(test_ds))) * 1000

    # ── Original Image Level Majority Voting ─────────────────────────────────────
    # Group patches by parent original image stem
    image_groups = {}
    for p, lbl, pred, arch_lbl in zip(all_paths, all_labels, all_preds, all_arch_labels):
        # Extract base image name (e.g. A1.30.Pol-Mas_01 from A1.30.Pol-Mas_01_patch_1_2.jpg)
        p_name = Path(p).name
        base_img = p_name.rsplit("_patch_", 1)[0] if "_patch_" in p_name else p_name
        
        if base_img not in image_groups:
            image_groups[base_img] = {"true_label": lbl, "patch_preds": [], "patch_arch": []}
        
        image_groups[base_img]["patch_preds"].append(pred)
        image_groups[base_img]["patch_arch"].append(arch_lbl)

    img_true_labels, img_pred_labels = [], []
    for b_img, data in image_groups.items():
        # Keep only patches labeled/predicted as Architectural (1)
        valid_preds = [p for p, a in zip(data["patch_preds"], data["patch_arch"]) if a == 1]
        if len(valid_preds) == 0:
            # Fallback to all patch predictions if no architectural patch found
            valid_preds = data["patch_preds"]
        
        # Majority vote
        vote = max(set(valid_preds), key=valid_preds.count)
        img_true_labels.append(data["true_label"])
        img_pred_labels.append(vote)

    image_level_acc = accuracy_score(img_true_labels, img_pred_labels) if img_true_labels else 0.0
    image_level_macro_f1 = f1_score(img_true_labels, img_pred_labels, average="macro", zero_division=0) if img_true_labels else 0.0

    subset_results = {}
    for sub_name, indices in subset_indices.items():
        if len(indices) == 0:
            continue
        sub_labels = [all_labels[i] for i in indices]
        sub_preds  = [all_preds[i] for i in indices]

        sub_acc = accuracy_score(sub_labels, sub_preds)
        sub_macro_f1 = f1_score(sub_labels, sub_preds, average="macro", zero_division=0)
        sub_weighted_f1 = f1_score(sub_labels, sub_preds, average="weighted", zero_division=0)

        subset_results[sub_name] = {
            "num_images":  len(indices),
            "accuracy":    round(float(sub_acc), 4),
            "macro_f1":    round(float(sub_macro_f1), 4),
            "weighted_f1": round(float(sub_weighted_f1), 4),
        }

    # ── Overall Metrics ────────────────────────────────────────────────────────
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0
    )
    macro_f1    = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm          = confusion_matrix(all_labels, all_preds)

    primary_std = subset_results.get("Standard (Primary Focus)", {
        "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4), "weighted_f1": round(weighted_f1, 4), "num_images": len(test_ds)
    })

    metrics = {
        "model":            model_key,
        "primary_standard": primary_std,
        "test_subsets":     subset_results,
        "test_accuracy":    round(acc, 4),
        "macro_f1":         round(macro_f1, 4),
        "weighted_f1":      round(weighted_f1, 4),
        "ms_per_image":     round(ms_per_image, 2),
        "num_test_images":  len(test_ds),
        "per_class":        {cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"], 4),
            "f1":        round(report[cls]["f1-score"], 4),
            "support":   report[cls]["support"],
        } for cls in CLASSES},
    }

    results_dir_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_path = results_dir_path / f"{model_key}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix CSV
    cm_df = pd.DataFrame(cm, index=CLASSES, columns=CLASSES)
    cm_path = results_dir_path / f"{model_key}_confusion.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    # Save misclassified examples
    misclassified = [
        {"path": p, "true": CLASSES[t], "pred": CLASSES[pr],
         "confidence": round(max(prb), 4)}
        for p, t, pr, prb in zip(all_paths, all_labels, all_preds, all_probs)
        if t != pr
    ]
    mis_df = pd.DataFrame(misclassified)
    mis_path = results_dir_path / f"{model_key}_misclassified.csv"
    mis_df.to_csv(mis_path, index=False, encoding="utf-8-sig")

    print(f"\n  [PRIMARY FOCUS] Standardized Buildings Acc={primary_std['accuracy']:.4f}  F1={primary_std['macro_f1']:.4f} ({primary_std['num_images']} patches)")
    for sub_k, sub_v in subset_results.items():
        if sub_k != "Standard (Primary Focus)":
            print(f"  [SUBSET] {sub_k:26s} Acc={sub_v['accuracy']:.4f}  F1={sub_v['macro_f1']:.4f} ({sub_v['num_images']} patches)")
    print(f"  [OVERALL] Combined Test Acc={acc:.4f}  Macro-F1={macro_f1:.4f}  Speed={ms_per_image:.1f}ms/img")
    print(f"  Saved -> {metrics_path.name}, {cm_path.name}, {mis_path.name}")

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate heritage classification models.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing processed data")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to load checkpoints from")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory to save evaluation results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results_dir_path = Path(args.results_dir) if args.results_dir else RESULTS_DIR

    all_results = []
    for model_key in MODEL_REGISTRY:
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_key}")
        print(f"{'='*50}")
        result = evaluate_model(
            model_key,
            device,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            results_dir=args.results_dir
        )
        if result:
            all_results.append(result)

    if all_results:
        summary_path = results_dir_path / "all_results_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nAll results saved to {summary_path}")


if __name__ == "__main__":
    main()

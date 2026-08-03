"""
evaluate.py — Load best checkpoints and evaluate all models on standardized & reference test sets.

Usage:
    .venv\\Scripts\\python.exe execution/training/evaluate.py

Outputs (saved to .tmp/results/):
    {model_name}_metrics.json      — accuracy, precision, recall, F1 per class + weighted/macro
    {model_name}_confusion.csv     — confusion matrix
    {model_name}_misclassified.csv — misclassified examples per class
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


def load_model(model_key: str, checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, int]:
    timm_id = MODEL_REGISTRY[model_key]
    extra = {"img_size": 224} if model_key == "dinov2" else {}
    ckpt = torch.load(checkpoint_path, map_location=device)
    actual_input_size = MODEL_INPUT_SIZES[model_key]
    
    try:
        model = timm.create_model(timm_id, pretrained=False, num_classes=NUM_CLASSES, **extra)
        model.load_state_dict(ckpt["model_state"])
    except Exception as e:
        print(f"Warning: Could not load '{timm_id}' ({e}). Attempting fallback architecture...")
        fallback_ids = {
            "swinv2": "swinv2_tiny_window8_256.ms_in1k",
            "dinov2": "vit_small_patch14_dinov2.lvd142m"
        }
        fallback_id = fallback_ids.get(model_key, timm_id)
        if model_key == "swinv2" and fallback_id == "swinv2_tiny_window8_256.ms_in1k":
            actual_input_size = 256
        model = timm.create_model(fallback_id, pretrained=False, num_classes=NUM_CLASSES, **extra)
        model.load_state_dict(ckpt["model_state"])

    model.to(device)
    model.eval()
    print(f"Loaded '{model_key}' from epoch {ckpt.get('epoch', '?')} "
          f"(phase {ckpt.get('phase', '?')}, val_acc={ckpt.get('val_acc', 0.0):.4f})")
    return model, actual_input_size


def evaluate_dataset(model: nn.Module, loader: DataLoader, device: torch.device):
    all_preds, all_labels, all_paths = [], [], []
    all_arch_labels = []
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

    patch_acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    patch_macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else 0.0
    patch_weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) if all_labels else 0.0

    # ── Original Image Level Voting ──────────────────────────────────────────────
    image_groups = {}
    for p, lbl, pred, arch_lbl in zip(all_paths, all_labels, all_preds, all_arch_labels):
        p_name = Path(p).name
        base_img = p_name.rsplit("_patch_", 1)[0] if "_patch_" in p_name else p_name
        
        if base_img not in image_groups:
            image_groups[base_img] = {"true_label": lbl, "patch_preds": [], "patch_arch": []}
        
        image_groups[base_img]["patch_preds"].append(pred)
        image_groups[base_img]["patch_arch"].append(arch_lbl)

    img_true_labels, img_pred_labels = [], []
    for b_img, data in image_groups.items():
        # Keep only architectural patches (sublabel == 1)
        valid_preds = [p for p, a in zip(data["patch_preds"], data["patch_arch"]) if a == 1]
        if len(valid_preds) == 0:
            valid_preds = data["patch_preds"]
        
        vote = max(set(valid_preds), key=valid_preds.count)
        img_true_labels.append(data["true_label"])
        img_pred_labels.append(vote)

    image_level_acc = accuracy_score(img_true_labels, img_pred_labels) if img_true_labels else 0.0
    image_level_macro_f1 = f1_score(img_true_labels, img_pred_labels, average="macro", zero_division=0) if img_true_labels else 0.0

    report = classification_report(
        all_labels, all_preds, target_names=CLASSES, output_dict=True, zero_division=0
    ) if all_labels else {}

    cm = confusion_matrix(all_labels, all_preds) if all_labels else []

    res = {
        "patch_accuracy": round(float(patch_acc), 4),
        "patch_macro_f1": round(float(patch_macro_f1), 4),
        "patch_weighted_f1": round(float(patch_weighted_f1), 4),
        "image_voting_accuracy": round(float(image_level_acc), 4),
        "image_voting_macro_f1": round(float(image_level_macro_f1), 4),
        "num_patches": len(all_labels),
        "num_images": len(image_groups),
        "report": report,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_paths": all_paths,
        "all_probs": all_probs,
        "inference_time_sec": round(float(inference_time), 2)
    }
    return res


def evaluate_model(model_key: str, device: torch.device, max_samples_per_class: int = None, data_dir: str = None, checkpoint_dir: str = None, results_dir: str = None):
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

    model, input_size = load_model(model_key, checkpoint_path, device)
    norm_mean, norm_std = MODEL_NORM_STATS[model_key]
    batch_size = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    test_ds = HeritageDataset(data_dir_path, "test", input_size, norm_mean, norm_std,
                             max_samples_per_class=max_samples_per_class)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    ref_ds = HeritageDataset(data_dir_path, "reference_test", input_size, norm_mean, norm_std,
                            max_samples_per_class=max_samples_per_class)
    ref_loader = DataLoader(ref_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Evaluating '{model_key}' on Standard Test Set...")
    std_res = evaluate_dataset(model, test_loader, device)

    print(f"Evaluating '{model_key}' on Reference Test Set...")
    ref_res = evaluate_dataset(model, ref_loader, device)

    ms_per_image = (std_res["inference_time_sec"] / max(1, std_res["num_patches"])) * 1000

    metrics = {
        "model": model_key,
        "standard_test": {
            "patch_accuracy": std_res["patch_accuracy"],
            "patch_macro_f1": std_res["patch_macro_f1"],
            "image_voting_accuracy": std_res["image_voting_accuracy"],
            "image_voting_macro_f1": std_res["image_voting_macro_f1"],
            "num_patches": std_res["num_patches"],
            "num_images": std_res["num_images"]
        },
        "reference_test": {
            "patch_accuracy": ref_res["patch_accuracy"],
            "patch_macro_f1": ref_res["patch_macro_f1"],
            "image_voting_accuracy": ref_res["image_voting_accuracy"],
            "image_voting_macro_f1": ref_res["image_voting_macro_f1"],
            "num_patches": ref_res["num_patches"],
            "num_images": ref_res["num_images"]
        },
        "ms_per_image": round(ms_per_image, 2),
        "per_class": {cls: {
            "precision": round(std_res["report"][cls]["precision"], 4) if cls in std_res["report"] else 0.0,
            "recall":    round(std_res["report"][cls]["recall"], 4) if cls in std_res["report"] else 0.0,
            "f1":        round(std_res["report"][cls]["f1-score"], 4) if cls in std_res["report"] else 0.0,
            "support":   std_res["report"][cls]["support"] if cls in std_res["report"] else 0,
        } for cls in CLASSES}
    }

    results_dir_path.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_path = results_dir_path / f"{model_key}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save confusion matrix CSV
    if len(std_res["confusion_matrix"]) > 0:
        cm_df = pd.DataFrame(std_res["confusion_matrix"], index=CLASSES, columns=CLASSES)
        cm_path = results_dir_path / f"{model_key}_confusion.csv"
        cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print(f"\n  [STANDARD TEST SET]  Patch Acc={std_res['patch_accuracy']:.4f} | Image Voting Acc={std_res['image_voting_accuracy']:.4f} ({std_res['num_images']} buildings)")
    print(f"  [REFERENCE TEST SET] Patch Acc={ref_res['patch_accuracy']:.4f} | Image Voting Acc={ref_res['image_voting_accuracy']:.4f} ({ref_res['num_images']} buildings)")
    print(f"  Saved metrics -> {metrics_path.name}")

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

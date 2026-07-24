"""
compare_models.py — Print and save a side-by-side comparison table of all models.

Usage:
    .venv\\Scripts\\python.exe execution/training/compare_models.py

Requires:
    evaluate.py to have been run first (reads .tmp/results/{model}_metrics.json)

Output:
    Console table + .tmp/results/model_comparison.csv
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR  = PROJECT_ROOT / ".tmp" / "results"

# Known FLOPs and param counts from paper / timm
MODEL_SPECS = {
    "resnet50": {"params_M": 25.6,  "flops_G": 4.1},
    "vit":      {"params_M": 86.0,  "flops_G": 17.6},
    "dinov2":   {"params_M": 22.0,  "flops_G": 4.6},
    "swinv2":   {"params_M": 28.0,  "flops_G": 4.5},
    "effnet":   {"params_M": 21.5,  "flops_G": 2.9},
    "convnext": {"params_M": 28.6,  "flops_G": 4.5},
}

# Display names
DISPLAY_NAMES = {
    "resnet50": "ResNet-50",
    "vit":      "ViT-B/16",
    "dinov2":   "DINOv2-S",
    "swinv2":   "Swin-V2-T",
    "effnet":   "EfficientNet-V2-S",
    "convnext": "ConvNeXt-Tiny",
}

# Model type labels
MODEL_TYPES = {
    "resnet50": "CNN (baseline)",
    "vit":      "Transformer (flat)",
    "dinov2":   "Transformer (self-sup.)",
    "swinv2":   "Transformer (hierarchical)",
    "effnet":   "CNN (Fused-MBConv)",
    "convnext": "Modern CNN (Depthwise 7x7)",
}


def main():
    rows = []

    for model_key, specs in MODEL_SPECS.items():
        metrics_path = RESULTS_DIR / f"{model_key}_metrics.json"

        if not metrics_path.exists():
            print(f"[MISSING] {model_key}_metrics.json — run evaluate.py first.")
            row = {
                "Model":        DISPLAY_NAMES[model_key],
                "Type":         MODEL_TYPES[model_key],
                "Params (M)":   specs["params_M"],
                "FLOPs (G)":    specs["flops_G"],
                "Test Acc.":    "N/A",
                "Macro F1":     "N/A",
                "Weighted F1":  "N/A",
                "Speed (ms/img)": "N/A",
            }
        else:
            with open(metrics_path, "r", encoding="utf-8") as f:
                m = json.load(f)

            primary = m.get("primary_standard", {"accuracy": m['test_accuracy'], "macro_f1": m['macro_f1']})
            subsets = m.get("test_subsets", {})
            hist = subsets.get("Historic Old Pics", {"accuracy": 0.0, "macro_f1": 0.0})
            other = subsets.get("Other / Needs Edit", {"accuracy": 0.0, "macro_f1": 0.0})

            row = {
                "Model":           DISPLAY_NAMES[model_key],
                "Type":            MODEL_TYPES[model_key],
                "Params (M)":      specs["params_M"],
                "FLOPs (G)":       specs["flops_G"],
                "Standard Acc (Primary)": f"{primary['accuracy']:.4f}",
                "Standard F1 (Primary)":  f"{primary['macro_f1']:.4f}",
                "Historic F1":     f"{hist['macro_f1']:.4f}",
                "Other F1":        f"{other['macro_f1']:.4f}",
                "Overall Acc":     f"{m['test_accuracy']:.4f}",
                "Speed (ms)":      f"{m['ms_per_image']:.1f}",
            }

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Console output ─────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  HERITAGE ARCHITECTURE CLASSIFICATION — MODEL COMPARISON (PRIMARY: STANDARDIZED BUILDINGS)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    # ── Per-class breakdown ────────────────────────────────────────────────────
    print("\nPer-class F1 scores:\n")
    class_rows = []
    classes = ["A1", "A2", "B1", "B2"]
    for model_key in MODEL_SPECS:
        metrics_path = RESULTS_DIR / f"{model_key}_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        crow = {"Model": DISPLAY_NAMES[model_key]}
        for cls in classes:
            crow[cls] = f"{m['per_class'][cls]['f1']:.4f}"
        class_rows.append(crow)

    if class_rows:
        class_df = pd.DataFrame(class_rows)
        print(class_df.to_string(index=False))

    # ── Save CSV ───────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "model_comparison.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nComparison table saved to: {out_path}")


if __name__ == "__main__":
    main()

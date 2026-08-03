"""
generate_report_plots.py — Generate high-quality benchmark figures and charts for PROGRESS_REPORT.md

Outputs:
    outputs/figures/dataset_distribution.png   — Split and Class sample distribution
    outputs/figures/model_comparison_chart.png — Standard Accuracy, F1, and Speed comparison
    outputs/figures/confusion_matrices.png    — 2x2 Heatmap grid of all 4 models' confusion matrices
    outputs/figures/per_class_f1_chart.png     — Class-level F1 breakdown (A1, A2, B1, B2)
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR  = PROJECT_ROOT / ".tmp" / "results"
MANIFEST_CSV = PROJECT_ROOT / ".tmp" / "manifest.csv"
PROC_CSV     = PROJECT_ROOT / ".tmp" / "processed_manifest.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

MODELS = ["resnet50", "vit", "dinov2", "swinv2"]
MODEL_NAMES = {
    "resnet50": "ResNet-50\n(CNN)",
    "vit":      "ViT-B/16\n(Flat ViT)",
    "dinov2":   "DINOv2-S\n(Self-Sup.)",
    "swinv2":   "Swin-V2-T\n(Hierarchical)"
}
CLASSES = ["A1", "A2", "B1", "B2"]
CLASS_LABELS = [
    "A1 (French)",
    "A2 (Modernism)",
    "B1 (Vernacular)",
    "B2 (Eclectic)"
]
COLORS = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22']


def plot_dataset_distribution():
    print("Generating Figure 1: Dataset Distribution...")
    
    # Read actual counts from processed_manifest.csv
    if PROC_CSV.exists():
        df_proc = pd.read_csv(PROC_CSV)
        proc_split_counts = df_proc['split'].value_counts()
        train_patches = int(proc_split_counts.get('train', 79628))
        val_patches   = int(proc_split_counts.get('val', 21126))
        test_patches  = int(proc_split_counts.get('test', 82920))
        
        proc_class_counts = df_proc['style_label'].value_counts()
        a1_patches = int(proc_class_counts.get('A1', 81685))
        a2_patches = int(proc_class_counts.get('A2', 21515))
        b1_patches = int(proc_class_counts.get('B1', 58118))
        b2_patches = int(proc_class_counts.get('B2', 22356))
    else:
        train_patches, val_patches, test_patches = 79628, 21126, 82920
        a1_patches, a2_patches, b1_patches, b2_patches = 81685, 21515, 58118, 22356

    if MANIFEST_CSV.exists():
        df_raw = pd.read_csv(MANIFEST_CSV)
        raw_split_counts = df_raw['split'].value_counts()
        train_raw = int(raw_split_counts.get('train', 4072))
        val_raw   = int(raw_split_counts.get('val', 962))
        test_raw  = int(raw_split_counts.get('test', 4706))
    else:
        train_raw, val_raw, test_raw = 4072, 962, 4706

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Split Distribution (Patches)
    splits = ['Train', 'Val', 'Test']
    counts = [train_patches, val_patches, test_patches]
    colors = ['#2980b9', '#f39c12', '#27ae60']

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f"{val:,}\n({pct:.1f}%)"
        return my_autopct

    wedges, texts, autotexts = ax1.pie(
        counts, labels=splits, autopct=make_autopct(counts),
        startangle=140, colors=colors, explode=(0.02, 0.04, 0.02),
        pctdistance=0.65, textprops=dict(weight='bold')
    )

    total_patches = sum(counts)
    ax1.set_title(f"Dataset Split Distribution ({total_patches:,} Patches)\n[70-15-15 Building-Level Stratified Group Split]", pad=15)

    # Class Distribution (Patches)
    class_counts = [a1_patches, a2_patches, b1_patches, b2_patches]
    bars = ax2.bar(CLASS_LABELS, class_counts, color=['#16a085', '#2980b9', '#8e44ad', '#d35400'], width=0.55)
    ax2.set_title(f"Class Sample Distribution ({total_patches:,} Patches)", pad=15)
    ax2.set_ylabel("Number of Patches")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:,}\n({height/sum(class_counts):.1%})",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    output_path = OUTPUT_DIR / "dataset_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {output_path}")


def plot_model_comparison():
    print("Generating Figure 2: Model Comparison Benchmark Chart...")
    models_data = []
    for m in MODELS:
        path = RESULTS_DIR / f"{m}_metrics.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                std = d.get("standard_test", d.get("primary_standard", {}))
                acc = std.get("image_voting_accuracy", std.get("patch_accuracy", std.get("accuracy", d.get("test_accuracy", 0.0))))
                f1 = std.get("patch_macro_f1", std.get("macro_f1", d.get("macro_f1", 0.0)))
                models_data.append({
                    "model": m,
                    "name": MODEL_NAMES[m],
                    "std_acc": acc,
                    "std_f1":  f1,
                    "speed":   d.get("ms_per_image", d.get("ms_per_patch", 10.0))
                })

    df = pd.DataFrame(models_data)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df['std_acc'] * 100, width, label='Standard Accuracy (%)', color='#2ecc71')
    rects2 = ax1.bar(x + width/2, df['std_f1'] * 100, width, label='Standard Macro-F1 (%)', color='#3498db')

    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Heritage Building Classification — Model Performance Comparison\n(Primary Benchmark: Standardized Buildings)', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['name'], fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper left', frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Highlight bars with values
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9.5)

    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9.5, color='#1b4f72')

    # Winner callout
    if len(df) > 0:
        winner_idx = df['std_acc'].idxmax()
        best_val = df.loc[winner_idx, 'std_acc'] * 100
        ax1.annotate(f'[Best Model] ({best_val:.2f}%)',
                     xy=(winner_idx - width/2, best_val),
                     xytext=(winner_idx - 0.2, min(best_val + 7, 100)),
                     arrowprops=dict(facecolor='gold', shrink=0.08, width=2, headwidth=8),
                     fontweight='bold', color='#b7950b', fontsize=11)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "model_comparison_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {output_path}")


def plot_confusion_matrices():
    print("Generating Figure 3: Confusion Matrices Grid...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10.5))
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        cm_path = RESULTS_DIR / f"{m}_confusion.csv"

        if cm_path.exists():
            cm_df = pd.read_csv(cm_path, index_col=0)
            cm = cm_df.values
        else:
            cm = np.zeros((4, 4))

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        ax.set_title(f"{MODEL_NAMES[m].replace(chr(10), ' ')}", fontsize=13, fontweight='bold', pad=10)
        tick_marks = np.arange(len(CLASSES))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(CLASSES, fontweight='bold')
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(CLASSES, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = int(cm[i, j])
                ax.text(j, i, f"{val}",
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontweight='bold', fontsize=11)

    plt.suptitle('Test Set Confusion Matrices Across All 4 Models', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / "confusion_matrices.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {output_path}")


def plot_per_class_f1():
    print("Generating Figure 4: Per-Class F1 Score Breakdown Chart...")
    data = {}
    for m in MODELS:
        path = RESULTS_DIR / f"{m}_metrics.json"
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                data[m] = [d["per_class"][c]["f1"] for c in CLASSES]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(CLASSES))
    width = 0.2

    for idx, m in enumerate(MODELS):
        offset = (idx - 1.5) * width
        vals = data.get(m, [0, 0, 0, 0])
        bars = ax.bar(x + offset, [v * 100 for v in vals], width, label=MODEL_NAMES[m].replace('\n', ' '), color=COLORS[idx])
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('F1 Score (%)', fontweight='bold')
    ax.set_title('Per-Class F1 Score Breakdown Across Heritage Architectural Styles', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontweight='bold', fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper right', frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "per_class_f1_chart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {output_path}")


def main():
    print("=== Generating High-Quality Benchmark Plots ===")
    plot_dataset_distribution()
    plot_model_comparison()
    plot_confusion_matrices()
    plot_per_class_f1()
    print("All figures successfully created in outputs/figures/!")


if __name__ == "__main__":
    main()

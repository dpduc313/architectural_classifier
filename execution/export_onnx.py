"""
export_onnx.py — Export all trained PyTorch models to ONNX format for architecture visualization.
"""

import os
import sys
from pathlib import Path
import torch

# Add execution/training directory to path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from execution.training.train import build_model
from execution.training.dataset import MODEL_INPUT_SIZES

CHECKPOINT_DIR = PROJECT_ROOT / ".tmp" / "checkpoints"
ONNX_DIR = PROJECT_ROOT / ".tmp" / "checkpoints_ONNX"

def export_all_models():
    # Create the output directory
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"ONNX export directory created: {ONNX_DIR}")

    model_keys = ["resnet50", "vit", "dinov2", "swinv2", "effnet", "convnext"]

    for key in model_keys:
        print("\n" + "="*50)
        print(f"Processing model: {key}")
        print("="*50)
        
        # 1. Build the model structure
        try:
            model = build_model(key)
            model.eval()  # Set model to evaluation mode (important for dropout/batchnorm)
        except Exception as e:
            print(f"Error building model {key}: {e}")
            continue

        # 2. Try to load the trained weights
        ckpt_path = CHECKPOINT_DIR / f"{key}_best.pt"
        if ckpt_path.exists():
            print(f"Loading weights from {ckpt_path.name}...")
            try:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print("Successfully loaded trained weights.")
            except Exception as e:
                print(f"Warning: Could not load weights for {key} due to error: {e}")
                print("Exporting the default model architecture instead (structures will be identical).")
        else:
            print(f"No checkpoint found at {ckpt_path}. Exporting base architecture.")

        # 3. Create dummy input with the correct resolution
        img_size = MODEL_INPUT_SIZES.get(key, 224)
        dummy_input = torch.randn(1, 3, img_size, img_size)
        print(f"Using dummy input size: {dummy_input.shape}")

        # 4. Export to ONNX format
        onnx_path = ONNX_DIR / f"{key}_architecture.onnx"
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=14,  # Use standard opset version 14
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            print(f"Successfully exported {key} to ONNX: {onnx_path.name}")
        except Exception as e:
            print(f"Error exporting {key} to ONNX: {e}")

if __name__ == "__main__":
    export_all_models()
    print("\n" + "="*50)
    print("Export process finished.")
    print(f"Files are saved under: {ONNX_DIR}")
    print("Drag and drop these .onnx files into https://netron.app to view the graphs.")
    print("="*50)

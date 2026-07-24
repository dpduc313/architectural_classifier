"""
gradcam_vis.py — Generate Grad-CAM heatmap overlays for Swin-V2 on test images.

Usage:
    .venv\\Scripts\\python.exe execution/training/gradcam_vis.py

Requires:
    - Swin-V2 checkpoint at .tmp/checkpoints/swinv2_best.pt
    - processed_data/test/ to exist
    - Package: pip install grad-cam

Output:
    outputs/gradcam/{class_name}/{filename}_gradcam.jpg
"""

import random
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("ERROR: 'grad-cam' package not found.")
    print("Install with: .venv\\Scripts\\pip.exe install grad-cam")
    raise

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import CLASSES, MODEL_INPUT_SIZES
from train import MODEL_REGISTRY, CHECKPOINT_DIR, NUM_CLASSES

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed_data"
OUTPUT_DIR    = PROJECT_ROOT / "outputs" / "gradcam"
SAMPLES_PER_CLASS = 5  # how many images per class to visualise
RANDOM_SEED   = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_swinv2_target_layer(model) -> list:
    """
    Target the norm layer at the end of the last Swin stage.
    GradCAM needs a layer that outputs a spatial feature map (H×W×C).
    For Swin-V2 in timm, that is model.layers[-1].blocks[-1].norm1
    """
    try:
        target_layer = model.layers[-1].blocks[-1].norm1
        return [target_layer]
    except AttributeError:
        # Fallback: use the last layer norm before global pooling
        print("Warning: could not find standard Swin target layer. Using norm layer fallback.")
        return [model.norm]


def preprocess_image(img_path: Path, input_size: int = 256):
    """
    Returns:
        tensor:    (1, C, H, W) normalised tensor for model input
        rgb_float: (H, W, 3) float32 numpy in [0, 1] for overlay
    """
    transform = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    unnorm_transform = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),  # no normalise — for overlay
    ])
    img_pil = Image.open(img_path).convert("RGB")
    tensor    = transform(img_pil).unsqueeze(0)
    rgb_float = unnorm_transform(img_pil).permute(1, 2, 0).numpy()
    return tensor, rgb_float


def generate_gradcam(model_key: str = "swinv2"):
    checkpoint_path = CHECKPOINT_DIR / f"{model_key}_best.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}. Run train.py --model {model_key} first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = MODEL_INPUT_SIZES[model_key]

    # Load model
    timm_id = MODEL_REGISTRY[model_key]
    model = timm.create_model(timm_id, pretrained=False, num_classes=NUM_CLASSES)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print(f"Loaded {model_key} from checkpoint (val_acc={ckpt['val_acc']:.4f})")

    target_layers = get_swinv2_target_layer(model)
    cam = GradCAM(model=model, target_layers=target_layers)

    random.seed(RANDOM_SEED)

    for class_name in CLASSES:
        class_dir = PROCESSED_DIR / "test" / class_name
        if not class_dir.exists():
            print(f"  Skipping {class_name}: test dir not found.")
            continue

        all_imgs = sorted(class_dir.glob("*.jpg"))
        samples  = random.sample(all_imgs, min(SAMPLES_PER_CLASS, len(all_imgs)))

        out_dir = OUTPUT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in samples:
            tensor, rgb_float = preprocess_image(img_path, input_size)
            tensor = tensor.to(device)

            label_idx = CLASSES.index(class_name)
            targets   = [ClassifierOutputTarget(label_idx)]

            # Generate heatmap
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            heatmap = grayscale_cam[0]  # (H, W)

            overlay = show_cam_on_image(rgb_float, heatmap, use_rgb=True)

            # Save side-by-side: original | heatmap overlay
            orig_uint8    = (rgb_float * 255).astype(np.uint8)
            overlay_uint8 = overlay.astype(np.uint8)

            # Add class/pred label to overlay
            with torch.no_grad():
                logits = model(tensor)
                pred_idx = logits.argmax(dim=1).item()
                pred_cls = CLASSES[pred_idx]
            label_text = f"True: {class_name}  Pred: {pred_cls}"
            cv2.putText(overlay_uint8, label_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            combined = np.concatenate([orig_uint8, overlay_uint8], axis=1)
            out_path = out_dir / f"{img_path.stem}_gradcam.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")

    print(f"\nGrad-CAM visualisations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_gradcam(model_key="swinv2")

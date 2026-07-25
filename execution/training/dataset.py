import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CLASSES = ["A1", "A2", "B1", "B2"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ImageNet normalisation (used by SwinV2, DINOv2, ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ViT (augreg) uses 0.5/0.5 normalisation
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD  = [0.5, 0.5, 0.5]

# Input resolution per model
# DINOv2 native is 518 but we cap at 224 (pass img_size=224 to timm)
MODEL_INPUT_SIZES = {
    "swinv2":   256,
    "vit":      224,
    "dinov2":   224,
    "resnet50": 224,
    "effnet":   224,
    "convnext": 224,
}

# Per-model normalisation stats
MODEL_NORM_STATS = {
    "swinv2":   (IMAGENET_MEAN, IMAGENET_STD),
    "vit":      (VIT_MEAN, VIT_STD),
    "dinov2":   (IMAGENET_MEAN, IMAGENET_STD),
    "resnet50": (IMAGENET_MEAN, IMAGENET_STD),
    "effnet":   (IMAGENET_MEAN, IMAGENET_STD),
    "convnext": (IMAGENET_MEAN, IMAGENET_STD),
}


def get_transforms(split: str, input_size: int = 224,
                   norm_mean=None, norm_std=None) -> transforms.Compose:
    """
    Returns a torchvision transform pipeline for the given split.

    Args:
        split:      'train', 'val', or 'test'
        input_size: image size expected by the model
        norm_mean:  channel mean for normalisation
        norm_std:   channel std for normalisation
    """
    if norm_mean is None:
        norm_mean = IMAGENET_MEAN
    if norm_std is None:
        norm_std = IMAGENET_STD

    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(int(input_size * 256 / 224)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])


class HeritageDataset(Dataset):
    """
    Dataset that reads directly from the processed_data/ folder structure.

    Args:
        processed_dir: path to processed_data/
        split:         'train', 'val', or 'test'
        input_size:    image size (224 or 256 depending on model)
        norm_mean:     per-model normalisation mean
        norm_std:      per-model normalisation std
    """

    def __init__(self, processed_dir: str, split: str,
                 input_size: int = 224, norm_mean=None, norm_std=None,
                 max_samples_per_class: int = None):
        self.transform = get_transforms(split, input_size, norm_mean, norm_std)
        self.samples: list[tuple[Path, int]] = []

        split_dir = Path(processed_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                "Make sure preprocess_images.py has been run first."
            )

        for class_name in CLASSES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            imgs = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                imgs.extend(class_dir.glob(ext))
            imgs = sorted(imgs)
            if max_samples_per_class is not None and len(imgs) > max_samples_per_class:
                # Subsample deterministically with fixed step stride for even coverage across buildings
                step = len(imgs) / max_samples_per_class
                indices = [int(i * step) for i in range(max_samples_per_class)]
                imgs = [imgs[i] for i in indices]
            for img_path in imgs:
                self.samples.append((img_path, CLASS_TO_IDX[class_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {split_dir}. "
                             "Check that preprocess_images.py completed successfully.")

        print(f"[HeritageDataset] {split}: {len(self.samples)} patches across "
              f"{len(CLASSES)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: could not open {img_path}: {e}. Returning zeros.")
            image = Image.new("RGB", (224, 224), color=0)
        image = self.transform(image)
        return image, label, str(img_path)


def get_class_weights(dataset: HeritageDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted CrossEntropyLoss.
    Helps with class imbalance.
    """
    counts = torch.zeros(len(CLASSES))
    for _, label, _ in dataset:
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    weights = weights / weights.sum() * len(CLASSES)
    return weights

"""
Data Loader — Folder-based Train / Val / Test Split
====================================================
Local dataset folder ထဲက images တွေကို train / val / test ခွဲပေးမယ်။

Supported folder structures:
  (A) Already split:     dataset/train/class_a/  dataset/test/class_b/  ...
  (B) Single folder:     dataset/images/class_a/  dataset/images/class_b/  ...
                          → auto split into train / val / test

Usage:
  from data_loader import create_dataloaders
  train_loader, val_loader, test_loader, class_names = create_dataloaders("path/to/dataset")
"""

import os
import shutil
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


# ============================
# Configuration
# ============================
IMG_SIZE = 224
BATCH_SIZE = 32
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)
PIN_MEMORY = torch.cuda.is_available()

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ============================
# Transforms
# ============================
def get_train_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_eval_transform(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize(int(img_size / 0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ============================
# Dataset Class
# ============================
class ImageFolderDataset(Dataset):
    """
    Folder structure: root/class_name/image.jpg
    """
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if Path(fname).suffix.lower() in VALID_EXTENSIONS:
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================
# Folder Structure Detection
# ============================
def detect_structure(data_dir):
    """
    Dataset folder structure ကို detect လုပ်မယ်:
      - "pre_split" : train/ (+ optional val/, test/) folders ရှိပြီးသား
      - "flat"      : class folders ပဲ ရှိတယ်, split မလုပ်ရသေး
    """
    data_dir = Path(data_dir)
    subdirs = {d.name for d in data_dir.iterdir() if d.is_dir()}

    if "train" in subdirs:
        return "pre_split"
    return "flat"


# ============================
# Auto Split (flat folder → train/val/test)
# ============================
def auto_split_folder(src_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    src_dir/class_name/*.jpg  →  dest_dir/train/class_name/
                                  dest_dir/val/class_name/
                                  dest_dir/test/class_name/

    Original files ကို copy လုပ်တာ (move မဟုတ်ဘူး)။
    dest_dir ရှိပြီးသားဆိုရင် skip လုပ်မယ်။
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    # Already split ဖြစ်ပြီးသားဆိုရင် skip
    if (dest_dir / "train").exists():
        print(f"Split already exists at {dest_dir}, skipping.")
        return

    random.seed(seed)
    class_names = sorted(d.name for d in src_dir.iterdir() if d.is_dir())

    for split in ["train", "val", "test"]:
        for cls in class_names:
            (dest_dir / split / cls).mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for cls in class_names:
        files = [
            f for f in (src_dir / cls).iterdir()
            if f.suffix.lower() in VALID_EXTENSIONS
        ]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            for f in split_files:
                shutil.copy2(f, dest_dir / split_name / cls / f.name)
                total_copied += 1

        print(f"  {cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    print(f"Total: {total_copied} files copied to {dest_dir}")


# ============================
# DataLoader Creation
# ============================
def create_dataloaders(
    data_dir,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
):
    """
    Dataset path ပေးရုံနဲ့ train/val/test DataLoader ၃ ခု ပြန်ပေးမယ်။

    Args:
        data_dir:    dataset root path
        img_size:    input image size
        batch_size:  batch size
        train_ratio: (flat folder only) train split ratio
        val_ratio:   (flat folder only) val split ratio
        test_ratio:  (flat folder only) test split ratio

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    data_dir = Path(data_dir)
    structure = detect_structure(data_dir)
    print(f"Dataset: {data_dir} | Structure: {structure}")

    if structure == "pre_split":
        train_dir = data_dir / "train"
        val_dir = data_dir / "val" if (data_dir / "val").exists() else None
        test_dir = data_dir / "test" if (data_dir / "test").exists() else None
    else:
        # Flat folder → auto split
        split_dir = data_dir / "_split"
        print(f"Auto-splitting to {split_dir} ...")
        auto_split_folder(data_dir, split_dir, train_ratio, val_ratio, test_ratio)
        train_dir = split_dir / "train"
        val_dir = split_dir / "val"
        test_dir = split_dir / "test"

    # Train dataset
    train_dataset = ImageFolderDataset(train_dir, transform=get_train_transform(img_size))
    class_names = train_dataset.class_names
    num_classes = len(class_names)

    # Val dataset — မရှိရင် train ထဲက random_split ခွဲမယ်
    if val_dir and val_dir.exists():
        val_dataset = ImageFolderDataset(val_dir, transform=get_eval_transform(img_size))
    else:
        val_size = int(len(train_dataset) * 0.15)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        # val subset ကို eval transform သုံးမယ်
        val_dataset_eval = ImageFolderDataset(train_dir, transform=get_eval_transform(img_size))
        val_dataset.dataset = val_dataset_eval
        print(f"  No val/ folder — split from train: train={train_size}, val={val_size}")

    # Test dataset — မရှိရင် None
    test_dataset = None
    if test_dir and test_dir.exists():
        test_dataset = ImageFolderDataset(test_dir, transform=get_eval_transform(img_size))

    # DataLoaders
    loader_kwargs = dict(batch_size=batch_size, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset else None

    # Summary
    print(f"\nClasses: {num_classes} → {class_names}")
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}", end="")
    if test_dataset:
        print(f" | Test: {len(test_dataset):,}")
    else:
        print(" | Test: (none)")
    print(f"Batch size: {batch_size} | Workers: {NUM_WORKERS}")

    return train_loader, val_loader, test_loader, class_names


# ============================
# Standalone Usage
# ============================
if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../dataset/your_dataset"

    train_loader, val_loader, test_loader, class_names = create_dataloaders(data_path)

    # Sanity check
    images, labels = next(iter(train_loader))
    print(f"\nSanity check — batch shape: {tuple(images.shape)}, labels: {tuple(labels.shape)}")

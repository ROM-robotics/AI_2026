"""
Transfer Learning Template — MobileNet Family
=================================================
MobileNet V2 / V3-Small / V3-Large ကိုသုံးပြီး transfer learning လုပ်မယ်။
- Local path ထဲက dataset load မယ်
- Classifier only train မယ် (backbone frozen)
- Full model fine-tune မယ်
- Weight only + Full model save မယ်
"""

import os
import shutil
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI-less backend (script mode)
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from itertools import cycle


# ============================
# Configuration
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path
DATA_DIR = Path(    r"C:\Users\PredatorNeo\Desktop\AI-Class\AI_2026\Day12 CNN tf learning\groupA\hta_mobilenet-food-5-classes\food_5_classes"
)

# Train / Val / Test split ratios (flat folder 용)
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS_CLASSIFIER = 10
NUM_EPOCHS_FINETUNE = 5
LR_CLASSIFIER = 1e-3
LR_FINETUNE = 1e-4
VAL_RATIO = 0.15

# ImageNet normalization
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

NUM_WORKERS = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)
PIN_MEMORY = DEVICE.type == "cuda"

# MobileNet variant ရွေးပါ
# "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"
MODEL_NAME = "mobilenet_v2"

SAVE_BASE = Path("saved_models")


# ============================
# MobileNet variant → (model_fn, weights, input_size)
# ============================
MOBILENET_VARIANTS = {
    "mobilenet_v2":       (models.mobilenet_v2,       models.MobileNet_V2_Weights.IMAGENET1K_V2,       224),
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 224),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, 224),
}


def get_img_size(variant=MODEL_NAME):
    return MOBILENET_VARIANTS[variant][2]


# ============================
# Dataset
# ============================
class ImageFolderDataset(Dataset):
    """Local folder structure: root/class_name/image.jpg"""
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.class_names = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================
# Transforms
# ============================
def get_train_transform(img_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_test_transform(img_size):
    resize = int(img_size / 0.875)  # center crop ratio
    return transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ============================
# Auto Split (flat folder → train / val / test)
# ============================
def auto_split(src_dir, dest_dir, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=42):
    """src_dir/class/*.jpg → dest_dir/train|val|test/class/*.jpg  (copy, not move)"""
    src_dir, dest_dir = Path(src_dir), Path(dest_dir)
    if (dest_dir / "train").exists():
        print(f"Split already exists at {dest_dir} — skipping.")
        return

    random.seed(seed)
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    class_names = sorted(d.name for d in src_dir.iterdir() if d.is_dir())

    for split in ("train", "val", "test"):
        for cls in class_names:
            (dest_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in class_names:
        files = [f for f in (src_dir / cls).iterdir() if f.suffix.lower() in valid_ext]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        splits  = {
            "train": files[:n_train],
            "val":   files[n_train:n_train + n_val],
            "test":  files[n_train + n_val:],
        }
        for split_name, split_files in splits.items():
            for f in split_files:
                shutil.copy2(f, dest_dir / split_name / cls / f.name)
        print(f"  {cls}: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")


# ============================
# DataLoaders
# ============================
def create_dataloaders(train_dir, test_dir, img_size):
    train_dataset = ImageFolderDataset(train_dir, transform=get_train_transform(img_size))
    test_dataset  = ImageFolderDataset(test_dir,  transform=get_test_transform(img_size))

    val_size   = int(len(train_dataset) * VAL_RATIO)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    train_loader = DataLoader(train_subset, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_subset,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names

    print(f"Classes: {num_classes} | Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, num_classes, class_names


def create_dataloaders_from_flat(data_dir, img_size):
    """Flat folder (class dirs only) → auto split → DataLoaders"""
    data_dir  = Path(data_dir)
    split_dir = data_dir / "_split"
    print(f"Auto-splitting {data_dir} → {split_dir}")
    auto_split(data_dir, split_dir)
    return create_dataloaders(split_dir / "train", split_dir / "test", img_size)


# ============================
# Model — MobileNet Family
# ============================
def create_model(num_classes, variant=MODEL_NAME):
    model_fn, weights, _ = MOBILENET_VARIANTS[variant]
    model = model_fn(weights=weights)

    # Backbone freeze
    for param in model.parameters():
        param.requires_grad = False

    # MobileNet V2:       model.classifier = [Dropout, Linear]
    # MobileNet V3 small/large: model.classifier = [Linear, Hardswish, Dropout, Linear]
    if variant == "mobilenet_v2":
        in_features = model.classifier[1].in_features
    else:  # v3_small / v3_large
        in_features = model.classifier[-1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )

    model = model.to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{variant}] Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")
    return model


# ============================
# Training Functions
# ============================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, phase_name):
    best_val_acc = 0.0
    best_state = None

    print(f"\n{'='*60}")
    print(f"{phase_name}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " ✓"

        elapsed = time.time() - start
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.1f}s{marker}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc


# ============================
# Phase 1: Classifier Only
# ============================
def train_classifier_only(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR_CLASSIFIER)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    return train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS_CLASSIFIER, "Phase 1: Classifier Only (backbone frozen)"
    )


# ============================
# Phase 2: Full Fine-tune
# ============================
def finetune_full(model, train_loader, val_loader):
    for param in model.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n], "lr": LR_FINETUNE / 10},
        {"params": model.classifier.parameters(), "lr": LR_FINETUNE},
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen all — trainable params: {trainable:,}")

    return train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS_FINETUNE, "Phase 2: Full Fine-tune (all layers)"
    )


# ============================
# Evaluation Helpers
# ============================
def collect_predictions(model, loader):
    """test_loader မှ all predictions + probabilities + labels ကောက်မယ်"""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(len(class_names) + 2, len(class_names) + 2))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved: {save_path}")


def plot_roc_curve(labels, probs, class_names, save_path):
    n_classes = len(class_names)
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    colors = cycle(plt.cm.tab10.colors)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr["micro"], tpr["micro"],
            label=f"micro-avg (AUC={roc_auc['micro']:.3f})",
            color="red", linestyle=":", linewidth=2)
    for i, (cls, color) in enumerate(zip(class_names, colors)):
        ax.plot(fpr[i], tpr[i], color=color,
                label=f"{cls} (AUC={roc_auc[i]:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ROC curve saved:        {save_path}")


def save_classification_report(labels, preds, class_names, save_path):
    report = classification_report(labels, preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    save_path.write_text(report, encoding="utf-8")
    print(f"  Classification report:  {save_path}")


# ============================
# Save Model + Results
# ============================
def save_results(model, test_loader, class_names, variant=MODEL_NAME):
    """saved_models/{variant}/ ထဲမှာ weights, full model, plots, report save မယ်"""
    out_dir = SAVE_BASE / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Weights only
    weights_path = out_dir / f"weights_{variant}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved:      {weights_path} ({weights_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 2) Full model (structure + weights)
    full_path = out_dir / f"full_{variant}.pth"
    torch.save(model, full_path)
    print(f"Full model saved:   {full_path} ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 3) Collect predictions
    print("\nGenerating evaluation plots...")
    labels, preds, probs = collect_predictions(model, test_loader)

    # 4) Confusion Matrix
    plot_confusion_matrix(labels, preds, class_names,
                          out_dir / "confusion_matrix.png")

    # 5) ROC Curve
    plot_roc_curve(labels, probs, class_names,
                   out_dir / "roc_curve.png")

    # 6) Classification Report
    save_classification_report(labels, preds, class_names,
                               out_dir / "classification_report.txt")

    print(f"\nAll results saved to: {out_dir}")


# ============================
# Main
# ============================
def main():
    print(f"Device: {DEVICE}")

    img_size = get_img_size(MODEL_NAME)
    print(f"Model: {MODEL_NAME} | Input size: {img_size}x{img_size}")

    # 1. Data  (flat folder → auto split)
    train_loader, val_loader, test_loader, num_classes, class_names = \
        create_dataloaders_from_flat(DATA_DIR, img_size)

    # 2. Model
    model = create_model(num_classes, variant=MODEL_NAME)

    # 3. Classifier only training
    train_classifier_only(model, train_loader, val_loader)

    # 4. Full fine-tuning
    finetune_full(model, train_loader, val_loader)

    # 5. Test evaluation
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} ({test_acc*100:.1f}%)")

    # 6. Save model + plots + report
    save_results(model, test_loader, class_names, variant=MODEL_NAME)


if __name__ == "__main__":
    main()

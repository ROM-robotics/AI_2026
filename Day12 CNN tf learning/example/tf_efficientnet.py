"""
Transfer Learning Template — EfficientNet Family
=================================================
EfficientNet B0–B7 / V2-S/M/L ကိုသုံးပြီး transfer learning လုပ်မယ်။
- Local path ထဲက dataset load မယ်
- Classifier only train မယ် (backbone frozen)
- Full model fine-tune မယ်
- Weight only + Full model save မယ်
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path


# ============================
# Configuration
# ============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset path — ကိုယ့် local dataset path ပြောင်းပါ
DATA_DIR = Path(r"../../dataset/your_dataset")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

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

# EfficientNet variant ရွေးပါ
# V1: "efficientnet_b0" ~ "efficientnet_b7"
# V2: "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"
MODEL_NAME = "efficientnet_b0"

SAVE_DIR = Path("saved_models")


# ============================
# EfficientNet variant → (model_fn, weights, input_size)
# ============================
EFFICIENTNET_VARIANTS = {
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 224),
    "efficientnet_b1": (models.efficientnet_b1, models.EfficientNet_B1_Weights.IMAGENET1K_V2, 240),
    "efficientnet_b2": (models.efficientnet_b2, models.EfficientNet_B2_Weights.IMAGENET1K_V1, 260),
    "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1, 300),
    "efficientnet_b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.IMAGENET1K_V1, 380),
    "efficientnet_b5": (models.efficientnet_b5, models.EfficientNet_B5_Weights.IMAGENET1K_V1, 456),
    "efficientnet_b6": (models.efficientnet_b6, models.EfficientNet_B6_Weights.IMAGENET1K_V1, 528),
    "efficientnet_b7": (models.efficientnet_b7, models.EfficientNet_B7_Weights.IMAGENET1K_V1, 600),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.IMAGENET1K_V1, 384),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights.IMAGENET1K_V1, 480),
    "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights.IMAGENET1K_V1, 480),
}


def get_img_size(variant=MODEL_NAME):
    return EFFICIENTNET_VARIANTS[variant][2]


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
# DataLoaders
# ============================
def create_dataloaders(train_dir, test_dir, img_size):
    train_dataset = ImageFolderDataset(train_dir, transform=get_train_transform(img_size))
    test_dataset = ImageFolderDataset(test_dir, transform=get_test_transform(img_size))

    val_size = int(len(train_dataset) * VAL_RATIO)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    num_classes = len(train_dataset.class_names)
    class_names = train_dataset.class_names

    print(f"Classes: {num_classes} | Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, num_classes, class_names


# ============================
# Model — EfficientNet Family
# ============================
def create_model(num_classes, variant=MODEL_NAME):
    model_fn, weights, _ = EFFICIENTNET_VARIANTS[variant]
    model = model_fn(weights=weights)

    # Backbone freeze
    for param in model.parameters():
        param.requires_grad = False

    # EfficientNet classifier head: model.classifier = [Dropout, Linear]
    in_features = model.classifier[1].in_features
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
# Save Model
# ============================
def save_model(model, variant=MODEL_NAME):
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Weights only
    weights_path = SAVE_DIR / f"weights_{variant}.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Weights saved: {weights_path} ({weights_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 2) Weights + model structure
    full_path = SAVE_DIR / f"full_{variant}.pth"
    torch.save(model, full_path)
    print(f"Full model saved: {full_path} ({full_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ============================
# Main
# ============================
def main():
    print(f"Device: {DEVICE}")

    img_size = get_img_size(MODEL_NAME)
    print(f"Model: {MODEL_NAME} | Input size: {img_size}x{img_size}")

    # 1. Data
    train_loader, val_loader, test_loader, num_classes, class_names = \
        create_dataloaders(TRAIN_DIR, TEST_DIR, img_size)

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

    # 6. Save
    save_model(model, variant=MODEL_NAME)


if __name__ == "__main__":
    main()

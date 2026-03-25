"""
Inference — EfficientNet Food-5 Classifier
==========================================
saved_models/ ထဲက trained model ကို load ပြီး
image တစ်ပုံ (သို့) folder တစ်ခုကို inference လုပ်မယ်။

Usage:
  # Single image
  python inference.py --image path/to/image.jpg

  # Folder (class sub-dirs ရှိရင် accuracy ပါ ထုတ်ပေးမယ်)
  python inference.py --folder food_5_classes/_split/test

  # Model ရွေးချယ်ခြင်း
  python inference.py --image img.jpg --model weights   # state_dict (default)
  python inference.py --image img.jpg --model full      # full model
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ============================
# Paths
# ============================
SAVE_DIR   = Path(r"C:\Users\RAIDER\Desktop\ai\vision.ai\lecture\tf_examples\saved_models\efficientnet_b0")
VARIANT    = "efficientnet_b0"
NUM_CLASSES = 5
CLASS_NAMES = ["chicken_wings", "club_sandwich", "donuts", "pizza", "sushi"]

WEIGHTS_PATH = SAVE_DIR / f"weights_{VARIANT}.pth"
FULL_PATH    = SAVE_DIR / f"full_{VARIANT}.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# EfficientNet input size map
# ============================
IMG_SIZE_MAP = {
    "efficientnet_b0": 224, "efficientnet_b1": 240, "efficientnet_b2": 260,
    "efficientnet_b3": 300, "efficientnet_b4": 380, "efficientnet_b5": 456,
    "efficientnet_b6": 528, "efficientnet_b7": 600,
    "efficientnet_v2_s": 384, "efficientnet_v2_m": 480, "efficientnet_v2_l": 480,
}
IMG_SIZE = IMG_SIZE_MAP[VARIANT]

# ============================
# Transform
# ============================
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

infer_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE / 0.875)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ============================
# Model Loaders
# ============================
def load_weights_model(weights_path, variant=VARIANT, num_classes=NUM_CLASSES):
    """weights only (.pth) → model architecture ကို rebuild ပြီး weights load"""
    variant_map = {
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b1": models.efficientnet_b1,
        "efficientnet_b2": models.efficientnet_b2,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_b4": models.efficientnet_b4,
        "efficientnet_b5": models.efficientnet_b5,
        "efficientnet_b6": models.efficientnet_b6,
        "efficientnet_b7": models.efficientnet_b7,
        "efficientnet_v2_s": models.efficientnet_v2_s,
        "efficientnet_v2_m": models.efficientnet_v2_m,
        "efficientnet_v2_l": models.efficientnet_v2_l,
    }
    model_fn = variant_map[variant]
    model = model_fn(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
    )
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"[weights] Loaded: {weights_path}")
    return model


def load_full_model(full_path):
    """full model (.pth) → torch.load ဖြင့် တစ်ခါထဲ load"""
    model = torch.load(full_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE).eval()
    print(f"[full]    Loaded: {full_path}")
    return model


# ============================
# Single Image Inference
# ============================
def predict_image(model, img_path, class_names=CLASS_NAMES, topk=3):
    """Image တစ်ပုံကို predict လုပ်ပြီး top-k results ပြမယ်"""
    img = Image.open(img_path).convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(DEVICE)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)                              # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]           # (num_classes,)

    topk_probs, topk_idxs = probs.topk(min(topk, len(class_names)))

    print(f"\nImage : {Path(img_path).name}")
    print(f"{'Rank':<6} {'Class':<20} {'Confidence':>12}")
    print("-" * 40)
    for rank, (idx, prob) in enumerate(zip(topk_idxs, topk_probs), 1):
        marker = " ◀" if rank == 1 else ""
        print(f"{rank:<6} {class_names[idx.item()]:<20} {prob.item()*100:>10.2f}%{marker}")

    pred_class = class_names[topk_idxs[0].item()]
    pred_conf  = topk_probs[0].item()
    return pred_class, pred_conf


# ============================
# Folder Inference
# ============================
def predict_folder(model, folder_path, class_names=CLASS_NAMES):
    """
    Folder structure:
      (A) folder/class_name/img.jpg  → per-class accuracy ပါ ထုတ်ပေးမယ်
      (B) folder/img.jpg             → prediction ပဲ ထုတ်ပေးမယ်
    """
    folder = Path(folder_path)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    # Structure detect
    sub_dirs = [d for d in folder.iterdir() if d.is_dir()]
    has_class_dirs = len(sub_dirs) > 0

    correct, total = 0, 0
    results = []

    if has_class_dirs:
        print(f"\nFolder  : {folder}  (class sub-dirs detected → accuracy mode)")
        print(f"{'Class':<20} {'Predicted':<20} {'Conf':>8}  {'✓/✗':>4}")
        print("-" * 58)
        for class_dir in sorted(sub_dirs):
            true_class = class_dir.name
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_ext]
            for img_path in images:
                pred, conf = _predict_single_silent(model, img_path, class_names)
                ok = (pred == true_class)
                correct += ok
                total   += 1
                mark = "✓" if ok else "✗"
                results.append((true_class, pred, conf, ok))
                print(f"{true_class:<20} {pred:<20} {conf*100:>7.1f}%  {mark:>4}")

        print(f"\n{'='*58}")
        print(f"Overall Accuracy: {correct}/{total}  ({correct/total*100:.1f}%)")

        # Per-class summary
        print(f"\n{'Class':<20} {'Correct':>8} {'Total':>8} {'Acc':>8}")
        print("-" * 48)
        for cls in class_names:
            cls_results = [r for r in results if r[0] == cls]
            c = sum(r[3] for r in cls_results)
            t = len(cls_results)
            acc = c / t if t else 0.0
            print(f"{cls:<20} {c:>8} {t:>8} {acc*100:>7.1f}%")

    else:
        print(f"\nFolder  : {folder}  (flat images)")
        images = [f for f in folder.iterdir() if f.suffix.lower() in valid_ext]
        for img_path in sorted(images):
            predict_image(model, img_path, class_names, topk=1)


def _predict_single_silent(model, img_path, class_names):
    """print 없이 (class, confidence) 만 반환"""
    img    = Image.open(img_path).convert("RGB")
    tensor = infer_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx  = probs.argmax().item()
    return class_names[idx], probs[idx].item()


# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser(description="EfficientNet Food-5 Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Single image path")
    group.add_argument("--folder", type=str, help="Folder path (flat or class sub-dirs)")
    parser.add_argument(
        "--model", choices=["weights", "full"], default="weights",
        help="weights = state_dict load  |  full = torch.load (default: weights)"
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-k results (single image only)")
    args = parser.parse_args()

    print(f"Device  : {DEVICE}")
    print(f"Variant : {VARIANT}  |  Classes: {CLASS_NAMES}")

    # Load model
    if args.model == "full":
        model = load_full_model(FULL_PATH)
    else:
        model = load_weights_model(WEIGHTS_PATH)

    # Run inference
    if args.image:
        predict_image(model, args.image, CLASS_NAMES, topk=args.topk)
    else:
        predict_folder(model, args.folder, CLASS_NAMES)


if __name__ == "__main__":
    main()

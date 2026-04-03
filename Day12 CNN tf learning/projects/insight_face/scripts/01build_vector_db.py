"""
build_vector_db.py
==================
data/gallery ထဲရှိ လူတစ်ဦးချင်းစီ၏ ပုံများမှ မျက်နှာများကို detect လုပ်ပြီး
feature vector (embedding) များ ထုတ်ကာ data/vector_db/ directory ထဲတွင်

    embeddings.pkl  — {name: mean_embedding}  dict  (pickle format)
    embeddings.npz  — numpy archive           (vectors [N,512], labels [N,])

နှစ်မျိုးနှင့် သိမ်းသည်။

Usage:
    python build_vector_db.py
    python build_vector_db.py --gallery ../data/gallery --out ../data/vector_db
"""

import argparse
import os
import pickle

import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(SCRIPT_DIR, "..")

MODEL_NAME  = "buffalo_l"
DET_SIZE    = (640, 640)
PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]
MODEL_ROOT  = os.path.join(BASE_DIR, "models")

DEFAULT_GALLERY = os.path.join(BASE_DIR, "data", "gallery")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "data", "vector_db")
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def l2_normalize(emb: np.ndarray) -> np.ndarray:
    n = norm(emb)
    return emb / n if n > 0 else emb


def load_model() -> FaceAnalysis:
    root = os.path.abspath(MODEL_ROOT)
    print(f"[Model Root] {root}")
    app = FaceAnalysis(name=MODEL_NAME, root=root, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print(f"[✓] Model loaded: {MODEL_NAME}\n")
    return app


def get_best_embedding(app: FaceAnalysis, img: np.ndarray):
    """Image မှ detection confidence အမြင့်ဆုံး မျက်နှာ၏ L2-normalized embedding ကို return လုပ်သည်။"""
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score if f.det_score is not None else 0.0)
    return l2_normalize(best.embedding)


# ─────────────────────────────────────────────
# Gallery Processing
# ─────────────────────────────────────────────
def build_embeddings(app: FaceAnalysis, gallery_dir: str) -> dict:
    """
    gallery_dir/
    ├── Person_A/  ← subfolder name = label
    │   ├── img1.jpg
    │   └── img2.jpg
    └── Person_B/
        └── img1.jpg

    Returns:
        { name: mean_embedding (512-d, L2-normalized) }
    """
    gallery_dir = os.path.abspath(gallery_dir)
    if not os.path.isdir(gallery_dir):
        raise FileNotFoundError(f"Gallery directory not found: {gallery_dir}")

    result = {}

    for person_name in sorted(os.listdir(gallery_dir)):
        person_dir = os.path.join(gallery_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        print(f"[Person] {person_name}")
        embeddings = []

        for fname in sorted(os.listdir(person_dir)):
            if os.path.splitext(fname)[-1].lower() not in IMG_EXTS:
                continue

            img_path = os.path.join(person_dir, fname)
            img = cv2.imread(img_path)
            emb = get_best_embedding(app, img)
            if emb is not None:
                embeddings.append(emb)
                print(f"  [+] {fname}  →  embedding ({emb.shape[0]}-d)")
            else:
                print(f"  [-] {fname}  →  no face detected, skipped")

        if embeddings:
            mean_emb = l2_normalize(np.mean(embeddings, axis=0))
            result[person_name] = mean_emb
            print(f"  [✓] {person_name}: mean embedding from {len(embeddings)} image(s)\n")
        else:
            print(f"  [✗] {person_name}: no valid embeddings, skipped\n")

    return result


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
def save_pkl(embeddings: dict, out_dir: str) -> str:
    """
    embeddings.pkl  →  { name (str): mean_embedding (np.ndarray 512-d) }
    """
    path = os.path.join(out_dir, "embeddings.pkl")
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    return path


def save_npz(embeddings: dict, out_dir: str) -> str:
    """
    embeddings.npz  →  numpy archive
        vectors : np.ndarray  shape (N, 512)   — stacked mean embeddings
        labels  : np.ndarray  shape (N,)        — corresponding person names
    """
    labels  = np.array(list(embeddings.keys()))
    vectors = np.stack(list(embeddings.values()), axis=0)   # (N, 512)

    path = os.path.join(out_dir, "embeddings.npz")
    np.savez(path, vectors=vectors, labels=labels)
    return path


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build face vector DB from gallery.")
    parser.add_argument(
        "--gallery",
        default=DEFAULT_GALLERY,
        help=f"Path to gallery directory (default: {DEFAULT_GALLERY})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for vector DB files (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Model load ──
    app = load_model()

    # ── Step 2: Extract embeddings ──
    print("=" * 50)
    print(f"[Gallery] {os.path.abspath(args.gallery)}")
    print("=" * 50)
    embeddings = build_embeddings(app, args.gallery)

    if not embeddings:
        print("[✗] No embeddings extracted. Exiting.")
        return

    # ── Step 3: Save ──
    print("=" * 50)
    pkl_path = save_pkl(embeddings, out_dir)
    npz_path = save_npz(embeddings, out_dir)

    print(f"[Saved] PKL  → {pkl_path}")
    print(f"[Saved] NPZ  → {npz_path}")
    print("=" * 50)
    print(f"[Done]  {len(embeddings)} person(s) saved.")


if __name__ == "__main__":
    main()

"""
02load_vector_db.py
===================
build_vector_db.py ဖြင့် သိမ်းထားသော vector DB ကို load ပြီး
embeddings.pkl နှင့် embeddings.npz နှစ်မျိုးစလုံးမှ vector များကို
ဖတ်ပြသည်။

Usage:
    python 02load_vector_db.py
    python 02load_vector_db.py --db ../data/vector_db
"""

import argparse
import os
import pickle

import numpy as np


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.join(SCRIPT_DIR, "..")
DEFAULT_DB_DIR  = os.path.join(BASE_DIR, "data", "vector_db")


# ─────────────────────────────────────────────
# Load PKL
# ─────────────────────────────────────────────
def load_pkl(db_dir: str) -> dict:
    """
    Returns:
        { name (str): mean_embedding (np.ndarray, 512-d) }
    """
    path = os.path.join(db_dir, "embeddings.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PKL file not found: {path}")
    with open(path, "rb") as f:           # "rb" = read binary
        data = pickle.load(f)
    return data


# ─────────────────────────────────────────────
# Load NPZ
# ─────────────────────────────────────────────
def load_npz(db_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        vectors : np.ndarray  shape (N, 512)
        labels  : np.ndarray  shape (N,)
    """
    path = os.path.join(db_dir, "embeddings.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found: {path}")
    data    = np.load(path, allow_pickle=True)
    vectors = data["vectors"]   # (N, 512)
    labels  = data["labels"]    # (N,)
    return vectors, labels


# ─────────────────────────────────────────────
# Print Helpers
# ─────────────────────────────────────────────
def print_pkl_summary(db: dict):
    print("\n" + "=" * 50)
    print("  PKL  —  embeddings.pkl")
    print("=" * 50)
    print(f"  Total persons : {len(db)}")
    print(f"  Type          : {type(db)}")
    print("-" * 50)
    for name, vec in db.items():
        print(f"  [{name}]")
        print(f"    shape  : {vec.shape}")
        print(f"    dtype  : {vec.dtype}")
        print(f"    min    : {vec.min():.6f}")
        print(f"    max    : {vec.max():.6f}")
        print(f"    norm   : {np.linalg.norm(vec):.6f}")
        print(f"    preview: {vec[:6]}  ...")
        print()


def print_npz_summary(vectors: np.ndarray, labels: np.ndarray):
    print("=" * 50)
    print("  NPZ  —  embeddings.npz")
    print("=" * 50)
    print(f"  vectors shape : {vectors.shape}   (N persons × 512 dims)")
    print(f"  labels  shape : {labels.shape}")
    print(f"  dtype         : {vectors.dtype}")
    print("-" * 50)
    for i, label in enumerate(labels):
        vec = vectors[i]
        print(f"  [{i}] {label}")
        print(f"    norm   : {np.linalg.norm(vec):.6f}")
        print(f"    preview: {vec[:6]}  ...")
        print()
    print("=" * 50)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Load and inspect the face vector DB.")
    parser.add_argument(
        "--db",
        default=DEFAULT_DB_DIR,
        help=f"Path to vector_db directory (default: {DEFAULT_DB_DIR})",
    )
    args = parser.parse_args()

    db_dir = os.path.abspath(args.db)
    print(f"[Vector DB] {db_dir}")

    # ── PKL ──
    pkl_db = load_pkl(db_dir)
    print_pkl_summary(pkl_db)

    # ── NPZ ──
    vectors, labels = load_npz(db_dir)
    print_npz_summary(vectors, labels)

    print(f"[Done]  {len(pkl_db)} person(s) loaded successfully.")


if __name__ == "__main__":
    main()

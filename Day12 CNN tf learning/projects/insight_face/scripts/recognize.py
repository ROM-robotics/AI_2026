"""
recognize.py — InsightFace Face Recognition Script
====================================================
Gallery ထဲမှ registered မျက်နှာများနှင့် query image ကို 
Cosine Similarity ဖြင့် 1:N matching လုပ်သည်။

Usage:
    # Gallery register (data/gallery/person_name/*.jpg)
    python recognize.py --mode register --gallery ../data/gallery

    # Recognize from image
    python recognize.py --mode recognize --query ../data/images/test.jpg --gallery ../data/gallery

    # Recognize from webcam
    python recognize.py --mode webcam --gallery ../data/gallery

Folder structure (gallery):
    data/gallery/
    ├── Alice/
    │   ├── alice_01.jpg
    │   └── alice_02.jpg
    └── Bob/
        ├── bob_01.jpg
        └── bob_02.jpg
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
MODEL_NAME       = "buffalo_l"
DET_SIZE         = (640, 640)
PROVIDERS        = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# scripts/ ၏ parent (insight_face/) ထဲက models/ ကို root အဖြစ် သုံးသည်
MODEL_ROOT       = os.path.join(os.path.dirname(__file__), "..", "models")
SIMILARITY_THRESH = 0.5             # Cosine similarity threshold
GALLERY_PKL      = "gallery_embeddings.pkl"

MATCH_COLOR   = (0, 255, 0)        # Green — matched
UNKNOWN_COLOR = (0, 0, 255)        # Red   — unknown
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# Initialize Model
# ─────────────────────────────────────────────
def load_model() -> FaceAnalysis:
    root = os.path.abspath(MODEL_ROOT)
    print(f"[Model Root] {root}")
    app = FaceAnalysis(name=MODEL_NAME, root=root, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print(f"[✓] Model loaded: {MODEL_NAME}")
    return app


# ─────────────────────────────────────────────
# Embedding Utilities
# ─────────────────────────────────────────────
def l2_normalize(emb: np.ndarray) -> np.ndarray:
    n = norm(emb)
    return emb / n if n > 0 else emb


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))   # assumes L2-normalized


def get_best_face_embedding(app: FaceAnalysis, img: np.ndarray):
    """Image မှ မျက်နှာ ရှာပြီး confidence အမြင့်ဆုံး face ၏ embedding return"""
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: f.det_score if f.det_score is not None else 0)
    return l2_normalize(best.embedding)


# ─────────────────────────────────────────────
# Gallery Operations
# ─────────────────────────────────────────────
def build_gallery(app: FaceAnalysis, gallery_dir: str) -> dict:
    """
    gallery_dir ထဲမှ person subfolder တစ်ခုချင်းစီ၏ images မှ
    embedding တွေကို တွက်ပြီး gallery dict return လုပ်သည်။
    """
    gallery = {}  # {name: [emb1, emb2, ...]}
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for person_name in os.listdir(gallery_dir):
        person_dir = os.path.join(gallery_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        embs = []
        for fname in os.listdir(person_dir):
            if os.path.splitext(fname)[-1].lower() not in img_exts:
                continue
            img_path = os.path.join(person_dir, fname)
            img      = cv2.imread(img_path)
            if img is None:
                continue
            emb = get_best_face_embedding(app, img)
            if emb is not None:
                embs.append(emb)
                print(f"  [+] {person_name}/{fname}")

        if embs:
            gallery[person_name] = embs
            print(f"  [✓] {person_name}: {len(embs)} embeddings registered")

    return gallery


def save_gallery(gallery: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(gallery, f)
    print(f"[Saved] Gallery → {path}")


def load_gallery(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[✗] Gallery not found: {path}  →  Run --mode register first.")
        return {}
    with open(path, "rb") as f:
        gallery = pickle.load(f)
    print(f"[✓] Gallery loaded: {len(gallery)} persons")
    return gallery


# ─────────────────────────────────────────────
# 1:N Matching
# ─────────────────────────────────────────────
def match_face(query_emb: np.ndarray, gallery: dict, threshold: float = SIMILARITY_THRESH):
    """
    Query embedding ကို gallery ထဲ person တိုင်းနဲ့ Cosine Similarity နှိုင်းယှဉ်သည်။
    Returns: (best_name, best_score) သို့မဟုတ် ("Unknown", score)
    """
    best_name  = "Unknown"
    best_score = -1.0

    for name, embs in gallery.items():
        # Person ၏ embedding တွေ average (mean embedding)
        mean_emb = l2_normalize(np.mean(embs, axis=0))
        sim      = cosine_similarity(query_emb, mean_emb)
        if sim > best_score:
            best_score = sim
            best_name  = name if sim >= threshold else "Unknown"

    return best_name, best_score


# ─────────────────────────────────────────────
# Draw Result
# ─────────────────────────────────────────────
def draw_result(frame: np.ndarray, faces: list, gallery: dict) -> np.ndarray:
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        emb  = l2_normalize(face.embedding)
        name, score = match_face(emb, gallery)
        matched = name != "Unknown"

        color = MATCH_COLOR if matched else UNKNOWN_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name}  [{score:.3f}]"
        cv2.putText(frame, label, (x1, y1 - 8), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if face.kps is not None:
            for kp in face.kps.astype(int):
                cv2.circle(frame, tuple(kp), 3, (0, 165, 255), -1)

    return frame


# ─────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────
def mode_register(app, gallery_dir):
    print(f"\n[Register] Building gallery from: {gallery_dir}")
    gallery = build_gallery(app, gallery_dir)
    if gallery:
        save_gallery(gallery, GALLERY_PKL)
    else:
        print("[✗] No faces found in gallery directory.")


def mode_recognize(app, query_path, gallery_dir):
    gallery = load_gallery(GALLERY_PKL)
    if not gallery:
        return

    frame = cv2.imread(query_path)
    if frame is None:
        print(f"[✗] Cannot read: {query_path}")
        return

    faces = app.get(frame)
    if not faces:
        print("[!] No faces detected.")
        return

    print(f"\n[Result] {os.path.basename(query_path)}  —  {len(faces)} face(s)")
    for i, face in enumerate(faces):
        emb  = l2_normalize(face.embedding)
        name, score = match_face(emb, gallery)
        print(f"  [{i}] → {name}  (score={score:.4f})")

    result = draw_result(frame.copy(), faces, gallery)
    out_path = os.path.splitext(query_path)[0] + "_recognized.jpg"
    cv2.imwrite(out_path, result)
    print(f"[Saved] {out_path}")

    cv2.imshow("InsightFace — Recognition", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def mode_webcam(app, gallery_dir):
    gallery = load_gallery(GALLERY_PKL)
    if not gallery:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[✗] Cannot open webcam.")
        return

    print("[Info] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces  = app.get(frame)
        result = draw_result(frame, faces, gallery)
        cv2.imshow("InsightFace — Recognition (Webcam)", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="InsightFace Face Recognition")
    parser.add_argument("--mode", choices=["register", "recognize", "webcam"],
                        default="webcam", help="Mode to run")
    parser.add_argument("--gallery", type=str, default="../data/gallery",
                        help="Gallery directory containing person subfolders")
    parser.add_argument("--query",   type=str, default=None,
                        help="Query image path (for --mode recognize)")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESH,
                        help=f"Cosine similarity threshold (default: {SIMILARITY_THRESH})")
    args = parser.parse_args()

    global SIMILARITY_THRESH
    SIMILARITY_THRESH = args.threshold

    app = load_model()

    if args.mode == "register":
        mode_register(app, args.gallery)
    elif args.mode == "recognize":
        if not args.query:
            print("[✗] --query path required for recognize mode.")
            return
        mode_recognize(app, args.query, args.gallery)
    elif args.mode == "webcam":
        mode_webcam(app, args.gallery)


if __name__ == "__main__":
    main()

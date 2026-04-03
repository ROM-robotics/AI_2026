"""
04live_inference_antispoof.py
==============================
Webcam မှ real-time face recognition + MiniFASNetV2 Anti-Spoofing။
vector_db/embeddings.pkl ထဲမှ vector များနှင့် Cosine Similarity ဖြင့် တိုက်စစ်သည်။
မျက်နှာ FAKE ဖြစ်ပါက recognition ကို bypass လုပ်ပြီး FAKE ဟုသာ ပြသသည်။

Usage:
    python 04live_inference_antispoof.py
    python 04live_inference_antispoof.py --db ../data/vector_db --threshold 0.5 --cam 0
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from numpy.linalg import norm
from insightface.app import FaceAnalysis

# project root ကို sys.path ထဲ ထည့်ပြီး scripts.liveness import သုံးသည်
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from scripts.liveness import LivenessDetector


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.join(SCRIPT_DIR, "..")

MODEL_NAME  = "buffalo_l"
DET_SIZE    = (640, 640)
PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]
MODEL_ROOT  = os.path.join(BASE_DIR, "models")

DEFAULT_DB_DIR    = os.path.join(BASE_DIR, "data", "vector_db")
SIMILARITY_THRESH = 0.5

FONT          = cv2.FONT_HERSHEY_SIMPLEX
MATCH_COLOR   = (0, 220, 0)    # Green  — matched + REAL
UNKNOWN_COLOR = (0, 165, 255)  # Orange — unknown + REAL
FAKE_COLOR    = (0, 0, 220)    # Red    — FAKE (spoof)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def l2_normalize(emb: np.ndarray) -> np.ndarray:
    n = norm(emb)
    return emb / n if n > 0 else emb


def load_model() -> FaceAnalysis:
    root = os.path.abspath(MODEL_ROOT)
    app  = FaceAnalysis(name=MODEL_NAME, root=root, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print(f"[✓] InsightFace model loaded: {MODEL_NAME}")
    return app


def load_vector_db(db_dir: str) -> dict:
    """
    Returns: { name (str): mean_embedding (np.ndarray 512-d) }
    """
    path = os.path.join(db_dir, "embeddings.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[✗] Vector DB not found: {path}\n"
            "     Run build_vector_db.py first."
        )
    with open(path, "rb") as f:
        db = pickle.load(f)
    print(f"[✓] Vector DB loaded: {len(db)} person(s)  →  {path}")
    return db


def match_face(query_emb: np.ndarray, db: dict, threshold: float):
    """
    query_emb ကို db ထဲ persons တိုင်းနှင့် Cosine Similarity နှိုင်းယှဉ်သည်။
    Returns: (name, score)
    """
    best_name  = "Unknown"
    best_score = -1.0

    for name, db_emb in db.items():
        sim = float(np.dot(query_emb, db_emb))   # L2-normalized → dot == cosine
        if sim > best_score:
            best_score = sim
            best_name  = name if sim >= threshold else "Unknown"

    return best_name, best_score


# ─────────────────────────────────────────────
# Draw
# ─────────────────────────────────────────────
def draw_faces(
    frame: np.ndarray,
    faces: list,
    db: dict,
    liveness_detector: LivenessDetector,
    threshold: float,
) -> np.ndarray:
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        # ── Step 1: Liveness check ──
        liv_label, liv_score = liveness_detector.predict(frame, face.bbox)
        is_real = (liv_label == "REAL")

        if is_real:
            # ── Step 2: Recognition (REAL face only) ──
            emb         = l2_normalize(face.embedding)
            name, score = match_face(emb, db, threshold)
            matched     = name != "Unknown"
            color       = MATCH_COLOR if matched else UNKNOWN_COLOR
            label       = f"{name}  [{score:.3f}]  REAL {liv_score:.2f}"
        else:
            # FAKE → recognition skip
            color = FAKE_COLOR
            label = f"FAKE  [{liv_score:.2f}]"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), FONT, 0.6, color, 2, cv2.LINE_AA)

        if face.kps is not None:
            for kp in face.kps.astype(int):
                cv2.circle(frame, tuple(kp), 3, (0, 165, 255), -1)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), FONT, 0.8, (0, 255, 255), 2)
    return frame


# ─────────────────────────────────────────────
# Webcam Loop
# ─────────────────────────────────────────────
def run_webcam(
    app: FaceAnalysis,
    db: dict,
    liveness_detector: LivenessDetector,
    cam_id: int,
    threshold: float,
):
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[✗] Cannot open camera: {cam_id}")
        return

    print("[Info] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Frame read failed.")
            break

        faces  = app.get(frame)
        result = draw_faces(frame, faces, db, liveness_detector, threshold)
        cv2.imshow("Live Face Recognition + Anti-Spoof", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Webcam face recognition with anti-spoofing.")
    parser.add_argument("--db",        default=DEFAULT_DB_DIR,    help="Path to vector_db directory")
    parser.add_argument("--threshold", default=SIMILARITY_THRESH, type=float, help="Cosine similarity threshold")
    parser.add_argument("--cam",       default=0,                 type=int,   help="Camera device index")
    args = parser.parse_args()

    app               = load_model()
    db                = load_vector_db(args.db)
    liveness_detector = LivenessDetector()

    print(f"[Info] Threshold : {args.threshold}")
    print(f"[Info] Camera    : {args.cam}")

    run_webcam(app, db, liveness_detector, args.cam, args.threshold)


if __name__ == "__main__":
    main()

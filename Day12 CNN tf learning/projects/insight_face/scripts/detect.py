"""
detect.py — InsightFace Face Detection Script
==============================================
Image / Video / Webcam မှ မျက်နှာများကို detect လုပ်ပြီး
Bounding Box, Landmarks, Age, Gender ကို output ထုတ်သည်။

Usage:
    # Image
    python detect.py --source ../data/images/test.jpg

    # Video file
    python detect.py --source ../data/videos/test.mp4

    # Webcam
    python detect.py --source 0

    # Save result
    python detect.py --source ../data/images/test.jpg --save
"""

import argparse
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME  = "buffalo_l"          # buffalo_l / buffalo_m / buffalo_s
DET_SIZE    = (640, 640)
PROVIDERS   = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# scripts/ ၏ parent (insight_face/) ထဲက models/ ကို root အဖြစ် သုံးသည်
# ~/.insightface/models/ ကို override လုပ် သည်
MODEL_ROOT  = os.path.join(os.path.dirname(__file__), "..", "models")

GENDER_MAP  = {1: "Male", 0: "Female"}
BOX_COLOR   = (0, 255, 0)          # Green
KPS_COLOR   = (0, 0, 255)          # Red
TEXT_COLOR  = (255, 255, 255)      # White
FONT        = cv2.FONT_HERSHEY_SIMPLEX


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
# Draw on Frame
# ─────────────────────────────────────────────
def draw_faces(frame: np.ndarray, faces: list) -> np.ndarray:
    for face in faces:
        # Bounding Box
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        # Age & Gender label
        age    = int(face.age) if face.age is not None else -1
        gender = GENDER_MAP.get(int(face.gender), "?") if face.gender is not None else "?"
        det_score = f"{face.det_score:.2f}" if face.det_score is not None else "?"
        label = f"{gender}, {age}y  [{det_score}]"
        cv2.putText(frame, label, (x1, y1 - 8), FONT, 0.55, TEXT_COLOR, 1, cv2.LINE_AA)

        # Landmarks (5 points)
        if face.kps is not None:
            for kp in face.kps.astype(int):
                cv2.circle(frame, tuple(kp), 3, KPS_COLOR, -1)

    # Face count
    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), FONT, 0.9, (0, 255, 255), 2)
    return frame


# ─────────────────────────────────────────────
# Process Single Image
# ─────────────────────────────────────────────
def process_image(app: FaceAnalysis, path: str, save: bool = False):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[✗] Cannot read image: {path}")
        return

    faces = app.get(frame)
    print(f"\n[Result] {os.path.basename(path)}")
    print(f"  Detected faces: {len(faces)}")
    for i, face in enumerate(faces):
        age    = int(face.age) if face.age is not None else -1
        gender = GENDER_MAP.get(int(face.gender), "?") if face.gender is not None else "?"
        score  = f"{face.det_score:.4f}" if face.det_score is not None else "?"
        emb    = face.embedding.shape if face.embedding is not None else None
        print(f"  [{i}] Gender={gender}  Age={age}  Score={score}  Embedding={emb}")

    result = draw_faces(frame.copy(), faces)

    if save:
        out_path = os.path.splitext(path)[0] + "_detected.jpg"
        cv2.imwrite(out_path, result)
        print(f"[Saved] {out_path}")
    else:
        cv2.imshow("InsightFace — Detection", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Process Video / Webcam
# ─────────────────────────────────────────────
def process_video(app: FaceAnalysis, source, save: bool = False):
    # source: file path string or int (webcam index)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[✗] Cannot open source: {source}")
        return

    writer = None
    if save and isinstance(source, str):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.splitext(source)[0] + "_detected.mp4"
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"[Recording] → {out_path}")

    print("[Info] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces  = app.get(frame)
        result = draw_faces(frame, faces)

        if writer:
            writer.write(result)

        cv2.imshow("InsightFace — Detection", result)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print("[Saved] Video written.")
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="InsightFace Face Detection")
    parser.add_argument("--source", type=str, default="0",
                        help="Image/video path or webcam index (default: 0)")
    parser.add_argument("--save", action="store_true",
                        help="Save result instead of displaying")
    args = parser.parse_args()

    app = load_model()

    # Source type detection
    source = args.source
    img_exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    ext        = os.path.splitext(source)[-1].lower()

    if ext in img_exts:
        process_image(app, source, save=args.save)
    elif ext in video_exts or source.isdigit():
        source = int(source) if source.isdigit() else source
        process_video(app, source, save=args.save)
    else:
        print(f"[✗] Unsupported source: {source}")


if __name__ == "__main__":
    main()

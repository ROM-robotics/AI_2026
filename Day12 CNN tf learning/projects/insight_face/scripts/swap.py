"""
swap.py — InsightFace Face Swap Script
=======================================
Source မျက်နှာကို Target image / video ထဲမှ မျက်နှာနဲ့ swap လုပ်သည်။
insightface ၏ inswapper_128.onnx model ကို သုံးသည်။

Requirements:
    - inswapper_128.onnx → ../models/ ထဲမှာ ထားပါ
      Download: https://github.com/deepinsight/insightface/releases
    - onnxruntime-gpu (သို့) onnxruntime

Usage:
    # Image swap
    python swap.py --source ../data/images/source.jpg --target ../data/images/target.jpg

    # Video swap
    python swap.py --source ../data/images/source.jpg --target ../data/videos/target.mp4

    # Webcam swap (real-time)
    python swap.py --source ../data/images/source.jpg --target 0

    # Swap all faces (default: swap face index 0 only)
    python swap.py --source ../data/images/source.jpg --target ../data/images/target.jpg --all-faces
"""

import argparse
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_NAME    = "buffalo_l"
DET_SIZE      = (640, 640)
PROVIDERS     = ["CUDAExecutionProvider", "CPUExecutionProvider"]

# scripts/ ၏ parent (insight_face/) ထဲက models/ ကို root အဖြစ် သုံးသည်
MODEL_ROOT    = os.path.join(os.path.dirname(__file__), "..", "models")
SWAPPER_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "inswapper_128.onnx")
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ─────────────────────────────────────────────
# Initialize Models
# ─────────────────────────────────────────────
def load_models():
    # Face analyzer
    root = os.path.abspath(MODEL_ROOT)
    print(f"[Model Root] {root}")
    app = FaceAnalysis(name=MODEL_NAME, root=root, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    print(f"[✓] FaceAnalysis loaded: {MODEL_NAME}")

    # Swapper model
    swapper_path = os.path.abspath(SWAPPER_MODEL)
    if not os.path.exists(swapper_path):
        print(f"[✗] Swapper model not found: {swapper_path}")
        print("    Download inswapper_128.onnx and place it in ../models/")
        print("    https://github.com/deepinsight/insightface/releases")
        return app, None

    swapper = get_model(swapper_path, providers=PROVIDERS)
    print(f"[✓] Swapper model loaded: inswapper_128.onnx")
    return app, swapper


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def get_best_face(app: FaceAnalysis, img: np.ndarray):
    """Confidence 가장 높은 face return"""
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score if f.det_score is not None else 0)


def swap_faces(app, swapper, frame: np.ndarray, source_face, swap_all: bool = False) -> np.ndarray:
    """
    Frame ထဲ detect ရသော မျက်နှာတွေကို source_face နဲ့ swap လုပ်သည်။
    swap_all=True → မျက်နှာ အားလုံး swap
    swap_all=False → confidence အမြင့်ဆုံး မျက်နှာ တစ်ခုသာ swap
    """
    if swapper is None:
        return frame

    faces = app.get(frame)
    if not faces:
        return frame

    target_faces = faces if swap_all else [max(faces, key=lambda f: f.det_score or 0)]
    result = frame.copy()

    for target_face in target_faces:
        result = swapper.get(result, target_face, source_face, paste_back=True)

    return result


# ─────────────────────────────────────────────
# Process Image
# ─────────────────────────────────────────────
def process_image(app, swapper, source_path: str, target_path: str, swap_all: bool):
    # Source face
    src_img  = cv2.imread(source_path)
    if src_img is None:
        print(f"[✗] Cannot read source: {source_path}")
        return
    source_face = get_best_face(app, src_img)
    if source_face is None:
        print("[✗] No face detected in source image.")
        return
    print(f"[✓] Source face detected  (score={source_face.det_score:.3f})")

    # Target image
    tgt_img = cv2.imread(target_path)
    if tgt_img is None:
        print(f"[✗] Cannot read target: {target_path}")
        return

    # Swap
    result   = swap_faces(app, swapper, tgt_img, source_face, swap_all)
    out_path = os.path.splitext(target_path)[0] + "_swapped.jpg"
    cv2.imwrite(out_path, result)
    print(f"[Saved] {out_path}")

    # Side-by-side preview
    h = max(src_img.shape[0], tgt_img.shape[0], result.shape[0])

    def resize_h(img, h):
        r = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), h))

    preview = np.hstack([
        resize_h(src_img,  h),
        resize_h(tgt_img,  h),
        resize_h(result,   h),
    ])
    labels = ["Source", "Target", "Swapped"]
    w_each = preview.shape[1] // 3
    for i, lbl in enumerate(labels):
        cv2.putText(preview, lbl, (i * w_each + 10, 30),
                    FONT, 1.0, (0, 255, 255), 2)

    cv2.imshow("InsightFace — Face Swap", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
# Process Video / Webcam
# ─────────────────────────────────────────────
def process_video(app, swapper, source_path: str, target, swap_all: bool):
    # Source face
    src_img = cv2.imread(source_path)
    if src_img is None:
        print(f"[✗] Cannot read source: {source_path}")
        return
    source_face = get_best_face(app, src_img)
    if source_face is None:
        print("[✗] No face detected in source image.")
        return
    print(f"[✓] Source face detected  (score={source_face.det_score:.3f})")

    is_webcam = isinstance(target, int)
    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        print(f"[✗] Cannot open: {target}")
        return

    writer = None
    if not is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.path.splitext(str(target))[0] + "_swapped.mp4"
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        print(f"[Recording] → {out_path}")

    print("[Info] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = swap_faces(app, swapper, frame, source_face, swap_all)

        if writer:
            writer.write(result)

        cv2.imshow("InsightFace — Face Swap", result)
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
    parser = argparse.ArgumentParser(description="InsightFace Face Swap")
    parser.add_argument("--source", type=str, required=True,
                        help="Source face image path (whose face to use)")
    parser.add_argument("--target", type=str, required=True,
                        help="Target image / video path or webcam index (0)")
    parser.add_argument("--all-faces", action="store_true",
                        help="Swap all detected faces (default: highest-confidence face only)")
    args = parser.parse_args()

    app, swapper = load_models()
    if swapper is None:
        return

    img_exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    ext        = os.path.splitext(args.target)[-1].lower()

    if ext in img_exts:
        process_image(app, swapper, args.source, args.target, args.all_faces)
    elif ext in video_exts or args.target.isdigit():
        target = int(args.target) if args.target.isdigit() else args.target
        process_video(app, swapper, args.source, target, args.all_faces)
    else:
        print(f"[✗] Unsupported target: {args.target}")


if __name__ == "__main__":
    main()

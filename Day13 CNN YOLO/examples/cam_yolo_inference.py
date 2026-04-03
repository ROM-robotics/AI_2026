import cv2
from ultralytics import YOLO
import os
import time

# Load YOLOv8 model
model = YOLO("best.pt")
model.fuse()

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Optional: set fixed resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create directory to save frames
os.makedirs("captured_frames", exist_ok=True)

# Create a resizable window
window_name = "YOLOv8 Live Camera"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Store original dimensions
    h, w = frame.shape[:2]

    # Resize frame for YOLO input
    new_w = 640
    new_h = int(h * new_w / w)
    resized = cv2.resize(frame, (new_w, new_h))

    # Run YOLO detection
    results = model(resized, conf=0.3, device='cuda' if model.device.type == 'cuda' else 'cpu')

    # Create fresh copy for drawing
    annotated = frame.copy()

    # Extract data
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    # Class names
    class_names = model.names

    # Scale from resized to original frame
    scale_x = w / new_w
    scale_y = h / new_h

    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

        # Choose label
        if score < 0.5:
            label = "unknown"
            color = (0, 0, 255)  # red
        else:
            label = class_names.get(cls, str(cls))
            color = (0, 255, 0)  # green

        # Draw box and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    # Save annotated frame if detection exists
    if len(boxes) > 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(f"captured_frames/detected_{timestamp}.jpg", annotated)

    # Show the annotated frame
    cv2.imshow(window_name, annotated)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()

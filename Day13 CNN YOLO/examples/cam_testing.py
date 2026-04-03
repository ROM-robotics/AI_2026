import cv2

# Open default camera
cap = cv2.VideoCapture(0)

# # Set camera resolution to 4K
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# # Check if the resolution was set correctly
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(f"✅ Camera resolution set to {width}x{height}")

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame (stream end?). Exiting ...")
        break

    # Flip frame horizontally (optional)
    frame = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow('Test', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()

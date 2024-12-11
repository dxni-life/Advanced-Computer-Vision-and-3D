import torch
import cv2
from pathlib import Path

# Define paths
yolo_repo_path = Path("/Users/dani/yolov5")  # Path to YOLOv5 repository
model_path = "/Users/dani/best.pt"


# Load the YOLOv5 model
print("Loading the YOLOv5 model...")
model = torch.hub.load(yolo_repo_path.as_posix(), 'custom', path=model_path, source='local')

# Open the webcam
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()
print("Webcam initialized successfully. Starting real-time detection...")

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Perform inference on the frame
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Get detections (x1, y1, x2, y2, confidence, class)

    # Iterate over detections
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)

        # Calculate the center of the bounding box (landmark location)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw a dot at the center of the detection
        cv2.circle(frame, (cx, cy), radius=5, color=(0, 255, 0), thickness=-10)  # Green dot

        # Optionally, display the class name (landmark type)
        label = f"Class {int(cls)}"
        cv2.putText(frame, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display the frame with detections
    cv2.imshow("Real-Time Landmark Detection", frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
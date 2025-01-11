import cv2
from PIL import Image
from util import get_limits

yellow = [0, 0, 225]  # yellow in BGR colorspace
cap = cv2.VideoCapture(0)  # Try 0, 1, 2 based on available cameras

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Failed to capture frame. Reinitializing...")
        cap.release()
        cap = cv2.VideoCapture(0)  # Try different indices
        continue

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

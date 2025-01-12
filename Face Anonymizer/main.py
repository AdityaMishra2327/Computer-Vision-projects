import os
import argparse
import cv2
import mediapipe as mp


def process_img(img, face_detection):
    """Process an image to detect and blur faces."""
    H, W, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur detected faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img


def main():
    parser = argparse.ArgumentParser(description="Face detection and blurring tool.")
    parser.add_argument("--mode", default="webcam", choices=["image", "video", "webcam"], help="Mode of operation")
    parser.add_argument("--filePath", default="./data/testVideo.mp4", help="Path to image or video file")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if args.mode == "image":
            # Read image
            if not os.path.exists(args.filePath):
                print(f"Error: File '{args.filePath}' not found.")
                return

            img = cv2.imread(args.filePath)
            img = process_img(img, face_detection)

            # Save processed image
            output_path = os.path.join(output_dir, 'output.png')
            cv2.imwrite(output_path, img)
            print(f"Processed image saved to: {output_path}")

        elif args.mode == "video":
            # Process video file
            if not os.path.exists(args.filePath):
                print(f"Error: File '{args.filePath}' not found.")
                return

            cap = cv2.VideoCapture(args.filePath)
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to read video file.")
                cap.release()
                return

            output_video_path = os.path.join(output_dir, 'output.mp4')
            output_video = cv2.VideoWriter(
                output_video_path,
                cv2.VideoWriter_fourcc(*'MP4V'),
                25,
                (frame.shape[1], frame.shape[0])
            )

            while ret:
                frame = process_img(frame, face_detection)
                output_video.write(frame)
                ret, frame = cap.read()

            cap.release()
            output_video.release()
            print(f"Processed video saved to: {output_video_path}")

        elif args.mode == "webcam":
            # Process webcam stream
            cap = cv2.VideoCapture(0)  # Change index to select a different camera
            if not cap.isOpened():
                print("Error: Unable to access the webcam.")
                return

            print("Press 'q' to quit.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Unable to capture frame from webcam.")
                    break

                frame = process_img(frame, face_detection)
                cv2.imshow('Webcam Feed', frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

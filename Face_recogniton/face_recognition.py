import cv2
import numpy as np
import os
from datetime import datetime


class FaceRecognizer:
    def __init__(self):
        # Load the face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.labels = {}
        self.label_ids = {}
        self.current_id = 0

    def prepare_training_data(self, person1_path, person2_path):
        """Prepare training data from images"""
        training_data = []
        labels = []

        def process_person_directory(directory):
            if not os.path.exists(directory):
                print(f"Directory not found: {directory}")
                return

            person_name = os.path.basename(directory)
            if person_name not in self.label_ids:
                self.label_ids[person_name] = self.current_id
                self.labels[self.current_id] = person_name
                self.current_id += 1

            label_id = self.label_ids[person_name]

            for img_name in os.listdir(directory):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(directory, img_name)
                    try:
                        # Read image in grayscale
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            print(f"Error loading image: {img_path}")
                            continue

                        # Detect faces
                        faces = self.face_cascade.detectMultiScale(
                            img,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 30)
                        )

                        for (x, y, w, h) in faces:
                            roi = img[y:y + h, x:x + w]
                            roi = cv2.resize(roi, (150, 150))
                            training_data.append(roi)
                            labels.append(label_id)
                            print(f"Processed: {img_name}")

                    except Exception as e:
                        print(f"Error processing {img_name}: {str(e)}")

        # Process both directories
        process_person_directory(person1_path)
        process_person_directory(person2_path)

        return training_data, labels

    def train(self, person1_path, person2_path):
        """Train the face recognizer"""
        print("\nPreparing training data...")
        training_data, labels = self.prepare_training_data(person1_path, person2_path)

        if not training_data:
            print("No faces detected in training images!")
            return False

        print(f"\nTraining with {len(training_data)} images...")
        self.recognizer.train(training_data, np.array(labels))
        print("Training complete!")
        return True

    def run_recognition(self):
        """Run real-time face recognition"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\nStarting face recognition... Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Process each face
            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (150, 150))

                # Predict
                try:
                    label_id, confidence = self.recognizer.predict(roi)
                    name = self.labels.get(label_id, "Unknown")

                    # Convert confidence to percentage (lower is better in LBPH)
                    confidence = round(100 - confidence, 2)

                    # Draw rectangle around face
                    color = (0, 255, 0) if confidence > 50 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Put name and confidence
                    text = f"{name} ({confidence}%)"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                except Exception as e:
                    print(f"Prediction error: {str(e)}")

            # Display frame
            cv2.imshow('Face Recognition', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    # Create recognizer
    recognizer = FaceRecognizer()

    # Define paths
    base_dir = "training_images"
    person1_path = os.path.join(base_dir, "person1")
    person2_path = os.path.join(base_dir, "person2")

    # Check/create directories
    for path in [base_dir, person1_path, person2_path]:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

    # Check for training images
    if not any(os.scandir(person1_path)) or not any(os.scandir(person2_path)):
        print("\nError: Training images not found!")
        print("\nPlease add training images to:")
        print(f"1. {person1_path}")
        print(f"2. {person2_path}")
        print("\nEach directory should contain several clear face images of the person.")
        return

    # Train and run
    if recognizer.train(person1_path, person2_path):
        recognizer.run_recognition()


if __name__ == "__main__":
    main()
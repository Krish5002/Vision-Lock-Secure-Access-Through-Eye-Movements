import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def get_images_and_labels():
    dataset_path = "dataset"
    face_samples, ids = [], []

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("❌ No user data found! Register a user first.")
        return [], []

    for user_folder in os.listdir(dataset_path):
        user_id = int(user_folder.split("_")[1])  # Extract User ID
        folder_path = os.path.join(dataset_path, user_folder)

        for image in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image)
            gray_img = Image.open(img_path).convert("L")
            img_numpy = np.array(gray_img, "uint8")
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(user_id)

    return face_samples, ids

def train_model():
    faces, ids = get_images_and_labels()
    
    if len(faces) == 0:
        print("❌ No training data found! Register a user first.")
        return
    
    recognizer.train(faces, np.array(ids))
    os.makedirs("trainer", exist_ok=True)  # Ensure trainer directory exists
    recognizer.write("trainer/trainer.yml")
    print("✅ Training complete! Model is updated.")
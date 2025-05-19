import cv2
import numpy as np
import json
import dlib
import random
import time
import os
from tkinter import messagebox

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
trainer_path = "trainer/trainer.yml"
os.makedirs("trainer", exist_ok=True)  # Ensure trainer directory exists

# Load user database
users_path = "database.json"
if os.path.exists(users_path):
    with open(users_path, "r") as f:
        try:
            users = json.load(f)
        except json.JSONDecodeError:
            users = {}
else:
    users = {}

# Load Haar Cascade
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise FileNotFoundError("Haar cascade file not loaded properly")
except Exception as e:
    raise RuntimeError(f"Failed to load Haar cascade: {str(e)}")

# Load dlib shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
except Exception as e:
    raise RuntimeError(f"Failed to load dlib models: {str(e)}")

# Define possible eye gestures
GESTURES = ["Blink", "Look Left", "Look Right"]

def initialize_recognizer():
    """Initialize the face recognizer with error handling"""
    global recognizer
    try:
        if os.path.exists(trainer_path) and os.path.getsize(trainer_path) > 0:
            recognizer.read(trainer_path)
            return True
        return False
    except Exception as e:
        print(f"Error loading trainer: {str(e)}")
        return False

# Initialize the recognizer when module loads
model_loaded = initialize_recognizer()


def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR) for blink detection."""
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))  # Vertical distance
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))  # Vertical distance
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))  # Horizontal distance
    return (A + B) / (2.0 * C)

def get_iris_position(eye_region):
    """Find the darkest spot in the eye (pupil) to detect iris position."""
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours (detect dark region)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find largest contour (iris)
        iris_contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(iris_contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return cx, cy  # Return X, Y position of iris
    return None, None

def detect_eye_movement(landmarks, gesture, frame):
    """Detect eye movement using iris tracking and plot iris."""
    
    # Extract eye landmarks
    left_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye_pts = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    
    # Convert points to numpy arrays
    left_eye_pts = np.array(left_eye_pts, dtype=np.int32)
    right_eye_pts = np.array(right_eye_pts, dtype=np.int32)
    
    # Get bounding rectangles for both eyes
    x1, y1, w1, h1 = cv2.boundingRect(left_eye_pts)
    x2, y2, w2, h2 = cv2.boundingRect(right_eye_pts)
    
    # Extract eye regions from the frame
    left_eye_region = frame[y1:y1+h1, x1:x1+w1]
    right_eye_region = frame[y2:y2+h2, x2:x2+w2]
    
    # Find iris positions in both eyes
    left_iris_x, left_iris_y = get_iris_position(left_eye_region)
    right_iris_x, right_iris_y = get_iris_position(right_eye_region)
    
    if left_iris_x is None or right_iris_x is None:
        return False  # Couldn't detect iris properly

    # Normalize iris position relative to eye width
    left_eye_shift = left_iris_x / w1
    right_eye_shift = right_iris_x / w2

    # Plot iris position as a dot
    cv2.circle(frame, (x1 + left_iris_x, y1 + left_iris_y), 3, (255, 0, 0), -1)  # Blue dot for left eye
    cv2.circle(frame, (x2 + right_iris_x, y2 + right_iris_y), 3, (255, 0, 0), -1)  # Blue dot for right eye

    # Debugging display
    cv2.putText(frame, f"L: {left_eye_shift:.2f} | R: {right_eye_shift:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Blink detection
    left_ear = eye_aspect_ratio(left_eye_pts)
    right_ear = eye_aspect_ratio(right_eye_pts)
    avg_ear = (left_ear + right_ear) / 2.0

    if gesture == "Blink":
        return avg_ear < 0.2  # Blink detected

    # Eye movement detection thresholds
    if gesture == "Look Left" and left_eye_shift > 0.7 and right_eye_shift > 0.7:
        return True  # Eyes moved right (looking left)

    if gesture == "Look Right" and left_eye_shift < 0.3 and right_eye_shift < 0.3:
        return True  # Eyes moved left (looking right)

    return False

def recognize_face(user_id=None):
    if not model_loaded:
        return "failure", None
    
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    correct_gesture = False
    requested_gesture = random.choice(GESTURES)
    recognized = False
    username = None

    while time.time() - start_time < 5:  # Run for 5 seconds
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_roi = gray[y:y+h, x:x+w]

            id, confidence = recognizer.predict(face_roi)

            # If user_id is provided (from login), only match that specific user
            if user_id is not None:
                if str(id) == str(user_id) and confidence < 80:
                    recognized = True
                    username = users.get(str(id), {}).get("name", f"User {id}")
                    box_color = (0, 255, 0)  # Green
                    label = f"{username} ({round(confidence, 2)})"
                    
                    if detect_eye_movement(landmarks, requested_gesture, frame):
                        correct_gesture = True
                else:
                    box_color = (0, 0, 255)  # Red
                    label = "Unauthorized"
            else:
                # Original behavior when no user_id is provided
                if confidence < 70:
                    recognized = True
                    username = users.get(str(id), {}).get("name", f"User {id}")
                    box_color = (0, 255, 0)  # Green
                    label = f"{username} ({round(confidence, 2)})"
                    
                    if detect_eye_movement(landmarks, requested_gesture, frame):
                        correct_gesture = True
                else:
                    box_color = (0, 0, 255)  # Red
                    label = "Unknown"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            cv2.putText(frame, f"Do: {requested_gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Face Authentication", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return results
    if recognized:
        if correct_gesture:
            return "success", username
        else:
            return "idle", username
    return "failure", None

if __name__ == "__main__":
    # Maintain original standalone behavior
    status, username = recognize_face()
    if status == "success":
        messagebox.showinfo("Success", f"✅ Welcome {username}!")
    elif status == "idle":
        messagebox.showwarning("Idle", "⏳ User is idle!")
    else:
        messagebox.showerror("Error", "⛔ Unknown user detected!")
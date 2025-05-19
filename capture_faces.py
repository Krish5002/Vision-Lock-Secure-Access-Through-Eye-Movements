import cv2
import os

def capture_face_images(user_id, save_path="dataset", num_images=20):
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    user_folder = os.path.join(save_path, f"user_{user_id}")

    # Check if user already exists
    if os.path.exists(user_folder):
        print(f"⚠️ User {user_id} is already registered! Skipping registration.")
        return False

    os.makedirs(user_folder)

    count = 0
    while count < num_images:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            file_name = f"{user_folder}/{count}.jpg"
            cv2.imwrite(file_name, gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Capturing Faces - Press Q to Quit", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

    if count > 0:
        print(f"✅ {count} images saved for User ID: {user_id}")
        return True
    else:
        print("❌ No images captured. Try again.")
        return False
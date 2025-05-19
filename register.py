import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if face_cascade.empty():
    raise FileNotFoundError("⚠️ Haarcascade file not found!")

def register_face(user_id):
    cap = cv2.VideoCapture(0)
    count = 0
    save_path = f"dataset/user_{user_id}"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    while count < 40:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"{save_path}/{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Register Face", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ {count} images saved for User ID: {user_id}")
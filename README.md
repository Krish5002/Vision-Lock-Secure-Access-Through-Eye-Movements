# Vision-Lock-Secure-Access-Through-Eye-Movements
This is a face authentication system built using Python, OpenCV, and dlib, enhanced with eye gesture recognition like blinking, looking left, and looking right. The system includes secure user registration and login with encrypted passwords, facial recognition, and real-time iris tracking for additional security verification.

**📌 Features**

    ✅ User Registration with encrypted password and face capture

    🔒 Secure Login with UserID and password verification

    🧠 Face recognition using LBPH algorithm

    👁️ Random Eye Gesture Verification (Blink, Look Left, Look Right)

    🎯 Trained model for accurate face recognition

    💾 User data stored in database.json

    🧪 Real-time feedback and error handling using Tkinter GUI

**🛠️Technologies Used**

    Python 3.x

    OpenCV

    dlib (for face landmarks detection)

    Tkinter (for GUI)

    NumPy

    JSON (for data storage)

**Project Structure**

FaceAuthSystem/

├── main.py                # GUI with Registration & Login

├── Recognizer.py          # Face recognition and eye gesture detection

├── Train_model.py         # Trains model using registered faces

├── Pupil.py               # Iris detection logic

├── database.json          # Stores user details (UserID, name, hashed password)

├── shape_predictor_68_face_landmarks.dat # Required dlib model

├── trainer/
│   └── trainer.yml        # Trained face recognition model

└── haarcascade_frontalface_default.xml   # Face detection model (OpenCV)

**🚀 How to Run**


**1. Install dependencies**

      pip install opencv-python opencv-contrib-python dlib numpy Pillow

**2. Download dlib shape predictor**

      Download shape_predictor_68_face_landmarks.dat from:
      https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
      Place it in your project directory.

**3. Run the App**

      python main.py

**🧑‍💻 User Flow
🔐 Registration**

    Enter User ID, Name, and Password

    Face is captured and model is trained

    Password is hashed before saving to database.json

**🔓 Login**

    Enter valid User ID and Password

    Face recognition is performed

    System asks for random eye gesture (Blink, Look Left, or Look Right)

    Success only if face matches and correct gesture is performed

**🔐 Security Features**

    Passwords are stored as SHA-256 hashes

    Users are authenticated using both face data and live eye gesture

    Unrecognized users or incorrect gestures are blocked

**🧪 Sample Output Messages**

    ✅ "Welcome [Username]!" – Face recognized and gesture correct

    ⏳ "User is idle!" – Face matched, but wrong/missing gesture

    ⛔ "Face recognition failed!" – Unknown or mismatched face

**📣 Notes**

    Make sure your webcam is connected and working.

    Ensure proper lighting for accurate detection.

    The first time registration should be done before login.

**📜 License**

This project is open-source and free to use for academic or personal purposes.

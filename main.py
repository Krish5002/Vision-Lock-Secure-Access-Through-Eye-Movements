import tkinter as tk
from tkinter import messagebox
import os
import json
import hashlib
from register import register_face
from train_model import train_model
from recognizer import recognize_face, model_loaded

# Load user database
user_db = "database.json"
if not os.path.exists(user_db):
    with open(user_db, "w") as f:
        json.dump({}, f)

# GUI Window
root = tk.Tk()
root.title("Face Authentication System")
root.geometry("400x400")

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def register():
    user_id = entry_id.get()
    username = entry_name.get()
    password = entry_password.get()
    
    if not user_id or not username or not password:
        messagebox.showerror("Error", "All fields are required!")
        return
    
    # Load current users
    with open(user_db, "r") as f:
        users = json.load(f)
    
    if user_id in users:
        messagebox.showerror("Error", "User ID already exists!")
        return
    
    # Store user details with hashed password
    users[user_id] = {
        "name": username,
        "password": hash_password(password)
    }

    # Save updated user database
    with open(user_db, "w") as f:
        json.dump(users, f)

    messagebox.showinfo("Info", f"User '{username}' registered with ID {user_id}. Capture your face now.")
    
    register_face(user_id)  # Capture face
    train_model()  # Train model immediately after capturing
    messagebox.showinfo("Success", "Face Registered & Model Trained Successfully!")

def login():
    user_id = entry_login_id.get()
    password = entry_login_password.get()
    
    if not user_id or not password:
        messagebox.showerror("Error", "Both User ID and Password are required!")
        return
    
    # Load users
    with open(user_db, "r") as f:
        users = json.load(f)
    
    if user_id not in users:
        messagebox.showerror("Error", "Invalid User ID!")
        return
    
    hashed_password = hash_password(password)
    if users[user_id]["password"] != hashed_password:
        messagebox.showerror("Error", "Invalid Password!")
        return
    
    if not model_loaded:
        messagebox.showerror("Error", "Face recognition model not trained yet!")
        return
    
    username = users[user_id]["name"]
    status, _ = recognize_face(user_id)
    
    if status == "success":
        messagebox.showinfo("Welcome", f"Welcome {username}!")
    elif status == "idle":
        messagebox.showwarning("Warning", "User is idle!")
    else:
        messagebox.showerror("Error", "Face recognition failed!")

# Registration Frame
reg_frame = tk.LabelFrame(root, text="Registration", padx=10, pady=10)
reg_frame.pack(pady=10)

tk.Label(reg_frame, text="User ID:").grid(row=0, column=0, sticky="e")
entry_id = tk.Entry(reg_frame)
entry_id.grid(row=0, column=1, pady=5)

tk.Label(reg_frame, text="Username:").grid(row=1, column=0, sticky="e")
entry_name = tk.Entry(reg_frame)
entry_name.grid(row=1, column=1, pady=5)

tk.Label(reg_frame, text="Password:").grid(row=2, column=0, sticky="e")
entry_password = tk.Entry(reg_frame, show="*")
entry_password.grid(row=2, column=1, pady=5)

tk.Button(reg_frame, text="Register", command=register).grid(row=3, columnspan=2, pady=10)

# Login Frame
login_frame = tk.LabelFrame(root, text="Login", padx=10, pady=10)
login_frame.pack(pady=10)

tk.Label(login_frame, text="User ID:").grid(row=0, column=0, sticky="e")
entry_login_id = tk.Entry(login_frame)
entry_login_id.grid(row=0, column=1, pady=5)

tk.Label(login_frame, text="Password:").grid(row=1, column=0, sticky="e")
entry_login_password = tk.Entry(login_frame, show="*")
entry_login_password.grid(row=1, column=1, pady=5)

tk.Button(login_frame, text="Login", command=login).grid(row=2, columnspan=2, pady=10)

root.mainloop()
import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import asyncio
import json
import pyrebase

firebase_config = {
    "apiKey": "AIzaSyAMRydlD04Ui3p7IEIDDpEjLJjnxgDoDsQ",
    "authDomain": "nodell-c25fc.firebaseapp.com",
    "databaseURL": "https://nodell-c25fc-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "nodell-c25fc.appspot.com"
}
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

settings = {
    "pixel_to_cm": 1.5,
    "roi": [[456, 570], [168, 448], [332, 297], [500, 333]]
}

def update_settings():
    try:
        px = float(pixel_entry.get())
        roi = json.loads(roi_entry.get())
        settings["pixel_to_cm"] = px
        settings["roi"] = roi
        db.child("settings").set(settings)
        messagebox.showinfo("✅ Success", "Settings updated to Firebase")
    except Exception as e:
        messagebox.showerror("❌ Error", f"Gagal: {e}")

def run_async_func(func, *args):
    threading.Thread(target=lambda: asyncio.run(func(*args))).start()

def select_file():
    path = filedialog.askopenfilename(filetypes=[("Video/Images", "*.mp4 *.avi *.jpg *.png")])
    if path:
        import client_file
        run_async_func(client_file.process_file, path)

def start_camera():
    import client_camera
    run_async_func(client_camera.send_video)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🚦 AI Traffic Controller")
app.geometry("520x420")

ctk.CTkLabel(app, text="🚦 AI Traffic Controller", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)

ctk.CTkLabel(app, text="Pixel to CM:", anchor="w").pack(fill="x", padx=30)
pixel_entry = ctk.CTkEntry(app, placeholder_text="e.g. 1.5")
pixel_entry.insert(0, str(settings["pixel_to_cm"]))
pixel_entry.pack(fill="x", padx=30, pady=(5, 15))

ctk.CTkLabel(app, text="ROI (JSON array of points):", anchor="w").pack(fill="x", padx=30)
roi_entry = ctk.CTkEntry(app, placeholder_text="e.g. [[x1,y1],[x2,y2],...]")
roi_entry.insert(0, json.dumps(settings["roi"]))
roi_entry.pack(fill="x", padx=30, pady=(5, 20))

update_btn = ctk.CTkButton(app, text="💾 Update Firebase", command=update_settings, corner_radius=12)
update_btn.pack(pady=5, padx=30, fill="x")

btn_frame = ctk.CTkFrame(app, fg_color="transparent")
btn_frame.pack(pady=25)

cam_btn = ctk.CTkButton(btn_frame, text="🎥 Use Camera", command=start_camera, width=180, height=40, corner_radius=15)
file_btn = ctk.CTkButton(btn_frame, text="📁 Use File", command=select_file, width=180, height=40, corner_radius=15)

cam_btn.grid(row=0, column=0, padx=15)
file_btn.grid(row=0, column=1, padx=15)

ctk.CTkLabel(app, text="Made with ❤️  + YOLOv8 + Firebase", font=ctk.CTkFont(size=10)).pack(pady=10)

app.mainloop()

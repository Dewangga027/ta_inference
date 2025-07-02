from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
import pyrebase
import json

firebase_config = {
    "apiKey": "AIzaSyAMRydlD04Ui3p7IEIDDpEjLJjnxgDoDsQ",
    "authDomain": "nodell-c25fc.firebaseapp.com",
    "databaseURL": "https://nodell-c25fc-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "nodell-c25fc.appspot.com"
}
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()
torch.backends.cudnn.benchmark = True

def get_pixel_to_cm():
    return float(db.child("settings").child("pixel_to_cm").get().val() or 1.0)

def get_roi():
    roi = db.child("settings").child("roi").get().val()
    if not roi:
        return np.array([[0, 0]])
    return np.array(roi, dtype=np.int32)

def resize_frame(frame, target_width=480):
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame, (w, h), (w, h)
    scale = target_width / w
    new_w = target_width
    new_h = int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, (w, h), (new_w, new_h)

def resize_roi(roi, original_size, new_size):
    orig_w, orig_h = original_size
    new_w, new_h = new_size
    scale_x = new_w / 640
    scale_y = new_h / 640
    return np.array([[int(x * scale_x), int(y * scale_y)] for x, y in roi], dtype=np.int32)

TONASE_KG = {
    "sedan":        2000,
    "truk ringan":  5100,
    "truk sedang":  8300,
    "truk berat":  25000,
    "bus kecil":    5100,
    "bus besar":    9000
}

def classify(label, w_cm):
    if label == "bus":
        return "bus besar" if w_cm > 600 else "bus kecil"
    if label == "truck":
        if w_cm > 700: return "truk berat"
        if w_cm > 500: return "truk sedang"
        return "truk ringan"
    if label == "car":
        return "sedan"
    return None

app = FastAPI()
model = YOLO("yolov8n.pt").to("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model.half()
print(f"✔ Loaded YOLO on {model.device}")

def process_yolo(frame, roi, pixel_to_cm):
    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cv2.pointPolygonTest(roi, (cx, cy), False) < 0:
            continue

        label = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        w_px = x2 - x1
        w_cm = w_px * pixel_to_cm
        kategori = classify(label, w_cm)
        berat = TONASE_KG.get(kategori, 0)

        info = f"{kategori} | {berat/1000:.1f}t" if kategori else "unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, info, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return frame

def overlay_info(frame, roi, prev_time):
    fps = 1.0 / (time.time() - prev_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.polylines(frame, [roi], isClosed=True, color=(0, 0, 255), thickness=2)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")
    pixel_to_cm = get_pixel_to_cm()
    raw_roi = get_roi()
    prev_time = time.time()

    try:
        while True:
            raw_bytes = await websocket.receive_bytes()
            npimg = np.frombuffer(raw_bytes, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if frame is None:
                print("⚠️ Failed to decode image")
                continue
            
            frame, orig_size, new_size = resize_frame(frame)
            roi = resize_roi(raw_roi, orig_size, new_size)
            frame = process_yolo(frame, roi, pixel_to_cm)
            overlay_info(frame, roi, prev_time)
            prev_time = time.time()

            _, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            await websocket.send_bytes(encoded.tobytes())

    except WebSocketDisconnect:
        print("⚠️ Client disconnected")
    except Exception as e:
        print(f"🔥 Error: {e}")

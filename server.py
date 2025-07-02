from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import time
import torch
import pyrebase
import json
from base64 import b64decode

firebase_config = {
    "apiKey": "AIzaSyAMRydlD04Ui3p7IEIDDpEjLJjnxgDoDsQ",
    "authDomain": "nodell-c25fc.firebaseapp.com",
    "databaseURL": "https://nodell-c25fc-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "nodell-c25fc.appspot.com"
}
firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

def get_pixel_to_cm():
    return float(db.child("settings").child("pixel_to_cm").get().val() or 1.0)

def get_roi():
    roi = db.child("settings").child("roi").get().val()
    if not roi: return np.array([[0, 0]])
    return np.array(roi, dtype=np.int32)

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
model = YOLO("yolov8n.pt")
model.to("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"✔ Loaded YOLO on {model.device}")

def decode_and_parse(msg):
    try:
        parsed = json.loads(msg)
        b64_data = parsed.get("data", "")
        pixel_to_cm = float(parsed.get("pixel_to_cm", get_pixel_to_cm()))
        roi_list = parsed.get("roi", get_roi().tolist())
        roi = np.array(roi_list, dtype=np.int32)

        frame_data = b64decode(b64_data + '=' * (-len(b64_data) % 4))
        npimg = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            print("⚠️ Failed to decode frame")
        return frame, pixel_to_cm, roi
    except Exception as e:
        print(f"🔥 JSON decode error: {e}")
        return None, 1.0, np.array([[0, 0]])

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

async def send_back_frame(websocket, frame):
    _, encoded = cv2.imencode('.jpg', frame)
    await websocket.send_text(base64.b64encode(encoded.tobytes()).decode())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Client connected")
    prev_time = time.time()

    try:
        while True:
            msg = await websocket.receive_text()
            frame, pixel_to_cm, roi = decode_and_parse(msg)
            if frame is None:
                continue

            frame = process_yolo(frame, roi, pixel_to_cm)
            overlay_info(frame, roi, prev_time)
            prev_time = time.time()

            cv2.imshow("YOLO Server GUI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            await send_back_frame(websocket, frame)

    except WebSocketDisconnect:
        print("⚠️ Client disconnected")
    except Exception as e:
        print(f"🔥 Error: {e}")
    finally:
        cv2.destroyAllWindows()

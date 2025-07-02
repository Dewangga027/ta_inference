import cv2
import numpy as np
from ultralytics import YOLO
from kalman_tracker import KalmanFilter1D
import csv
from collections import defaultdict
import os

# === Konfigurasi & Inisialisasi ===
MODEL_PATH     = "best.pt"
VIDEO_PATH     = "test_video.mp4"
OUTPUT_FOLDER  = "output_video"
OUTPUT_VIDEO   = os.path.join(OUTPUT_FOLDER, "processed_output.mp4")
OUTPUT_CSV     = "vehicle_load_log.csv"
PIXEL_TO_CM    = 1.5
TRACKER_CFG    = "bytetrack.yaml"

# Pastikan folder output ada
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TONASE_KG = {
    "sedan":        2000,
    "truk ringan":  5100,
    "truk sedang":  8300,
    "truk berat":  25000,
    "bus kecil":    5100,
    "bus besar":    9000
}

# Inisialisasi YOLO dan VideoCapture
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Setup Output Video
out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (640, 640))  # karena resize ke 640x640

# ROI polygon
polygon_roi = np.array([
    [456.5, 570.0],
    [168.5, 448.0],
    [332.0, 297.5],
    [500.5, 333.0]
], dtype=np.int32)

def bbox_in_roi(x1, y1, x2, y2, roi):
    corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    inside = sum(cv2.pointPolygonTest(roi, (int(x), int(y)), False) >= 0 for x, y in corners)
    return inside >= 2

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

trackers = {}
seen_vehicles = {}
counts = defaultdict(int)

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1

    frame = cv2.resize(frame, (640, 640))
    results = model.track(frame, tracker=TRACKER_CFG, persist=True)[0]

    cv2.polylines(frame, [polygon_roi], True, (0, 0, 255), 2)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if not bbox_in_roi(x1, y1, x2, y2, polygon_roi):
            continue

        track_id = int(box.id[0])
        label = model.names[int(box.cls[0])]
        w_px = x2 - x1

        if track_id not in trackers:
            trackers[track_id] = KalmanFilter1D()

        w_smooth = trackers[track_id].step(w_px)
        w_cm = w_smooth * PIXEL_TO_CM
        category = classify(label, w_cm)
        weight = TONASE_KG.get(category, 0)

        if track_id not in seen_vehicles and category:
            seen_vehicles[track_id] = {
                "frame": frame_idx,
                "label": label,
                "category": category,
                "weight": weight
            }
            counts[category] += 1

        txt = f"{category} | {weight / 1000:.2f} t" if category else "unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, txt, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Estimator", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Simpan ke CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Track ID", "First Frame", "Label", "Category", "Weight (kg)"])
    for tid, info in seen_vehicles.items():
        writer.writerow([tid, info["frame"], info["label"], info["category"], info["weight"]])

    writer.writerow([])
    total = len(seen_vehicles)
    avg_weight = sum(v["weight"] for v in seen_vehicles.values()) / total if total else 0
    writer.writerow(["Total Unique Vehicles", total])
    writer.writerow(["Average Weight (kg)", f"{avg_weight:.2f}"])
    writer.writerow([])
    writer.writerow(["Counts per Category"])
    for cat, cnt in counts.items():
        writer.writerow([cat, cnt])

print(f"✔ Saved CSV log to {OUTPUT_CSV}")
print(f"✔ Saved processed video to {OUTPUT_VIDEO}")

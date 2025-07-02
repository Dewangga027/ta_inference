import cv2
import numpy as np
from ultralytics import YOLO
from kalman_tracker import KalmanFilter1D
import csv
from collections import defaultdict
import os
import time
import subprocess
import torch

MODEL_PATH     = "yolov8n.pt"
VIDEO_PATH     = "mobil.MP4"
OUTPUT_FOLDER  = "output_video"
OUTPUT_VIDEO   = os.path.join(OUTPUT_FOLDER, "processed_output.mp4")
OUTPUT_CSV     = "vehicle_load_log.csv"
PIXEL_TO_CM    = 1.5
TRACKER_CFG    = "bytetrack.yaml"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TONASE_KG = {
    "sedan":        2000,
    "truk ringan":  5100,
    "truk sedang":  8300,
    "truk berat":  25000,
    "bus kecil":    5100,
    "bus besar":    9000
}

model = YOLO(MODEL_PATH)
model.to("cuda:0")
if torch.cuda.is_available():
    model.half()
    
print(f"✔ Loaded model {MODEL_PATH} on {model.device}")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

orig_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
orig_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

base_width = 480
new_height = int((base_width / orig_w) * orig_h)
resize_dim = (base_width, new_height)

scale_x = base_width / 640
scale_y = new_height / 640

out = cv2.VideoWriter(OUTPUT_VIDEO,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, resize_dim)

polygon_roi_original = np.array([
    [456.5, 570.0],
    [168.5, 448.0],
    [332.0, 297.5],
    [500.5, 333.0]
], dtype=np.int32)
polygon_roi = np.round(polygon_roi_original * np.array([scale_x, scale_y])).astype(np.int32)

def bbox_in_roi(x1, y1, x2, y2, roi):
    corners = [(x1,y1), (x2,y1), (x2,y2), (x1,y2)]
    inside = sum(cv2.pointPolygonTest(roi, (int(x), int(y)), False) >= 0 for x, y in corners)
    return inside >= 2

def get_gpu_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                 '--format=csv,nounits,noheader'],
                                stdout=subprocess.PIPE, text=True)
        gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
        return f"GPU: {gpu_util}% | Mem: {mem_used}/{mem_total} MB"
    except Exception as e:
        return "GPU Info: N/A"


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
TRACK_EVERY_N = 3

frame_idx = 0
while cap.isOpened():
    start_time = time.time()

    ret, frame = cap.read()

    if not ret: break
    frame_idx += 1

    frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
        
    if frame_idx % TRACK_EVERY_N == 0:
        results = model.track(frame, tracker=TRACKER_CFG, persist=True, verbose=False, conf=0.25)[0]
    else:
        results = model.predict(frame, stream=False)[0]

    cv2.polylines(frame, [polygon_roi], True, (0, 0, 255), 2)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        if not bbox_in_roi(x1, y1, x2, y2, polygon_roi):
            continue

        if box.id is None:
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
        if txt != "unknown":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, txt, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            

    elapsed_time = time.time() - start_time
    fps_text = f"FPS: {1 / elapsed_time:.2f}"
    cv2.putText(frame, fps_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    start_time = time.time()
    gpu_text = get_gpu_usage()
    cv2.putText(frame, gpu_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)
    cv2.imshow("Estimator", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

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

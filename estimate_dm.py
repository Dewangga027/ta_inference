from ultralytics import YOLO
import cv2
import csv
import os

# ==== PARAMETER ====
MODEL_PATH = "best.pt"
IMAGE_PATH = "test.png"
OUTPUT_CSV = "output_estimasi.csv"
CONF_THRESHOLD = 0.4

# Skala referensi pada dua titik Y (diukur manual sebelumnya)
# Misalnya: skala lebih besar di bawah (dekat kamera)
REFERENCE_SCALE = {
    "top_y": 0, "top_scale": 1.0,        # pixel Y paling atas
    "bottom_y": 720, "bottom_scale": 2.5 # pixel Y paling bawah
}

# Fungsi untuk hitung skala dinamis
def get_dynamic_scale(y_center):
    top_y = REFERENCE_SCALE["top_y"]
    bottom_y = REFERENCE_SCALE["bottom_y"]
    top_scale = REFERENCE_SCALE["top_scale"]
    bottom_scale = REFERENCE_SCALE["bottom_scale"]
    
    ratio = (y_center - top_y) / (bottom_y - top_y)
    scale = top_scale + ratio * (bottom_scale - top_scale)
    return scale

# === Load model dan gambar ===
model = YOLO(MODEL_PATH)
results = model(IMAGE_PATH)[0]
img = results.orig_img.copy()

# === Inisialisasi data untuk CSV ===
data_csv = [["ID", "Label", "Conf", "Width(cm)", "Height(cm)", "X1", "Y1", "X2", "Y2"]]
id_counter = 1

# === Proses setiap deteksi ===
for box in results.boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]
    conf = float(box.conf[0])
    if conf < CONF_THRESHOLD:
        continue

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    w_px = x2 - x1
    h_px = y2 - y1
    y_center = (y1 + y2) / 2

    # Skala dinamis
    scale = get_dynamic_scale(y_center)
    w_cm = w_px * scale
    h_cm = h_px * scale

    # Gambar bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"{label} ({w_cm:.1f}x{h_cm:.1f} cm)"
    cv2.putText(img, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    print(f"{label} - W:{w_cm:.1f} cm, H:{h_cm:.1f} cm, Conf:{conf:.2f}")

    data_csv.append([id_counter, label, f"{conf:.2f}", f"{w_cm:.1f}", f"{h_cm:.1f}", x1, y1, x2, y2])
    id_counter += 1

# === Tampilkan gambar terdeteksi, resize agar fit window ===
def resize_to_fit(img, max_w=1280, max_h=720):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    return cv2.resize(img, (int(w * scale), int(h * scale)))

resized = resize_to_fit(img)
cv2.imshow("Detected", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# === Simpan ke CSV ===
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data_csv)

print(f"[✅] Estimasi disimpan ke {OUTPUT_CSV}")
from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("best.pt")  # Pastikan path sesuai

# Input image path
image_path = "test.png"

# Perform detection with 640x640 size
results = model(image_path, imgsz=640)

# Plot annotated results (bounding boxes & labels)
annotated_image = results[0].plot()

# Resize to exactly 640x640 if needed (optional)
annotated_image_resized = cv2.resize(annotated_image, (640, 640))

# Display the result (optional)
cv2.imshow("YOLOv8 Detection 640x640", annotated_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("output_detected_640.png", annotated_image_resized)
print("✅ Hasil disimpan sebagai 'output_detected_640.png'")
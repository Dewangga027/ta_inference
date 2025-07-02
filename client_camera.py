import cv2
import asyncio
import websockets
import numpy as np
import pyrebase

firebase_config = {
    "apiKey": "AIzaSyAMRydlD04Ui3p7IEIDDpEjLJjnxgDoDsQ",
    "authDomain": "nodell-c25fc.firebaseapp.com",
    "databaseURL": "https://nodell-c25fc-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "nodell-c25fc.appspot.com"
}

firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

def fetch_settings():
    data = db.child("settings").get().val()
    pixel_to_cm = data.get("pixel_to_cm", 1.0)
    roi = np.array(data.get("roi", []), dtype=np.int32)
    return pixel_to_cm, roi

async def send_video():
    uri = "wss://bfdd-120-188-75-254.ngrok-free.app/ws"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera tidak dapat dibuka.")
        return

    async with websockets.connect(uri, max_size=2**25) as websocket:
        print("📡 Terhubung ke server WebSocket.")
        pixel_to_cm, roi = fetch_settings()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Gagal membaca frame.")
                break

            _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            await websocket.send(encoded.tobytes())

            response = await websocket.recv()
            npimg = np.frombuffer(response, dtype=np.uint8)
            result_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            if result_frame is not None:
                cv2.imshow("📷 YOLO Processed Frame", result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Dihentikan oleh user.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(send_video())

import cv2
import asyncio
import websockets
import numpy as np
import pyrebase
import os

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

async def process_file(path):
    uri = "wss://bfdd-120-188-75-254.ngrok-free.app/ws"
    async with websockets.connect(uri, max_size=2**25) as websocket:
        print(f"📁 Sending file: {path}")
        is_image = path.lower().endswith((".jpg", ".jpeg", ".png"))
        
        if is_image:
            frame = cv2.imread(path)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            await websocket.send(buffer.tobytes())
            result = await websocket.recv()
            npimg = np.frombuffer(result, dtype=np.uint8)
            processed = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            if processed is not None:
                cv2.imshow("Processed Image", processed)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("❌ Failed to decode processed image.")

        else:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print("❌ Failed to open video.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                await websocket.send(buffer.tobytes())

                result = await websocket.recv()
                npimg = np.frombuffer(result, dtype=np.uint8)
                processed = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                if processed is not None:
                    cv2.imshow("Processed Video Frame", processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        asyncio.run(process_file(sys.argv[1]))
    else:
        print("⚠️ Usage: python client_file.py <path_to_image_or_video>")

import cv2
import base64
import asyncio
import websockets
import numpy as np
import time
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

def fetch_settings():
    data = db.child("settings").get().val()
    pixel_to_cm = data.get("pixel_to_cm", 1.0)
    roi = np.array(data.get("roi", []), dtype=np.int32)
    return pixel_to_cm, roi

async def send_video():
    uri = "wss://bfdd-120-188-75-254.ngrok-free.app/ws"
    async with websockets.connect(uri, max_size=2**25) as websocket:
        cap = cv2.VideoCapture(0)
        pixel_to_cm, roi = fetch_settings()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            _, buffer = cv2.imencode('.jpg', frame)
            b64_frame = base64.b64encode(buffer).decode()
            payload = json.dumps({
                "type": "frame",
                "data": b64_frame,
                "pixel_to_cm": pixel_to_cm,
                "roi": roi.tolist(),
                "source": "camera"
            })
            await websocket.send(payload)

            result = await websocket.recv()
            img_bytes = base64.b64decode(result)
            npimg = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow("Processed Camera Frame", npimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(send_video())

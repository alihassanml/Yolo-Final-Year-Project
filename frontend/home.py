from fastapi import FastAPI, WebSocket
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load YOLO model (update to the path of your trained model)
model = YOLO("../model/best.pt")  # Adjust the path to your trained YOLO model

@app.get("/")
def get_index():
    return templates.TemplateResponse("index.html", {"request": {}})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            # Receive image from client
            data = await websocket.receive_text()
            img_data = data.split(",")[1]
            img_bytes = base64.b64decode(img_data)

            # Convert bytes to OpenCV image
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Perform detection
            results = model.predict(source=frame, save=False)  # Perform inference
            detections = results[0].boxes  # Bounding box results

            # Draw detections
            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])  # Class index
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Encode processed image back to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            # Send back to client
            await websocket.send_text(encoded_frame)

        except Exception as e:
            print("Connection closed:", e)
            break

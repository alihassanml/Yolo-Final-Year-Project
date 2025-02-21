from fastapi import FastAPI, File, UploadFile, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
import cv2
import numpy as np
from ultralytics import YOLO
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from model import prediction
from typing import List
import datetime
from database import Base, engine
from sqlalchemy.sql import func
from playsound import playsound
import threading


app = FastAPI()
templates = Jinja2Templates(directory="templates")
cap = cv2.VideoCapture(0)


Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


model = YOLO("../model/best5.pt")
class_name = ['Drinking', 'Eating', 'Violence', 'Sleeping', 'Smoking', 'Walking', 'Weapon']



def play_alarm():
    playsound('../alram/alram.mp3')


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



alert_classes = ['Violence', 'Smoking', 'Weapon']

def generate_frames():
    while True:
        ret, video = cap.read()
        if not ret:
            break
        results = model(video)
        annotated_frame = results[0].plot()

        for box in results[0].boxes:
            cls = int(box.cls)
            class_name = model.names[cls]
            
            if class_name in alert_classes:
                print(f"Alert! Detected: {class_name}")
                # threading.Thread(target=play_alarm, daemon=True).start()


        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')



@app.get("/start_video")
def start_video():
    global cap
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return {"message": "Video started"}

@app.get("/stop_video")
def stop_video():
    global cap
    if cap.isOpened():
        cap.release()
    return {"message": "Video stopped"}




from pydantic import BaseModel

class PredictionResponse(BaseModel):
    id: int
    name: str
    accuracy: str
    time: str

    class Config:
        orm_mode = True

@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(db: Session = Depends(get_db)):
    predictions = db.query(prediction).all()
    return predictions


@app.get("/class-counts")
async def get_class_counts(db: Session = Depends(get_db)):
    counts = (
        db.query(prediction.name, func.count(prediction.name).label("count"))
        .group_by(prediction.name)
        .all()
    )
    return [{"name": name, "count": count} for name, count in counts]



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
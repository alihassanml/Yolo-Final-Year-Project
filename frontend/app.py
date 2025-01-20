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


app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


model = YOLO("../model/best5.pt")
class_name = ['Drinking', 'Eating', 'Violence', 'Sleeping', 'Smoking', 'Walking', 'Weapon']


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = box.cls[0]

            db_prediction = prediction(
                name=class_name[int(label)],
                accuracy=f"{confidence:.2f}",
                time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            db.add(db_prediction)
            db.commit()

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image, f"{class_name[int(label)]} {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

    _, buffer = cv2.imencode(".jpg", image)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")


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

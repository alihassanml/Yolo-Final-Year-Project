from ultralytics import YOLO
import cv2
from playsound import playsound
import threading

def play_alarm():
    playsound('alram/alram.mp3')

model = YOLO('model/best5.pt')

print("Model classes:", model.names)

alert_classes = ['Violence', 'Smoking', 'Weapon']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break  

    results = model(frame)
    result = results[0]

    for box in result.boxes:
        cls = int(box.cls)
        class_name = model.names[cls]
        
        if class_name in alert_classes:
            print(f"Alert! Detected: {class_name}")
            # sleep(10)
            threading.Thread(target=play_alarm, daemon=True).start()

    annotated_frame = result.plot() 
    cv2.imshow('YOLO Inference', annotated_frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

from ultralytics import YOLO
import cv2

model = YOLO('model/best5.pt')
url  = 'ea.jpeg'
results = model(url)

result = results[0]

image_with_detections = result.plot()

cv2.imshow('Detections', image_with_detections)
# cv2.imwrite('./predict/violence.png',image_with_detections)

cv2.waitKey(0)
cv2.destroyAllWindows()

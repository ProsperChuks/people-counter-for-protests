import cv2
import math
from ultralytics import YOLO

model = YOLO('model_weights/yolov8n.pt')
label = model.names

window = cv2.VideoCapture(0)
window.set(3, 820)
window.set(4, 580)

while True:
    status, img = window.read()
    predictions = model(img, stream=True)

    for pred in predictions:
        boxes = pred.boxes
        for box in boxes:
            if label[int(box.cls)] == 'person': # quick but not an advised fix
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (50, 169, 62), 2)

                # text
                class_label = label[int(box.cls)]
                conf_val = '%.2f'%(box.conf[0] * 100)
                text = f'{class_label} {str(conf_val)}%'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                color = (0, 0, 255)
                thickness = 1
                (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + text_width), int(y1 - text_height - 5)), (0, 255, 0), -1)  
                cv2.putText(img, text, (int(x1), int(y1 - 5)), font, font_scale, color, thickness)

    if not status:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.imshow('frame', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

window.release() 
cv2.destroyAllWindows() 

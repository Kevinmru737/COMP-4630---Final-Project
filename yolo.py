# https://docs.ultralytics.com/tasks/detect/#how-do-i-train-a-yolo26-model-on-my-custom-dataset

from ultralytics import YOLO
import torch
import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt


# Load a pretrained model
# Load with explicit task
model = YOLO("best.pt", task="pose")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Train the model on your custom dataset
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml
#model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)

# https://www.youtube.com/watch?v=oA85M9JHsW0

knownObjects = dict()
def drawDetections(img, detections, threshold):
    boxes = detections.boxes

    for box in boxes:  
      if float(box.conf[0]) > threshold:

      
          objClass = int(box.cls[0])
          if objClass not in knownObjects.keys():
            knownObjects[objClass] = (random.randint(0,255),random.randint(0,255),random.randint(0,255),random.randint(0,255))

          x1,y1,x2,y2 = map(int, box.xyxy[0])
          boxLabel = f'{detections.names[objClass]}'
          boxColor = knownObjects[objClass]
          print(boxLabel)

          cv.rectangle(img, (x1, y1), (x2,y2), boxColor)
          (fontW, fontH), baseline = cv.getTextSize(boxLabel, cv.FONT_HERSHEY_COMPLEX, 1, 1)

          cv.rectangle(img, (x1, y1), (x1 + fontW + 5, y1 - baseline - fontH), boxColor, -1)
          cv.rectangle(img, (x1, y1), (x1 + fontW, y1 - baseline - fontH), boxColor, 2)

          cv.putText(img, boxLabel, (x1 + 10, y1 - baseline + 5), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 1)



def drawDetectionsPose(img, detections, threshold):
    boxes = detections.boxes
    keypoints = detections.keypoints  # pose-specific

    for i, box in enumerate(boxes):
        if float(box.conf[0]) > threshold:
            objClass = int(box.cls[0])
            if objClass not in knownObjects:
                knownObjects[objClass] = (
                    random.randint(0,255), random.randint(0,255), random.randint(0,255)
                )

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxLabel = detections.names[objClass]
            boxColor = knownObjects[objClass]

            cv.rectangle(img, (x1, y1), (x2, y2), boxColor, 2)
            cv.putText(img, boxLabel, (x1, y1 - 10), cv.FONT_HERSHEY_COMPLEX, 1, boxColor, 1)

            # Draw keypoints
            if keypoints is not None:
                kpts = keypoints.xy[i]  # shape: [num_keypoints, 2]
                for kp in kpts:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:  # skip missing keypoints
                        cv.circle(img, (kx, ky), 4, (0, 255, 0), -1)

# 1. Initialize the camera (0 is default, use 1 for external USB cams)
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    # Our operations on the frame come here
    results = model(frame)[0]

    # Display the resulting frame
    drawDetectionsPose(frame, results, threshold=0.5)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


#x1 = (x_center - width / 2) * img_width
#y1 = (y_center - height / 2) * img_height
#x2 = (x_center + width / 2) * img_width
#y2 = (y_center + height / 2) * img_height


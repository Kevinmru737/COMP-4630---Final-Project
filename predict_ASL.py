from ultralytics import YOLO
import torch
import numpy as np
import cv2 as cv
import random
from matplotlib import pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# training YOLO model from the guide below
# https://docs.ultralytics.com/tasks/detect/#how-do-i-train-a-yolo26-model-on-my-custom-dataset

# Load a YOLO model for hand detection
model_yolo = YOLO("best30.pt", task="pose")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# This architecture needs to be manually changed based on what the classification model was trained on
class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(24)
        )

    def forward(self, X):
        return self.network(X)


# Loading the classification model
model = BasicCNN().to(device)
model.load_state_dict(torch.load('model (1).pt', map_location=device))
model.eval()

# The current classification model only has 24 classes excluding j and z (index 9 and 25 from the alphabet)
class_names = [chr(ord('A') + i) for i in range(26) if i != 9 and i != 25]

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# https://docs.ultralytics.com/datasets/pose/hand-keypoints/#hand-landmarks

# Train the model on your custom dataset
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/hand-keypoints.yaml
#model.train(data="hand-keypoints.yaml", epochs=100, imgsz=640)

# drawDetections from the youtube video below
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

            # Draw keypoints -- commenting out for actual performance run
            #if keypoints is not None:
            #    kpts = keypoints.xy[i]  # shape: [num_keypoints, 2]
            #    for kp in kpts:
            #        kx, ky = int(kp[0]), int(kp[1])
            #        if kx > 0 and ky > 0:  # skip missing keypoints
            #            cv.circle(img, (kx, ky), 4, (0, 255, 0), -1)

    

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
    results = model_yolo(frame)[0]

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the hand from the frame
            hand_crop = frame[y1:y2, x1:x2]
            
            # Pass to classifier
            gray = cv.cvtColor(hand_crop, cv.COLOR_BGR2GRAY)
            pil_img = Image.fromarray(gray)
            input_tensor = transform(pil_img).unsqueeze(0).to(device)

            with torch.no_grad(): 
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                label = class_names[predicted.item()]
                conf_pct = confidence.item() * 100
                # Display the resulting frame
                cv.putText(frame, f'{label} ({conf_pct:.1f}%)', (10, 40),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    

    drawDetectionsPose(frame, results, threshold=0.5)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()


#notes for dimensions of box
#x1 = (x_center - width / 2) * img_width
#y1 = (y_center - height / 2) * img_height
#x2 = (x_center + width / 2) * img_width
#y2 = (y_center + height / 2) * img_height


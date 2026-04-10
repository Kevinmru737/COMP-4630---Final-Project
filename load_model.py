import numpy as np
import cv2 as cv
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BasicCNN().to(device)
model.load_state_dict(torch.load('garbage_model.pt', map_location=device))
model.eval()

class_names = [chr(ord('A') + i) for i in range(26) if i != 9 and i != 25]

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(gray)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        label = class_names[predicted.item()]
        conf_pct = confidence.item() * 100

    cv.putText(frame, f'{label} ({conf_pct:.1f}%)', (10, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
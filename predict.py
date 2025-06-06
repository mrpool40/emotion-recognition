import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import cv2
import numpy as np
import os

# ===  Load class labels from training folder
class_names = os.listdir("dataset/train")
class_names.sort()
NUM_CLASSES = len(class_names)

# ===  Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ===  Image transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ===  Open second camera (index 1)
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(" Failed to open second camera (index 1). Try another index.")
    exit()

print(" Using camera index 1 (second camera). Press 'q' to quit.")

# ===  Real-time loop
while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    # Preprocess
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    #  Display
    cv2.putText(frame, f"Emotion: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

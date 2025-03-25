import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from model import MaskDetectorCNN  # Import model from model.py

# Load Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Trained Model
model = MaskDetectorCNN().to(device)
model.load_state_dict(torch.load("model/mask_detector.pth", map_location=device))
model.eval()

# Class Labels
class_labels = ["With Mask", "Without Mask", "Improper Mask"]

# Define Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Model Prediction
        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, 1).item()
        
        label = class_labels[pred]
        color = (0, 255, 0) if label == "With Mask" else (0, 0, 255) if label == "Without Mask" else (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Face Mask Detection

## 📌 Overview
This project implements a **real-time face mask detection system** using **PyTorch** and **OpenCV**.  
It classifies faces into three categories:
1. ✅ **With Mask**
2. ❌ **Without Mask**
3. ⚠ **Improperly Worn Mask**  

A **Convolutional Neural Network (CNN)** is trained on a dataset of masked, unmasked, and improperly masked faces.  
The model is then used for **real-time detection via webcam input**.

---

## 📂 Project Structure
```
FaceMaskDetection/
├── dataset/ 
│ ├── with_mask/
│ ├── without_mask/
│ ├── improper_mask/ 
├──scripts/
│ ├── train.py                # CNN Model Definition, Train the CNN Model
│ ├── mask_detect.py          # Real-time mask detection script (Webcam)
├── model/mask_detector.pth 
├── requirements.txt
├── README.md
```
---

## 🔧 Installation & Setup

### **1️⃣ Install Dependencies**
Make sure you have **Python 3.8+** installed.  
Then, install the required libraries:
```
pip install -r requirements.txt
```
requirements.txt
```
torch
torchvision
opencv-python
pillow
numpy
```
2️⃣ Train the Model
If you don’t have a trained model yet, run:
```
python train.py
```
This will train the CNN on the dataset and save the model as:
```
model/mask_detector.pth
```
3️⃣ Run the Mask Detection Script
To start real-time mask detection via webcam, run:
```
python mask_detect.py
```
Press Q to exit the webcam window.

🎯 Features
✅ Deep Learning-based Mask Classification
✅ Real-time Face Detection with OpenCV
✅ Works on CPU & GPU (CUDA enabled)
✅ Fast Processing with Haar Cascade Classifier

📊 Dataset Information
The dataset contains three categories:

   1. With Mask (dataset/with_mask/)

   2. Without Mask (dataset/without_mask/)

   3. Improper Mask (dataset/improper_mask/)

replace the dataset in the corresponding folders in dataset main folder

💡 Future Improvements
🔹 Improve face detection with DNN-based models (e.g., SSD, YOLO)
🔹 Optimize CNN for better accuracy & faster inference
🔹 Deploy as a Flask API or Mobile App

📝 Author
Developed by Ashok Kedari









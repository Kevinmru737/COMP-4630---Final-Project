COMP 4630 - Final Project

# ✋ American Sign Language Recognition using YOLO-Pose & OpenCV

This project explores real-time **American Sign Language (ASL) recognition** using a combination of deep learning and classical computer vision techniques. It integrates a **YOLO-based pose estimation model** with **OpenCV Haar cascades** and a trained classifier on the **Sign Language MNIST dataset** to detect and classify hand gestures.

---

## 📌 Overview

The system is designed to:
- Detect hands in real-time video streams
- Extract key regions using OpenCV Haar cascade classifiers
- Apply a YOLO-pose-based model for improved spatial understanding
- Classify ASL gestures using a trained neural network
- Provide real-time predictions for alphabet signs

---

## 🧠 Key Components

### 1. Hand Detection (OpenCV)
We use a Haar cascade classifier for initial hand region detection:
- Fast and lightweight detection
- Provides bounding boxes for region-of-interest extraction

Reference:
- OpenCV Haar Cascade hand detector (`hand.xml`)

---

### 2. Pose Estimation (YOLO)
A YOLO-based pose model is used to improve robustness in hand localization:
- Handles variation in scale and orientation
- Improves detection in real-world conditions

---

### 3. Gesture Classification
We train a neural network on the **Sign Language MNIST dataset**:
- 28×28 grayscale hand images
- 24-class classification (A–Z excluding J and Z motion gestures)

Dataset:
- https://www.kaggle.com/datasets/datamunge/sign-language-mnist

---

## ⚙️ Pipeline

1. Capture video frame
2. Detect hand region (Haar cascade / YOLO-pose)
3. Crop and preprocess region
4. Normalize input image
5. Run classification model
6. Output predicted ASL character

---

## 🧪 Model Architecture

- CNN-based classifier (trained on Sign Language MNIST)
- YOLO-pose model for detection stage
- OpenCV preprocessing pipeline for segmentation

---

## 📊 Performance

- Improved robustness using YOLO-based detection vs classical-only methods
- Real-time inference capability depending on hardware
- Strong performance on clean MNIST-style inputs, with moderate degradation in real-world lighting conditions

*(Insert your actual accuracy / F1 score here if available)*

---

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt

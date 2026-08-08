# EmoTrack

> **A real-time deep learning-based facial emotion recognition system powered by MobileNetV2.**

---

## Overview

**EmoTrack** is a machine learning pipeline that classifies human facial expressions into seven distinct emotions. By utilizing transfer learning with a pre-trained **MobileNetV2** architecture, the model is highly optimized for accuracy while remaining lightweight enough to perform live, real-time emotion detection via a standard webcam.

## Features

- [x] **Deep Transfer Learning:** Built on top of MobileNetV2, ensuring rapid training and high-performance inference.
- [x] **Real-Time Webcam Inference:** Includes a standalone Python script to detect faces using OpenCV Haar Cascades and classify emotions on the fly.
- [x] **Class-Balanced Training:** Handles dataset imbalances dynamically with class weighting strategies during the training phase.
- [x] **Automated Data Splitting:** Provides utility scripts to cleanly divide raw datasets into `train`, `val`, and `test` splits (60/20/20).
- [x] **Comprehensive Evaluation:** Generates confusion matrices and classification reports to evaluate model performance thoroughly.

---

## Supported Emotions

The model is trained to recognize the following 7 core expressions:
- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprise**

---

## Quick Start & Installation

### 1. Prerequisites
Ensure you have Python installed, then install the required dependencies:

```bash
pip install tensorflow opencv-python numpy scikit-learn matplotlib
```

### 2. Dataset Preparation
This project uses the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset (or similar structured datasets). First, place all your raw image data in emotion-labeled folders, then use the provided script to create train/val/test splits.

```bash
# Splits your dataset into a 60-20-20 ratio
python split.py
```

Your data directory should now look like this:
```text
data/
├── train/
│   ├── angry/
│   ├── happy/
│   └── ...
├── val/
└── test/
```

### 3. Model Training
Open the Jupyter Notebook `emotion_detection_classbalanced.ipynb`. This notebook walks you through:
- Loading and augmenting data via `ImageDataGenerator`.
- Applying class weights for underrepresented emotions (like Disgust).
- Fine-tuning the MobileNetV2 architecture.
- Saving the compiled model to `emotion_mobilenet.h5`.

---

## Real-Time Inference (Webcam)

Once you have trained your model (or downloaded the pre-trained `emotion_mobilenet.h5`), you can run the live webcam tracker!

```bash
python webcamtest.py
```

- The script automatically detects faces using OpenCV's `haarcascade_frontalface_default.xml`.
- Each detected face is cropped, pre-processed, and passed to MobileNetV2.
- The predicted emotion and bounding box are drawn directly on your live video feed.
- **Press `Q`** to exit the camera window.

---

## Project Structure

| File                                    | Description                                                                 |
| --------------------------------------- | --------------------------------------------------------------------------- |
| `split.py`                              | Data pipeline script for generating train/val/test splits.                  |
| `emotion_detection_classbalanced.ipynb` | Core training notebook with augmentation, class weighting, and evaluation.  |
| `webcamtest.py`                         | OpenCV-based script for real-time face tracking and emotion classification. |
| `emotion_mobilenet.h5`                  | The compiled and trained Keras model (generated after training).            |

---

## Author
Developed as part of the **KDD Summer 2025** initiative.

Here’s a clean, professional `README.md` that documents both:

* `split.py` — for splitting the dataset
* `emotion_detection_classbalanced.ipynb` — for training the MobileNetV2 model with class balancing

---

### ✅ `README.md`

```markdown
# Emotion Detection using MobileNetV2

This project uses the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) facial emotion recognition dataset to train a deep learning model that can classify facial expressions into seven emotions using MobileNetV2.

## 📁 Dataset Preparation

Ensure your dataset is structured into three folders inside `data/`:

```

data/
├── train/
├── val/
└── test/

```

Each of these folders should contain **subfolders for each emotion**, e.g.:

```

train/
├── angry/
├── disgust/
├── fear/
├── happy/
├── neutral/
├── sad/
└── surprise/

````

### ⚙️ `split.py`

This script splits the full dataset into `train`, `val`, and `test` folders using a **60-20-20 ratio**.

#### Usage:

```bash
python split.py
````

* Make sure the original images are sorted into emotion-labeled folders before splitting.
* Adjust the source path and output path inside the script if needed.

---

## 🧠 Model Training

### 📓 `emotion_detection_classbalanced.ipynb`

This notebook:

* Loads the dataset using `ImageDataGenerator`
* Applies **data augmentation** for robustness
* Uses **MobileNetV2** with transfer learning
* Applies **class weighting** to handle data imbalance
* Trains and validates the model
* Evaluates on the test set using:

  * Accuracy
  * Classification report
  * Confusion matrix
* Supports **real-time webcam inference** with OpenCV

#### Highlights:

* Handles extreme class imbalance (e.g. `disgust` class is underrepresented)
* Built with TensorFlow / Keras and OpenCV
* Model is saved to disk as `emotion_mobilenet.h5`

---

## 🖥️ Requirements

Install the required libraries:

```bash
pip install tensorflow opencv-python scikit-learn matplotlib
```

---

## 📷 Real-Time Inference

Once the model is trained, the notebook uses your webcam to detect and classify facial expressions live.

Press `Q` to quit the webcam window.

---

## 📂 Files

| File                                    | Description                                    |
| --------------------------------------- | ---------------------------------------------- |
| `split.py`                              | Script to split original dataset into 60/20/20 |
| `emotion_detection_classbalanced.ipynb` | Main training and evaluation notebook          |
| `emotion_mobilenet.h5`                  | Saved trained model (after notebook execution) |

---

## 🧠 Emotions Covered

* 😠 Angry
* 🤢 Disgust
* 😱 Fear
* 😄 Happy
* 😐 Neutral
* 😢 Sad
* 😲 Surprise

---

## ✍️ Author

This project was developed as part of the **KDD Summer 2025** initiative.

---





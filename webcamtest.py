import os
import cv2
import numpy as np
import tensorflow as tf

# === PARAMETERS ===
IMAGE_SIZE = (128, 128)
NUM_EMOTIONS = 7
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
MODEL_PATH = r"D:\\OFFICE WORK\\KDD Summer 2025\\emotion-detection\data\\emotion_mobilenet.h5"

# === Load model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Face detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, IMAGE_SIZE)
        face_norm = face.astype('float32') / 255.0
        face_input = np.expand_dims(face_norm, axis=0)
        preds = model.predict(face_input)
        label = class_labels[np.argmax(preds)]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Importing the libraries
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import *
import numpy as np

# Load model
model = load_model('facial_emotion_recognition_model.h5')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Using webcam to have image for detecting face
# VideoCapture(i),
# i = 0: Using embedded laptop/PC webcam
# i = 1: Using external webcam
capture = cv2.VideoCapture(1)

while True:
    _, frame = capture.read()

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if faces is None:
        cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                predict = model.predict(roi)[0]
                label = class_labels[predict.argmax()]
                label_position = (x, y - 20)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion :3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    default='./models/merged_model.pkl',
    help='Path to the model file',
)
parser.add_argument(
    '--label',
    type=str,
    default='./models/merged_label.pkl',
    help='Path to the label encoder file',
)
args = parser.parse_args()

# Load saved model and encoder
model_path = args.model
label_path = args.label

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(label_path, 'rb') as f:
    le = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lms = hand_landmarks.landmark
            wrist_x, wrist_y, wrist_z = lms[0].x, lms[0].y, lms[0].z

            # Landmarks 1-20, interleaved x/y/z, wrist-normalized
            # Must match extract_landmarks.py exactly
            features = []
            for i in range(1, 21):
                features.append(lms[i].x - wrist_x)
                features.append(lms[i].y - wrist_y)
                features.append(lms[i].z - wrist_z)

            features = np.array(features, dtype=np.float32).reshape(1, -1)

            prediction = model.predict(features)
            gesture_name = le.inverse_transform(prediction)[0]

            cv2.putText(image, f"MOVE: {gesture_name}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('RPS Gesture Detector', image)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
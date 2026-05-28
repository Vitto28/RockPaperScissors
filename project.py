import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse
import time
import subprocess

p1_score = 0
p2_score = 0
game_result = "Hold thumbs up to play!"
is_playing = False
count_down_start = None
ready_timer = None

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='./models/rps_model.pkl')
parser.add_argument('--label', type=str, default='./models/rps_label.pkl')
args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = pickle.load(f)
with open(args.label, 'rb') as f:
    le = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Set to 2 so you can play Rock-Paper-Scissors with someone!
    # model_complexity=0,
    model_complexity=1, # Use the more accurate model for better results, especially with multiple hands
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    key = cv2.waitKey(5) & 0xFF    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # get frame dimensions
    h, w, _ = frame.shape

    p1_gesture = None
    p2_gesture = None

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lms = hand_landmarks.landmark

            # 1. Live Translation
            wrist_x, wrist_y = lms[0].x, lms[0].y
            # rel_x = np.array([lm.x - wrist_x for lm in lms])
            # rel_y = np.array([lm.y - wrist_y for lm in lms])

            # aspect ratio correction for webcam feed
            rel_x = np.array([(lm.x - wrist_x) * w for lm in lms])
            rel_y = np.array([(lm.y - wrist_y) * h for lm in lms])
            
            # 2. Live Scale Normalization
            scale = np.hypot(rel_x[9], rel_y[9])
            if scale < 1e-6:
                continue
            rel_x /= scale
            rel_y /= scale
            
            # Live Bypass for the thumb shape
            is_thumb_up_shape = (
                lms[8].y  > lms[6].y  and  
                lms[12].y > lms[10].y and  
                lms[16].y > lms[14].y and  
                lms[20].y > lms[18].y and  
                lms[4].y  < lms[5].y       
            )

            if is_thumb_up_shape:
                cos_a, sin_a = 1.0, 0.0
            else:    
                # 3. Live Rotation Normalization (Align Middle Finger straight up)
                angle = np.arctan2(rel_y[9], rel_x[9])
                rotation_angle = -angle - np.pi / 2
                cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
            
            features = []
            for i in range(1, 21):
                rot_x = rel_x[i] * cos_a - rel_y[i] * sin_a
                rot_y = rel_x[i] * sin_a + rel_y[i] * cos_a
                features.extend([rot_x, rot_y])
            
            # 4. Live Feature Engineering: Fingertip Distances
            # The goal is to properly distinguish between scissors and paper
            # The idea is to measure the distance between the fingertips and the wrist. When the hand is open (paper), these distances will be larger, and when the hand is in a fist (rock), they will be smaller. For scissors, we expect some fingers to be extended (index and middle) and others not (ring and pinky, possibly thumb too), resulting in a mixed pattern of distances.
            for i in [4, 8, 12, 16, 20]:
                dist = np.hypot(rel_x[i], rel_y[i])
                features.append(dist)
                
            features = np.array(features, dtype=np.float32).reshape(1, -1)

            # Get probability confidence arrays for presentation overlay requirements
            prob = model.predict_proba(features)[0]
            pred_class = np.argmax(prob)
            confidence = prob[pred_class]
            gesture_name = le.inverse_transform([pred_class])[0]

            # Detects each player based on which side of the screen they are
            assigned_gesture = "thumbs_up" if gesture_name in ["ok", "thumbs_up"] else gesture_name

            if lms[0].x < 0.5:
                p1_gesture = assigned_gesture
                cv2.putText(image, f"P1: {p1_gesture.upper()}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(image, f"({confidence*100:.1f}%)", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                p2_gesture = assigned_gesture
                cv2.putText(image, f"P2: {p2_gesture.upper()}", (w - 425, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(image, f"({confidence*100:.1f}%)", (w - 425, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Rectangle encasing each player gesture, displaying it with confidence (changes size accordingly)
            padding = 20
            x_pixels = [int(lm.x * w) for lm in lms]
            y_pixels = [int(lm.y * h) for lm in lms]
            x_min = max(0, min(x_pixels) - padding)
            y_min = max(0, min(y_pixels) - padding)
            x_max = min(w, max(x_pixels) + padding)
            y_max = min(h, max(y_pixels) + padding)
            box_color = (0, 255, 0) if lms[0].x < 0.5 else (0, 0, 255)
            label_text = f"{gesture_name.upper()} ({confidence*100:.1f}%)"
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, 2)
            cv2.putText(image, label_text, (x_min + 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    # 5. Game Preparation (waits for both players' thumbs up)
        if not is_playing and (p1_gesture == 'thumbs_up' and p2_gesture == 'thumbs_up'):
            if ready_timer == None:
                ready_timer = time.time()
            else:
                time_held = time.time() - ready_timer
                game_result = f"Hold thumbs to start round {int((1.0 - time_held) * 1000)}ms"
                if time_held >= 0.5:
                    count_down_start = time.time()
                    is_playing = True
                    game_result = 'Get Ready!'
                    ready_timer = None
        else:
            if not is_playing:
                ready_timer = None
                if game_result in [None, 'Get Ready!', 'Hold thumbs up to play!'] or 'Hold' in game_result:
                    game_result = 'Keep thumbs up to start!'

    # Game Logic
    #After pressing SPACE the countdown begins
    if is_playing:
        elapsed = time.time() - count_down_start
        if elapsed < 1:
            if game_result != "3...":
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
            game_result = "3..." 
        elif elapsed < 2:
            if game_result != "2...":
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
            game_result = "2..."
        elif elapsed < 3:
            if game_result != "1...":
                subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
            game_result = "1..."
        elif elapsed >= 3 and game_result in {"3...","2...","1...","Get Ready!"}:
            subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])
            if 'gesture_name' in locals():
                p1_choice = p1_gesture
                p2_choice = p2_gesture
            else:
                game_result = "Missing a hand. Try Again!"

            if p1_choice == None or p2_choice == None:
                game_result = "Missing a hand. Try Again!"
            elif p1_choice == p2_choice:
                game_result = "Tie Game!"
            elif (p1_choice == "rock" and p2_choice == "scissors") or \
                (p1_choice == "scissors" and p2_choice == "paper") or \
                (p1_choice == "paper" and p2_choice == "rock"):
                game_result = "Player 1 Wins!"
                p1_score += 1
            else:
                game_result = "Player 2 Wins!"
                p2_score += 1
            is_playing = False

    # Game restarts if SPACE is pressed again
    if key == ord(' ') and game_result == ("Player 1 Wins!" or "Player 2 Wins!"):
        is_playing = False

    #Static Markers for Score and Game Status
    cv2.putText(image, f"Game Status: {game_result}", (w // 2 - 200, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(image, f"P1 Score: {p1_score}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(image, f"P2 Score: {p2_score}", (w - 425, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('RPS Gesture Detector', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
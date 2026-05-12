import cv2
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
# Use the standard "solutions" entry point
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,       # <--- ADD THIS: Use 0 for Mac stability
    min_detection_confidence=0.5, # <--- LOWER THIS slightly to 0.5
    min_tracking_confidence=0.5
)

# 2. Initialize Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Prepare the Frame
    # MediaPipe needs RGB, but OpenCV captures in BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # To improve performance, optionally mark the image as not writeable
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    # 4. Draw Landmarks and Extract Data
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Convert back for display
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # This draws the lines/dots on your hand in the preview
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # This is where you'll eventually extract the (x, y) coordinates
            # to feed into your Rock-Paper-Scissors model
            for id, lm in enumerate(hand_landmarks.landmark):
                # lm.x and lm.y are normalized (0.0 to 1.0)
                pass 

    # 5. Display the output
    cv2.imshow('Hand Tracking Preview', image)

    if cv2.waitKey(5) & 0xFF == 27: # Press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1) 
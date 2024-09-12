import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize MediaPipe Hands for gesture detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained face detection model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the age and gender models
age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')

# Mean values and list of age ranges and genders
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Use the default webcam
cap = cv2.VideoCapture(0)

# Gesture history for smoothing
history = deque(maxlen=15)

# Function to count fingers raised using relative positions
def count_fingers(hand_landmarks):
    fingers = []
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)  # Thumb is up
    else:
        fingers.append(0)  # Thumb is down

    for i in range(1, 5):
        if hand_landmarks.landmark[mp_hands.HandLandmark(i * 4)].y < hand_landmarks.landmark[mp_hands.HandLandmark(i * 4 - 2)].y:
            fingers.append(1)  # Finger is up
        else:
            fingers.append(0)  # Finger is down
    return fingers

# Function to recognize gestures based on fingers raised
def recognize_gesture(fingers):
    if len(fingers) != 5:
        return "Unknown Gesture"
    if fingers == [0, 1, 1, 0, 0]:
        return "Peace Sign"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "High Five"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 0, 0, 0, 1]:
        return "OK Sign"
    else:
        return "Unknown Gesture"

# Main loop to capture webcam frames and process
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture video frame.")
        break

    # Flip frame for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]

        # Draw rectangle around face and add text for gender and age
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = f'{gender}, {age}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Convert the frame to RGB for hand detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Count the number of raised fingers
            fingers_raised = count_fingers(hand_landmarks)

            # Display finger count on the frame
            finger_count = sum(fingers_raised)
            cv2.putText(frame, f'Fingers: {finger_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Recognize gestures based on raised fingers
            gesture = recognize_gesture(fingers_raised)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Age, Gender, and Hand Gesture Recognizer', frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

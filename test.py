# pylint: disable=E1101

import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Get realtime webcam feed
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make detections
        results = holistic.process(img)
        print(results.face_landmarks)

        # convert back to bgr
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_holistic.FACE_CONNECTIONS
        )

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        cv2.imshow("Model detections", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

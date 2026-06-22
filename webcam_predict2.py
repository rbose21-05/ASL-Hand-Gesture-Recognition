import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("asl_model_2.keras")

class_names = ["A", "B", "C", "D", "E"]
prediction_history = deque(maxlen=15)
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    num_hands=1
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    options
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = hand_landmarker.detect(mp_image)

    stable_prediction = "?"
    confidence = 0.0

    h, w, _ = frame.shape

    if len(result.hand_landmarks) > 0:

        landmarks = result.hand_landmarks[0]

        x_coords = []
        y_coords = []

        for landmark in landmarks:
            x_coords.append(int(landmark.x * w))
            y_coords.append(int(landmark.y * h))

        padding = 40

        x1 = max(0, min(x_coords) - padding)
        y1 = max(0, min(y_coords) - padding)

        x2 = min(w, max(x_coords) + padding)
        y2 = min(h, max(y_coords) + padding)

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:

            img = cv2.resize(roi, (224, 224))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            prediction = model.predict(img, verbose=0)

            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))

            if confidence > 0.80:
                prediction_history.append(predicted_class)

            if len(prediction_history) > 0:
                stable_prediction = max(
                    set(prediction_history),
                    key=prediction_history.count
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            #cv2.imshow("Hand Crop", roi)

    if confidence < 0.90:
        text = "No hand detected"
    else:
        text = f"{stable_prediction} ({confidence:.2%})"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (160, 160)

print("Loading model...")
model = keras.models.load_model("best_cnn.keras")

with open("class_names.json") as f:
    class_names = json.load(f)

print(f"Loaded model with {len(class_names)} classes.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Press SPACE to capture and identify, Q to quit.")

last_result = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    if last_result:
        y = 30
        for name, prob in last_result:
            text = f"{name}: {prob*100:.1f}%"
            cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            y += 30

    cv2.putText(display, "SPACE: capture   Q: quit", (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Khana Lens - Nepali Food Recognition", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        x = img.astype("float32")
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x, verbose=0)[0]
        top_idx = np.argsort(preds)[::-1][:3]
        last_result = [(class_names[i], preds[i]) for i in top_idx]

        print("Prediction:")
        for name, prob in last_result:
            print(f"  {name}: {prob*100:.1f}%")

cap.release()
cv2.destroyAllWindows()
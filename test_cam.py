import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Opened:", cap.isOpened())

time.sleep(1)  # give the driver a moment before even reading

if cap.isOpened():
    for i in range(60):
        ret, frame = cap.read()
        if ret:
            brightness = np.mean(frame)
            print(f"Frame {i}: read={ret}, mean brightness={brightness:.2f}")
        else:
            print(f"Frame {i}: read={ret}")
        time.sleep(0.05)

    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_final.png", frame)
        print("Saved test_final.png, brightness:", np.mean(frame))

cap.release()
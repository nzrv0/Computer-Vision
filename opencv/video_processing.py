import cv2 as cv
import numpy as np
from pathlib import Path
import os

path = Path("./videos")
video_path = str(path / "sample.mp4")
assert True == os.path.exists(video_path)


cap = cv.VideoCapture(0)  # for capturing a camera
# cap = cv.VideoCapture(video_path)
fourcc = cv.VideoWriter_fourcc(*"XVID")
video_writer = cv.VideoWriter("output.avi", fourcc, 20.0, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()

    frame = cv.flip(frame, 0)

    video_writer.write(frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("frame", gray)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

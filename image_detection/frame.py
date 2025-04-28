import cv2 as cv
from pathlib import Path
from helpers import get_path
import os
from edge_detector import pipeline

path = get_path("videos")
ll_path = os.listdir(path)
video = path / ll_path[2]


cap = cv.VideoCapture(video)
# Create a window that you can resize
cv.namedWindow("Video", cv.WINDOW_NORMAL)

# Set a specific size
cv.resizeWindow("Video", 1200, 800)  # width x height

while cap.isOpened():
    ret, frame = cap.read()
    # edges = cv.Canny(frame, 100, 200)
    frame = cv.resize(frame, None, fx=0.5, fy=0.5)

    # gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = pipeline(frame)
    cv.imshow("Video", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()

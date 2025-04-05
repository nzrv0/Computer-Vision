import cv2 as cv
import sys
from pathlib import Path
import os

path = Path("./images")
image_path = str(path / "sample.jpg")
assert True == os.path.exists(image_path)

imRead = cv.imread(image_path)

# with cv as imRead:
#     pass

cv.imshow("Display Window", imRead)

k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("sample_two.png", imRead)

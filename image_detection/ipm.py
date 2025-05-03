import cv2 as cv
from pathlib import Path
from helpers import get_path
import os
import json
import numpy as np

import matplotlib.pyplot as plt


def load_json():
    path = get_path("images")
    data = path / "152278451988069900.json"
    str_ = ""
    with open(data, "r") as fs:
        str_ += fs.read()
    return json.loads(str_)


def parse_json():
    data = load_json()
    arr = data["lane_lines"][-3]["uv"]

    x = np.array(arr[0])
    y = np.array(arr[1])
    return x[0], x[-1], y[0], y[-1]


def change_perspective(image):
    rows, cols, chn = image.shape
    size_rows = rows / 2
    gap = 60
    other_gap = 200
    # pts1 = np.float32(
    #     [
    #         [cols / 2 - gap, 0],
    #         [cols / 2 + gap, 0],
    #         [0, rows],
    #         [cols, rows],
    #     ]
    # )
    pts = np.array([[cols / 2 - gap, 0], [cols / 2 + gap, 0], [0, rows], [cols, rows]])
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    pts2 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]])
    M = cv.getPerspectiveTransform(rect, pts2)
    dst = cv.warpPerspective(
        image,
        M,
        (cols, rows),
        flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR | cv.WARP_FILL_OUTLIERS,
    )

    return dst


def get_image():
    path = get_path("images")
    image = path / "155362930437411700.jpg"
    image = cv.imread(image)
    return image


def draw_line(image, pt1, pt2):
    return cv.line(image, pt1, pt2, (255, 0, 0), 3)


# start_x, end_x, start_y, end_y = parse_json()
# print(start_x, end_x, start_y, end_y)
image = get_image()
image = change_perspective(image)
data = load_json()
cc = 0
dd = []
for item in data["lane_lines"]:
    visibilty = np.array(item["visibility"])
    vv = np.unique_counts(visibilty).counts
    if len(vv) < 2:
        vv = np.append(vv, 0)
    item["visibility"] = vv[1]
    dd.append(vv[1])

import math

dd = np.stack(dd)
dd = np.sort(dd, axis=0)
max_dd = np.max(dd)
index = np.where(dd == max_dd)
max_2d = dd[index[0][0] - 1]
for dl in data["lane_lines"]:
    visibilty = np.array(dl["visibility"])
    if visibilty == max_2d or visibilty == max_dd:
        start_x, end_x, start_y, end_y = (
            dl["uv"][0][0],
            dl["uv"][0][-1],
            dl["uv"][1][0],
            dl["uv"][1][-1],
        )
        image = draw_line(image, (int(start_x), int(start_y)), (int(end_x), int(end_y)))
        cv.putText(
            image,
            str(cc),
            (int(start_x), int(start_y)),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3,
        )

plt.imshow(image)
plt.show()

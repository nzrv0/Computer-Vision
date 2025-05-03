from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import json
import torch


def get_path(subpath: str) -> Path:
    path = Path("./data")
    path = path / subpath
    return path


def bird_eye(image):
    rows, cols, chn = image.shape
    gap = 60
    pts1 = np.float32(
        [
            [cols / 2 - gap, 0],
            [cols / 2 + gap, 0],
            [0, rows],
            [cols, rows],
        ]
    )

    pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    output = cv.warpPerspective(
        image,
        M,
        (cols, rows),
        flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR | cv.WARP_FILL_OUTLIERS,
    )

    return output


def get_cords(lines, device):
    cords = []
    for line in lines:
        arr = line["uv"]
        x = torch.Tensor(arr[0], device=device)
        y = torch.Tensor(arr[1], device=device)
        cords.append(torch.stack((x, y)))

    return cords


def load_json(path):
    data = ""
    with open(path, "r") as fs:
        data += fs.read()
    data = json.loads(data)
    return data["file_path"], data["lane_lines"]


def show_image(image):
    plt.imshow(image)
    plt.show()

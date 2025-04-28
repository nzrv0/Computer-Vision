from helpers import get_path
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys


def get_image():
    path = get_path("images")
    ll_path = os.listdir(path)
    image = path / ll_path[0]
    image = cv.imread(image)
    widht, height = image.shape[:2]
    # image[0] = widht // 2
    # image[1] = height // 2
    image = cv.resize(image, None, fx=0.077, fy=0.088, interpolation=cv.INTER_LINEAR)
    return image


def gray_image(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    return gray


def blur(image):
    blur_img = cv.GaussianBlur(image, (5, 5), 0)
    return blur_img


def canny(image):
    canny = cv.Canny(image, 70, 140)
    return canny


def region(image):
    height, width = image.shape[:2]
    triangle = np.array([[(0, height), (width // 2, height // 2), (width, height)]])

    mask = np.zeros_like(image)

    mask = cv.fillPoly(mask, triangle, 255)
    mask = cv.bitwise_and(image, mask)
    return mask


def average(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parametrs = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parametrs[0]
        y_int = parametrs[1]
        if slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def make_points(image, average):
    if not np.isnan(average).any():
        slope, y_int = average
        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))
        x1 = int((y1 - y_int) // slope)
        x2 = int((y2 - y_int) // slope)
    else:
        return np.array([1, 1, 1, 1])
    return np.array([x1, y1, x2, y2])


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    if lines is not None:
        print(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    return lines_image


image = get_image()


def pipeline(image):
    gray_img = gray_image(image)
    blur_img = blur(gray_img)
    canny_img = canny(blur_img)
    region_img = region(canny_img)

    lines = cv.HoughLinesP(
        region_img,
        rho=2,
        theta=np.pi / 180,
        threshold=30,
        lines=np.array([]),
        minLineLength=20,
        maxLineGap=150,
    )
    averaged_lines = average(image, lines)
    black_lines = display_lines(image, lines)
    lanes = cv.addWeighted(image, 1, black_lines, 1, 1)

    contours, hierarchy = cv.findContours(
        region_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
    )
    # cv.drawContours(image, contours, -1, (0, 255, 0), 3)

    return lanes


def display(image):
    cv.imshow("", image)
    key = cv.waitKey(0)
    if key == ord("q"):
        sys.exit()
    cv.destroyAllWindows()


lanes = pipeline(image)
if __name__ == "__main__":
    display(lanes)

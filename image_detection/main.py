from pathlib import Path
from selectivesearch import selective_search
import cv2 as cv
import matplotlib.pyplot as plt
import copy

path = Path("./data")
dogs_path = path / "dogs.jpg"

image = cv.imread(dogs_path)

print(image.shape)


new_height = int(image.shape[1] / 4)

new_width = int(image.shape[0] / 2)


resized_image = cv.resize(image, (new_width, new_height))
# plt.imshow((resized_image))
print(resized_image.shape)
# plt.show()
img_lbl, regions = selective_search(resized_image, scale=400, min_size=50, sigma=1.2)


candidates = set()

for r in regions:
    if r["rect"] in candidates or r["size"] < 200:
        continue

    x, y, w, h = r["rect"]

    if h == 0 or w == 0:
        continue

    if w / h > 1.2 or h / w > 1.2:
        continue
    candidates.add(r["rect"])

candidates_scaled = [
    (
        int(x * (image.shape[1] / new_width)),
        int(y * (image.shape[0] / new_height)),
        int(w * (image.shape[1] / new_width)),
        int(h * (image.shape[0] / new_height)),
    )
    for x, y, w, h in candidates
]
output_image = copy.copy(image)
for x, y, w, h in candidates_scaled:
    cv.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
plt.imshow(output_image)
plt.show()

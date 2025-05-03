from torch.utils.data import Dataset
import cv2 as cv
import os
from helpers import get_path, load_json, get_cords
import torch


class LaneDataset(Dataset):
    def __init__(self, image_dir, cords_dir, device):
        self.image_dir = get_path(image_dir)
        self.segments = os.listdir(self.image_dir)
        self.device = device

        self.cords_dir = get_path(cords_dir)
        self.cords = os.listdir(self.cords_dir)

    def __getitem__(self, index):
        dataset = []
        segment = self.image_dir / self.segments[index]
        line = self.cords_dir / self.cords[index]

        format_line = str(line).split("/")[-1]
        format_segment = str(segment).split("/")[-1]

        if format_line != format_segment:
            return 0

        lines_data = os.listdir(line)

        for _, item in enumerate(lines_data):
            line_path = line / item
            file_path, lines = load_json(line_path)
            file_path = file_path.split("/")[-1]

            # read image
            image_path = segment / file_path
            image = cv.imread(image_path)
            imageRgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_ch = imageRgb.transpose(2, 0, 1)
            tensor_image = torch.from_numpy(image_ch).float()

            # load lines
            cords = get_cords(lines, self.device)
            tensor_image = tensor_image.to(self.device)

            dataset.append({"image": tensor_image, "cords": cords})
        return dataset

    def __len__(self):
        return len(self.segments)

from glob import glob

import cv2
import numpy as np
import torch


class BGMaskDataset(torch.utils.data.Dataset):
    def __init__(self, path, image_width, image_height):
        super().__init__()
        self.path = path
        self.image_width = image_width
        self.image_height = image_height
        print(f"BG_Mask path: {path}")
        
        # First try to find masks in the specified path
        self._files = glob(f"{path}/images/*.png")
        if len(self._files) == 0:
            self._files = glob(f"{path}/*.JPG")
        if len(self._files) == 0:
            # If no files found, try looking in parent directories
            path = "/".join(path.split("/")[:-3])
            print(f"Looking for masks in parent directory: {path}")
            self._files = glob(f"{path}/*/images/*.png")
            
        if len(self._files) == 0:
            print(f"Warning: No mask files found in {path}")
            
        self._files.sort()
        print(f"Found {len(self._files)} mask files")

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index: int):
        print(f"Looking in index: {index} of {len(self._files)} in {self.path}")
        image_path = self._files[index]
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.image_width, self.image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255
        image = (image > 0.5).astype(float)
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, kernel)
        image = torch.FloatTensor(image)
        return np.array(image)

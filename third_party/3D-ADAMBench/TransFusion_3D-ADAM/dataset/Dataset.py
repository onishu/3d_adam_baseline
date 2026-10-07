import glob
import os
import cv2
import numpy as np
import tifffile
import torch
import torchvision
from PIL import Image

from utils.geo_utils import *


def get_faulty_imgs(dataset, files):
    a = []
    for f in files:
        m = get_mask_filename(dataset, f)
        if isinstance(m, list):
            mask_full = None
            for mi in m:
                mask = cv2.imread(mi, cv2.IMREAD_GRAYSCALE) / 255
                if mask_full is None:
                    mask_full = mask
                else:
                    mask_full = np.logical_or(mask_full, mask)
            mask = mask_full * 255
            mask = mask.astype(np.uint8)
        else:
            mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)

        if mask is not None and np.max(mask) > 0:
            a.append(False)
        else:
            a.append(True)
    a = np.array(a)
    return a


def get_files(dataset, category, test, path):
    print(f"Looking for dataset {dataset}, in path {path} with test: {test}")
    if dataset == "visa":
        if test:
            p = list([p for p in glob.glob(f"{path}/*/*.JPG")])
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty
        return list([p for p in glob.glob(f"{path}/*.JPG")])
    elif dataset == "mvtec":
        if test:
            p = list([p for p in glob.glob(f"{path}/*/*.png")])
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty
        return list([p for p in glob.glob(f"{path}/*.png")])
    elif dataset == "mvtec3d":
        if test:
            p = list([p for p in glob.glob(f"{path}/*/rgb/*.png")])
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty
        return list([p for p in glob.glob(f"{path}/rgb/*.png")])
    elif dataset == "adam3d":
        # anomalib / HuggingFace ADAM3D layout:
        #   <root>/<category>/{train,test}/<label>/{rgb,xyz,ground_truth}/<stem>.*
        if test:
            p = sorted(glob.glob(f"{path}/{category}/test/*/rgb/*.png"))
            is_faulty = get_faulty_imgs(dataset, np.array(p))
            return list(p), is_faulty
        return sorted(glob.glob(f"{path}/{category}/train/good/rgb/*.png"))


def rgb_to_xyz_path(rgb_path: str) -> str:
    from pathlib import Path

    p = Path(rgb_path)
    if p.parent.name != "rgb":
        raise ValueError(f"Expected .../rgb/<file>, got {rgb_path}")
    return str(p.parent.parent / "xyz" / f"{p.stem}.tiff")


def get_depth_files(test, path, category=None, dataset=None, rgb_files=None):
    if dataset == "adam3d":
        if rgb_files is None:
            raise ValueError("adam3d depth pairing requires rgb_files")
        return [rgb_to_xyz_path(p) for p in rgb_files]
    if test:
        p = list([p for p in glob.glob(f"{path}/*/test/*/tiff/*.tiff")])
        return list(p)
    return list([p for p in glob.glob(f"{path}/*/train/*/tiff/*.tiff")])


def get_mask_filename(dataset, filename):
    if dataset == "visa":
        return filename.replace("test", "ground_truth").replace(".JPG", ".png")
    elif dataset == "mvtec":
        return filename.replace("test", "ground_truth").replace(".png", "_mask.png")
    elif dataset == "mvtec3d":
        return filename.replace("rgb", "gt")
    elif dataset == "adam3d":
        # .../test/<defect>/rgb/000.png -> .../test/<defect>/ground_truth/000.png
        from pathlib import Path

        p = Path(filename)
        if p.parent.name != "rgb":
            raise ValueError(f"Expected .../rgb/<file>, got {filename}")
        return str(p.parent.parent / "ground_truth" / p.name)


def get_max_min_depth_img(image_path):
    image = tifffile.imread(image_path).astype(np.float32)
    image_t = (
        np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
    )
    image = image_t[:, :, 2]
    zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
    im_max = np.max(image)
    im_min = np.min(image * (1.0 - zero_mask) + 1000 * zero_mask)
    return im_min, im_max


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        datasetParameters: dict,
        test: bool,
    ):
        super().__init__()
        # Downsample by factor of 2 to match UNet's downsampling
        self.image_width = 320  # Changed from 1280
        self.image_height = 256  # Changed from 1024
        self.test = test

        self.category = datasetParameters["category"]
        self.dataset_type = datasetParameters["dataset_type"]

        #postfix = "test/" if test else "train/"
        #path = f"{path}/{self.category}/{postfix}"

        out = get_files(self.dataset_type, self.category, self.test, path)
        if isinstance(out, tuple):
            self._files = out[0]
            self.is_faulty = np.logical_not(out[1])
        else:
            self._files = out
            self.is_faulty = [False] * len(self._files)
        idx = np.argsort(self._files)
        self.is_faulty = list(np.array(self.is_faulty)[idx])
        self._files.sort()

        self.global_max, self.global_min = 1, 0
        self.d = "d" in datasetParameters["mode"]
        self.rgb = "rgb" in datasetParameters["mode"]
        self.depth_files = None
        if self.d:
            self.depth_files = get_depth_files(
                self.test,
                path,
                category=self.category,
                dataset=self.dataset_type,
                rgb_files=self._files,
            )
            if self.dataset_type != "adam3d":
                self.depth_files.sort()
            missing = [p for p in self.depth_files if not os.path.isfile(p)]
            if missing:
                raise FileNotFoundError(
                    f"Missing {len(missing)} XYZ files for {self.category}; "
                    f"example: {missing[0]}"
                )
            if not test:
                for image_path in self.depth_files:
                    im_min, im_max = get_max_min_depth_img(image_path)
                    self.global_min = min(self.global_min, im_min)
                    self.global_max = max(self.global_max, im_max)
                self.global_min = self.global_min * 0.9
                self.global_max = self.global_max * 1.1
            else:
                self.global_max = datasetParameters["global_max"]
                self.global_min = datasetParameters["global_min"]

    def __len__(self):
        return len(self._files)

    def get_image(self, file, mask=False, width=None, height=None):
        #Handle the case where the file is a directory
        #print(f"file: {file}, type: {type(file)}, len: {len(file)}, is list: {isinstance(file, list)}")
        img = Image.open(file)
        width = self.image_width if width is None else width
        height = self.image_height if height is None else height
        img = torchvision.transforms.Resize((width, height))(img)
        if mask:
            return torch.FloatTensor(np.array(img))
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack((img, img, img), axis=2)

        img = (img / 255.0 - 0.5) * 2
        img = img.transpose((2, 0, 1))

        return torch.FloatTensor(img)

    def get_depth_image(self, file, width=None, height=None):
        depth_img = tifffile.imread(file).astype(np.float32)
        width = self.image_width if width is None else width
        height = self.image_height if height is None else height
        depth_img = cv2.resize(
            depth_img, (width, height), 0, 0, interpolation=cv2.INTER_NEAREST
        )
        depth_img = np.array(depth_img)

        image = depth_img
        image_t = (
            np.array(image)
            .reshape((image.shape[0], image.shape[1], 3))
            .astype(np.float32)
        )
        image = image_t[:, :, 2]

        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))
        plane_mask = get_plane_mask(image_t)  # 0 is background, 1 is foreground
        plane_mask[:, :, 0] = plane_mask[:, :, 0] * (1.0 - zero_mask)
        plane_mask = fill_plane_mask(plane_mask)

        image = fill_depth_map(image)
        zero_mask = np.where(image == 0, np.ones_like(image), np.zeros_like(image))

        image = (image - self.global_max) / (self.global_max - self.global_min)

        image = image * (1.0 - zero_mask)

        image = np.expand_dims(image, 2)

        depth_img = image

        return torch.FloatTensor(depth_img.transpose(2,1,0)), torch.FloatTensor(
            np.squeeze(plane_mask)
        )

    def __getitem__(self, index: int):
        img = self.get_image(self._files[index], mask=False)

        has_anom = self.is_faulty[index]
        if has_anom:
            mask = get_mask_filename(self.dataset_type, self._files[index])
            mask = self.get_image(mask, mask=True)
        else:
            width = img.shape[1]
            height = img.shape[2]
            mask = torch.FloatTensor(np.zeros((width, height)))

        sample = {"image": img, "mask": mask, "index": index, "is_anomaly": has_anom}

        if self.depth_files is not None:
            d_img, plane_mask = self.get_depth_image(self.depth_files[index])
            sample["plane_mask"] = plane_mask

        if self.d and self.rgb:
            full_img = torch.cat([img, d_img], dim=0)
            sample["image"] = full_img
        elif self.d:
            sample["image"] = d_img
        return sample

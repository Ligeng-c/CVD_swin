import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train", flip_flag=1):
        self.transform = transforms.Compose(transforms_)
        if mode == "None":
            self.files = sorted(glob.glob(root+ "/*.*"))
        else:
            self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #self.image_names = np.sprite()
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        if mode == "new_test":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

        self.flip_flag = flip_flag
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])

        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
        if self.flip_flag==1:
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)

class ImageDataset_single(Dataset):
    def __init__(self, root, transforms_=None, mode="train", flip_flag=1):
        self.transform = transforms.Compose(transforms_)
        if mode == "None":
            self.files = sorted(glob.glob(root+ "/*.*"))
        else:
            self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #self.image_names = np.sprite()
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        if mode == "new_test":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

        self.flip_flag = flip_flag
    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        if self.flip_flag==1:
            if np.random.random() < 0.5:
                img = Image.fromarray(np.array(img)[:, ::-1, :], "RGB")

        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)

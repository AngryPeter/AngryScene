import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, mode="train"):
        super().__init__()
        # 定义数据增强方法
        self.pf = 0
        self.rt = 0
        self.rl = 0
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, "imgs") + "/*.*"))
            self.labels = sorted(glob.glob(os.path.join(root, "labels") + "/*.*"))
        else:
            self.labels = sorted(glob.glob(root + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        # 决定是否翻转
        self.pf = np.random.choice([0, 1])
        if self.mode == 'train':
            if np.random.choice([0, 1]) == 0:
                transforms = [
                    transform.Resize(size=(384, 512), mode=Image.BICUBIC),
                    transform.RandomHorizontalFlip(self.pf),
                    transform.ToTensor(),
                    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
            else:
                # 选择Crop
                self.rt = np.random.randint(0, 380)
                self.rl = np.random.randint(0, 512)
                transforms = [
                    transform.Crop(self.rt, self.rl, height=384, width=512),
                    transform.RandomHorizontalFlip(self.pf),
                    transform.ToTensor(),
                    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ]
        else:
            transforms = [
                transform.Resize(size=(384, 512), mode=Image.BICUBIC),
                transform.ToTensor(),
                transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]
        self.transforms = transform.Compose(transforms)

        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transforms(img_A)
        else:
            img_A = np.empty([1])
        img_B = self.transforms(img_B)

        return img_A, img_B, photo_id
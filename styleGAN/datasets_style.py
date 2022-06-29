import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
import jittor as jt
from PIL import Image
from math import sqrt

class ImageDataset(Dataset):
    def __init__(self, root, resolution, mode="train"):
        super().__init__()
        transforms = [
            # transform.Resize(size=(384, 512), mode=Image.BICUBIC),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        self.transforms = transform.Compose(transforms)
        self.mode = mode
        self.res = resolution
        self.labels = []
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, "new_imgs", str(self.res)) + "/*.*"))
            while not self.res == 2:
                self.labels.append(sorted(glob.glob(os.path.join(root, "new_labels", str(self.res)) + "/*.*")))
                self.res = int(self.res / 2)
            self.labels = np.array(self.labels)
            self.labels = self.labels.transpose(1, 0)
        else:
            while not self.res == 2:
                self.labels.append(sorted(glob.glob(os.path.join("../new_val", str(self.res)) + "/*.*")))
                self.res = int(self.res / 2)
            self.labels = np.array(self.labels)
            self.labels = self.labels.transpose(1, 0)
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_paths = self.labels[index % len(self.labels)]
        photo_id = label_paths[0].split('/')[-1][:-4]

        img_Bs = [Image.open(label_path) for label_path in label_paths]
        img_Bs = [Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2)) for img_B in img_Bs]

        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_Bs = [Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB") for img_B in img_Bs]
            img_A = self.transforms(img_A)

        else:
            img_A = np.empty([1])
        img_Bs = [self.transforms(img_B) for img_B in img_Bs]
        return img_A, img_Bs, photo_id
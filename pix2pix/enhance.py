import os
import cv2
import glob
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def clahe(image):
    # 限制对比度自适应直方图均衡化CLAHE
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return cv2.fastNlMeansDenoisingColored(image_clahe, 10)


os.makedirs("./train/clahe/", exist_ok=True)
files = sorted(glob.glob(os.path.join('./train', "imgs") + "/*.*"))
for file in tqdm(files):
    image = Image.open(file)
    image = Image.fromarray(clahe(np.asarray(image)))
    photo_id = file.split('/')[-1][:-4]
    image.save('./train/clahe/' + photo_id + '.jpg')

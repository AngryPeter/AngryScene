from PIL import Image
import os
from tqdm import tqdm
import numpy as np

path = './results/scene/test_latest/images/'

for image_file in tqdm(os.listdir(path)):
    if str(image_file)[-len('_synthesized_image.jpg'):] != '_synthesized_image.jpg':
        continue

    # 打开文件
    image_path = os.path.join(path, image_file)
    image = Image.open(image_path)

    # 调整大小
    image = image.resize((512, 384), Image.LANCZOS)

    #目标路径
    target_path = os.path.join('./results_resized', str(image_file)[0:-len('_synthesized_image.jpg')] + '.jpg')
    os.makedirs('./results_resized', exist_ok=True)

    # 保存
    image.save(target_path)

from PIL import Image
import os
from tqdm import tqdm
import numpy as np

test_path = '../val_A'
image_root = '../dataset/train/imgs'
label_root = '../dataset/train/labels'

# 处理测试集
for image_file in tqdm(os.listdir(test_path)):
    # 打开文件
    image_path = os.path.join(test_path, image_file)
    image = Image.open(image_path).convert('L')

    # 调整大小
    image = image.resize((512, 256), Image.LANCZOS)

    # 处理label>28的值
    image = np.array(image)
    image[image > 28] = 28
    image = Image.fromarray(np.array(image).astype("uint8"))

    #目标路径
    target_path = os.path.join('./test_label', image_file)
    os.makedirs('./test_label', exist_ok=True)

    # 保存
    image.save(target_path)

# 处理训练集
for filename in tqdm(os.listdir(image_root)):
    # 打开文件
    image_path = os.path.join(image_root, filename)
    label_path = os.path.join(label_root, filename)
    image = Image.open(image_path)
    label = Image.open(label_path).convert('L')

    # 调整大小
    image = image.resize((512, 256), Image.LANCZOS)
    label = label.resize((512, 256), Image.LANCZOS)

    # 处理label>28的值
    label = np.array(label)
    label[label > 28] = 28
    label = Image.fromarray(np.array(label).astype("uint8"))

    #目标路径
    image_target_path = os.path.join('./train_img', str(filename)[0:-len('.jpg')] + '.png')
    label_target_path = os.path.join('./train_label', filename)
    os.makedirs('./train_img', exist_ok=True)
    os.makedirs('./train_label', exist_ok=True)

    # 保存
    image.save(image_target_path)
    label.save(label_target_path)
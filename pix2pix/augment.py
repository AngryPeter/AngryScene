# 参考论文：Differentiable Augmentation for Data-Efficient GAN Training
# @inproceedings{zhao2020diffaugment,
#   title={Differentiable Augmentation for Data-Efficient GAN Training},
#   author={Zhao, Shengyu and Liu, Zhijian and Lin, Ji and Zhu, Jun-Yan and Han, Song},
#   booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
#   year={2020}
# }

import jittor as jt


def diff_augment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            # 选择使用的数据增强方式
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        # 使用连续的内存进行储存
        x = x
    return x


def rand_brightness(x):
    x = x + (jt.rand((x.size(0), 1, 1, 1), dtype=x.dtype) - 0.5)
    return x


def rand_saturation(x):
    x_mean = jt.mean(x, dim=1, keepdims=True)
    x = (x - x_mean) * (jt.rand((x.size(0), 1, 1, 1), dtype=x.dtype) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = jt.mean(x, dims=(1, 2, 3), keepdims=True)
    x = (x - x_mean) * (jt.rand((x.size(0), 1, 1, 1), dtype=x.dtype) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = jt.randint(-shift_x, shift_x + 1, shape=[x.size(0), 1, 1])
    translation_y = jt.randint(-shift_y, shift_y + 1, shape=[x.size(0), 1, 1])
    grid_batch, grid_x, grid_y = jt.misc.meshgrid(
        jt.misc.arange(x.size(0),dtype='int32'),
        jt.misc.arange(x.size(2), dtype='int32'),
        jt.misc.arange(x.size(3), dtype='int32'),
    )
    grid_x = jt.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = jt.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = jt.nn.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1)[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = jt.randint(0, x.size(2) + (1 - cutout_size[0] % 2), shape=[x.size(0), 1, 1])
    offset_y = jt.randint(0, x.size(3) + (1 - cutout_size[1] % 2), shape=[x.size(0), 1, 1])
    grid_batch, grid_x, grid_y = jt.misc.meshgrid(
        jt.misc.arange(x.size(0), dtype='int32'),
        jt.misc.arange(cutout_size[0], dtype='int32'),
        jt.misc.arange(cutout_size[1], dtype='int32'),
    )
    grid_x = jt.clamp(grid_x + offset_x - cutout_size[0] // 2, min_v=0, max_v=x.size(2) - 1)
    grid_y = jt.clamp(grid_y + offset_y - cutout_size[1] // 2, min_v=0, max_v=x.size(3) - 1)
    mask = jt.ones([x.size(0), x.size(2), x.size(3)], dtype=x.dtype)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
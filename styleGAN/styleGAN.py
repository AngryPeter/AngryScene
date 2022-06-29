import os
import jittor as jt
import numpy as np
from model_style import StyledGenerator, Discriminator
import jittor.transform as transform
# from dataset import SymbolDataset
from datasets_style import ImageDataset

from tqdm import tqdm
import argparse
import math
import random
import cv2

jt.flags.use_cuda = True
jt.flags.log_silent = True

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def start_grad(model):
    for param in model.parameters():
        if 'running_mean' in param.name() or 'running_var' in param.name(): continue
        param.start_grad()

def save_image(img, path, nrow=10):
    N, C, W, H = img.shape
    if (N % nrow != 0):
        print("save_image error: N%nrow!=0")
        return
    img = img.transpose((1, 0, 2, 3))
    ncol = int(N / nrow)
    img2 = img.reshape([img.shape[0], -1, H])
    img = img2[:, :W * ncol, :]
    for i in range(1, int(img2.shape[1] / W / ncol)):
        img = np.concatenate([img, img2[:, W * ncol * i:W * ncol * (i + 1), :]], axis=2)
    min_ = img.min()
    max_ = img.max()
    img = (img - min_) / (max_ - min_) * 255
    img = img.transpose((1, 2, 0))
    if C == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, img)
    return img

@jt.single_process_scope()
def valid(epoch, images, photo_id, step=-1):
    """存储验证生成的图片"""
    os.makedirs(f"styleGAN/images/test_fake_imgs/{step}/epoch_{epoch}", exist_ok=True)
    # img_sample = images[:10]
    # img = save_image(img_sample, f"styleGAN/images/epoch_{epoch}_{step}_sample.png", nrow=5)

    images = ((images + 1) / 2 * 255).astype('uint8')

    for idx in range(images.shape[0]):
        cv2.imwrite(f"styleGAN/images/test_fake_imgs/{step}/epoch_{epoch}/{photo_id[idx]}.jpg",
                    images[idx].transpose(1, 2, 0)[:, :, ::-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    args = parser.parse_args()

    max_size  = 512
    init_step = int(math.log2(args.init_size) - 2)
    max_step  = int(math.log2(max_size) - 2)
    nsteps = max_step - init_step + 1

    lr = 1e-3
    mixing = False

    code_size = 64
    batch_size = {4: 512, 8: 256, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4, 512: 4, 1024: 2}
    batch_default = 32

    phase = 150_000
    max_iter = 100_000

    # transform = transform.Compose([
    #     transform.ToPILImage(),
    #     transform.RandomHorizontalFlip(),
    #     transform.ToTensor(),
    #     transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])

    netG = StyledGenerator(code_dim=code_size)
    netD = Discriminator(from_rgb_activate=True)
    g_running = StyledGenerator(code_size)
    g_running.eval()

    d_optimizer = jt.optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer = jt.optim.Adam(netG.generator.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({
        'params': netG.style.parameters(),
        'lr': lr * 0.01,
        'mult': 0.01,
        }
    )

    accumulate(g_running, netG, 0)

    if args.ckpt is not None:
        # ckpt = jt.load(args.ckpt)

        # netG.load_state_dict(ckpt['generator'])
        # netD.load_state_dict(ckpt['discriminator'])
        # g_running.load_state_dict(ckpt['g_running'])

        netG.load(f"styleGAN/saved_models/generator_{args.ckpt}.pkl")
        netD.load(f"styleGAN/saved_models/discriminator_{args.ckpt}.pkl")

        print('resuming from checkpoint .......')

    ## Actual Training
    step = init_step
    resolution = int(4 * 2 ** step)
    image_loader = ImageDataset(args.path, resolution, mode="train").set_attrs(
        batch_size=batch_size.get(resolution, batch_default),
        shuffle=True
    )
    train_loader = iter(image_loader)

    val_set = ImageDataset(args.path, resolution, mode="test").set_attrs(
                batch_size=1000, shuffle=False)
    val_loader = iter(val_set)

    # requires_grad(netG, False)
    stop_grad(netG)
    start_grad(netD)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0
    final_progress = False
    pbar = tqdm(range(max_iter))

    for i in pbar:
        alpha = min(1, 1 / phase * (used_sample + 1))
        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1
            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            try:
                # real_image = next(train_loader)
                _, valid_A, photo_id = next(val_loader)
            except (OSError, StopIteration):
                val_loader = iter(val_set)
                # real_image = next(train_loader)
                _, valid_A, photo_id = next(val_loader)

            with jt.no_grad():
                images = g_running(
                    jt.randn(1000, code_size), valid_A, step=step, alpha=alpha
                ).data

            valid(i, images, photo_id, step-1)

            image_loader = ImageDataset(args.path, resolution, mode="train").set_attrs(
                batch_size=batch_size.get(resolution, batch_default),
                shuffle=True
            )
            train_loader = iter(image_loader)

            val_set = ImageDataset(args.path, resolution, mode="test").set_attrs(
                batch_size=1000,
                shuffle=False
            )
            val_loader = iter(val_set)

            # jt.save(
            #     {
            #         'generator': netG.state_dict(),
            #         'discriminator': netD.state_dict(),
            #         'g_running': g_running.state_dict(),
            #     },
            #     f'checkpoint/train_step-{ckpt_step}.model',
            # )

            netG.save(os.path.join(f"styleGAN/saved_models/generator_{i}.pkl"))
            netD.save(os.path.join(f"styleGAN/saved_models/discriminator_{i}.pkl"))

        try:
            # real_image = next(train_loader)
            real_B, real_A, _ = next(train_loader)
        except (OSError, StopIteration):
            train_loader = iter(image_loader)
            # real_image = next(train_loader)
            real_B, real_A, _ = next(train_loader)

        real_B.start_grad() # imgs
        for A in real_A:
            A.start_grad()
            # A.requires_grad = True # labels:list:big to small
        b_size = real_B.size(0)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_AB = jt.contrib.concat((real_A[0], real_B), 1)
        # print(real_AB.shape)
        # print(real_A[0].shape)
        # print(real_B.shape)

        # print(type(real_B))
        # print(real_B.dtype)
        # a=jt.float32([1,2,3])

        # print(a.numpy())
        # print(real_A[0])
        # print(real_AB)
        # print(jt.concat((real_A[0], real_B), 1))
        # print(real_A.shape, real_B.shape, real_AB.shape)
        real_scores = netD(real_AB, step=step, alpha=alpha)
        real_predict = jt.nn.softplus(-real_scores).mean()
        grad_real = jt.grad(real_scores.sum(), real_AB)
        grad_penalty = (
            grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty

        if i % 10 == 0:
            grad_loss_val = grad_penalty.item()

        if mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.chunk(jt.randn(4, b_size, code_size),4, 0)
            gen_in1 = [jt.squeeze(gen_in11, 0), jt.squeeze(gen_in12, 0)]
            gen_in2 = [jt.squeeze(gen_in21, 0), jt.squeeze(gen_in22, 0)]
        else:
            gen_in1, gen_in2 = jt.chunk(jt.randn(2, b_size, code_size), 2, 0)
            gen_in1 = jt.squeeze(gen_in1, 0)
            gen_in2 = jt.squeeze(gen_in2, 0)

        fake_B = netG(gen_in1, real_A,step=step, alpha=alpha)
        # print(fake_B.shape)
        # print(real_A[0].shape)
        fake_AB = jt.contrib.concat((real_A[0], fake_B), 1)
        fake_predict = netD(fake_AB, step=step, alpha=alpha)
        fake_predict = jt.nn.softplus(fake_predict).mean()

        if i % 10 == 0:
            disc_loss_val = (real_predict + fake_predict).item()

        loss_D = real_predict + grad_penalty + fake_predict
        d_optimizer.step(loss_D)

        # optimize generator
        # ------------------
        #  Train Generators
        # ------------------
        # requires_grad(netG, True)
        start_grad(netG)
        stop_grad(netD)
        # requires_grad(netD, False)

        fake_B = netG(gen_in2, real_A, step=step, alpha=alpha)
        fake_AB = jt.contrib.concat((real_A[0], fake_B), 1)
        predict = netD(fake_AB, step=step, alpha=alpha)
        loss_G = jt.nn.softplus(-predict).mean()

        if i % 10 == 0:
            gen_loss_val = loss_G.item()

        g_optimizer.step(loss_G)

        accumulate(g_running, netG)
        # requires_grad(netG, False)
        # requires_grad(netD, True)
        start_grad(netD)
        stop_grad(netG)

        used_sample += real_B.shape[0]

        if (i + 1) % 1000 == 0:
            # images = np.array([])
            try:
                # real_image = next(train_loader)
                _, valid_A, photo_id = next(val_loader)
            except (OSError, StopIteration):
                val_loader = iter(val_set)
                # real_image = next(train_loader)
                _, valid_A, photo_id = next(val_loader)

            with jt.no_grad():
                images = g_running(
                    jt.randn(1000, code_size), valid_A, step=step, alpha=alpha
                ).data

            valid(i, images, photo_id)

            # jt.save_image(
            #     jt.concat(images, 0),
            #     f'sample/{str(i + 1).zfill(6)}.png',
            #     nrow=gen_i,
            #     normalize=True,
            #     range=(-1, 1),
            # )

        if (i + 1) % 10000 == 0:
            jt.save(g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.pkl')

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )
        pbar.set_description(state_msg)

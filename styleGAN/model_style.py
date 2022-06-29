import math
import jittor as jt
from jittor import Function as F
import numpy as np
import random
from math import sqrt

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight[0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        delattr(module, name)
        setattr(module, name + '_orig', weight)
        module.register_pre_forward_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)

def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualConv2d(jt.Module):
    def __init__(self, *args, **kwargs):
        self.trans = False
        try:
            # print(1)
            if kwargs['conv_transpose2d']:
                self.trans = True
                kwargs.pop('conv_transpose2d')
                # print(2)
                conv = jt.nn.ConvTranspose(*args, **kwargs)
                # print(3)
        except:
            conv = jt.nn.Conv2d(*args, **kwargs)
            jt.init.gauss_(conv.weight, 0, 1)
            jt.init.constant_(conv.bias,0)

        self.conv = equal_lr(conv)

    def execute(self, input):
        # if self.trans:
            # print("fanjuanji:")
        # else:
            # print("juanji")
        # print(input.shape)
        # print(self.conv(input).shape)
        return self.conv(input)

class EqualLinear(jt.Module):
    def __init__(self, in_dim, out_dim):
        linear = jt.nn.Linear(in_dim, out_dim)
        jt.init.gauss_(linear.weight, 0, 1)
        jt.init.constant_(linear.bias, 0)

        self.linear = equal_lr(linear)

    def execute(self, input):
        return self.linear(input)

class BlurFunctionBackward(jt.Function):
    def execute(self, grad_output, kernel, kernel_flip):
        self.saved_tensors = kernel, kernel_flip

        grad_input = jt.nn.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    def grad(self, gradgrad_output):
        kernel, kernel_flip = self.saved_tensors

        grad_input = jt.nn.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None

class BlurFunction(jt.Function):
    def execute(self, input, kernel, kernel_flip):
        self.saved_tensors = kernel, kernel_flip

        output = jt.nn.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    def grad(self, grad_output):
        kernel, kernel_flip = self.saved_tensors

        grad_input = BlurFunctionBackward().execute(grad_output, kernel, kernel_flip)

        return grad_input, None, None

blur = BlurFunction().apply

class Blur(jt.Module):
    def __init__(self, channel, upsample_factor=True):
        weight = jt.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype='float32')
        weight = weight.reshape(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = jt.flip(weight, [2, 3])

        self._weight = weight.repeat(channel, 1, 1, 1)
        self._weight_flip = weight_flip.repeat(channel, 1, 1, 1)

    def execute(self, input):
        # return jt.nn.conv2d(input, self.weight, padding=1, groups=input.shape[1])
        return blur(input, self._weight, self._weight_flip)

class FusedDownsample(jt.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        self.weight = jt.randn(out_channel, in_channel, kernel_size, kernel_size)
        self.bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.pad = padding

    def execute(self, input):
        weight = jt.nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]  +
            weight[:, :, :-1, 1:] +
            weight[:, :, 1:, :-1] +
            weight[:, :, :-1, :-1]
        ) / 4

        out = jt.nn.conv2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out

class FusedUpsample(jt.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        self.weight = jt.randn(in_channel, out_channel, kernel_size, kernel_size)
        self.bias = jt.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)
        self.pad = padding

    def execute(self, input):
        weight = jt.nn.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:] +
            weight[:, :, :-1, 1:] +
            weight[:, :, 1:, :-1] +
            weight[:, :, :-1, :-1]
        ) / 4

        out = jt.nn.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)
        return out

class ConvBlock(jt.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False
    ):
        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = jt.nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            jt.nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = jt.nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    jt.nn.LeakyReLU(0.2),
                )
            else:
                self.conv2 = jt.nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    jt.nn.AvgPool2d(2),
                    jt.nn.LeakyReLU(0.2),
                )
        else:
            self.conv2 = jt.nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                jt.nn.LeakyReLU(0.2),
            )
    def execute(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        return out

class Discriminator(jt.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        self.progression = jt.nn.ModuleList(
            [
                ConvBlock( 16,  32, 3, 1, downsample=True, fused=fused),  # 512
                ConvBlock( 32,  64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock( 64, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                ConvBlock(513, 512, 3, 1, 4, 0),
            ]
        )
        
        def make_from_rgb(out_channel):
            # if from_rgb_activate:
            # return jt.nn.Sequential(
            #     jt.nn.Conv2d(6, out_channels=513, kernel_size=3, stride=1, padding=1),
            #     jt.nn.ReLU()
            # )
            # else:
            #     return jt.nn.Conv2d(6, out_channels=out_channel, kernel_size=3, stride=1, padding=1)
            if from_rgb_activate:
                return jt.nn.Sequential(EqualConv2d(6, out_channel, 1), jt.nn.LeakyReLU(0.2))

            else:
                return EqualConv2d(6, out_channel, 1)

        self.from_rgb = jt.nn.ModuleList(
            [
                make_from_rgb(16),
                make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512),
            ]
        )
        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)

    def execute(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = np.std(out.numpy(), axis=0)
                mean_std = jt.array(out_std.mean())
                mean_std = mean_std.expand((out.size(0), 1, 4, 4))
                out = jt.concat([out, mean_std], 1)

            out = self.progression[index](out)
            # print(out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_rgb = jt.nn.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)

                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        return out

class ConstantInput(jt.Module):
    def __init__(self, channel, size=4):
        self.input = jt.randn(1, channel, size, size)

    def execute(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out

class NoiseInjection(jt.Module):
    def __init__(self, channel):
        self.weight = jt.zeros((1, channel, 1, 1))

    def execute(self, image, noise):
        # print(image.shape)
        # print(self.weight.shape)
        # print(noise.shape)
        return image + self.weight * noise

class AdaptiveInstanceNorm(jt.nn.Module):
    def __init__(self, in_channel, style_dim):
        self.norm = jt.nn.InstanceNorm2d(in_channel, affine=False)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def execute(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out

class EqualConv2d_(jt.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        lr_mul=1,
        bias=True,
        bias_init=0,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.weight = jt.nn.Parameter(
            jt.randn(out_channel, in_channel, kernel_size, kernel_size).divide(lr_mul)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            if bias_init:
                self.bias = jt.nn.Parameter(jt.ones(out_channel))
            else:
                self.bias = jt.nn.Parameter(jt.zeros(out_channel))

            self.lr_mul = lr_mul
        else:
            self.lr_mul = None

    def execute(self, input):
        if self.lr_mul != None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        out = jt.nn.conv2d(
            input,
            self.weight * self.scale,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
        )
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ModulatedConv2d(jt.nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        normalize_mode,
        blur_kernel,
        upsample=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        # if upsample:
        #     factor = 2
        #     p = (len(blur_kernel) - factor) - (kernel_size - 1)
        #     pad0 = (p + 1) // 2 + factor - 1
        #     pad1 = p // 2 + 1
        #     self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / sqrt(fan_in)
        self.padding = kernel_size // 2
        self.weight = jt.nn.Parameter(
            jt.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.normalize_mode = normalize_mode
        if normalize_mode == "InstanceNorm2d":
            self.norm = jt.nn.InstanceNorm2d(in_channel, affine=False)
        elif normalize_mode == "BatchNorm2d":
            self.norm = jt.nn.BatchNorm2d(in_channel, affine=False)

        self.beta = None

        self.gamma = EqualConv2d_(
            style_dim,
            in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            bias_init=1,
        )

        self.beta = EqualConv2d_(
            style_dim,
            in_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=True,
            bias_init=0,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample})"
        )

    def execute(self, input, stylecode):
        # print(input.shape)

        assert stylecode is not None
        batch, in_channel, height, width = input.shape
        repeat_size = input.shape[3] // stylecode.shape[3]

        gamma = self.gamma(stylecode)
        if self.beta:
            beta = self.beta(stylecode)
        else:
            beta = 0

        weight = self.scale * self.weight
        weight = jt.misc.repeat(weight,(batch, 1, 1, 1, 1))

        if self.normalize_mode in ["InstanceNorm2d", "BatchNorm2d"]:
            input = self.norm(input)
        elif self.normalize_mode == "LayerNorm":
            input = jt.nn.LayerNorm(input.shape[1:], elementwise_affine=False)(input)
        elif self.normalize_mode == "GroupNorm":
            input = jt.nn.GroupNorm(2 ** 3, input.shape[1:], affine=False)(input)
        elif self.normalize_mode == None:
            pass
        else:
            print("not implemented normalization")
        # print("happy")
        if self.upsample:
            input = jt.nn.ConvTranspose(self.in_channel, self.in_channel, kernel_size=4, stride=2, \
                 padding=1)(input)
        input = input * gamma + beta
        # 卷积
        out = jt.nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)(input)

        # weight = jt.reshape(weight,(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size))

        # if self.upsample:
        #     input = jt.reshape(input, (1, batch * in_channel, height, width))
        #     # input.view(1, batch * in_channel, height, width)
        #     weight = jt.reshape(weight, (
        #         batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
        #     ))
        #     # weight = weight.view(
        #     #     batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
        #     # )
        #     weight = weight.transpose(1, 2).reshape(
        #         batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
        #     )
        #     out = jt.nn.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
        #     _, _, height, width = out.shape
        #     out = jt.reshape(out, (batch, self.out_channel, height, width))
        #     # out.view(batch, self.out_channel, height, width)
        #     # print(height, width, out.shape)

        #     # out = self.blur(out)

        # else:
        # # if not self.upsample:
        #     input = jt.reshape(input, (1, batch * in_channel, height, width))
        #     # input.view(1, batch * in_channel, height, width)
        #     out = jt.nn.conv2d(input, weight, padding=self.padding, groups=batch)
        #     _, _, height, width = out.shape
        #     out = jt.reshape(out, (batch, self.out_channel, height, width))
        # # out.view(batch, self.out_channel, height, width)
            # print(height, width, out.shape)

        # print("zheli")
        # print(out.shape)
        return out

class StyledConv(jt.nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel,
        normalize_mode,
        upsample=False,
    ):
        super().__init__()
        # TODO:注意上采样
        # self.upsamle = upsample
        # if upsample:
        #     self.up_conv = jt.nn.ConvTranspose(
        #         in_channels = in_channel,
        #         out_channels = in_channel,
        #         kernel_size=4,
        #         padding=1,
        #         stride=2,)

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )

    def execute(self, input, style):
        # if self.upsamle:
        #     input = self.up_conv(input)
        out = self.conv(input, style)
        out = jt.nn.LeakyReLU(0.2)(out)
        return out

class StyledConvBlock(jt.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
        blur_kernel=[1, 3, 3, 1],
        normalize_mode="BatchNorm2d",
    ):
        self.initial = initial
        if initial:
            self.init = ConstantInput(in_channel)
            self.conv1 = StyledConv(
                in_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                upsample=False,
                normalize_mode=normalize_mode,
            )
        else:
            self.conv1 = StyledConv(
                in_channel,
                out_channel,
                3,
                style_dim,
                blur_kernel=blur_kernel,
                upsample=True,
                normalize_mode=normalize_mode,
            )

        self.conv2 = StyledConv(
            out_channel,
            out_channel,
            3,
            style_dim,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
        )
        # else:
        #     if upsample:
        #         if fused:
        #             self.conv1 = jt.nn.Sequential(
        #                 FusedUpsample(
        #                     in_channel, out_channel, kernel_size, padding=padding
        #                 ),
        #                 Blur(out_channel),
        #             )
        #         else:
        #             self.conv1 = jt.nn.Sequential(
        #                 jt.nn.Upsample(scale_factor=2, mode='nearest'),
        #                 EqualConv2d(
        #                     in_channel, out_channel, kernel_size, padding=padding
        #                 ),
        #                 Blur(out_channel),
        #             )
        #     else:
        #         self.conv1 = EqualConv2d(
        #             in_channel, out_channel, kernel_size, padding=padding
        #         )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        # self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        # self.lrelu1 = jt.nn.LeakyReLU(0.2)

        # self.conv2  = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        # self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        # self.lrelu2 = jt.nn.LeakyReLU(0.2)

    def execute(self, input, style, noise):
        # out = self.conv1(input)
        # out = self.noise1(out, noise)
        # out = self.lrelu1(out)
        # out = self.adain1(out, style)

        # out = self.conv2(out)
        # out = self.noise2(out, noise)
        # out = self.lrelu2(out)
        # out = self.adain2(out, style)
        if self.initial:
            # print("const")
            # print(input.shape)
            input = self.init(input)
            # print(input.shape)
        # print(input.shape)
        # print(self.conv1)
        out = self.conv1(input, style[0])
        out = self.noise1(out, noise)
        out = self.conv2(out, style[1])
        out = self.noise2(out, noise)

        return out

class PixelNorm(jt.Module):
    def __init__(self):
        pass

    def execute(self, input):
        return input / jt.sqrt(jt.mean(input ** 2, dim=1, keepdims=True) + 1e-8)

class ConvLayer(jt.nn.Sequential):
    """
    style 的卷积和上采样
    """
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        layers = []

        if upsample:
            # stride = 2
            # self.padding = 0
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size=4,
                    padding=1,
                    stride=2,
                    conv_transpose2d=True,
                )
            )
            layers.append(
                EqualConv2d(
                    out_channel,
                    out_channel,
                    kernel_size,
                    padding=1,
                    stride=1,
                )
            )

            # factor = 2
            # p = (len(blur_kernel) - factor) - (kernel_size - 1)
            # pad0 = (p + 1) // 2 + factor - 1
            # pad1 = p // 2 + 1

            # layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))
        else:
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=1,
                    stride=1,
                )
            )
        layers.append(jt.nn.LeakyReLU(0.2))
        super().__init__(*layers)

class Generator(jt.Module):
    def __init__(self, code_dim, fused=True):
        self.progression = jt.nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),   # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
                StyledConvBlock(128,  64, 3, 1, upsample=True, fused=fused),  # 256
                StyledConvBlock( 64,  32, 3, 1, upsample=True, fused=fused),  # 512
                # StyledConvBlock( 32,  16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        ## TODO： EqualConv2d
        self.to_rgb = jt.nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                # EqualConv2d(16, 3, 1),
            ]
        )

        # style（w） 使用的卷积（上采样）
        # style_code: w+
        self.convs_latent = jt.nn.ModuleList()
        self.channels = {
               4: 512,
               8: 512,
              16: 512,
              32: 512,
              64: 512,
             128: 256,
             256: 128,
             512:  64,
            1024:  32,
        }
        style_dim = 512
        stylecode_dim = self.channels[4]

        self.convs_latent.append(
            ConvLayer(style_dim, stylecode_dim, 3)
        )
        self.convs_latent.append(
            ConvLayer(stylecode_dim, stylecode_dim, 3)
        )

        for i in range(3, 11):
            stylecode_dim_prev = self.channels[2 ** (i - 1)]
            stylecode_dim_next = self.channels[2 ** i]
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_prev,
                    stylecode_dim_next,
                    kernel_size=3,
                    # upsample=True,
                )
            )
            self.convs_latent.append(
                ConvLayer(
                    stylecode_dim_next,
                    stylecode_dim_next,
                    kernel_size=3,
                )
            )
        # self.num_stylecodes = self.log_size * 2 - 2 * (self.start_index - 2)
        # the number of AdaIN layer(stylecodes)

        # 处理labels的卷积：输入labels，输出m_i
        label_convs = []
        for i in self.channels.keys():
            label_convs.append(EqualConv2d(3, self.channels[i], 3, padding=1))
            label_convs.append(EqualConv2d(3, self.channels[i], 3, padding=1))
        self.label_convs = jt.nn.ModuleList(label_convs)

    def execute(self, style, labels, noise, step=0, alpha=-1, mixing_range=(-1, -1)):
        # style = [512, 4, 4]
        # label = [8, size, size]
        style_codes = []
        conditions = []
        for i in range(len(labels)):
            # print(labels[i].shape)
            # print(self.label_convs[i])
            # 改成不一样的conditions
            conditions.append(self.label_convs[2*i](labels[len(labels) - 1 - i]))
            conditions.append(self.label_convs[2*i+1](labels[len(labels) - 1 - i]))

        old_style_code = style
        for i in range(2 * step + 2):
            # style_code : w+
            style_code = conditions[i] + old_style_code
            # print(i,":")
            # print(style_code.shape)
            style_code = self.convs_latent[i](style_code)
            style_codes.append(style_code)
            old_style_code = style_code
            if i % 2 == 1:
                old_style_code = jt.nn.ConvTranspose(
                    in_channels = self.channels[2 ** (int(i / 2) + 2)],
                    out_channels = self.channels[2 ** (int(i / 2) + 3)],
                    kernel_size=4,
                    padding=1,
                    stride=2,)(old_style_code)

        out = noise[0]
        # print("nimeihsiba")
        # print(len(style_codes))

        # if len(style) < 2:
        #     inject_index = [len(self.progression) + 1]
        # else:
        #     inject_index = sorted(random.sample(list(range(step)), len(style) - 1))

        # crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            # if mixing_range == (-1, -1):
            #     if crossover < len(inject_index) and i > inject_index[crossover]:
            #         crossover = min(crossover + 1, len(style))
            #     style_step = style[crossover]
            # else:
            #     if mixing_range[0] <= i <= mixing_range[1]:
            #         style_step = style[1]
            #     else:
            #         style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            style_step = [style_codes[2*i], style_codes[2*i+1]]
            # print(i,":")
            # print(style_codes[2*i].shape, style_codes[2*i+1].shape)
            out = conv(out, style_step, noise[i])

            if i == step:
                # print(out.shape)
                out = to_rgb(out)
                # print(out.shape)
                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = jt.nn.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out
                # print(out.shape)
                break
        # print(out.shape)
        return out

class StyledGenerator(jt.Module):
    def __init__(self, code_dim=64, n_mlp=8):
        self.generator = Generator(code_dim)

        layers = [PixelNorm()]
        for i in range(n_mlp - 1):
            layers.append(EqualLinear(code_dim, 2 * code_dim))
            layers.append(jt.nn.LeakyReLU(0.2))
            code_dim *= 2
        layers.append(EqualLinear(code_dim, code_dim))
        layers.append(jt.nn.LeakyReLU(0.2))

        self.style = jt.nn.Sequential(*layers)  #全连接

    def execute(
        self,
        input,  # latent z
        label,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
    ):

        ## 1
        styles = []
        if type(input) not in (list, tuple):
            input = [input]
        # print(self.style(input[0]).shape)
        batch = input[0].shape[0]
        # z -mlp-> w
        style = jt.reshape(self.style(input[0]), (batch, 512, 4, 4))   # w： [512, 4, 4]


        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(jt.randn(batch, 1, size, size))

        if mean_style is not None:
            styles_norm = []
            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        return self.generator(style, label, noise, step, alpha, mixing_range=mixing_range)

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdims=True)

        return style

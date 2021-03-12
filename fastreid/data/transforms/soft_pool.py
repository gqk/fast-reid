# -*- coding: utf-8 -*-

from typing import Union, Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image


def adaptive_soft_pool2d(
    input: torch.Tensor,
    output_size: Union[int, Tuple[int, int]],
    beta: float = 1.0,
):
    weight = (input * beta).exp()

    a = F.adaptive_avg_pool2d(input * weight, output_size)
    b = F.adaptive_avg_pool2d(weight, output_size)

    return a / b


def soft_pool2d(
    input: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Optional[Union[int, Tuple[int, int]]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    beta: float = 1.0,
):
    weight = (input * beta).exp()
    input = input * weight

    a = F.avg_pool2d(
        input, kernel_size=kernel_size, stride=stride, padding=padding
    )
    b = F.avg_pool2d(
        weight, kernel_size=kernel_size, stride=stride, padding=padding
    )

    return a / b


def soft_resize(
    input: Union[Image.Image, torch.Tensor],
    size: Union[int, Tuple[int, int]],
    interpolation: int = 2,
    beta: float = 1.0,
):
    reverse_transform = lambda x: x
    if not isinstance(input, torch.Tensor):
        assert isinstance(input, Image.Image)
        reverse_transform = partial(TF.to_pil_image, mode=input.mode)
        input = TF.to_tensor(input)

    size = [size] if isinstance(size, int) else size

    assert isinstance(size, (list, tuple))

    size_h, size_w = (size * 2)[:2]
    h, w = input.shape[-2:]

    downsample = h > size_h
    if len(size) < 2:
        if w < h:
            size_h = int(size_w * h / w)
            downsample = w > size_w
        else:
            size_w = int(size_h * w / h)

        if (w <= h and w == size_w) or (h <= w and h == size_h):
            return reverse_transform(input)

    if not downsample:
        return TF.resize(
            reverse_transform(input), size=size, interpolation=interpolation
        )

    output = adaptive_soft_pool2d(input, (size_h, size_w), beta)
    return reverse_transform(output)


class AdaptiveSoftPool2d(nn.Module):
    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        beta: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.beta = beta

    def forward(self, input):
        return adaptive_soft_pool2d(input, self.output_size, self.beta)


class SoftPool2d(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        beta: float = 1.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.beta = beta

    def forward(self, input: torch.Tensor):
        return soft_pool2d(input, self.kernel_size, self.stride, self.padding)


class SoftResize(nn.Module):
    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]],
        interpolation: int = 2,
        beta: float = 1.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.interpolation = interpolation
        self.beta = beta

    def forward(self, input: Union[Image.Image, torch.Tensor]):
        return soft_resize(
            input, self.output_size, self.interpolation, self.beta
        )

from typing import Union, Callable

import cv2
import numpy
import torch
import torchvision.transforms as tv_transforms
from PIL import Image

DEFAULT_TRANSFORMS: list[Union[torch.nn.Module, tv_transforms.ToTensor()]] = [
    tv_transforms.RandomRotation(degrees=25, expand=True),
    tv_transforms.RandomPerspective(distortion_scale=0.4, p=1),
    tv_transforms.GaussianBlur(kernel_size=(7, 13)),
    tv_transforms.ColorJitter(brightness=0.7, contrast=0.4)
]


class TransformsParams:
    """
    Class to store the parameters of the transforms

    :param image: PIL.Image.Image or torch.Tensor
    :param p: probability of the image being transformed. Default is 0.5
    :param transforms: list of torch.nn.Module, if None use the DEFAULT_TRANSFORMS
    """

    def __init__(self,
                 image: list[Union[Image.Image, torch.Tensor]],
                 p: float = 0.5,
                 transforms: list[torch.nn.Module] = None):
        self.image = image
        self.probability = p
        self.transforms = transforms or DEFAULT_TRANSFORMS


def compose(transforms_params: TransformsParams) -> Union[torch.Tensor, Callable]:
    """
    return a generator of transforms.Compose

    :param transforms_params: if None use the DEFAULT_TRANSFORMS
    :return: Generator of transforms.Compose
    """
    transforms = transforms_params.transforms if transforms_params else DEFAULT_TRANSFORMS[:]
    if len(transforms) == 0:
        raise ValueError('Empty transforms')

    if isinstance(transforms_params.image, Image.Image) and not isinstance(transforms[0], torch.Tensor):
        transforms = [tv_transforms.ToTensor()] + transforms
        return tv_transforms.Compose(transforms)(transforms_params.image)

    if not isinstance(transforms_params.image, torch.Tensor):
        raise ValueError(f'Invalid image type: {type(transforms_params.image)}')

    return tv_transforms.Compose(transforms)


print(type(compose(TransformsParams(image=[Image.open('6_test/1.jpg')]))))

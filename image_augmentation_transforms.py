from typing import Union, Generator

import cv2
import numpy
import torch
import torchvision.transforms as tv_transforms
from PIL import Image

DEFAULT_TRANSFORMS: list[list[any]] = [
    [
        tv_transforms.RandomRotation(degrees=25, expand=True)
    ],
    [
        tv_transforms.RandomPerspective(distortion_scale=0.4, p=1)
    ],
    [
        tv_transforms.GaussianBlur(kernel_size=(7, 13))
    ],
    [
        tv_transforms.ColorJitter(brightness=0.7, contrast=0.4)
    ]
]


def transforms_generator(image: Union[str, Image.Image, numpy.ndarray], transforms: list[torch.nn.Module] = None) \
        -> Generator[tv_transforms.Compose, None, None]:
    """
    return a generator of transforms.Compose

    :param image: file path or PIL.Image.Image or numpy.ndarray
    :param transforms: if None or empty will use the default_transforms
    :return: Generator of transforms.Compose
    """

    if isinstance(image, Image.Image):
        image = numpy.asarray(image)
    elif isinstance(image, str):
        if not image.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise ValueError(f'Invalid image file: {image}')
        # convert from non-RGB to RGB
        image = cv2.imread(image)[:, :, ::-1]
    elif not isinstance(image, numpy.ndarray):
        raise ValueError(f'Invalid image type: {type(image)}')

    transforms_: list[list[any]] = []
    if transforms is None or len(transforms) == 0:
        transforms_ = DEFAULT_TRANSFORMS[:]
    else:
        for transform in transforms:
            transforms_.append([transform])
    for transform in transforms_:
        if not isinstance(transform[0], tv_transforms.ToPILImage) and not isinstance(image, Image.Image):
            transform.insert(0, tv_transforms.ToPILImage())
        yield tv_transforms.Compose(transform)(image)

import random
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as tv_transforms

"""
    Default parameters explanation:
        Some of them are based on the torch.nn.Module library's default values.
        
        Other values have been carefully selected based on practical testing, 
        incorporating factors such as rotate, blur, brightness, contrast,erasing, 
        and real-world baseball card photography scenarios. 
        These values reflect a combination of objective measurements and the subjective judgment of developers.
        
    For example:
        why is degrees=15? Greater than 15 maybe too much, less than 15 may not be enough.
        It's a subjective judgment based on the experience of the developers.
"""
DEFAULT_TRANSFORMS: list[torch.nn.Module] = [
    tv_transforms.RandomRotation(degrees=15, expand=True),
    tv_transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
    tv_transforms.ColorJitter(brightness=0.6, contrast=0.3),
    tv_transforms.GaussianBlur(kernel_size=(7, 13)),
    tv_transforms.RandomErasing(p=0.5, scale=(0.01, 0.03), ratio=(0.1, 0.5))
]


class DataAugmentationProcessor:
    """
    Class to store the parameters of the transforms

    :param transforms: list of torch.nn.Module, if None will use the default transforms,
                        [RandomRotation, RandomPerspective, ColorJitter, GaussianBlur, RandomErasing]
    :param kwargs:
            Attributes of subclasses of torch.nn.Module.
            Examples: **kwargs = {
                            "degrees": 15,
                            "expand": True,
                            "distortion_scale": 0.3,
                            "p": 0.5,
                            "brightness": 0.6,
                            "contrast": 0.3,
                            "kernel_size": (7, 13),
                            "scale": (0.02, 0.33),
                            "ratio": (0.3, 3.3),
                        }
            torchvision.transforms.RandomRotation: degrees, expand
            torchvision.transforms.RandomPerspective: distortion_scale,p
            torchvision.transforms.ColorJitter: brightness, contrast
                ...
            // degrees (sequence or number): Range of degrees to select from.
                        If degrees is a number instead of sequence like (min, max), the range of degrees
                        will be (-degrees, +degrees).
            // expand (bool, optional): Optional expansion flag.
                        If true, expands the output to make it large enough to hold the entire rotated image.
                        If false or omitted, make the output image the same size as the input image.
                        Note that the expand flag assumes rotation around the center and no translation.
            // distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
                        Default is 0.5.
            // p (float): probability of the image being transformed. Default value is 0.5.
            ... etc.
    """

    def __init__(self, transforms: list[torch.nn.Module] = None, **kwargs):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = self.default_transforms(**kwargs)

    @staticmethod
    def default_transforms(**kwargs):
        default_transforms = []
        for t in DEFAULT_TRANSFORMS:
            for k, v in kwargs.items():
                if hasattr(t, k) and v is not None:
                    setattr(t, k, v)
            default_transforms.append(t)

        return default_transforms

    def compose(self, p: float = 0.5) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Compose the transforms with the image

        :param p: float, probability of binomial distribution, default is 0.5
        :return: Callable

        """
        transforms = self.transforms
        if transforms is None or len(transforms) == 0:
            return tv_transforms.Compose([])

        transforms = random.sample(transforms, np.random.binomial(len(transforms), p))
        return tv_transforms.Compose(transforms)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply the transforms to the image

        :param image: torch.Tensor, image tensor
        :return: torch.Tensor, transformed image tensor
        """
        return self.compose()(image)



from typing import Callable

import torch
import torchvision.transforms as tv_transforms
from PIL import Image

DEFAULT_TRANSFORMS = [
    tv_transforms.RandomRotation(degrees=20, expand=True),
    tv_transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    tv_transforms.ColorJitter(brightness=1, contrast=0.5),
    tv_transforms.GaussianBlur(kernel_size=(7, 13))
]


class DataAugmentationParams:
    """
    Class to store the parameters of the transforms

    :param transforms: list of torch.nn.Module, if None will use the default transforms,
                        [RandomRotation, RandomPerspective, ColorJitter, GaussianBlur]
    :param p: probability(from 0 to 1.0) of the image being transformed.
                        Default is 0.5, only set to transforms that have a probability.
                        If transforms is not None, will be ignored.
    :param image: PIL.Image.Image or torch.Tensor
    :param kwargs:
            Attributes of subclasses of torch.nn.Module.
            Examples:
                torchvision.transforms.RandomRotation: degrees, expand
                torchvision.transforms.RandomPerspective: distortion_scale,p
                torchvision.transforms.ColorJitter: brightness, contrast
                ...
            **kwargs = {
                            "degrees": 20,
                            "expand": True,
                            "distortion_scale": 0.5,
                            "p": 0.5,
                            "brightness": 1,
                            "contrast": 0.5
                        }
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

    def __init__(self, transforms: list[torch.nn.Module] = None, image: Image.Image | torch.Tensor = None, **kwargs):
        if transforms:
            self.transforms = transforms
        else:
            self.transforms = self.default_transforms(**kwargs)
        self.image = image

    @staticmethod
    def default_transforms(**kwargs):
        default_transforms = []
        for t in DEFAULT_TRANSFORMS:
            for k, v in kwargs.items():
                if hasattr(t, k) and v is not None:
                    setattr(t, k, v)
            default_transforms.append(t)

        return default_transforms


def compose(dap: DataAugmentationParams, call_compose: bool = False) \
        -> torch.Tensor | Callable[[Image.Image | torch.Tensor], torch.Tensor]:
    """
    Compose the transforms with the image
    :param dap: DataAugmentationParams
    :param call_compose: bool,
                        if false or transforms_params.image is None return the compose function,
                        otherwise return the transformed torch.Tensor
    :return: Callable

    """
    transforms = dap.transforms
    if transforms is None or len(transforms) == 0:
        return tv_transforms.Compose([])
    image = dap.image
    if image is None:
        return tv_transforms.Compose(transforms)

    if isinstance(image, Image.Image) and transforms[0] != tv_transforms.ToTensor():
        transforms = [tv_transforms.ToTensor()] + transforms

    if call_compose:
        return tv_transforms.Compose(transforms)(image)
    return tv_transforms.Compose(transforms)

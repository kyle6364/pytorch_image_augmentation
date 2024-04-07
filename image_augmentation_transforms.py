import random
from typing import Callable, Union

import torch
import torchvision.transforms as tv_transforms
from PIL import Image

"""
    Default parameters explanation:
        Some of them are based on the torch.nn.Module library's default values.
        
        Other values have been carefully selected based on practical testing, 
        incorporating factors such as rotate, blur, brightness, contrast,erasing, 
        and real-world baseball card photography scenarios. 
        These values reflect a combination of objective measurements and the subjective judgment of developers.
        
    For example:
        why is degrees=20? Greater than 20 maybe too much, less than 20 may not be enough.
        It's a subjective judgment based on the experience of the developers.
"""
DEFAULT_TRANSFORMS: list[torch.nn.Module] = [
    tv_transforms.RandomRotation(degrees=20, expand=True),
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
    :param image: PIL.Image.Image or torch.Tensor or list[PIL.Image.Image] or list[torch.Tensor]
    :param kwargs:
            Attributes of subclasses of torch.nn.Module.
            Examples: **kwargs = {
                            "degrees": 20,
                            "expand": True,
                            "distortion_scale": 0.5,
                            "p": 0.5,
                            "brightness": 1,
                            "contrast": 0.5,
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

    def __init__(self,
                 transforms: list[torch.nn.Module] = None,
                 image: Union[Image.Image, torch.Tensor, list[Image.Image], list[torch.Tensor]] = None,
                 **kwargs):
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

    def compose(self) \
            -> Union[
                Callable[[Union[Image.Image, torch.Tensor]], torch.Tensor],
                list[Callable[[Union[Image.Image, torch.Tensor]], torch.Tensor]]]:
        """
        Compose the transforms with the image

        if self.`image` is list then return list of Compose function
        else return Compose function

        :return: Callable

        """
        transforms = self.transforms
        if transforms is None or len(transforms) == 0:
            return tv_transforms.Compose([])
        image = self.image
        if image is None:
            return tv_transforms.Compose(transforms)

        original_transforms = transforms[:]
        transforms = self.__add_tensor_to_pipeline(image, transforms)

        if not isinstance(image, list):
            return tv_transforms.Compose(transforms)

        image_size = len(image)
        if image_size == 0:
            return tv_transforms.Compose(transforms)

        transforms_size = len(original_transforms)
        quotient = transforms_size // image_size
        remainder = transforms_size % image_size

        """
        We can aim to evenly distribute the given number of images among the specified transformations. 
        For example, if you have 100 images and 4 transformations (a, b, c, d), 
        we can allocate 25 images for each transformation: 25 for a, 25 for b, 25 for c, and 25 for d. 
        This distribution ensures an equal representation of each transformation type across the dataset.
        """
        compose = []
        if quotient > 0:
            for e in original_transforms:
                compose += [tv_transforms.Compose(e) for _ in range(quotient)]
        if remainder > 0:
            compose += [tv_transforms.Compose(t) for t in random.sample(original_transforms, remainder)]
        random.shuffle(compose)

        return compose

    @staticmethod
    def __add_tensor_to_pipeline(image, transforms):
        if isinstance(image, Image.Image) and transforms[0] != tv_transforms.ToTensor():
            return [tv_transforms.ToTensor()] + transforms
        if isinstance(image, list) and len(image) > 0:
            return DataAugmentationProcessor.__add_tensor_to_pipeline(image[0], transforms)
        return transforms

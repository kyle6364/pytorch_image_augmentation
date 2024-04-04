import os

import torch
import torchvision.transforms.functional as tv_convert
from PIL import Image
from matplotlib import pyplot as plt

import image_augmentation_transforms as iat


def _show_image(image, picture_name: str):
    plt.imshow(image)
    plt.title(picture_name)
    plt.show()


def _test_and_show(directory: str):
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            dap = iat.DataAugmentationParams(image=Image.open(os.path.join(root, file)))
            image: torch.Tensor = iat.compose(dap, call_compose=True)
            _show_image(tv_convert.to_pil_image(image), file)


if __name__ == '__main__':
    _test_and_show("6_test")

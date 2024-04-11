import os

import torch
import torchvision.transforms.functional as tv_convert
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

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
            source_image = Image.open(os.path.join(root, file))
            dap = iat.DataAugmentationProcessor(image=source_image, **{"p": 1})
            # dap.transforms.clear()
            to_tensor = transforms.ToTensor()
            tensor_image = to_tensor(source_image)
            compose = dap.compose()
            image = compose(tensor_image)
            _show_image(tv_convert.to_pil_image(image), file)


if __name__ == '__main__':
    _test_and_show("6_test")

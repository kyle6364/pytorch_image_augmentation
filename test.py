import os

from matplotlib import pyplot as plt

import image_augmentation_transforms as iat

from PIL import Image


def _show_image(image, picture_name: str):
    plt.imshow(image)
    plt.title(picture_name)
    plt.show()


def _test_and_show(directory: str):
    transform_names = []
    for t in iat.DEFAULT_TRANSFORMS:
        transform_names.append(t[0].__class__.__name__)
    for root, _, files in os.walk(directory):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            transforms = iat.compose(Image.open(os.path.join(root, file)))
            transforms()
            # for i, e in enumerate():
            #     _show_image(e, f'{transform_names[i]} Image')


if __name__ == '__main__':
    _test_and_show("6_test")

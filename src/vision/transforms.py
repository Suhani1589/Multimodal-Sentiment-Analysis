from __future__ import annotations

from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_image_transform(image_size: int = 224, train: bool = False):
    """Build ImageNet-compatible transforms."""
    ops = []
    if train:
        ops.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    else:
        ops.extend([transforms.Resize((image_size, image_size))])
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)

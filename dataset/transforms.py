"""Transform builders for hand age datasets."""
from __future__ import annotations

from torchvision import transforms


def build_transforms(img_size: int):
    """Return train/test transforms for a given square image size."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            #transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=(-360, 360)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_transform, test_transform


def build_normals_transform(img_size: int):
    """Return a transform for normal map images (resize + decode to [-1, 1] only).

    No spatial augmentations (flip/rotate) are applied because those would
    invalidate the normal vector directions. ColorJitter is also skipped.
    Normalize(0.5, 0.5) maps [0,1] ToTensor output to [-1,1].
    """
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

"""Transform builders for hand age datasets."""
from __future__ import annotations

from torchvision import transforms


def build_transforms(img_size: int):
    """Return train/test transforms for a given square image size."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            
            # Photometric robustness
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2, hue=0.02),
            transforms.RandomGrayscale(p=0.15),

            # Camera / focus robustness
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),

            transforms.ToTensor(),
            
            # Occlusion robustness (forces not relying on tiny regions)
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value='random'),

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

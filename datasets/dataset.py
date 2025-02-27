from dataclasses import dataclass
from typing import Any, Callable, List, Tuple
from torch.utils.data import DataLoader, Dataset
from numpy import ndarray
import os


class DataItem:
    def __init__(self, path: str, to_augment: bool, label: int):
        self.path = path
        self.to_augment = to_augment
        self.label = label


class DatasetGenerator(Dataset):
    def __init__(
        self,
        data,
        transform: Callable[[DataItem], ndarray],
        augment: Callable[[ndarray, ndarray], ndarray],
    ) -> None:
        super(DatasetGenerator, self).__init__()

        self.data: List[DataItem] = data
        self.transform: Callable[[DataItem], ndarray] = transform
        self.augment: Callable[[ndarray, ndarray], ndarray] = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[ndarray, ndarray]:
        img, label = self.transform(self.data[idx])
        image_path = self.data[idx].path

        if self.data[idx].to_augment:
            img, label = self.augment(img, label)

        if img is None or label is None:
            print(f"Warning: No data found for index {idx}")
        
        return img, label

    # def __getitem__(self, idx) -> Tuple[ndarray, ndarray]:
    #     image_data = self.data[idx]
    #     image_path = image_data.path

    #     img, label = self.transform(image_data)

    #     # Handle missing/corrupt images
    #     if img is None or label is None:
    #         print(f"Warning: Failed to load image at index {idx}: {image_path}")
    #         return self.__getitem__((idx + 1) % len(self.data))  # Skip and try the next image

    #     # Apply augmentation if required
    #     if image_data.to_augment:
    #         img, label = self.augment(img, label)

    #     return img, label


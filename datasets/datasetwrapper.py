import os
import cv2
import numpy as np
from typing import List, Any
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image
import io

from dataset.dataset import DataItem, DatasetGenerator


class DatasetWrapper:
    CLASS_NAMES = ["bonafide", "morph"]
    # label = 0 for bonafide 

    def __init__(
        self,
        root_dir: str,
        height: int = 224,
        width: int = 224,
        classes: int = 2,
    ) -> None:
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.classes = classes

    def loop_through_dir(
        self,
        dir: str,
        label: int,
        augment_times: int = 0,
    ) -> List[DataItem]:
        allowed_extensions = {".jpg", ".png", ".jpeg"}
        items: List[DataItem] = []

        cnt = 0
        for image_path in os.listdir(dir):
            if os.path.splitext(image_path)[1].lower() in allowed_extensions:
                image_path = os.path.join(dir, image_path)
                if cnt == 0:
                    print(image_path)
                    cnt = 3
                items.append(DataItem(image_path, False, label))
                items.extend(
                    DataItem(image_path, True, label) for _ in range(augment_times)
                )

        return items

    def transform(self, data: DataItem) -> Any:
        image = cv2.imread(data.path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {data.path}")
            return None, None  # Or handle this case as needed

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        image = (image - image.min()) / ((image.max() - image.min()) or 1.0)
        image = np.transpose(image.astype("float32"), axes=(2, 0, 1))

        label = np.zeros((self.classes), dtype=np.float32)
        label[data.label] = 1  # One-hot encoding

        return image, label

    def augment(self, image: np.ndarray, label: np.ndarray) -> Any:
        
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected image to be a numpy array, but got {type(image)}")

        
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.25),
                A.VerticalFlip(p=0.25),
                A.RandomBrightnessContrast(p=0.2),
                A.InvertImg(p=0.05),
                A.PixelDropout(p=0.02),
                RandomJPEGCompressionAlbumentations(
                    quality_min=50, quality_max=100, p=0.5
                ),
            ]
        )

        transformed = transform(image=image)
        transformed_image = transformed["image"]

        return transformed_image, label

    def augment_bonafide(self, images: List[np.ndarray], labels: List[np.ndarray]):
        """
        Augments all bonafide images in the dataset.
        Args:
            images (List[np.ndarray]): List of bonafide images.
            labels (List[np.ndarray]): Corresponding labels for the images.
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: Augmented images with labels.
        """
        augmented = []
        for img, lbl in zip(images, labels):
            
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            
            augmented_img, augmented_lbl = self.augment(img, lbl)
            augmented.append((augmented_img, augmented_lbl))
        return augmented

    def get_image_count(self, dataset_type: str) -> int:
        """
        Counts the number of images in the specified dataset type ('bonafide' or 'morph').
        Args:
            dataset_type (str): The type of dataset to count ('bonafide' or 'morph').
        Returns:
            int: The count of images in the specified dataset.
        """
        root_dir = os.path.join(self.root_dir, dataset_type)
        count = 0
        for root, _, files in os.walk(root_dir):
            count += len([f for f in files if f.endswith(('.jpg', '.png'))])
        return count

    def get_dataset(
        self,
        split_type,
        augment_times: int,
        batch_size: int,
        morph_types: List[str],
        num_models: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        data: List[DataItem] = []
        for morph_type in morph_types:
            for label, cid in enumerate(self.CLASS_NAMES):
                augment_count = augment_times * 2 if cid == "bonafide" else augment_times
                if cid == "morph":
                    cid = f"morph/{morph_type}/facedetect"

                data.extend(
                    self.loop_through_dir(
                        os.path.join(self.root_dir, cid, split_type),
                        label,
                        augment_count,
                    )
                )

        # Shuffle and divide data among models
        np.random.shuffle(data)
        datasets = []
        split_size = len(data) // num_models
        for i in range(num_models):
            subset = data[i * split_size : (i + 1) * split_size]
            datasets.append(
                DataLoader(
                    DatasetGenerator(subset, self.transform, self.augment),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            )

        if len(datasets) == 1:
            return datasets[0]
        return datasets

    def get_train_dataset(
        self,
        augment_times: int,
        batch_size: int,
        morph_types: List[str],
        num_models: int,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        return self.get_dataset(
            "train", augment_times, batch_size, morph_types, num_models, shuffle, num_workers
        )

    def get_test_dataset(
        self,
        augment_times: int,
        batch_size: int,
        morph_types: List[str],
        num_models: int,
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        return self.get_dataset(
            "test", augment_times, batch_size, morph_types, num_models, shuffle, num_workers
        )


class RandomJPEGCompressionAlbumentations(A.ImageOnlyTransform):
    def __init__(self, quality_min=30, quality_max=90, p=0.5, always_apply=False):
        super(RandomJPEGCompressionAlbumentations, self).__init__(p, always_apply)
        assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max

    def apply(self, img, **params):
        # Randomly select a JPEG quality
        quality = np.random.randint(self.quality_min, self.quality_max)

        # Apply JPEG compression to the image
        buffer = io.BytesIO()

        # Ensure the input image is uint8 type before saving
        img = (img * 255).astype(np.uint8)

        # Ensure the image is in [H, W, C] format for saving as JPEG
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]

        pil_img = Image.fromarray(img)  # Convert numpy image to PIL
        pil_img.save(buffer, format="JPEG", quality=quality)

        # Reload the image from the buffer
        buffer.seek(0)
        img = np.array(Image.open(buffer))

        # Convert back to [C, H, W] format
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0

        return img

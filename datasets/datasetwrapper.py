import random
import os
from os.path import isfile
import cv2
import numpy as np
from typing import List, Any
import albumentations as A
from torch.utils.data import DataLoader
from PIL import Image
import io

from datasets.dataset import DataItem, DatasetGenerator


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
        depth_dir: str,
        label: int,
        augment_times: int = 0,
    ) -> List[DataItem]:
        allowed_extensions = {".jpg", ".png", ".jpeg"}
        items: List[DataItem] = []

        cnt = 0
        for image_path in os.listdir(dir):
            if os.path.splitext(image_path)[1].lower() in allowed_extensions:
                image_path = os.path.join(dir, image_path)
                depth_path = os.path.join(depth_dir, image_path) 
                if os.path.isfile(image_path) and os.path.isfile(depth_path):
                    if cnt == 0:
                        print(image_path)
                        cnt = 3
                    items.append(DataItem(image_path, depth_path, False, label))
                    items.extend(
                        DataItem(image_path, depth_path, True, label) for _ in range(augment_times)
                    )
                else:
                    print(f"check color and depth image pair for {image_path}")

        return items

    def transform(self, data: DataItem) -> Any:
        color_img = cv2.imread(data.path, cv2.IMREAD_COLOR)
        if color_img is None:
            print(f"Failed to load image: {data.path}")
            return None, None  

        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, (self.width, self.height))
        color_img = (color_img - color_img.min()) / ((color_img.max() - color_img.min()) or 1.0)
        color_img = np.transpose(color_img.astype("float32"), axes=(2, 0, 1))

        depth_img = cv2.imread(data.depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"Failed to load depth image: {data.depth_path}")
            return None, None

        depth_img = depth_img.astype("float32")
        depth_img = cv2.resize(depth_img, (self.width, self.height))
        depth_img = (depth_img - depth_img.min()) / ((depth_img.max() - depth_img.min()) or 1.0)
        if len(depth_img.shape) == 2:
            depth_img = np.expand_dims(depth_img, axis=-1)  # (H, W, 1)
            depth_img = np.repeat(depth_img, 3, axis=-1)   # (H, W, 3)

        depth_img = np.transpose(depth_img, (2, 0, 1))  # (3, H, W)

        label = np.zeros((self.classes), dtype=np.float32)
        label[data.label] = 1  # One-hot encoding

        return color_img, depth_img, label



    def augment(self,color_img: np.ndarray, depth_img: np.ndarray, label: np.ndarray) -> Any:
        
        if not isinstance(color_img, np.ndarray):
            raise TypeError(f"Expected image to be a numpy array, but got {type(color_img)}")
        if not isinstance(depth_img, np.ndarray):
            raise TypeError(f"Expected image to be a numpy array, but got {type(depth_img)}")

        # Define the Transformations
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
            ],
            additional_targets={"depth": "image"}  # Register depth image for simultaneous transformation
        )

        # Apply the transformations
        transformed = transform(image=color_img, depth=depth_img)

        # Get Transformed Images
        transformed_color = transformed["image"]
        transformed_depth = transformed["depth"]

        
        # transform = A.Compose(
        #     [
        #         A.HorizontalFlip(p=0.25),
        #         A.VerticalFlip(p=0.25),
        #         A.RandomBrightnessContrast(p=0.2),
        #         A.InvertImg(p=0.05),
        #         A.PixelDropout(p=0.02),
        #         RandomJPEGCompressionAlbumentations(
        #             quality_min=50, quality_max=100, p=0.5
        #         ),
        #     ]
        # )

        # transformed = transform(image=image)
        # transformed_image = transformed["image"]

        return transformed_color, transformed_depth, label


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
        
        if morph_types == ['3D_morph']:
            print(morph_types)
            data: List[DataItem] = []
            for morph_type in morph_types:
                for label, cid in enumerate(self.CLASS_NAMES):
                    augment_count = augment_times * 2 if cid == "bonafide" else augment_times
                    if cid == "morph":
                        cid = "Morph"
                    elif cid == "bonafide":
                        cid = "Bona"

                    data.extend(
                        self.loop_through_dir(
                            os.path.join(self.root_dir, cid, "Color"),
                            os.path.join(self.root_dir, cid, "Depth"),
                            label,
                            augment_count,
                        )
                    )

            # Shuffle and divide data among models
            random.shuffle(data)
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

        else:   
            data: List[DataItem] = []
            for morph_type in morph_types:
                for label, cid in enumerate(self.CLASS_NAMES):
                    augment_count = augment_times * 2 if cid == "bonafide" else augment_times
                    if cid == "morph":
                        cid = f"morph/{morph_type}"
                        
                    depth_dir = self.root_dir.replace("color", "depth")

                    data.extend(
                        self.loop_through_dir(
                            os.path.join(self.root_dir, cid, split_type),
                            os.path.join(depth_dir, cid, split_type),
                            label,
                            augment_count,
                        )
                    )

            # Shuffle and divide data among models
            random.shuffle(data)
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
        quality = np.random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img = (img * 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  

        pil_img = Image.fromarray(img)  
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        img = np.array(Image.open(buffer))
        img = img.transpose(2, 0, 1)
        img = img.astype(np.float32) / 255.0

        return img

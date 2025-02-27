import os
import glob
from re import split
import cv2
import numpy as np
from typing import List, Any
import albumentations as A
from sympy import root
from torch.utils.data import DataLoader
from PIL import Image
import io
import torch

from dataset.dataset import DataItem, DatasetGenerator


class DatasetWrapper:
    CLASS_NAMES = ["bonafide", "morph"]

    def __init__(
        self,
        root_dir: str,
        morph_type: str,
        printer: str = "digital",
        morph_dir=None,
        height: int = 224,
        width: int = 224,
        classes: int = 2,
    ) -> None:
        self.root_dir = root_dir
        self.morph_type = morph_type
        if morph_dir != None:
            self.morph_dir = morph_dir
        else:
            self.morph_dir = root_dir
        self.height = height
        self.width = width
        self.classes = classes
        self.printer = printer

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
                    print(label, image_path)
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
            return None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height))
        image = (image - image.min()) / ((image.max() - image.min()) or 1.0)
        image = np.transpose(image.astype("float32"), axes=(2, 0, 1))

        # if data.label is None or not (0 <= data.label < self.classes):
        #     print(f"Invalid label {data.label} for image {data.path}")
        #     return None, None
        
        label = np.zeros((self.classes), dtype=np.float32)
        label[data.label] = 1

        return image, label


    def augment(self, image: np.ndarray, label: np.ndarray) -> Any:
        # print(f"Image type before augmentation: {type(image)}")

        
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
        # if isinstance(image, torch.Tensor):
        #     image = image.numpy()  # Convert tensor to numpy array
        # if isinstance(image, np.ndarray):
        #     image = np.moveaxis(image, 0, -1) 
        transformed = transform(image=image)
        transformed_image = transformed["image"]

        return transformed_image, label

    def get_dataset(
        self,
        split_type,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ):

        if self.printer == "digital":
            
            if morph_type in ['African_MAD_DB',  'CASIA_MAD_DB',  'Caucasian_MAD_DB',  'Indian_MAD_DB']:
                data: List[DataItem] = []
                for label, cid in enumerate(self.CLASS_NAMES):
                    augment_count = augment_times * 2 if cid == "bonafide" else augment_times
                    root_dir = self.root_dir
                    
                    if cid == "morph":
                        cid = "Morph/FaceDetect"
                    elif cid == "bonafide":
                        # cid = "Bona"
                        cid = "Bona/FaceDetect"
                    if split_type == "test":
                        split_type = "Test"
                    elif split_type == "train":
                        split_type = "Train"
                        
                    data.extend(
                        self.loop_through_dir(
                            os.path.join(root_dir, morph_type, split_type, cid ),
                            label,
                            augment_count,
                        )
                    )

                return DataLoader(
                    DatasetGenerator(data, self.transform, self.augment),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                )
                
            
            else:
                data: List[DataItem] = []
                for label, cid in enumerate(self.CLASS_NAMES):
                    augment_count = augment_times * 2 if cid == "bonafide" else augment_times
                    root_dir = self.root_dir
                
                    if cid == "morph":
                        cid = f"morph/{morph_type}/facedetect"
                        if self.morph_dir != self.root_dir:
                            root_dir = self.morph_dir
                            cid = ""
                    if cid == "bonafide":
                        cid = "bonafide/facedetect"   

                    data.extend(
                        self.loop_through_dir(
                            os.path.join(root_dir, cid, split_type),
                            label,
                            augment_count,
                        )
                    )

                return DataLoader(
                    DatasetGenerator(data, self.transform, self.augment),
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                )
            
        elif self.printer == "dnp":
            data: List[DataItem] = []
            for label, cid in enumerate(self.CLASS_NAMES):
                augment_count = augment_times * 2 if cid == "bonafide" else augment_times
                root_dir = self.root_dir
                
                if cid == "morph":
                    cid = "mor"
                elif cid == "bonafide":
                    cid = "bon"
                    
                data.extend(
                    self.loop_through_dir(
                        os.path.join(root_dir, morph_type, split_type, cid ),
                        label,
                        augment_count,
                    )
                )

            return DataLoader(
                DatasetGenerator(data, self.transform, self.augment),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            )
            
        else:
            raise ValueError("check printer value in datasetwrapper")
            

    def get_multiple_dataset(
        self,
        split_type,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
    ):
        data: List[DataItem] = []
        morphs = morph_type.split(".")
        for morph in morphs:
            for label, cid in enumerate(self.CLASS_NAMES):
                if morph == "post_process":
                    print(self.morph_dir)
                    self.morph_dir = (
                        "/home/ubuntu/volume/data/PostProcess_Data/digital/morph/after"
                    )
                    print(self.morph_dir)
                # print(f"morph_dir = {self.morph_dir}")

                augment_count = (
                    augment_times * 2 if cid == "bonafide" else augment_times
                )
                root_dir = self.root_dir
                if cid == "morph":
                    cid = f"morph/{morph}"
                    if self.morph_dir != self.root_dir:
                        root_dir = self.morph_dir
                        cid = ""
                        self.morph_dir = self.root_dir

                # print(f"root dir: {root_dir}")
                print(os.path.join(root_dir, cid, split_type))
                data.extend(
                    self.loop_through_dir(
                        os.path.join(root_dir, cid, split_type),
                        label,
                        augment_count,
                    )
                )

                # print(len(data))

        return DataLoader(
            DatasetGenerator(data, self.transform, self.augment),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_train_dataset(
        self,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
        multiple: bool = False,
    ):
        if multiple:
            return self.get_multiple_dataset(
                "train", augment_times, batch_size, morph_type, shuffle, num_workers
            )
        return self.get_dataset(
            "train", augment_times, batch_size, morph_type, shuffle, num_workers
        )

    def get_test_dataset(
        self,
        augment_times: int,
        batch_size: int,
        morph_type: str,
        shuffle: bool = True,
        num_workers: int = 8,
        multiple: bool = False,
    ):
        if multiple:
            return self.get_multiple_dataset(
                "train", augment_times, batch_size, morph_type, shuffle, num_workers
            )
        return self.get_dataset(
            "test", augment_times, batch_size, morph_type, shuffle, num_workers
        )


class RandomJPEGCompression(object):
    def __init__(self, quality_min=30, quality_max=90, p=0.5):
        assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if np.random.rand(1) > self.p:
            return img
        # Choose a random quality for JPEG compression
        quality = np.random.randint(self.quality_min, self.quality_max)

        # Save the image to a bytes buffer using JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)

        # Reload the image from the buffer
        img = Image.open(buffer)
        return img


# class RandomJPEGCompressionAlbumentations(A.ImageOnlyTransform):
#     def __init__(self, quality_min=30, quality_max=90, p=0.5, always_apply=False):
#         super(RandomJPEGCompressionAlbumentations, self).__init__(p, always_apply)
#         assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
#         self.quality_min = quality_min
#         self.quality_max = quality_max

#     def apply(self, img, **params):
#         # Randomly select a JPEG quality
#         quality = np.random.randint(self.quality_min, self.quality_max)

#         # Apply JPEG compression to the image
#         buffer = io.BytesIO()
#         # print(f"uegfruyjsegf {type(img)}")
#         img = (img * 255).astype(np.uint8)
#         img_reshaped = img.reshape((224, 224, 3))
#         pil_img = Image.fromarray(img_reshaped)  # Convert numpy image to PIL
#         pil_img.save(buffer, format="JPEG", quality=quality)

#         # Reload the image from the buffer
#         buffer.seek(0)
#         img = np.array(Image.open(buffer))
#         # img = torch.tensor(img).permute(2, 0, 1)
#         return img


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

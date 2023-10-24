import os
from pathlib import Path
from typing import List

import albumentations as A
import albumentations.pytorch
import numpy as np
import rasterio
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset

no_augmentation = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256, p=1, always_apply=True),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

# Non destructive transformations - Dehidral group D4
nodestructive_pipe = A.OneOf(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        # Experimental augmentations below
        # A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5), # Mild perturbations to position and orientation
        # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5) #  Simulates small deformations in the image
    ],
    p=1,
)

weak_augmentation = A.Compose(
    [
        A.PadIfNeeded(min_height=256, min_width=256, p=1, always_apply=True),
        nodestructive_pipe,
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)

# imagenet normalisation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = A.Normalize(mean=mean, std=std, p=1)


class VCDDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        img_ids: List[str],
        augmentation: A.Compose = no_augmentation,
        nbands: int = 3,
    ):
        """
        Venus change detection dataset, currently assumed to be RGB with 8 bit imagery.
        Args:
            root_dir (str): Path to the root directory of the dataset
            split (str): One of "train", "val", or "test"
            img_ids (List[str]): List of image ids to use for the dataset
            augmentation (A.Compose, optional): Augmentation pipeline to apply to the images. Defaults to no_augmentation.
            nbands (int, optional): Number of bands to use from each image. Defaults to 3 - RGB.
        """

        # Initialize a list to store valid subfolders
        self.split = split
        self.image_ids = img_ids
        self.augmentation = augmentation
        self.nbands = nbands
        self.valid_subfolders = []

        # Iterate through the subdirectories in the root directory
        for subfolder in os.listdir(root_dir):
            if not subfolder in self.image_ids:
                continue
            subfolder_path = os.path.join(root_dir, subfolder)

            # Check if the subfolder is a directory
            if os.path.isdir(subfolder_path):
                # Check if the subfolder contains the required image files
                required_files = {"before.tif", "after.tif", "mask.tif"}
                subfolder_files = set(os.listdir(subfolder_path))

                if required_files.issubset(subfolder_files):
                    self.valid_subfolders.append(Path(subfolder_path))

    def __len__(self):
        return len(self.valid_subfolders)

    def read_and_preprocess_image(self, image_path):
        """
        Read and preprocess an image from the specified file path.

        Args:
            image_path (str): The path to the image file.

        Returns:
            np.ndarray: The preprocessed image as a NumPy array. If the image is single-channel,
            the returned array will have shape (height, width). If it's multi-channel, it will have
            shape (height, width, channels).

        """
        with rasterio.open(image_path) as src:
            image = src.read() / 255.0  # Divide by 255.0 for normalization
            image = image.astype(np.float32)

            # Check if it's a single-channel image, and if so, drop the channel dimension
            if image.shape[0] == 1:
                image = image[0]  # Drop the channel dimension
                return image
            image = np.transpose(image, (1, 2, 0))  # Transpose CxHxW to HxWxC
            return image

    def __getitem__(self, idx):
        before_path = Path(self.valid_subfolders[idx] / "before.tif")
        after_path = Path(self.valid_subfolders[idx] / "after.tif")
        mask_path = Path(self.valid_subfolders[idx] / "mask.tif")

        before_image = self.read_and_preprocess_image(before_path)
        after_image = self.read_and_preprocess_image(after_path)
        mask = self.read_and_preprocess_image(mask_path)

        # Normalisation did not improve result
        # before_image = normalize(image=before_image)["image"] # normalise the RGB only
        # after_image = normalize(image=after_image)["image"] # normalise the RGB only

        # before_image and after_image are HxWxC numpy arrays
        image = np.concatenate([before_image, after_image], axis=2)

        # Apply augmentations
        data = {"image": image, "mask": mask}
        augmented = self.augmentation(**data)  # returns tensors
        image, mask = augmented["image"], augmented["mask"]

        image1 = image[: self.nbands, :, :]
        image2 = image[self.nbands :, :, :]
        mask = mask.unsqueeze(0)  # add a channel dimension
        return {
            "image1": image1.float(),
            "image2": image2.float(),
            "mask": mask.float(),
            "img_id": self.valid_subfolders[idx].name,
        }


class VCDDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        train_folders: List[str],
        val_folders: List[str],
        test_folders: List[str],
        num_workers: int,
        batch_size: int,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.train_folders = train_folders
        self.val_folders = val_folders
        self.test_folders = test_folders
        self.num_workers = num_workers
        self.batch_size = batch_size

    def setup(self, stage=None):
        if not stage in ["fit", "test", None]:
            raise ValueError(f"Invalid stage: {stage}")
        if stage == "fit" or stage is None:
            self.train_dataset = VCDDataset(
                self.root_dir,
                split="train",
                img_ids=self.train_folders,
                augmentation=weak_augmentation,
            )
            self.val_dataset = VCDDataset(
                self.root_dir,
                split="val",
                img_ids=self.val_folders,
                augmentation=weak_augmentation,
            )

        if stage == "test" or stage is None:
            self.test_dataset = VCDDataset(
                self.root_dir,
                split="test",
                img_ids=self.test_folders,
                augmentation=no_augmentation,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
        )

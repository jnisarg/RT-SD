import os

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CityscapesDataModule(L.LightningDataModule):

    def __init__(
        self,
        root,
        train_size=(1024, 1024),
        test_size=(1024, 2048),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        ignore_index=255,
        train_batch_size=8,
        test_batch_size=1,
        num_workers=8,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        self.dataset_kwargs = {
            "root": root,
            "train_size": train_size,
            "test_size": test_size,
            "mean": mean,
            "std": std,
            "ignore_index": ignore_index,
        }

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CityscapesDataset(mode="train", **self.dataset_kwargs)

        self.val_dataset = CityscapesDataset(mode="eval", **self.dataset_kwargs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


class CityscapesDataset(Dataset):
    def __init__(
        self,
        root,
        mode="train",
        train_size=(1024, 1024),
        test_size=(1024, 2048),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        ignore_index=255,
    ):
        super().__init__()
        self.root = root
        self.mode = mode
        self.train_size = train_size
        self.test_size = test_size
        self.mean = mean
        self.std = std
        self.ignore_index = ignore_index

        assert mode in ["train", "eval", "test"], f"Invalid mode: {mode}"

        with open(os.path.join(root, f"{mode}.txt"), "r", encoding="utf-8") as fr:
            self.samples = fr.read().splitlines()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_path, mask_path = sample.split()

        info = {"name": os.path.basename(image_path)}

        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        image, mask = self._transforms(image, mask)

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(self.mean, self.std)(image)
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask, info

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(os.path.join(self.root, image_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        return cv2.imread(os.path.join(self.root, mask_path), cv2.IMREAD_GRAYSCALE)

    def _transforms(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ):
        transform = self._get_transform()
        transformed = transform(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"]

        mask = self._map_mask(mask)

        return image, mask

    def _get_transform(self) -> A.Compose:
        if self.mode == "train":
            return A.Compose(
                [
                    A.RandomScale(scale_limit=0.5),
                    A.PadIfNeeded(
                        min_height=self.train_size[0],
                        min_width=self.train_size[1],
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                        mask_value=self.ignore_index,
                    ),
                    A.RandomCrop(
                        height=self.train_size[0],
                        width=self.train_size[1],
                        always_apply=True,
                    ),
                    A.HorizontalFlip(p=0.5),
                ]
            )

        return A.Compose(
            [
                A.Resize(
                    height=self.test_size[0],
                    width=self.test_size[1],
                    always_apply=True,
                ),
            ]
        )

    def _map_mask(self, mask: np.ndarray) -> np.ndarray:
        temp = mask.copy()
        for k, v in self._get_mask_mapping().items():
            mask[temp == k] = v
        return mask

    def _get_mask_mapping(self):
        return {
            -1: self.ignore_index,
            0: self.ignore_index,
            1: self.ignore_index,
            2: self.ignore_index,
            3: self.ignore_index,
            4: self.ignore_index,
            5: self.ignore_index,
            6: self.ignore_index,
            7: 0,
            8: 1,
            9: self.ignore_index,
            10: self.ignore_index,
            11: 2,
            12: 3,
            13: 4,
            14: self.ignore_index,
            15: self.ignore_index,
            16: self.ignore_index,
            17: 5,
            18: self.ignore_index,
            19: 6,
            20: 7,
            21: 8,
            22: 9,
            23: 10,
            24: 11,
            25: 12,
            26: 13,
            27: 14,
            28: 15,
            29: self.ignore_index,
            30: self.ignore_index,
            31: 16,
            32: 17,
            33: 18,
        }

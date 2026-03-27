import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from transforms.segmentation import build_image_transform, build_mask_transform


class RoadDataset(Dataset):
    def __init__(self, image_dir, image_size=1024, augment=True, require_masks=True):
        self.image_dir = image_dir
        self.image_size = image_size
        self.augment = augment
        self.require_masks = require_masks

        sat_ids = {f.replace("_sat.jpg", "") for f in os.listdir(image_dir) if f.endswith("_sat.jpg")}
        mask_ids = {f.replace("_mask.png", "") for f in os.listdir(image_dir) if f.endswith("_mask.png")}

        if require_masks:
            self.files = sorted(sat_ids & mask_ids)
        else:
            self.files = sorted(sat_ids)

        self.transform = build_image_transform(image_size)
        self.mask_transform = build_mask_transform(image_size)

    def _apply_geometric_augment(self, image, mask):
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            image = torch.rot90(image, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        if random.random() > 0.4:
            angle = random.uniform(-12.0, 12.0)
            max_shift = int(0.05 * self.image_size)
            translate_x = random.randint(-max_shift, max_shift)
            translate_y = random.randint(-max_shift, max_shift)
            scale = random.uniform(0.95, 1.05)

            image = TF.affine(
                image,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[translate_x, translate_y],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
            )

        return image, mask

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]

        sat_path = os.path.join(self.image_dir, f"{file_id}_sat.jpg")
        image = Image.open(sat_path).convert("RGB")
        image = torch.as_tensor(self.transform(image), dtype=torch.float32)

        mask_path = os.path.join(self.image_dir, f"{file_id}_mask.png")
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")
            mask = torch.as_tensor(self.mask_transform(mask), dtype=torch.float32)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        if self.augment:
            image, mask = self._apply_geometric_augment(image, mask)

        return image, mask


def load_eval_samples(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_ids = data.get("train_ids", [])
    valid_ids = data.get("valid_ids", [])
    if len(train_ids) != 10 or len(valid_ids) != 10:
        raise ValueError(f"Expected 10 train_ids and 10 valid_ids in {config_path}")

    return train_ids, valid_ids


def build_train_valid_datasets(train_dir, image_size, valid_split, seed):
    full_train_no_aug = RoadDataset(train_dir, image_size=image_size, augment=False, require_masks=True)
    full_train_aug = RoadDataset(train_dir, image_size=image_size, augment=True, require_masks=True)

    indices = list(range(len(full_train_no_aug)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    valid_size = int(len(indices) * valid_split)
    valid_size = max(1, valid_size)
    valid_size = min(valid_size, len(indices) - 1)

    valid_indices = indices[:valid_size]
    train_indices = indices[valid_size:]

    train_dataset = Subset(full_train_aug, train_indices)
    valid_dataset = Subset(full_train_no_aug, valid_indices)
    return train_dataset, valid_dataset


def build_dataloaders(train_dataset, valid_dataset, batch_size, num_workers):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from training.dataset import RoadDataset
from utils.visualization import draw_overlay

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "unet", "augmentation_preview")


def tensor_to_hwc_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return image.clip(0.0, 1.0)


def tensor_to_hw_mask(tensor):
    return tensor.detach().cpu().squeeze(0).numpy()


def build_id_to_index(dataset):
    return {file_id: idx for idx, file_id in enumerate(dataset.files)}

def save_preview_for_file(file_id, idx, base_dataset, aug_dataset, num_variants):
    original_image, original_mask = base_dataset[idx]

    n_cols = 1 + num_variants # Original + variants
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # If n_cols == 1, axes is not an array, but we have 4 so it's fine.
    
    # ORIGINAL
    draw_overlay(
        axes[0], 
        tensor_to_hwc_image(original_image), 
        tensor_to_hw_mask(original_mask), 
        title="ORIGINAL"
    )

    # AUGMENTATIONS
    for i in range(num_variants):
        aug_image, aug_mask = aug_dataset[idx]
        col = i + 1
        
        draw_overlay(
            axes[col],
            tensor_to_hwc_image(aug_image),
            tensor_to_hw_mask(aug_mask),
            title=f"AUGMENTATION EX {i + 1}"
        )

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"augment_overlay_4_in_a_row_{file_id}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 4-in-a-row geometric augmentation overlays")
    parser.add_argument("--file-id", type=str, default="948521", help="Target file ID to preview")
    parser.add_argument("--num-variants", type=int, default=3, help="How many augmented variants to show next to original")
    parser.add_argument("--image-size", type=int, default=512, help="Preview resize")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_dataset = RoadDataset(TRAIN_DIR, image_size=args.image_size, augment=False, require_masks=True)
    aug_dataset = RoadDataset(TRAIN_DIR, image_size=args.image_size, augment=True, require_masks=True)

    id_to_index = build_id_to_index(base_dataset)

    # Fallback to the first available ID if the requested one isn't found
    file_id = args.file_id
    if file_id not in id_to_index:
        print(f"File ID {file_id} not found in train set. Using first available.")
        file_id = base_dataset.files[0]
        
    save_preview_for_file(
        file_id=file_id,
        idx=id_to_index[file_id],
        base_dataset=base_dataset,
        aug_dataset=aug_dataset,
        num_variants=args.num_variants,
    )


if __name__ == "__main__":
    main()

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from training.dataset import RoadDataset, load_eval_samples

DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
EVAL_SAMPLES_PATH = os.path.join(PROJECT_ROOT, "src", "configs", "eval_samples.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "augmentation_preview")


def tensor_to_hwc_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return image.clip(0.0, 1.0)


def tensor_to_hw_mask(tensor):
    return tensor.detach().cpu().squeeze(0).numpy()


def build_id_to_index(dataset):
    return {file_id: idx for idx, file_id in enumerate(dataset.files)}


def save_preview_for_file(file_id, idx, base_dataset, aug_dataset, num_variants):
    original_image, original_mask = base_dataset[idx]

    n_cols = 2 + 2 * num_variants
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.8))

    axes[0].imshow(tensor_to_hwc_image(original_image))
    axes[0].set_title("ORIGINAL IMAGE")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_hw_mask(original_mask), cmap="gray")
    axes[1].set_title("ORIGINAL MASK")
    axes[1].axis("off")

    for i in range(num_variants):
        aug_image, aug_mask = aug_dataset[idx]
        image_col = 2 + 2 * i
        mask_col = image_col + 1

        axes[image_col].imshow(tensor_to_hwc_image(aug_image))
        axes[image_col].set_title(f"AUG IMAGE {i + 1}")
        axes[image_col].axis("off")

        axes[mask_col].imshow(tensor_to_hw_mask(aug_mask), cmap="gray")
        axes[mask_col].set_title(f"AUG MASK {i + 1}")
        axes[mask_col].axis("off")

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"augment_preview_{file_id}.png")
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate geometric augmentation preview images")
    parser.add_argument("--num-samples", type=int, default=4, help="How many train IDs to preview")
    parser.add_argument("--num-variants", type=int, default=4, help="How many augmented variants per sample")
    parser.add_argument("--image-size", type=int, default=256, help="Preview resize")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_dataset = RoadDataset(TRAIN_DIR, image_size=args.image_size, augment=False, require_masks=True)
    aug_dataset = RoadDataset(TRAIN_DIR, image_size=args.image_size, augment=True, require_masks=True)

    eval_train_ids, _ = load_eval_samples(EVAL_SAMPLES_PATH)
    id_to_index = build_id_to_index(base_dataset)

    selected_ids = [file_id for file_id in eval_train_ids if file_id in id_to_index][: args.num_samples]
    if len(selected_ids) < args.num_samples:
        missing = args.num_samples - len(selected_ids)
        fallback_ids = [file_id for file_id in base_dataset.files if file_id not in selected_ids]
        selected_ids.extend(random.sample(fallback_ids, min(missing, len(fallback_ids))))

    for file_id in selected_ids:
        save_preview_for_file(
            file_id=file_id,
            idx=id_to_index[file_id],
            base_dataset=base_dataset,
            aug_dataset=aug_dataset,
            num_variants=args.num_variants,
        )

    print(f"Saved {len(selected_ids)} preview files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

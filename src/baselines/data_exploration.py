import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from utils.visualization import load_label_from_id, load_satellite_image, save_segmentation_grid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "baselines")


def load_image(path):
    return np.array(Image.open(path))


def plot_samples(n_samples=4):
    train_dir = os.path.join(DATA_DIR, "train")
    files = [f.replace("_sat.jpg", "") for f in os.listdir(train_dir) if f.endswith("_sat.jpg")]
    files = random.sample(files, min(n_samples, len(files)))

    samples = []
    for file_id in files:
        sat_img = load_satellite_image(train_dir, file_id)
        mask_img = load_label_from_id(train_dir, file_id)
        samples.append({
            "image": sat_img,
            "label": mask_img,
            "overlay": mask_img,
        })

    save_segmentation_grid(samples, os.path.join(LOG_DIR, "data_samples.png"), dpi=150, bbox_inches=None)


def analyze_dataset():
    splits = ["train", "valid", "test"]

    print("Dataset Analysis")
    print("=" * 50)

    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue

        sat_files = [f for f in os.listdir(split_dir) if f.endswith("_sat.jpg")]
        mask_files = [f for f in os.listdir(split_dir) if f.endswith("_mask.png")]

        print(f"\n{split.upper()}:")
        print(f"  Images: {len(sat_files)}")
        print(f"  Masks: {len(mask_files)}")

    print("\n" + "=" * 50)
    print("Class Distribution")
    print("=" * 50)

    class_df = pd.read_csv(os.path.join(DATA_DIR, "class_dict.csv"))
    print(class_df.to_string(index=False))


def analyze_masks():
    train_dir = os.path.join(DATA_DIR, "train")
    files = [f.replace("_mask.png", "") for f in os.listdir(train_dir) if f.endswith("_mask.png")]

    road_pixels = []
    total_pixels = []

    for file_id in files[:100]:
        mask = load_label_from_id(train_dir, file_id)

        road_pixels.append(np.sum(mask > 0))
        total_pixels.append(mask.size)

    road_ratio = np.array(road_pixels) / np.array(total_pixels)

    print("\nMask Statistics (first 100 samples):")
    print(f"  Mean road ratio: {road_ratio.mean():.2%}")
    print(f"  Std road ratio: {road_ratio.std():.2%}")
    print(f"  Min road ratio: {road_ratio.min():.2%}")
    print(f"  Max road ratio: {road_ratio.max():.2%}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(road_ratio, bins=30, edgecolor="black")
    plt.xlabel("Road Pixel Ratio")
    plt.ylabel("Count")
    plt.title("Distribution of Road Coverage")

    plt.subplot(1, 2, 2)
    sample_mask = load_label_from_id(train_dir, files[0])
    plt.hist(sample_mask.flatten(), bins=50, edgecolor="black")
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    plt.title("Mask Pixel Value Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "mask_analysis.png"), dpi=150)


def check_image_sizes():
    splits = ["train", "valid", "test"]

    sizes = []
    for split in splits:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.exists(split_dir):
            continue

        files = [f for f in os.listdir(split_dir) if f.endswith("_sat.jpg")][:10]
        for file_name in files:
            img = load_image(os.path.join(split_dir, file_name))
            sizes.append(img.shape)

    unique_sizes = list(set(sizes))
    print(f"\nImage Sizes: {unique_sizes}")


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    analyze_dataset()
    check_image_sizes()
    plot_samples(4)
    analyze_masks()

import json
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "baselines")
EVAL_SAMPLES_PATH = os.path.join(PROJECT_ROOT, "src", "configs", "eval_samples.json")
RANDOM_SEED = 42
ANALYSIS_SAMPLE_SIZE = 50

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(LOG_DIR, exist_ok=True)


def load_image(path):
    return np.array(Image.open(path))


def load_eval_samples(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_ids = data.get("train_ids", [])
    valid_ids = data.get("valid_ids", [])
    if len(train_ids) != 10 or len(valid_ids) != 10:
        raise ValueError(f"Expected 10 train_ids and 10 valid_ids in {config_path}")
    return train_ids, valid_ids


def apply_kmeans(sat_img, kmeans):
    h, w, _ = sat_img.shape
    pixels = sat_img.reshape(-1, 3)
    labels = kmeans.predict(pixels)
    centers = kmeans.cluster_centers_
    return labels.reshape(h, w), centers


def find_road_cluster(centers, method="brightest"):
    if method == "brightest":
        brightness = centers.mean(axis=1)
        return np.argmax(brightness)
    if method == "grayscale":
        variance = (
            np.abs(centers[:, 0] - centers[:, 1])
            + np.abs(centers[:, 1] - centers[:, 2])
            + np.abs(centers[:, 0] - centers[:, 2])
        )
        return np.argmin(variance)
    if method == "highest_g":
        return np.argmax(centers[:, 1])
    return 0


def calculate_iou(pred, gt):
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    if union == 0:
        return 0.0
    return intersection / union


def calculate_precision_recall(pred, gt):
    tp = np.logical_and(pred > 0, gt > 0).sum()
    fp = np.logical_and(pred > 0, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt > 0).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def save_standardized_grid(file_ids, split_dir, include_label, output_path, best_kmeans, best_method):
    n_cols = 4 if include_label else 3
    fig, axes = plt.subplots(len(file_ids), n_cols, figsize=(4 * n_cols, 3.6 * len(file_ids)))
    if len(file_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, file_id in enumerate(file_ids):
        sat_path = os.path.join(split_dir, f"{file_id}_sat.jpg")
        sat_img = load_image(sat_path)

        labels, centers = apply_kmeans(sat_img, best_kmeans)
        road_cluster = find_road_cluster(centers, best_method)
        pred_mask = (labels == road_cluster).astype(np.uint8) * 255

        axes[i, 0].imshow(sat_img)
        axes[i, 0].axis("off")

        pred_col = 1
        if include_label:
            mask_path = os.path.join(split_dir, f"{file_id}_mask.png")
            mask_img = load_image(mask_path)
            if len(mask_img.shape) == 3:
                mask_img = mask_img[:, :, 0]
            axes[i, 1].imshow(mask_img, cmap="gray")
            axes[i, 1].axis("off")
            pred_col = 2

        axes[i, pred_col].imshow(pred_mask, cmap="gray")
        axes[i, pred_col].axis("off")

        overlay_col = pred_col + 1
        axes[i, overlay_col].imshow(sat_img)
        overlay_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
        axes[i, overlay_col].imshow(overlay_mask, cmap="autumn", alpha=0.45)
        axes[i, overlay_col].axis("off")

        if i == 0:
            axes[i, 0].set_title("IMAGE")
            if include_label:
                axes[i, 1].set_title("LABEL")
            axes[i, pred_col].set_title("PREDICT")
            axes[i, overlay_col].set_title("OVERLAY")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    train_dir = os.path.join(DATA_DIR, "train")
    valid_dir = os.path.join(DATA_DIR, "valid")
    eval_train_ids, eval_valid_ids = load_eval_samples(EVAL_SAMPLES_PATH)

    all_files = [f.replace("_sat.jpg", "") for f in os.listdir(train_dir) if f.endswith("_sat.jpg")]
    sample_files = random.sample(all_files, min(ANALYSIS_SAMPLE_SIZE, len(all_files)))

    print(f"Loading {len(sample_files)} images for K-Means analysis...")

    images = []
    masks = []
    for file_id in sample_files:
        sat_path = os.path.join(train_dir, f"{file_id}_sat.jpg")
        mask_path = os.path.join(train_dir, f"{file_id}_mask.png")

        sat_img = load_image(sat_path)
        mask_img = load_image(mask_path)
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]

        images.append(sat_img)
        masks.append(mask_img)

    all_pixels = []
    for sat_img in images[:10]:
        pixels = sat_img.reshape(-1, 3)
        idx = np.random.choice(len(pixels), min(10000, len(pixels)), replace=False)
        all_pixels.append(pixels[idx])

    all_pixels = np.vstack(all_pixels)
    print(f"K-Means training on {len(all_pixels):,} pixels")

    k_values = [2, 3, 4, 5, 6]
    kmeans_models = {}
    for k in k_values:
        print(f"  Fitting K-Means with k={k}...")
        km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=100)
        km.fit(all_pixels)
        kmeans_models[k] = km

    test_files = eval_train_ids
    test_images = []
    test_masks = []
    for file_id in test_files:
        sat_path = os.path.join(train_dir, f"{file_id}_sat.jpg")
        mask_path = os.path.join(train_dir, f"{file_id}_mask.png")
        sat_img = load_image(sat_path)
        mask_img = load_image(mask_path)
        if len(mask_img.shape) == 3:
            mask_img = mask_img[:, :, 0]
        test_images.append(sat_img)
        test_masks.append(mask_img)

    results = []
    for k in k_values:
        kmeans = kmeans_models[k]
        for method in ["brightest", "grayscale", "highest_g"]:
            ious = []
            precisions = []
            recalls = []

            for sat_img, mask_img in zip(test_images, test_masks):
                labels, centers = apply_kmeans(sat_img, kmeans)
                road_cluster = find_road_cluster(centers, method)
                pred_mask = (labels == road_cluster).astype(np.uint8) * 255

                iou = calculate_iou(pred_mask, mask_img)
                precision, recall = calculate_precision_recall(pred_mask, mask_img)
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)

            results.append({
                "k": k,
                "method": method,
                "IoU": float(np.mean(ious)),
                "Precision": float(np.mean(precisions)),
                "Recall": float(np.mean(recalls)),
                "centers": kmeans.cluster_centers_,
            })

    best_result = max(results, key=lambda x: x["IoU"])
    best_k = best_result["k"]
    best_method = best_result["method"]
    best_kmeans = kmeans_models[best_k]

    fig, axes = plt.subplots(len(test_images), 4, figsize=(16, 4 * len(test_images)))
    for i, (sat_img, mask_img) in enumerate(zip(test_images, test_masks)):
        labels, centers = apply_kmeans(sat_img, best_kmeans)
        road_cluster = find_road_cluster(centers, best_method)
        pred_mask = (labels == road_cluster).astype(np.uint8) * 255

        axes[i, 0].imshow(sat_img)
        axes[i, 0].set_title("Satellite")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask_img, cmap="gray")
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_mask, cmap="gray")
        axes[i, 2].set_title("K-Means Prediction")
        axes[i, 2].axis("off")

        cluster_img = centers[labels].astype(np.uint8).reshape(sat_img.shape)
        axes[i, 3].imshow(cluster_img)
        axes[i, 3].set_title(f"K={best_k} Clusters")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "kmeans_results.png"), dpi=150)

    save_standardized_grid(
        eval_train_ids,
        train_dir,
        include_label=True,
        output_path=os.path.join(LOG_DIR, "kmeans_predictions_train.png"),
        best_kmeans=best_kmeans,
        best_method=best_method,
    )
    save_standardized_grid(
        eval_valid_ids,
        valid_dir,
        include_label=False,
        output_path=os.path.join(LOG_DIR, "kmeans_predictions_valid.png"),
        best_kmeans=best_kmeans,
        best_method=best_method,
    )

    print("\n" + "=" * 60)
    print("SUMMARY: K-Means for Road Detection")
    print("=" * 60)
    print(f"Best configuration: K={best_result['k']}, method={best_result['method']}")
    print(f"Best IoU: {best_result['IoU']:.4f}")
    print(f"Best Precision: {best_result['Precision']:.4f}")
    print(f"Best Recall: {best_result['Recall']:.4f}")


if __name__ == "__main__":
    main()

import json
import os
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import stats
from utils.visualization import load_label_from_id, load_satellite_image, save_segmentation_grid

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
LOG_DIR = os.path.join(PROJECT_ROOT, "outputs", "baselines")
EVAL_SAMPLES_PATH = os.path.join(PROJECT_ROOT, "src", "configs", "eval_samples.json")
RANDOM_SEED = 42
ANALYSIS_SAMPLE_SIZE = 50
MAX_KS_SAMPLES = 50000

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


def sample_for_ks(data, max_samples=MAX_KS_SAMPLES):
    if len(data) <= max_samples:
        return data
    idx = np.random.choice(len(data), size=max_samples, replace=False)
    return data[idx]


def find_optimal_threshold_ks(data1, data2, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0, 256)

    ks_stats = []
    for threshold in thresholds:
        group_below = data1[data1 <= threshold]
        group_above = data2[data2 > threshold]
        if len(group_below) == 0 or len(group_above) == 0:
            ks_stats.append(0)
            continue

        ks_result: Any = stats.ks_2samp(group_below, group_above)
        ks_stats.append(float(ks_result[0]))

    best_idx = np.argmax(ks_stats)
    return thresholds[best_idx], ks_stats


def find_optimal_range_ks(road_data, non_road_data, step=10):
    best_range = (0, 255)
    best_ks = 0

    for low in range(0, 256, step):
        for high in range(low + step, 256, step):
            road_in_range = road_data[(road_data >= low) & (road_data <= high)]
            non_road_in_range = non_road_data[(non_road_data >= low) & (non_road_data <= high)]
            if len(road_in_range) < 100 or len(non_road_in_range) < 100:
                continue

            road_outside = road_data[(road_data < low) | (road_data > high)]
            non_road_outside = non_road_data[(non_road_data < low) | (non_road_data > high)]
            if len(road_outside) < 100 or len(non_road_outside) < 100:
                continue

            ks1_result: Any = stats.ks_2samp(road_in_range, non_road_in_range)
            ks2_result: Any = stats.ks_2samp(road_outside, non_road_outside)
            combined_ks = float(ks1_result[0]) + float(ks2_result[0])
            if combined_ks > best_ks:
                best_ks = combined_ks
                best_range = (low, high)

    return best_range, best_ks


def apply_ks_threshold(sat_img, r_thresh, g_thresh, b_thresh, mode="above"):
    r, g, b = sat_img[:, :, 0], sat_img[:, :, 1], sat_img[:, :, 2]
    if mode == "above":
        mask = (r >= r_thresh) & (g >= g_thresh) & (b >= b_thresh)
    elif mode == "below":
        mask = (r <= r_thresh) & (g <= g_thresh) & (b <= b_thresh)
    elif mode == "range":
        mask = (
            (r >= r_thresh[0])
            & (r <= r_thresh[1])
            & (g >= g_thresh[0])
            & (g <= g_thresh[1])
            & (b >= b_thresh[0])
            & (b <= b_thresh[1])
        )
    else:
        mask = (r >= r_thresh) | (g >= g_thresh) | (b >= b_thresh)
    return mask.astype(np.uint8) * 255


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


def save_standardized_grid(file_ids, split_dir, include_label, output_path, strategy):
    samples = []
    for file_id in file_ids:
        sat_img = load_satellite_image(split_dir, file_id)

        pred_mask = apply_ks_threshold(
            sat_img,
            strategy["r_thresh"],
            strategy["g_thresh"],
            strategy["b_thresh"],
            strategy["mode"],
        )

        sample = {
            "image": sat_img,
            "prediction": pred_mask,
            "overlay": pred_mask,
        }
        if include_label:
            sample["label"] = load_label_from_id(split_dir, file_id)
        samples.append(sample)

    save_segmentation_grid(samples, output_path, dpi=150, bbox_inches=None)


def main():
    train_dir = os.path.join(DATA_DIR, "train")
    valid_dir = os.path.join(DATA_DIR, "valid")
    eval_train_ids, eval_valid_ids = load_eval_samples(EVAL_SAMPLES_PATH)
    all_files = [f.replace("_sat.jpg", "") for f in os.listdir(train_dir) if f.endswith("_sat.jpg")]
    sample_files = random.sample(all_files, min(ANALYSIS_SAMPLE_SIZE, len(all_files)))

    road_pixels_r, road_pixels_g, road_pixels_b = [], [], []
    non_road_pixels_r, non_road_pixels_g, non_road_pixels_b = [], [], []

    for file_id in sample_files:
        sat_path = os.path.join(train_dir, f"{file_id}_sat.jpg")
        sat_img = load_image(sat_path)
        mask_img = load_label_from_id(train_dir, file_id)

        road_mask = mask_img > 0
        non_road_mask = mask_img == 0
        road_pixels_r.append(sat_img[:, :, 0][road_mask])
        road_pixels_g.append(sat_img[:, :, 1][road_mask])
        road_pixels_b.append(sat_img[:, :, 2][road_mask])
        non_road_pixels_r.append(sat_img[:, :, 0][non_road_mask])
        non_road_pixels_g.append(sat_img[:, :, 1][non_road_mask])
        non_road_pixels_b.append(sat_img[:, :, 2][non_road_mask])

    road_r = np.concatenate(road_pixels_r)
    road_g = np.concatenate(road_pixels_g)
    road_b = np.concatenate(road_pixels_b)
    non_road_r = np.concatenate(non_road_pixels_r)
    non_road_g = np.concatenate(non_road_pixels_g)
    non_road_b = np.concatenate(non_road_pixels_b)

    road_r_ks = sample_for_ks(road_r)
    road_g_ks = sample_for_ks(road_g)
    road_b_ks = sample_for_ks(road_b)
    non_road_r_ks = sample_for_ks(non_road_r)
    non_road_g_ks = sample_for_ks(non_road_g)
    non_road_b_ks = sample_for_ks(non_road_b)

    print(f"KS sample sizes -> road: {len(road_r_ks):,}, non-road: {len(non_road_r_ks):,}")

    r_thresh, _ = find_optimal_threshold_ks(road_r_ks, non_road_r_ks)
    g_thresh, _ = find_optimal_threshold_ks(road_g_ks, non_road_g_ks)
    b_thresh, _ = find_optimal_threshold_ks(road_b_ks, non_road_b_ks)
    r_range, _ = find_optimal_range_ks(road_r_ks, non_road_r_ks)
    g_range, _ = find_optimal_range_ks(road_g_ks, non_road_g_ks)
    b_range, _ = find_optimal_range_ks(road_b_ks, non_road_b_ks)

    ks_strategies = [
        {"name": "KS: R>G>B thresholds", "r_thresh": r_thresh, "g_thresh": g_thresh, "b_thresh": b_thresh, "mode": "above"},
        {"name": "KS: R>G>B below", "r_thresh": r_thresh, "g_thresh": g_thresh, "b_thresh": b_thresh, "mode": "below"},
        {"name": "KS: Range RGB", "r_thresh": r_range, "g_thresh": g_range, "b_thresh": b_range, "mode": "range"},
        {"name": "KS: Any channel above", "r_thresh": r_thresh, "g_thresh": g_thresh, "b_thresh": b_thresh, "mode": "any"},
        {"name": "Manual: Bright (150,150,150)", "r_thresh": 150, "g_thresh": 150, "b_thresh": 150, "mode": "above"},
    ]

    results = []
    for strategy in ks_strategies:
        ious, precisions, recalls = [], [], []
        for file_id in eval_train_ids:
            sat_path = os.path.join(train_dir, f"{file_id}_sat.jpg")
            sat_img = load_image(sat_path)
            mask_img = load_label_from_id(train_dir, file_id)

            pred_mask = apply_ks_threshold(
                sat_img,
                strategy["r_thresh"],
                strategy["g_thresh"],
                strategy["b_thresh"],
                strategy["mode"],
            )
            iou = calculate_iou(pred_mask, mask_img)
            precision, recall = calculate_precision_recall(pred_mask, mask_img)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

        results.append(
            {
                "strategy": strategy["name"],
                "IoU": float(np.mean(ious)),
                "Precision": float(np.mean(precisions)),
                "Recall": float(np.mean(recalls)),
            }
        )

    best_result = max(results, key=lambda x: x["IoU"])
    best_strategy = next(s for s in ks_strategies if s["name"] == best_result["strategy"])

    save_standardized_grid(
        eval_train_ids,
        train_dir,
        include_label=True,
        output_path=os.path.join(LOG_DIR, "rgb_threshold_predictions_train.png"),
        strategy=best_strategy,
    )
    save_standardized_grid(
        eval_valid_ids,
        valid_dir,
        include_label=False,
        output_path=os.path.join(LOG_DIR, "rgb_threshold_predictions_valid.png"),
        strategy=best_strategy,
    )

    print("\n" + "=" * 60)
    print("SUMMARY: RGB Thresholding for Road Detection")
    print("=" * 60)
    print(f"Best Strategy: {best_result['strategy']}")
    print(f"IoU: {best_result['IoU']:.4f}")
    print(f"Precision: {best_result['Precision']:.4f}")
    print(f"Recall: {best_result['Recall']:.4f}")


if __name__ == "__main__":
    main()

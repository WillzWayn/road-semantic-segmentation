import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from transforms.segmentation import build_inference_transform


DEFAULT_IMAGE_TITLE = "Satellite Image"
DEFAULT_LABEL_TITLE = "Ground Truth"
DEFAULT_PREDICTION_TITLE = "Predicted Mask"
DEFAULT_OVERLAY_TITLE = "Overlay Prediction"


def load_satellite_image(split_dir, file_id, image_size=None):
    sat_path = os.path.join(split_dir, f"{file_id}_sat.jpg")
    image = Image.open(sat_path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size))
    return np.array(image)


def load_mask_image(mask_path, image_size=None, binary=False):
    mask = Image.open(mask_path).convert("L")
    if image_size is not None:
        mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)

    mask_np = np.array(mask)
    if binary:
        return (mask_np > 0).astype(np.uint8)
    return mask_np


def load_label_from_id(split_dir, file_id, image_size=None, binary=False):
    mask_path = os.path.join(split_dir, f"{file_id}_mask.png")
    return load_mask_image(mask_path, image_size=image_size, binary=binary)


def build_overlay_mask(mask):
    return np.ma.masked_where(np.asarray(mask) == 0, np.asarray(mask))


def draw_image(ax, image, title=None):
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=10)


def draw_mask(ax, mask, title=None, cmap="gray"):
    ax.imshow(mask, cmap=cmap)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=10)


def draw_overlay(ax, image, mask, title=None, cmap="autumn", alpha=0.45):
    ax.imshow(image)
    ax.imshow(build_overlay_mask(mask), cmap=cmap, alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, pad=10)


def _normalize_grid_axes(axes, n_rows, n_cols):
    if n_rows == 1:
        return np.expand_dims(axes, axis=0)
    if n_cols == 1:
        return np.expand_dims(axes, axis=1)
    return axes


def save_segmentation_grid(samples, out_path, title=None, dpi=150, bbox_inches="tight"):
    if not samples:
        raise ValueError("samples must contain at least one item")

    panel_order = ["image", "label", "prediction", "overlay"]
    active_panels = [panel for panel in panel_order if any(sample.get(panel) is not None for sample in samples)]
    if not active_panels:
        raise ValueError("samples must include at least one visualization panel")

    panel_titles = {
        "image": DEFAULT_IMAGE_TITLE,
        "label": DEFAULT_LABEL_TITLE,
        "prediction": DEFAULT_PREDICTION_TITLE,
        "overlay": DEFAULT_OVERLAY_TITLE,
    }
    figure_width = 4 * len(active_panels)
    figure_height = 3.6 * len(samples)
    fig, axes = plt.subplots(len(samples), len(active_panels), figsize=(figure_width, figure_height))
    axes = _normalize_grid_axes(axes, len(samples), len(active_panels))

    for row_idx, sample in enumerate(samples):
        for col_idx, panel in enumerate(active_panels):
            ax = axes[row_idx, col_idx]
            panel_title = panel_titles[panel] if row_idx == 0 else None

            if panel == "image":
                draw_image(ax, sample["image"], title=panel_title)
                continue

            if panel == "label":
                draw_mask(ax, sample["label"], title=panel_title)
                continue

            if panel == "prediction":
                draw_mask(ax, sample["prediction"], title=panel_title)
                continue

            overlay_mask = sample.get("overlay")
            if overlay_mask is None:
                overlay_mask = sample.get("prediction")
            if overlay_mask is None:
                raise ValueError("overlay panel requires 'overlay' or 'prediction' data")

            draw_overlay(ax, sample["image"], overlay_mask, title=panel_title)

    if title:
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98 + (0.02 if len(samples) <= 4 else 0))

    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.92 if len(samples) > 4 else 0.88)
    plt.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
    plt.close(fig)


def predict_mask_from_id(model, split_dir, file_id, image_size, device, threshold=0.5):
    sat_path = os.path.join(split_dir, f"{file_id}_sat.jpg")
    image_pil = Image.open(sat_path).convert("RGB")

    inference_transform = build_inference_transform(image_size)
    input_tensor = torch.as_tensor(inference_transform(image_pil), dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_prob = torch.sigmoid(output[0, 0]).cpu().numpy()
        pred_mask = (pred_prob > threshold).astype(np.uint8)

    image_np = np.array(image_pil.resize((image_size, image_size)))
    return image_np, pred_mask


def save_prediction_grid(model, file_ids, split_dir, out_path, include_label, image_size, device, threshold=0.5, title=None):
    samples = []
    for file_id in file_ids:
        image_np, pred_mask = predict_mask_from_id(
            model,
            split_dir,
            file_id,
            image_size,
            device,
            threshold=threshold,
        )

        sample = {
            "image": image_np,
            "prediction": pred_mask,
            "overlay": pred_mask,
        }
        if include_label:
            sample["label"] = load_label_from_id(
                split_dir,
                file_id,
                image_size=image_size,
            )

        samples.append(sample)

    save_segmentation_grid(
        samples,
        out_path,
        title=title,
        dpi=150,
        bbox_inches="tight",
    )

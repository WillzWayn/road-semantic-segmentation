import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from transforms.segmentation import build_inference_transform


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
    n_cols = 4 if include_label else 3
    fig, axes = plt.subplots(len(file_ids), n_cols, figsize=(4 * n_cols, 3.6 * len(file_ids)))
    if title:
        fig.suptitle(title, fontsize=20, fontweight="bold", y=0.98 + (0.02 if len(file_ids) <= 4 else 0))

    if len(file_ids) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, file_id in enumerate(file_ids):
        image_np, pred_mask = predict_mask_from_id(
            model,
            split_dir,
            file_id,
            image_size,
            device,
            threshold=threshold,
        )

        axes[i, 0].imshow(image_np)
        axes[i, 0].axis("off")

        pred_col = 1
        if include_label:
            mask_path = os.path.join(split_dir, f"{file_id}_mask.png")
            label = np.array(
                Image.open(mask_path).convert("L").resize(
                    (image_size, image_size),
                    resample=Image.Resampling.NEAREST,
                )
            )
            axes[i, 1].imshow(label, cmap="gray")
            axes[i, 1].axis("off")
            pred_col = 2

        axes[i, pred_col].imshow(pred_mask, cmap="gray")
        axes[i, pred_col].axis("off")

        overlay_col = pred_col + 1
        axes[i, overlay_col].imshow(image_np)
        overlay_mask = np.ma.masked_where(pred_mask == 0, pred_mask)
        axes[i, overlay_col].imshow(overlay_mask, cmap="autumn", alpha=0.45)
        axes[i, overlay_col].axis("off")

        if i == 0:
            axes[i, 0].set_title("Satellite Image", fontsize=14, pad=10)
            if include_label:
                axes[i, 1].set_title("Ground Truth", fontsize=14, pad=10)
            axes[i, pred_col].set_title("Predicted Mask", fontsize=14, pad=10)
            axes[i, overlay_col].set_title("Overlay Prediction", fontsize=14, pad=10)

    plt.tight_layout()
    if title:
        fig.subplots_adjust(top=0.92 if len(file_ids) > 4 else 0.88)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

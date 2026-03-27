import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from models.unet import UNet
from training.dataset import load_eval_samples
from training.utils import save_prediction_grid


def main():
    parser = argparse.ArgumentParser(description="Run local U-Net predictions from downloaded checkpoint")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_unet.pth"),
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Inference image size",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "train", "both"],
        default="valid",
        help="Which split to render",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid threshold used for binary mask prediction",
    )
    parser.add_argument(
        "--train-output",
        default="unet_predictions_train_local.png",
        help="Output filename for train visualization inside outputs/unet",
    )
    parser.add_argument(
        "--valid-output",
        default="unet_predictions_valid_local.png",
        help="Output filename for valid visualization inside outputs/unet",
    )
    args = parser.parse_args()

    train_dir = os.path.join(PROJECT_ROOT, "dataset", "train")
    valid_dir = os.path.join(PROJECT_ROOT, "dataset", "valid")
    eval_samples_path = os.path.join(PROJECT_ROOT, "src", "configs", "eval_samples.json")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "unet")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = UNet(in_channels=3, out_channels=1, base_channels=64).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_train_ids, eval_valid_ids = load_eval_samples(eval_samples_path)

    if args.split in {"train", "both"}:
        save_prediction_grid(
            model=model,
            file_ids=eval_train_ids,
            split_dir=train_dir,
            out_path=os.path.join(out_dir, args.train_output),
            include_label=True,
            image_size=args.image_size,
            device=device,
            threshold=args.threshold,
        )
        print(f"Saved: outputs/unet/{args.train_output}")

    if args.split in {"valid", "both"}:
        save_prediction_grid(
            model=model,
            file_ids=eval_valid_ids,
            split_dir=valid_dir,
            out_path=os.path.join(out_dir, args.valid_output),
            include_label=False,
            image_size=args.image_size,
            device=device,
            threshold=args.threshold,
        )
        print(f"Saved: outputs/unet/{args.valid_output}")


if __name__ == "__main__":
    main()

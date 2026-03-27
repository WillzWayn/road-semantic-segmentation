import argparse
import os
import sys
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from training.dataset import load_eval_samples
from training.utils import save_prediction_grid

def main():
    parser = argparse.ArgumentParser(description="Generate 3 and 4 example predictions for U-Net and DeepLabV3")
    parser.add_argument("--image-size", type=int, default=512, help="Inference image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold")
    args = parser.parse_args()

    unet_checkpoint = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_unet.pth")
    deeplab_checkpoint = os.path.join(PROJECT_ROOT, "outputs", "checkpoints", "best_deeplabv3.pth")

    valid_dir = os.path.join(PROJECT_ROOT, "dataset", "valid")
    eval_samples_path = os.path.join(PROJECT_ROOT, "src", "configs", "eval_samples.json")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    _, eval_valid_ids = load_eval_samples(eval_samples_path)

    # 1. Run U-Net
    if os.path.exists(unet_checkpoint):
        print("Running U-Net predictions...")
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "unet")
        os.makedirs(out_dir, exist_ok=True)
        
        unet = UNet(in_channels=3, out_channels=1, base_channels=64).to(device)
        state_dict = torch.load(unet_checkpoint, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()

        for num_examples in [3, 4]:
            out_path = os.path.join(out_dir, f"unet_prediction_{num_examples}_valid.png")
            save_prediction_grid(
                model=unet,
                file_ids=eval_valid_ids[:num_examples],
                split_dir=valid_dir,
                out_path=out_path,
                include_label=False,
                image_size=args.image_size,
                device=device,
                threshold=args.threshold,
                title=f"U-Net Validation Predictions ({num_examples} Examples)",
            )
            print(f"Saved: {out_path}")
    else:
        print(f"U-Net checkpoint not found at {unet_checkpoint}")

    # 2. Run DeepLabV3
    if os.path.exists(deeplab_checkpoint):
        print("\nRunning DeepLabV3 predictions...")
        out_dir = os.path.join(PROJECT_ROOT, "outputs", "deeplabv3")
        os.makedirs(out_dir, exist_ok=True)
        
        deeplab = DeepLabV3(in_channels=3, out_channels=1, pretrained=True).to(device)
        state_dict = torch.load(deeplab_checkpoint, map_location=device)
        deeplab.load_state_dict(state_dict, strict=False)
        deeplab.eval()

        for num_examples in [3, 4]:
            out_path = os.path.join(out_dir, f"deeplabv3_prediction_{num_examples}_valid.png")
            save_prediction_grid(
                model=deeplab,
                file_ids=eval_valid_ids[:num_examples],
                split_dir=valid_dir,
                out_path=out_path,
                include_label=False,
                image_size=args.image_size,
                device=device,
                threshold=args.threshold,
                title=f"DeepLabV3 Validation Predictions ({num_examples} Examples)",
            )
            print(f"Saved: {out_path}")
    else:
        print(f"DeepLabV3 checkpoint not found at {deeplab_checkpoint}")


if __name__ == "__main__":
    main()

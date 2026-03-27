import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from models.unet import UNet
from losses.segmentation import build_default_criterion
from training.dataset import build_dataloaders, build_train_valid_datasets, load_eval_samples
from training.utils import calculate_iou, save_prediction_grid


def get_config(image_size=256, batch_size=8, epochs=30, lr=1e-4):
    class Config:
        DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
        TRAIN_DIR = os.path.join(DATA_DIR, "train")
        VALID_DIR = os.path.join(DATA_DIR, "valid")
        OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

        model_suffix = f"_{image_size}" if image_size != 256 else ""
        CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
        LOG_DIR = os.path.join(OUTPUTS_DIR, f"unet{model_suffix}")
        EPOCH_PREDICTIONS_DIR = os.path.join(LOG_DIR, "epoch_predictions")
        EVAL_SAMPLES_PATH = os.path.join(PROJECT_ROOT, "src", "configs", "eval_samples.json")
        CHECKPOINT_NAME = f"best_unet{model_suffix}.pth"

        IN_CHANNELS = 3
        OUT_CHANNELS = 1
        BASE_CHANNELS = 64

        BATCH_SIZE = batch_size
        NUM_EPOCHS = epochs
        LEARNING_RATE = lr
        IMAGE_SIZE = image_size
        VALID_SPLIT = 0.15
        SEED = 42
        NUM_WORKERS = 8

        DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    return Config()


def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    total_iou = 0.0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=False)
    for _, (images, masks) in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device=='cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_iou = calculate_iou(outputs, masks)
        total_loss += loss.item()
        total_iou += batch_iou
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{batch_iou:.4f}")

    return total_loss / len(dataloader), total_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, total=len(dataloader), desc="Valid", leave=False)
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            with torch.cuda.amp.autocast(enabled=device=='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            batch_iou = calculate_iou(outputs, masks)

            total_loss += loss.item()
            total_iou += batch_iou
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{batch_iou:.4f}")

    return total_loss / len(dataloader), total_iou / len(dataloader)


def train(config):
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.EPOCH_PREDICTIONS_DIR, exist_ok=True)

    eval_train_ids, eval_valid_ids = load_eval_samples(config.EVAL_SAMPLES_PATH)

    train_dataset, valid_dataset = build_train_valid_datasets(
        train_dir=config.TRAIN_DIR,
        image_size=config.IMAGE_SIZE,
        valid_split=config.VALID_SPLIT,
        seed=config.SEED,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    train_loader, valid_loader = build_dataloaders(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    model = UNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_channels=config.BASE_CHANNELS,
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = build_default_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=config.DEVICE=='cuda')

    train_losses = []
    valid_losses = []
    train_ious = []
    valid_ious = []
    best_iou = 0.0

    print("Starting training...")
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")

        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, scaler)
        valid_loss, val_iou = validate(model, valid_loader, criterion, config.DEVICE)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_ious.append(train_iou)
        valid_ious.append(val_iou)

        print(
            f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, Valid IoU: {val_iou:.4f}"
        )

        epoch_tag = f"epoch_{epoch + 1:03d}"
        save_prediction_grid(
            model=model,
            file_ids=eval_train_ids,
            split_dir=config.TRAIN_DIR,
            out_path=os.path.join(config.EPOCH_PREDICTIONS_DIR, f"{epoch_tag}_train.png"),
            include_label=True,
            image_size=config.IMAGE_SIZE,
            device=config.DEVICE,
        )
        save_prediction_grid(
            model=model,
            file_ids=eval_valid_ids,
            split_dir=config.VALID_DIR,
            out_path=os.path.join(config.EPOCH_PREDICTIONS_DIR, f"{epoch_tag}_valid.png"),
            include_label=False,
            image_size=config.IMAGE_SIZE,
            device=config.DEVICE,
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME))
            print(f"  Saved best model with IoU: {best_iou:.4f}")

        scheduler.step(valid_loss)

    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.savefig(os.path.join(config.LOG_DIR, "training_curves.png"))

    print("\nSaving final standardized inference on fixed comparison samples...")

    save_prediction_grid(
        model=model,
        file_ids=eval_train_ids,
        split_dir=config.TRAIN_DIR,
        out_path=os.path.join(config.LOG_DIR, "unet_predictions_train.png"),
        include_label=True,
        image_size=config.IMAGE_SIZE,
        device=config.DEVICE,
    )

    save_prediction_grid(
        model=model,
        file_ids=eval_valid_ids,
        split_dir=config.VALID_DIR,
        out_path=os.path.join(config.LOG_DIR, "unet_predictions_valid.png"),
        include_label=False,
        image_size=config.IMAGE_SIZE,
        device=config.DEVICE,
    )

    print("\n" + "=" * 60)
    print("U-Net Training Complete")
    print("=" * 60)
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"Model saved to: {os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_NAME)}")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Road Segmentation")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    config = get_config(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(f"Using device: {config.DEVICE}")
    print(f"Training resolution: {config.IMAGE_SIZE}x{config.IMAGE_SIZE} with batch size {config.BATCH_SIZE}")
    train(config)


if __name__ == "__main__":
    main()

import typer
from typing_extensions import Annotated
from enum import Enum


class Split(str, Enum):
    train = "train"
    valid = "valid"
    both = "both"

from src.training.train_unet import get_config as get_unet_config, train as train_unet
from src.training.train_deeplabv3 import get_config as get_deeplabv3_config, train as train_deeplabv3
from src.evaluation.predict_unet_local import run_prediction as predict_unet
from src.evaluation.predict_deeplabv3_local import run_prediction as predict_deeplabv3

app = typer.Typer(help="Road segmentation CLI — train models and run predictions.")
train_app = typer.Typer(help="Train a segmentation model.")
predict_app = typer.Typer(help="Run predictions with a trained checkpoint.")

app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")


@train_app.command("unet")
def train_unet_cmd(
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 30,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 1e-4,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 8,
    image_size: Annotated[int, typer.Option(help="Input image size (256 or 512).")] = 256,
):
    """Train U-Net for road segmentation."""
    config = get_unet_config(image_size=image_size, batch_size=batch_size, epochs=epochs, lr=lr)
    train_unet(config)


@train_app.command("deeplabv3")
def train_deeplabv3_cmd(
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 30,
    lr: Annotated[float, typer.Option(help="Learning rate.")] = 1e-4,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 8,
):
    """Train DeepLabV3 for road segmentation."""
    config = get_deeplabv3_config(epochs=epochs, lr=lr, batch_size=batch_size)
    train_deeplabv3(config)


@predict_app.command("unet")
def predict_unet_cmd(
    checkpoint: Annotated[str, typer.Option(help="Path to .pth checkpoint.")] = "outputs/checkpoints/best_unet.pth",
    split: Annotated[Split, typer.Option(help="Dataset split: train, valid, or both.")] = Split.valid,
    threshold: Annotated[float, typer.Option(help="Binarization threshold.")] = 0.4,
    image_size: Annotated[int, typer.Option(help="Inference image size.")] = 256,
):
    """Run U-Net predictions on a dataset split."""
    predict_unet(checkpoint=checkpoint, split=split.value, threshold=threshold, image_size=image_size)


@predict_app.command("deeplabv3")
def predict_deeplabv3_cmd(
    checkpoint: Annotated[str, typer.Option(help="Path to .pth checkpoint.")] = "outputs/checkpoints/best_deeplabv3.pth",
    split: Annotated[Split, typer.Option(help="Dataset split: train, valid, or both.")] = Split.valid,
    threshold: Annotated[float, typer.Option(help="Binarization threshold.")] = 0.4,
    image_size: Annotated[int, typer.Option(help="Inference image size.")] = 256,
):
    """Run DeepLabV3 predictions on a dataset split."""
    predict_deeplabv3(checkpoint=checkpoint, split=split.value, threshold=threshold, image_size=image_size)

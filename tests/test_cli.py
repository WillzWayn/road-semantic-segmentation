"""Tests for the CLI — verifies commands parse args and call the right functions."""
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

runner = CliRunner()


def get_app():
    from src.cli import app
    return app


class TestTrainUnet:
    def test_defaults(self):
        with patch("src.cli.get_unet_config") as mock_config, \
             patch("src.cli.train_unet") as mock_train:
            mock_config.return_value = MagicMock()
            result = runner.invoke(get_app(), ["train", "unet"])
            assert result.exit_code == 0, result.output
            mock_config.assert_called_once_with(
                image_size=256, batch_size=8, epochs=30, lr=1e-4
            )
            mock_train.assert_called_once_with(mock_config.return_value)

    def test_custom_args(self):
        with patch("src.cli.get_unet_config") as mock_config, \
             patch("src.cli.train_unet") as mock_train:
            mock_config.return_value = MagicMock()
            result = runner.invoke(get_app(), [
                "train", "unet",
                "--epochs", "10",
                "--lr", "0.001",
                "--batch-size", "4",
                "--image-size", "512",
            ])
            assert result.exit_code == 0, result.output
            mock_config.assert_called_once_with(
                image_size=512, batch_size=4, epochs=10, lr=0.001
            )


class TestTrainDeeplabv3:
    def test_defaults(self):
        with patch("src.cli.get_deeplabv3_config") as mock_config, \
             patch("src.cli.train_deeplabv3") as mock_train:
            mock_config.return_value = MagicMock()
            result = runner.invoke(get_app(), ["train", "deeplabv3"])
            assert result.exit_code == 0, result.output
            mock_config.assert_called_once_with(epochs=30, lr=1e-4, batch_size=8)
            mock_train.assert_called_once_with(mock_config.return_value)

    def test_custom_args(self):
        with patch("src.cli.get_deeplabv3_config") as mock_config, \
             patch("src.cli.train_deeplabv3") as mock_train:
            mock_config.return_value = MagicMock()
            result = runner.invoke(get_app(), [
                "train", "deeplabv3",
                "--epochs", "5",
                "--lr", "0.0005",
                "--batch-size", "16",
            ])
            assert result.exit_code == 0, result.output
            mock_config.assert_called_once_with(epochs=5, lr=0.0005, batch_size=16)


class TestPredictUnet:
    def test_defaults(self):
        with patch("src.cli.predict_unet") as mock_predict:
            result = runner.invoke(get_app(), ["predict", "unet"])
            assert result.exit_code == 0, result.output
            mock_predict.assert_called_once_with(
                checkpoint="outputs/checkpoints/best_unet.pth",
                split="valid",
                threshold=0.4,
                image_size=256,
            )

    def test_custom_args(self):
        with patch("src.cli.predict_unet") as mock_predict:
            result = runner.invoke(get_app(), [
                "predict", "unet",
                "--checkpoint", "outputs/checkpoints/best_unet_512.pth",
                "--split", "train",
                "--threshold", "0.35",
                "--image-size", "512",
            ])
            assert result.exit_code == 0, result.output
            mock_predict.assert_called_once_with(
                checkpoint="outputs/checkpoints/best_unet_512.pth",
                split="train",
                threshold=0.35,
                image_size=512,
            )


class TestPredictDeeplabv3:
    def test_defaults(self):
        with patch("src.cli.predict_deeplabv3") as mock_predict:
            result = runner.invoke(get_app(), ["predict", "deeplabv3"])
            assert result.exit_code == 0, result.output
            mock_predict.assert_called_once_with(
                checkpoint="outputs/checkpoints/best_deeplabv3.pth",
                split="valid",
                threshold=0.4,
                image_size=256,
            )

    def test_custom_args(self):
        with patch("src.cli.predict_deeplabv3") as mock_predict:
            result = runner.invoke(get_app(), [
                "predict", "deeplabv3",
                "--checkpoint", "outputs/checkpoints/custom.pth",
                "--split", "train",
                "--threshold", "0.3",
                "--image-size", "512",
            ])
            assert result.exit_code == 0, result.output
            mock_predict.assert_called_once_with(
                checkpoint="outputs/checkpoints/custom.pth",
                split="train",
                threshold=0.3,
                image_size=512,
            )

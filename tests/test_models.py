"""Tests for road segmentation models and utilities."""

import pytest
import torch


class TestUNet:
    """Tests for U-Net model."""

    def test_unet_import(self):
        """Test that U-Net can be imported."""
        from src.models.unet import UNet

        assert UNet is not None

    def test_unet_forward_shape(self):
        """Test U-Net forward pass produces correct output shape."""
        from src.models.unet import UNet

        model = UNet(in_channels=3, out_channels=1, base_channels=32)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)


class TestDeepLabV3:
    """Tests for DeepLabV3 model."""

    def test_deeplabv3_import(self):
        """Test that DeepLabV3 can be imported."""
        from src.models.deeplabv3 import DeepLabV3

        assert DeepLabV3 is not None

    def test_deeplabv3_forward_shape(self):
        """Test DeepLabV3 forward pass produces correct output shape."""
        from src.models.deeplabv3 import DeepLabV3

        model = DeepLabV3(out_channels=1, pretrained=False)
        x = torch.randn(2, 3, 256, 256)
        output = model(x)

        assert output.shape == (2, 1, 256, 256)


class TestMetrics:
    """Tests for metric utilities."""

    def test_iou_calculation(self):
        """Test IoU calculation."""
        from src.utils.metrics import calculate_iou

        # Perfect prediction
        preds = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        targets = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])

        iou = calculate_iou(preds, targets)
        assert iou >= 0.99  # Should be close to 1.0


class TestLosses:
    """Tests for loss functions."""

    def test_default_criterion(self):
        """Test default criterion creation."""
        from src.losses.segmentation import build_default_criterion

        criterion = build_default_criterion()
        assert criterion is not None

    def test_bce_dice_loss_import(self):
        """Test BCE-Dice loss import."""
        from src.losses.segmentation import BCEDiceLoss

        assert BCEDiceLoss is not None

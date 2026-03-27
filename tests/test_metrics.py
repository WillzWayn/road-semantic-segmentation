"""Tests for metric utilities."""
import pytest
import torch

from src.utils.metrics import calculate_iou


class TestCalculateIoU:
    """Tests for IoU calculation."""

    def test_iou_perfect_match(self):
        """Test IoU is 1.0 for perfect prediction."""
        # Create prediction and target that match exactly
        pred = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 10  # logits >> 0
        target = torch.ones(2, 2)

        iou = calculate_iou(pred, target)

        assert iou == 1.0, f"Expected IoU=1.0 for perfect match, got {iou}"

    def test_iou_no_overlap(self):
        """Test IoU is 0.0 for no overlap."""
        # Prediction is all positive logits (predicts road everywhere)
        pred = torch.tensor([[10.0, 10.0], [10.0, 10.0]])
        # Target is all zeros (no road)
        target = torch.zeros(2, 2)

        iou = calculate_iou(pred, target)

        assert iou == 0.0, f"Expected IoU=0.0 for no overlap, got {iou}"

    def test_iou_partial_overlap(self):
        """Test IoU for partial overlap."""
        # Predict half correctly
        pred = torch.tensor([[10.0, 10.0], [-10.0, -10.0]])  # Predict road in top half
        target = torch.tensor([[1.0, 1.0], [1.0, 0.0]])  # Road in 3/4 of area

        # Prediction: top half (2 pixels)
        # Target: 3 pixels (top row + bottom-left)
        # Intersection: top half = 2 pixels
        # Union: all 3 road pixels = 3 pixels
        # IoU = 2/3 = 0.667
        iou = calculate_iou(pred, target)

        assert abs(iou - 0.667) < 0.01, f"Expected IoU~=0.667, got {iou}"

    def test_iou_empty_prediction_and_target(self):
        """Test IoU when both prediction and target are empty."""
        # Both predict no road
        pred = torch.tensor([[-10.0, -10.0], [-10.0, -10.0]])
        target = torch.zeros(2, 2)

        iou = calculate_iou(pred, target)

        # When both are empty, IoU should be 0 (no meaningful intersection/union)
        assert iou == 0.0, f"Expected IoU=0.0 for empty prediction/target, got {iou}"

    def test_iou_with_threshold(self):
        """Test that IoU respects threshold parameter."""
        # Logits around the default threshold (0.5 after sigmoid)
        pred = torch.tensor([[0.0, 1.0], [2.0, 3.0]])  # sigmoid gives [0.5, 0.73, 0.88, 0.95]
        target = torch.ones(2, 2)

        # With default threshold (0.5), all predictions should be positive
        iou_default = calculate_iou(pred, target, threshold=0.5)

        # With higher threshold (0.8), only 2 predictions should be positive
        iou_high = calculate_iou(pred, target, threshold=0.8)

        # IoU with high threshold should be lower (less true positives)
        assert iou_default >= iou_high, f"Default threshold IoU ({iou_default}) should be >= high threshold IoU ({iou_high})"

    def test_iou_batch_consistency(self):
        """Test IoU calculation is consistent across batch."""
        # Create same pattern repeated
        pred = torch.ones(3, 4, 4) * 10  # All road predictions
        target = torch.ones(3, 4, 4)  # All road targets

        iou = calculate_iou(pred, target)

        assert iou == 1.0, f"Expected IoU=1.0 for uniform batch, got {iou}"

    def test_iou_different_shapes(self):
        """Test IoU works with different tensor shapes."""
        # Test with larger tensors
        pred = torch.ones(1, 128, 128) * 10
        target = torch.ones(1, 128, 128)

        iou = calculate_iou(pred, target)

        assert iou == 1.0, f"Expected IoU=1.0 for 128x128 tensors, got {iou}"

    def test_iou_returns_scalar(self):
        """Test that IoU returns a scalar float, not a tensor."""
        pred = torch.randn(2, 64, 64)
        target = torch.randint(0, 2, (2, 64, 64)).float()

        iou = calculate_iou(pred, target)

        assert isinstance(iou, float), f"Expected float, got {type(iou)}"
        assert 0.0 <= iou <= 1.0, f"IoU should be in [0, 1], got {iou}"
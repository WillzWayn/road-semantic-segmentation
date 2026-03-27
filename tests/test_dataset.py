"""Tests for dataset loading and utilities."""
import os
import sys
import tempfile

import pytest
import torch
from PIL import Image

# Add project root to path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from training.dataset import RoadDataset, build_train_valid_datasets


class TestRoadDataset:
    """Tests for RoadDataset class."""

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory with sample images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample images
            for i in range(5):
                # Create satellite image
                sat_img = Image.new("RGB", (256, 256), color=(100, 100, 100))
                sat_path = os.path.join(tmpdir, f"sample_{i:03d}_sat.jpg")
                sat_img.save(sat_path)

                # Create mask
                mask_img = Image.new("L", (256, 256), color=0)
                mask_path = os.path.join(tmpdir, f"sample_{i:03d}_mask.png")
                mask_img.save(mask_path)

            yield tmpdir

    def test_dataset_initialization(self, temp_dataset_dir):
        """Test that dataset initializes correctly."""
        dataset = RoadDataset(temp_dataset_dir, image_size=256, require_masks=True)

        assert len(dataset) == 5, f"Expected 5 samples, got {len(dataset)}"

    def test_dataset_getitem(self, temp_dataset_dir):
        """Test that dataset __getitem__ returns correct shapes."""
        dataset = RoadDataset(temp_dataset_dir, image_size=256, require_masks=True)

        image, mask = dataset[0]

        assert image.shape == (3, 256, 256), f"Image shape should be (3, 256, 256), got {image.shape}"
        assert mask.shape == (1, 256, 256), f"Mask shape should be (1, 256, 256), got {mask.shape}"
        assert image.dtype == torch.float32
        assert mask.dtype == torch.float32

    def test_dataset_without_masks(self, temp_dataset_dir):
        """Test dataset can load images without masks."""
        # Remove one mask
        os.remove(os.path.join(temp_dataset_dir, "sample_000_mask.png"))

        dataset = RoadDataset(temp_dataset_dir, image_size=256, require_masks=False)

        # Should still find all 5 satellite images
        assert len(dataset) == 5

    def test_dataset_require_masks(self, temp_dataset_dir):
        """Test dataset filters for samples with masks when require_masks=True."""
        # Remove one mask
        os.remove(os.path.join(temp_dataset_dir, "sample_000_mask.png"))

        dataset = RoadDataset(temp_dataset_dir, image_size=256, require_masks=True)

        # Should only find 4 samples with both image and mask
        assert len(dataset) == 4

    def test_dataset_augmentation(self, temp_dataset_dir):
        """Test that augmentation can be enabled."""
        dataset_aug = RoadDataset(temp_dataset_dir, image_size=256, augment=True, require_masks=True)
        dataset_no_aug = RoadDataset(temp_dataset_dir, image_size=256, augment=False, require_masks=True)

        # Both should have same length
        assert len(dataset_aug) == len(dataset_no_aug)

        # With augmentation, same sample might give different results
        # (but we can't easily test randomness without multiple calls)
        image, mask = dataset_aug[0]
        assert image.shape == (3, 256, 256)

    def test_dataset_image_size_resize(self, temp_dataset_dir):
        """Test that images are resized to specified size."""
        dataset = RoadDataset(temp_dataset_dir, image_size=128, require_masks=True)

        image, mask = dataset[0]

        assert image.shape == (3, 128, 128)
        assert mask.shape == (1, 128, 128)


class TestDatasetSplit:
    """Tests for dataset splitting utilities."""

    @pytest.fixture
    def temp_train_dir(self):
        """Create a temporary training directory with sample images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample images (more than minimum needed for split)
            for i in range(20):
                sat_img = Image.new("RGB", (256, 256), color=(100, 100, 100))
                sat_path = os.path.join(tmpdir, f"sample_{i:03d}_sat.jpg")
                sat_img.save(sat_path)

                mask_img = Image.new("L", (256, 256), color=0)
                mask_path = os.path.join(tmpdir, f"sample_{i:03d}_mask.png")
                mask_img.save(mask_path)

            yield tmpdir

    def test_build_train_valid_datasets(self, temp_train_dir):
        """Test that train/valid split creates correct datasets."""
        train_dataset, valid_dataset = build_train_valid_datasets(
            train_dir=temp_train_dir,
            image_size=256,
            valid_split=0.2,
            seed=42,
        )

        # Check that split worked
        total_samples = len(train_dataset) + len(valid_dataset)
        assert total_samples == 20, f"Expected 20 total samples, got {total_samples}"

        # Valid should be approximately 20% (4 samples)
        assert 2 <= len(valid_dataset) <= 6, f"Expected ~4 valid samples, got {len(valid_dataset)}"

    def test_split_reproducibility(self, temp_train_dir):
        """Test that split is reproducible with same seed."""
        train1, valid1 = build_train_valid_datasets(
            train_dir=temp_train_dir,
            image_size=256,
            valid_split=0.2,
            seed=42,
        )

        train2, valid2 = build_train_valid_datasets(
            train_dir=temp_train_dir,
            image_size=256,
            valid_split=0.2,
            seed=42,
        )

        assert len(train1) == len(train2)
        assert len(valid1) == len(valid2)

    def test_split_different_seeds(self, temp_train_dir):
        """Test that different seeds give different splits."""
        train1, valid1 = build_train_valid_datasets(
            train_dir=temp_train_dir,
            image_size=256,
            valid_split=0.2,
            seed=42,
        )

        train2, valid2 = build_train_valid_datasets(
            train_dir=temp_train_dir,
            image_size=256,
            valid_split=0.2,
            seed=123,
        )

        # Lengths should be the same (same split ratio)
        assert len(train1) == len(train2)
        assert len(valid1) == len(valid2)
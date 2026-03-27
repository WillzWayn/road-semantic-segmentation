# Documentation Artifacts

This document lists the curated result artifacts kept in-repo for documentation and portfolio evidence.

## Baseline Results

### K-Means Clustering

- `outputs/baselines/kmeans_results.png` - K-means clustering visualization on satellite imagery

### Mask Analysis

- `outputs/baselines/mask_analysis.png` - Analysis of road mask distributions

## U-Net Results

- `outputs/unet/training_curves.png` - Training and validation loss/IoU curves over 30 epochs
- `outputs/unet/unet_predictions_valid_grid_01_06.png` - Validation predictions at various thresholds

### Training Performance

- Best validation IoU: 0.5729
- Optimal threshold: 0.40

## DeepLabV3 Results

- `outputs/deeplabv3/training_curves.png` - Training and validation loss/IoU curves
- `outputs/deeplabv3/deeplabv3_predictions_valid_grid_01_06.png` - Validation predictions at various thresholds

### Training Performance

- Best validation IoU: 0.3479

## Excluded Artifacts

The following are generated during training but excluded from version control:

- `outputs/checkpoints/*.pth` - Model weight files (~300MB)
- `outputs/*/epoch_predictions/` - Per-epoch prediction visualizations
- `outputs/baselines/data_samples.png` - Large data sample images
- `outputs/baselines/*_predictions_*.png` - Baseline prediction grids
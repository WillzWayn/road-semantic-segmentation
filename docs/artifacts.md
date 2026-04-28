# Documentation Artifacts

This document lists the curated artifacts kept in-repo for documentation and portfolio evidence.
Generated experiment outputs remain in `outputs/` locally and are not part of version control.

## README Assets

- `docs/images/dataset-examples.png` - Representative satellite/mask examples from the dataset
- `docs/images/unet-validation-predictions.png` - Curated U-Net validation predictions used in the README
- `docs/images/unet-training-curves.png` - U-Net training and validation curves
- `docs/images/deeplabv3-validation-predictions.png` - Curated DeepLabV3 validation predictions used in the README

## Supporting Documents

- `docs/Coursework_2122_COMP11069.pdf` - Course brief and original project framing
- `report/src/` - LaTeX source files for the written report and presentation
- `report/output/` - Compiled report and presentation PDFs

## Source of Truth For Generated Artifacts

The following files may be regenerated locally and are intentionally excluded from git:

- `outputs/checkpoints/*.pth` - Model weight files (~300MB)
- `outputs/*/epoch_predictions/` - Per-epoch prediction visualizations
- `outputs/baselines/*.png` - Baseline analyses and qualitative grids
- `outputs/unet/*.png` - Local prediction exports and training curves
- `outputs/deeplabv3/*.png` - Local prediction exports and training curves

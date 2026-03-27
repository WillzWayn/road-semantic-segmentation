import sys
import os

# Ensure src is in sys.path so we can import from utils.metrics
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# Remove the directory of this file from sys.path if it exists at index 0
# to prevent "import utils" from resolving to this file's folder (src/training)
this_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path[0] == this_dir:
    sys.path.pop(0)

from utils.metrics import calculate_iou
from utils.visualization import predict_mask_from_id, save_prediction_grid

__all__ = ["calculate_iou", "predict_mask_from_id", "save_prediction_grid"]

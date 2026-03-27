import torch.nn as nn
from .deeplabv3 import DeepLabV3
from .unet import UNet


def get_model(model_name, in_channels=3, out_channels=1, **kwargs):
    models = {
        'unet': UNet,
        'deeplabv3': DeepLabV3,
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name.lower()](in_channels=in_channels, out_channels=out_channels, **kwargs)


__all__ = ["UNet", "DeepLabV3", "get_model"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone='resnet50', pretrained=True):
        super().__init__()
        
        if backbone == 'resnet50':
            if pretrained:
                weights = DeepLabV3_ResNet50_Weights.DEFAULT
                self.model = deeplabv3_resnet50(weights=weights)
            else:
                self.model = deeplabv3_resnet50(weights=None)
        
        in_features = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_features, out_channels, kernel_size=1)

    def forward(self, x):
        return self.model(x)['out']


def get_deeplabv3_plus(in_channels=3, out_channels=1, backbone='resnet50', pretrained=True):
    return DeepLabV3(in_channels, out_channels, backbone, pretrained)

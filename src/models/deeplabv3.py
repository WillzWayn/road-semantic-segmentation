import torch.nn as nn
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone="resnet50", pretrained=True):
        super().__init__()

        if in_channels != 3:
            raise ValueError("DeepLabV3 currently expects in_channels=3.")

        if backbone == "resnet50":
            if pretrained:
                try:
                    weights = DeepLabV3_ResNet50_Weights.DEFAULT
                    self.model = deeplabv3_resnet50(weights=weights)
                except Exception:
                    self.model = deeplabv3_resnet50(weights=None)
            else:
                self.model = deeplabv3_resnet50(weights=None)
        else:
            raise ValueError("DeepLabV3 currently supports only backbone='resnet50'.")
        
        self.model.classifier = DeepLabHead(2048, out_channels)

    def forward(self, x):
        return self.model(x)['out']


def get_deeplabv3_plus(in_channels=3, out_channels=1, backbone='resnet50', pretrained=True):
    return DeepLabV3(in_channels, out_channels, backbone, pretrained)

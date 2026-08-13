import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50SpatialStem(nn.Module):
    """
    Modified ResNet-50 backbone that returns spatial feature maps instead of pooled vectors.
    """
    def __init__(self, in_channels: int = 3, return_layer: str = 'layer3'):
        super().__init__()
        # Load pretrained ImageNet weights
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Modify first convolution to accept in_channels instead of 3
        if in_channels != 3:
            original_conv = backbone.conv1
            self.conv1 = nn.Conv2d(
                in_channels, 
                original_conv.out_channels, 
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride, 
                padding=original_conv.padding, 
                bias=original_conv.bias is not None
            )
            # Initialize with random weights for the extra channels
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            # Copy original weights for the first 3 channels if possible (though input is totally different modality)
            # So random init is fine, but we'll use kaiming.
        else:
            self.conv1 = backbone.conv1
            
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.return_layer = return_layer
        
        # We output spatial feature maps, so no avgpool or fc
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.return_layer == 'layer4':
            x = self.layer4(x)
            
        return x

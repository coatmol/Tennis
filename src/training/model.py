import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PlayerDetectionModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Feature Extractor Backbone (ResNet18)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # Remove Average Pooling and FC layer (Keep up to layer4)
        # Input: [B, 3, 384, 640] -> Output Feature Map: [B, 512, 12, 20]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Convolutional Detection Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output 5 channels per grid cell: 1 confidence + 4 box parameters
            nn.Conv2d(128, 5, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        # Forward pass through backbone
        features = self.backbone(x)  # [Batch, 512, 12, 20]

        # Forward pass through head
        out = self.head(features)  # [Batch, 5, 12, 20]

        # Reshape output to [Batch, Grid_H, Grid_W, 5]
        out = out.permute(0, 2, 3, 1)

        # Separate confidence from bounding box offsets
        confidence = torch.sigmoid(out[..., 0])
        bbox_preds = out[..., 1:]

        return confidence, bbox_preds

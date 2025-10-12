# src\models\mobilenet_models.py

import torch.nn as nn
from torchvision import models


class MobileNetV2Model(nn.Module):
    """
    Wrapper for MobileNetV2 with support for transfer learning.

    Args:
        num_classes (int): Number of output classes for classification.
        pretrained (bool): Whether to use ImageNet-pretrained weights.

    Methods:
        forward(x): Returns model predictions for input x.
        freeze_backbone(): Freeze all blocks except the classifier.
        unfreeze_backbone(blocks=None): Unfreeze all or selected blocks.

    Usage example:
        model = MobileNetV2Model(num_classes=10)
        model.freeze_backbone()
        model.unfreeze_backbone(blocks=3)
    """

    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        self.model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        )
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """Freeze all blocks except the classifier."""
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def unfreeze_backbone(self, blocks=None):
        """
        Unfreeze backbone blocks partially or fully.
        Args:
            blocks: None = unfreeze all
                    int = last N blocks
                    list = list of block indices to unfreeze
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = False  # Freeze all first

        features = list(self.model.features.children())

        if blocks is None:
            # Unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific blocks
            if isinstance(blocks, int):
                blocks_to_unfreeze = features[-blocks:]
            elif isinstance(blocks, list):
                blocks_to_unfreeze = [features[i] for i in blocks]
            else:
                raise ValueError("blocks should be None, int, or list of indices.")

            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True

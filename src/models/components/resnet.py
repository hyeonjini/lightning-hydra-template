import torch.nn as nn
from torchvision import models

from importlib import import_module


def get_weights(name: str):
    weights = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "resnet152": models.ResNet152_Weights.DEFAULT,
        "resnext50_32x4d": models.ResNeXt50_32X4D_Weights.DEFAULT,
        "resnext101_32x8d": models.ResNeXt101_32X8D_Weights.DEFAULT,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights.DEFAULT,
        "wide_resnet101_2": models.Wide_ResNet101_2_Weights.DEFAULT,
    }
    return weights[name]

class ResNet(nn.Module):

    def __init__(
        self,
        name: str = "resnet18",
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:

        self.weights = None
        if pretrained:
            self.weights = get_weights(name)

        nn.Module.__init__(self)
        self.net = getattr(
            import_module("torchvision.models"),
            name,
        )(weights=self.weights)

        self.net.fc = nn.Linear(
            in_features=self.net.fc.in_features,
            out_features=num_classes,
            bias=True,
        )
    
    def forward(self, x):
        return self.net(x)
        
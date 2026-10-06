import torch
import torch.nn as nn
import torchvision


class CXRNet(nn.Module):
    """DenseNet-121 (ImageNet) with C sigmoid outputs; accepts grayscale B x 1 x H x W in [0, 1]."""
    def __init__(self, n_classes=14, pretrained=True):
        super().__init__()
        w = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = torchvision.models.densenet121(weights=w)
        self.net.classifier = nn.Linear(self.net.classifier.in_features, n_classes)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        return self.net(x.contiguous(memory_format=torch.channels_last))

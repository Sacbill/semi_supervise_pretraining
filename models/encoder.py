import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from config import config

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        # Use pretrained weights if specified in config
        weights = ResNet50_Weights.IMAGENET1K_V1 if config["use_pretrained_model"] else None
        resnet = resnet50(weights=weights)
        
        # Remove the final fully connected layer to use ResNet as a feature extractor
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # Forward pass through the encoder
        return self.encoder(x)


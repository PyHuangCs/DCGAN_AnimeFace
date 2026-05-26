"""DCGAN model definitions for anime face generation."""

import torch.nn as nn


LATENT_SIZE = 128


class Discriminator(nn.Module):
    """DCGAN discriminator for 64x64 RGB images."""

    def __init__(self, image_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # input: image_channels x 64 x 64
            nn.Conv2d(image_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_maps x 32 x 32

            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_maps*2 x 16 x 16

            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_maps*4 x 8 x 8

            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # feature_maps*8 x 4 x 4

            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    """DCGAN generator for 64x64 RGB images."""

    def __init__(self, latent_size: int = LATENT_SIZE, image_channels: int = 3, feature_maps: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # input: latent_size x 1 x 1
            nn.ConvTranspose2d(latent_size, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # feature_maps*8 x 4 x 4

            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # feature_maps*4 x 8 x 8

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # feature_maps*2 x 16 x 16

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # feature_maps x 32 x 32

            nn.ConvTranspose2d(feature_maps, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
            # image_channels x 64 x 64
        )

    def forward(self, z):
        return self.net(z)


def weights_init(module):
    """Initialize DCGAN weights following the original DCGAN convention."""
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)

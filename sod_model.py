"""
sod_model.py

Defines the CNN encoder-decoder model for Salient Object Detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),           
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),                  

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),           
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),                 
        )

    def forward(self, x):
        return self.block(x)


class SODNet(nn.Module):
    """
    Encoder-decoder CNN for Salient Object Detection.

    Improvements added:
    - BatchNorm
    - Dropout
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32, dropout=0.1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels, dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2, dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4, dropout)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8, dropout)

        # Decoder (simple, no skip connections)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4, base_channels * 4, dropout)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2, base_channels * 2, dropout)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels, base_channels, dropout)

        # Final 1-channel mask
        self.final_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        x3 = self.enc3(p2)
        p3 = self.pool3(x3)

        bottleneck = self.enc4(p3)

        # Decoder
        d3 = self.up3(bottleneck)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self.dec1(d1)

        out = self.final_conv(d1)
        return torch.sigmoid(out) 


if __name__ == "__main__":
    model = SODNet(in_channels=3, base_channels=32)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print("Input shape:", x.shape, "Output shape:", y.shape)

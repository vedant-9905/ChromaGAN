"""
core/models.py
==============
Defines the ChromaGAN architectures:
1. UNetColorGenerator: U-Net with skip connections mapping Luminance L -> Chrominance (a, b)
2. PatchGANColorDiscriminator: 70x70 PatchGAN critic scoring real vs generated (L, a, b) pairs
"""

import torch
import torch.nn as nn
from typing import List


class DownBlock(nn.Module):
    """Conv2d -> BatchNorm/InstanceNorm -> LeakyReLU(0.2)"""

    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample -> Conv2d -> BatchNorm -> ReLU -> (optional Dropout)"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetColorGenerator(nn.Module):
    """
    U-Net Generator for Black & White Photo Colorization.
    
    Ingests 1-channel normalized Luminance L in [-1, 1].
    Outputs 2-channel normalized Chrominance (a, b) in [-1, 1].
    Skip connections preserve exact high-frequency spatial details from the input photograph.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_filters: int = 64) -> None:
        super().__init__()
        f = base_filters

        # Encoder (Downsampling path)
        self.down1 = DownBlock(in_channels, f, normalize=False)  # 256 -> 128
        self.down2 = DownBlock(f, f * 2)                         # 128 -> 64
        self.down3 = DownBlock(f * 2, f * 4)                     # 64 -> 32
        self.down4 = DownBlock(f * 4, f * 8)                     # 32 -> 16
        self.down5 = DownBlock(f * 8, f * 8)                     # 16 -> 8
        self.down6 = DownBlock(f * 8, f * 8)                     # 8 -> 4
        self.down7 = DownBlock(f * 8, f * 8)                     # 4 -> 2
        self.down8 = DownBlock(f * 8, f * 8, normalize=False)    # 2 -> 1 (Bottleneck)

        # Decoder (Upsampling path with skip concatenation)
        self.up1 = UpBlock(f * 8, f * 8, dropout=0.5)            # 1 -> 2
        self.up2 = UpBlock(f * 16, f * 8, dropout=0.5)           # 2 -> 4
        self.up3 = UpBlock(f * 16, f * 8, dropout=0.5)           # 4 -> 8
        self.up4 = UpBlock(f * 16, f * 8)                        # 8 -> 16
        self.up5 = UpBlock(f * 16, f * 4)                        # 16 -> 32
        self.up6 = UpBlock(f * 8, f * 2)                         # 32 -> 64
        self.up7 = UpBlock(f * 4, f)                             # 64 -> 128

        # Final projection to 2 channels (a, b) at original resolution 256x256
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(f * 2, f, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
            nn.Conv2d(f, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Chrominance normalized to [-1, 1]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, l_channel: torch.Tensor) -> torch.Tensor:
        # Encoder forward pass with stored skip connections
        d1 = self.down1(l_channel)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward pass with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        # Recombine with first layer and project
        out = self.final_up(torch.cat([u7, d1], dim=1))
        return out


class PatchGANColorDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field).
    Evaluates joint condition of (L, a, b) to ensure realistic chromatic distribution.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 64) -> None:
        super().__init__()
        f = base_filters

        self.model = nn.Sequential(
            # Input: concatenated [L, a, b] -> 3 channels
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs patch-level real/fake logits."""
        return self.model(x)

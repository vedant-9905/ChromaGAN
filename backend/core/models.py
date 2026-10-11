"""
core/models.py
==============
Defines the RadSwin-WGAN Generator (Residual U-Net) and
Discriminator (PatchGAN with Wasserstein head).

Architecture Notes:
- Generator: ResUNet with skip connections, InstanceNorm, LeakyReLU, Tanh output
- Discriminator: 70x70 PatchGAN, no final activation (raw scores for WGAN)
"""

import torch
import torch.nn as nn
from typing import List


# ---------------------------------------------------------------------------
# Helper Blocks
# ---------------------------------------------------------------------------

class ConvINLeaky(nn.Module):
    """Conv2d → InstanceNorm2d → LeakyReLU(0.2)"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not use_norm)
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_ch, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Identity-shortcut residual block.
    Conv → IN → LReLU → Conv → IN → (+ skip)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class UpConvINLeaky(nn.Module):
    """Bilinear upsample + ConvINLeaky for decoder path."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvINLeaky(in_ch, out_ch, kernel=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


# ---------------------------------------------------------------------------
# Generator: Residual U-Net (ResUNet)
# ---------------------------------------------------------------------------

class ResUNetGenerator(nn.Module):
    """
    Multi-scale Residual U-Net Generator.

    Encoder: 4 downsampling stages (stride-2 convolutions)
    Bottleneck: 6 residual blocks
    Decoder: 4 upsampling stages with skip connections (channel-wise concat)
    Output: Tanh activation for normalized [-1, 1] range

    Args:
        in_channels  : input image channels (1 for grayscale CT)
        out_channels : output image channels (1 for grayscale CT)
        base_filters : base number of feature maps (default 64)
        n_res_blocks : number of residual blocks in bottleneck (default 6)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        n_res_blocks: int = 6,
    ) -> None:
        super().__init__()
        f = base_filters

        # ---- Encoder ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, f, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(f, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )  # H x W x f

        self.enc2 = ConvINLeaky(f, f * 2, kernel=4, stride=2, padding=1)   # H/2
        self.enc3 = ConvINLeaky(f * 2, f * 4, kernel=4, stride=2, padding=1)  # H/4
        self.enc4 = ConvINLeaky(f * 4, f * 8, kernel=4, stride=2, padding=1)  # H/8

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(f * 8) for _ in range(n_res_blocks)]
        )

        # ---- Decoder ----
        # Each decoder stage receives skip from encoder → in_ch = f*8 + f*8 etc.
        self.dec4 = UpConvINLeaky(f * 8 + f * 8, f * 4)   # concat with enc4 → f*16→f*4
        self.dec3 = UpConvINLeaky(f * 4 + f * 4, f * 2)
        self.dec2 = UpConvINLeaky(f * 2 + f * 2, f)
        self.dec1 = UpConvINLeaky(f + f, f)

        # ---- Output head ----
        self.output_conv = nn.Sequential(
            nn.Conv2d(f, out_channels, 7, 1, 3, bias=True),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)   # f
        e2 = self.enc2(e1)  # f*2
        e3 = self.enc3(e2)  # f*4
        e4 = self.enc4(e3)  # f*8

        # Bottleneck
        b = self.bottleneck(e4)  # f*8

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([b, e4], dim=1))   # f*4
        d3 = self.dec3(torch.cat([d4, e3], dim=1))  # f*2
        d2 = self.dec2(torch.cat([d3, e2], dim=1))  # f
        d1 = self.dec1(torch.cat([d2, e1], dim=1))  # f

        return self.output_conv(d1)


# ---------------------------------------------------------------------------
# Discriminator: PatchGAN (70×70 receptive field)
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for WGAN-GP.

    Receptive field ≈ 70×70. No final Sigmoid — outputs raw critic scores
    suitable for Wasserstein distance estimation.

    Args:
        in_channels : input channels (1 for single-channel CT)
        base_filters : base feature map count (default 64)
        n_layers    : number of downsampling Conv layers after input (default 3)
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_filters: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        # Input conv (no normalization on first layer per PatchGAN convention)
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_filters, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        f_prev = base_filters
        for n in range(1, n_layers + 1):
            f_curr = min(base_filters * (2 ** n), 512)
            stride = 1 if n == n_layers else 2
            layers += [
                nn.Conv2d(f_prev, f_curr, 4, stride, 1, bias=False),
                nn.InstanceNorm2d(f_curr, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            f_prev = f_curr

        # Final projection to single-channel map (no activation)
        layers.append(nn.Conv2d(f_prev, 1, 4, 1, 1, bias=True))

        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw patch-level critic scores (not probabilities)."""
        return self.model(x)

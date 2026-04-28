"""Small U-Net for image denoising."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    Small U-Net for single-channel image denoising.

    3-level encoder-decoder with skip connections and bilinear upsampling.
    Input and output are single-channel float32 images in [0, 1].

    Parameters
    ----------
    base_channels : int
        Number of channels in the first encoder block. Doubles at each level.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = base_channels

        # encoder
        self.enc1 = _conv_block(1, c)
        self.enc2 = _conv_block(c, c * 2)
        self.enc3 = _conv_block(c * 2, c * 4)

        # bottleneck
        self.bottleneck = _conv_block(c * 4, c * 8)

        # decoder
        self.dec3 = _conv_block(c * 8 + c * 4, c * 4)
        self.dec2 = _conv_block(c * 4 + c * 2, c * 2)
        self.dec1 = _conv_block(c * 2 + c, c)

        self.pool = nn.MaxPool2d(2)
        self.out = nn.Conv2d(c, 1, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # bottleneck
        b = self.bottleneck(self.pool(e3))

        # decoder with skip connections
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], dim=1))

        # residual: subtract predicted noise from input
        return torch.clamp(x - self.out(d1), 0, 1)

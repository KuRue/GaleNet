"""Simple CNN+Transformer model for hurricane forecasting."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """A basic residual convolutional block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + residual)


class HurricaneCNN(nn.Module):
    """CNN encoder + Transformer forecaster."""

    def __init__(self, in_channels: int, time_steps: int) -> None:
        super().__init__()
        self.time_steps = time_steps

        self.stem = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        blocks = [ResidualBlock(64) for _ in range(6)]
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(64, 512)

        self.transformer = nn.Transformer(
            d_model=512,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
        )

        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast hurricane tracks.

        ``(batch, time, channels, H, W) -> (batch, time, 4)``

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, time, channels, H, W)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(batch, time, 4)``.
        """

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.stem(x)
        x = self.encoder(x)
        x = self.pool(x).view(b * t, -1)
        x = self.proj(x)  # (B*T, 512)

        x = x.view(t, b, -1)
        out = self.transformer(x, x)
        out = out.view(t * b, -1)
        out = self.head(out)
        out = out.view(t, b, 4).transpose(0, 1)
        return out


__all__ = ["HurricaneCNN"]

"""Bochner time encoding — learnable Fourier features for temporal gaps.

Based on Bochner's theorem (TGAT, Xu et al. 2020): any positive-definite
kernel can be approximated via random Fourier features. Here we make the
frequencies and phases learnable so the network can adapt them to the
timescale of Solana transactions (~seconds to days).
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


class BochnerTimeEncoding(nn.Module):
    """Phi(dt) = (1/sqrt(d)) * cos(omega * dt + bias).

    omega and bias are learnable so the model picks up whatever
    periodicities matter (block times, daily patterns, etc.).
    """

    def __init__(self, d_time: int = 64):
        super().__init__()
        self.d_time = d_time

        # frequencies — init from N(0,1), will spread across scales
        self.omega = nn.Parameter(torch.randn(d_time))
        # phases — uniform on [0, 2pi] covers the full cosine cycle
        self.bias = nn.Parameter(torch.empty(d_time).uniform_(0, 2 * math.pi))

    def forward(self, dt: Tensor) -> Tensor:
        """
        Args:
            dt: [E] or [E, 1] — time deltas (seconds).
        Returns:
            [E, d_time] Fourier encoding.
        """
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)  # [E, 1]
        # normalise to days to keep float32 precision (raw seconds can be huge)
        dt_days = dt / 86400.0
        # [E, 1] * [d_time] + [d_time] -> [E, d_time]
        out = torch.cos(dt_days * self.omega + self.bias)
        return out * (1.0 / math.sqrt(self.d_time))

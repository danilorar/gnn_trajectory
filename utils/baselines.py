"""Non-learned trajectory baselines.

Kept separate from utils/models.py so the frozen model definitions stay unchanged.
"""

import torch
import torch.nn as nn


class ConstantVelocityBaseline(nn.Module):
    """Extrapolate the target's last observed displacement for 12 steps.

    Has no parameters; pass `device` explicitly to the evaluation helpers.
    """

    def forward(self, batch):
        # Node features start with 5 past xy positions, oldest to newest
        past = batch.x[batch.target_mask][:, :10].view(-1, 5, 2)
        current = past[:, -1]
        step = current - past[:, -2]  # displacement per 0.5 s
        k = torch.arange(1, 13, device=past.device, dtype=past.dtype)
        return current[:, None, :] + k[None, :, None] * step[:, None, :]

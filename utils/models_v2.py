"""v2 scene transformer: token encoders (the encoder and decoder come next)."""

import torch
import torch.nn as nn

from utils.dataset_v2 import AGENT_TYPES


def mlp(d_in, d_out, hidden=None):
    hidden = hidden or d_out
    return nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class AgentEncoder(nn.Module):
    """One token per agent from its 2 s history, type and size.

    History (T steps of [x, y, cos, sin, vx, vy]) is flattened with its validity flags,
    so step order is kept and missing steps are visible to the network.
    """

    def __init__(self, d_model=128, steps=5, step_dim=6):
        super().__init__()
        self.history = mlp(steps * (step_dim + 1), d_model)
        self.type_embedding = nn.Embedding(len(AGENT_TYPES), d_model)
        self.size = nn.Linear(2, d_model)
        self.out = mlp(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch):
        hist = batch["agent_hist"]                      # (B, A, T, 6)
        valid = batch["agent_hist_valid"].float()       # (B, A, T)
        x = torch.cat([hist * valid[..., None], valid[..., None]], dim=-1).flatten(2)  # (B, A, T*7)

        h = self.history(x) + self.type_embedding(batch["agent_type"]) + self.size(batch["agent_size"])
        h = self.norm(h + self.out(h))
        return h * batch["agent_mask"][..., None]       # padded slots -> 0


class LaneEncoder(nn.Module):
    """One token per lane segment, VectorNet-style: point MLP -> max-pool -> concat -> MLP -> max-pool.
    Point features [x, y, cos, sin] carry the lane direction; a learned flag marks lane connectors."""

    def __init__(self, d_model=128, point_dim=4):
        super().__init__()
        self.point = mlp(point_dim, d_model)
        self.point2 = mlp(2 * d_model, d_model)
        self.connector_embedding = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def masked_max(x, valid):
        return x.masked_fill(~valid[..., None], float("-inf")).max(dim=-2).values

    def forward(self, batch):
        points = batch["lane_points"]                   # (B, L, P, 4)
        valid = batch["lane_point_valid"]               # (B, L, P)
        has_points = valid.any(-1)                      # padded segments have no valid point

        h = self.point(points)                          # (B, L, P, d)
        pooled = self.masked_max(h, valid | ~has_points[..., None])
        h = self.point2(torch.cat([h, pooled[..., None, :].expand_as(h)], dim=-1))
        h = self.masked_max(h, valid | ~has_points[..., None])  # (B, L, d)

        h = self.norm(h + self.connector_embedding(batch["lane_is_connector"].long()))
        return h * batch["lane_mask"][..., None]        # padded slots -> 0

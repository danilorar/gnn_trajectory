"""v2 scene transformer: token encoders (the encoder and decoder come next)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


# ============================================================
# Scene encoder: transformer over agent + lane tokens with a relative-pose attention bias
# ============================================================


def token_poses(batch):
    """(x, y, theta) per token: agents = last valid history step, lanes = middle valid point.
    Returns agent poses (B, A, 3) and lane poses (B, L, 3)."""
    hist, hist_valid = batch["agent_hist"], batch["agent_hist_valid"]
    T = hist.shape[2]
    last = (hist_valid * torch.arange(1, T + 1, device=hist.device)).argmax(-1)       # last valid step
    agent = hist.gather(2, last[..., None, None].expand(-1, -1, 1, hist.shape[-1]))[:, :, 0]

    points, point_valid = batch["lane_points"], batch["lane_point_valid"]
    mid = (point_valid.sum(-1) - 1).clamp(min=0) // 2                                  # middle valid point
    lane = points.gather(2, mid[..., None, None].expand(-1, -1, 1, points.shape[-1]))[:, :, 0]

    def to_pose(s):
        return torch.stack([s[..., 0], s[..., 1], torch.atan2(s[..., 3], s[..., 2])], dim=-1)

    return to_pose(agent), to_pose(lane)


def rel_features(pose, pos_scale=50.0):
    """(B, L, 3) -> (B, L, L, 5): token j seen from token i (eq. 4.1).
    [dx, dy] of j in i's frame (divided by pos_scale), cos and sin of (theta_j - theta_i),
    and the distance (divided by pos_scale)."""
    x, y, th = pose.unbind(-1)
    dx = x[:, None, :] - x[:, :, None]          # [b, i, j] = x_j - x_i
    dy = y[:, None, :] - y[:, :, None]
    c, s = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy   # rotate by -theta_i
    dth = th[:, None, :] - th[:, :, None]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    return torch.stack([local_x / pos_scale, local_y / pos_scale, torch.cos(dth), torch.sin(dth),
                        dist / pos_scale], dim=-1)


class EncoderLayer(nn.Module):
    """Pre-LN transformer layer (eq. 3.2) whose attention takes an additive (B, H, L, L) bias."""

    def __init__(self, d_model=128, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x, bias):
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
                                           dropout_p=self.attn_dropout if self.training else 0.0)
        x = x + self.dropout(self.proj(h.transpose(1, 2).reshape(B, L, d)))
        return x + self.dropout(self.ffn(self.norm2(x)))


class SceneEncoder(nn.Module):
    """
    Transformer over all agent and lane tokens.

    Relative pose enters as a scalar bias per head instead of a full vector e_ij added to the keys:
        alpha^h_ij = softmax_j(q_i^h . k_j^h / sqrt(d_h) + b^h_ij),   b_ij = MLP(r_ij) in R^H
    This costs (B, H, L, L) instead of (B, L, L, d) (an attention bias, as in Wayformer-style models).
    The bias is computed once and shared by all layers.
    """

    def __init__(self, d_model=128, heads=8, layers=3, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.bias_mlp = nn.Sequential(nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, heads))
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, pad, pose):
        """tokens (B, L, d), pad (B, L) True = padding, pose (B, L, 3) -> (B, L, d)."""
        bias = self.bias_mlp(rel_features(pose)).permute(0, 3, 1, 2)           # (B, H, L, L)
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))           # padded keys
        x = tokens
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x) * ~pad[..., None]                                   # padded tokens -> 0

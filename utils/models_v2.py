"""v2 scene transformer: token encoders (the encoder and decoder come next)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset_v2 import AGENT_TYPES


def mlp(d_in, d_out, hidden=None):
    """Linear -> LayerNorm -> ReLU -> Linear. Pass hidden for small outputs (default hidden = d_out)."""
    hidden = hidden or d_out
    return nn.Sequential(nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class AgentEncoder(nn.Module):
    """One token per agent from its 2 s history, type and size.

    History (T steps of [x, y, cos, sin, vx, vy]) is flattened with its validity flags,
    so step order is kept and missing steps are visible to the network.
    """

    def __init__(self, d_model=128, steps=5, step_dim=6, pos_scale=1.0):
        super().__init__()
        self.pos_scale = pos_scale  # 50.0 = positions in 50 m units, as in the attention bias
        self.history = mlp(steps * (step_dim + 1), d_model)
        self.type_embedding = nn.Embedding(len(AGENT_TYPES), d_model)
        self.size = nn.Linear(2, d_model)
        self.out = mlp(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch):
        hist = batch["agent_hist"]                      # (B, A, T, 6)
        if self.pos_scale != 1.0:
            hist = torch.cat([hist[..., :2] / self.pos_scale, hist[..., 2:]], dim=-1)
        valid = batch["agent_hist_valid"].float()       # (B, A, T)
        x = torch.cat([hist * valid[..., None], valid[..., None]], dim=-1).flatten(2)  # (B, A, T*7)

        h = self.history(x) + self.type_embedding(batch["agent_type"]) + self.size(batch["agent_size"])
        h = self.norm(h + self.out(h))
        return h * batch["agent_mask"][..., None]       # padded slots -> 0


class LaneEncoder(nn.Module):
    """One token per lane segment, VectorNet-style: point MLP -> max-pool -> concat -> MLP -> max-pool.
    Point features [x, y, cos, sin] carry the lane direction; a learned flag marks lane connectors."""

    def __init__(self, d_model=128, point_dim=4, pos_scale=1.0):
        super().__init__()
        self.pos_scale = pos_scale
        self.point = mlp(point_dim, d_model)
        self.point2 = mlp(2 * d_model, d_model)
        self.connector_embedding = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def masked_max(x, valid):
        return x.masked_fill(~valid[..., None], float("-inf")).max(dim=-2).values

    def forward(self, batch):
        points = batch["lane_points"]                   # (B, L, P, 4)
        if self.pos_scale != 1.0:
            points = torch.cat([points[..., :2] / self.pos_scale, points[..., 2:]], dim=-1)
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


def rel_features(pose, pos_scale=50.0, adj=None):
    """(B, L, 3) -> (B, L, L, 5): token j seen from token i (eq. 4.1).
    [dx, dy] of j in i's frame (divided by pos_scale), cos and sin of (theta_j - theta_i),
    and the distance (divided by pos_scale).
    With adj (B, L, L) bool, adj[b, i, j] = "j succeeds i": two more features,
    "j is successor of i" and "j is predecessor of i" -> (B, L, L, 7)."""
    x, y, th = pose.unbind(-1)
    dx = x[:, None, :] - x[:, :, None]          # [b, i, j] = x_j - x_i
    dy = y[:, None, :] - y[:, :, None]
    c, s = torch.cos(th)[:, :, None], torch.sin(th)[:, :, None]
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy   # rotate by -theta_i
    dth = th[:, None, :] - th[:, :, None]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
    feats = [local_x / pos_scale, local_y / pos_scale, torch.cos(dth), torch.sin(dth), dist / pos_scale]
    if adj is not None:
        feats += [adj.float(), adj.transpose(1, 2).float()]
    return torch.stack(feats, dim=-1)


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

    def __init__(self, d_model=128, heads=8, layers=3, dropout=0.1, topology=False):
        super().__init__()
        self.heads = heads
        self.topology = topology
        self.bias_mlp = nn.Sequential(nn.Linear(7 if topology else 5, 64), nn.ReLU(), nn.Linear(64, heads))
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens, pad, pose, adj=None):
        """tokens (B, L, d), pad (B, L) True = padding, pose (B, L, 3), adj (B, L, L) if topology -> (B, L, d)."""
        bias = self.bias_mlp(rel_features(pose, adj=adj if self.topology else None)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))           # padded keys
        x = tokens
        for layer in self.layers:
            x = layer(x, bias)
        return self.norm(x) * ~pad[..., None]                                   # padded tokens -> 0


# ============================================================
# Anchor decoder (eqs. 5.1-5.2), loss (eqs. 6.1-6.2) and the full v2 model
# ============================================================

class DecoderLayer(nn.Module):
    """Pre-LN: self-attention between the K mode queries, cross-attention to the scene, FFN."""

    def __init__(self, d_model=128, heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d_model, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, scene, pad):
        h = self.norm1(q)
        q = q + self.dropout(self.self_attn(h, h, h, need_weights=False)[0])
        h = self.norm2(q)
        q = q + self.dropout(self.cross_attn(h, scene, scene, key_padding_mask=pad, need_weights=False)[0])
        return q + self.dropout(self.ffn(self.norm3(q)))


class AnchorDecoder(nn.Module):
    """K anchor queries q_k = MLP(anchor_k) + emb_k + target_token (eq. 5.1).
    Each query outputs a 12-step trajectory and a score; softmax over K gives the mode probabilities (eq. 5.2)."""

    def __init__(self, anchors, d_model=128, heads=8, layers=2, future_steps=12, dropout=0.1, pos_scale=50.0):
        super().__init__()
        self.register_buffer("anchors", torch.as_tensor(anchors, dtype=torch.float32))  # (K, 2)
        K = len(anchors)
        self.pos_scale = pos_scale
        self.future_steps = future_steps
        self.anchor_mlp = mlp(2, d_model)
        self.mode_embedding = nn.Embedding(K, d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.traj_head = mlp(d_model, future_steps * 2, hidden=d_model)
        self.score_head = mlp(d_model, 1, hidden=d_model)

    def forward(self, scene, pad):
        """scene (B, N, d) with the target token at index 0 -> traj (B, K, F, 2), logits (B, K)."""
        B = scene.shape[0]
        K = len(self.anchors)
        q = (self.anchor_mlp(self.anchors / self.pos_scale) + self.mode_embedding.weight)[None]  # (1, K, d)
        q = q + scene[:, :1]                                                                  # + target token
        for layer in self.layers:
            q = layer(q, scene, pad)
        q = self.norm(q)
        traj = self.traj_head(q).view(B, K, self.future_steps, 2) * self.pos_scale  # predicted in 50 m units
        logits = self.score_head(q).squeeze(-1)
        return traj, logits


def v2_loss(traj, logits, target, target_valid, anchors, score_weight=1.0, assign="predicted"):
    """Eqs. 6.1-6.2: L = SmoothL1(traj_k*, Y) over valid steps + score_weight * CE(logits, k*).

    assign="anchor":    k* = anchor closest to the true endpoint
    assign="predicted": k* = predicted mode with the lowest ADE over the whole path
                        (winner-takes-all as in v1, HiVT, QCNet); anchors then only initialise the queries.
    The same k* is the probability target."""
    valid = target_valid.float()
    if assign == "anchor":
        k_star = torch.cdist(target[:, -1:], anchors[None].expand(len(target), -1, -1)).squeeze(1).argmin(-1)
    elif assign == "predicted":
        with torch.no_grad():
            err = (traj - target[:, None]).norm(dim=-1)                                       # (B, K, F)
            ade = (err * valid[:, None]).sum(-1) / valid.sum(-1, keepdim=True).clamp(min=1)    # (B, K)
            k_star = ade.argmin(-1)
    else:
        raise ValueError(f"unknown assign: {assign}")
    best = traj[torch.arange(len(traj), device=traj.device), k_star]                                      # (B, F, 2)
    reg = F.smooth_l1_loss(best, target, reduction="none").sum(-1)                                        # (B, F)
    reg = (reg * valid).sum() / valid.sum().clamp(min=1)
    cls = F.cross_entropy(logits, k_star)
    return reg + score_weight * cls, {"reg": reg.item(), "cls": cls.item()}


class V2Predictor(nn.Module):
    """Agent + lane encoders -> scene transformer -> anchor decoder."""

    def __init__(self, anchors, d_model=128, heads=8, encoder_layers=3, decoder_layers=2, dropout=0.1,
                 assign="predicted", topology=False, scale_inputs=False):
        super().__init__()
        self.assign = assign
        self.topology = topology
        pos_scale = 50.0 if scale_inputs else 1.0
        self.agent_encoder = AgentEncoder(d_model, pos_scale=pos_scale)
        self.lane_encoder = LaneEncoder(d_model, pos_scale=pos_scale)
        self.scene_encoder = SceneEncoder(d_model, heads, encoder_layers, dropout, topology=topology)
        self.decoder = AnchorDecoder(anchors, d_model, heads, decoder_layers, dropout=dropout)

    def forward(self, batch):
        agent_pose, lane_pose = token_poses(batch)
        tokens = torch.cat([self.agent_encoder(batch), self.lane_encoder(batch)], dim=1)
        pad = ~torch.cat([batch["agent_mask"], batch["lane_mask"]], dim=1)
        adj = None
        if self.topology:
            # Token-level successor matrix: lane-lane links from lane_adj, zero for any pair with an agent
            A = batch["agent_mask"].shape[1]
            N = tokens.shape[1]
            adj = torch.zeros(len(tokens), N, N, dtype=torch.bool, device=tokens.device)
            adj[:, A:, A:] = batch["lane_adj"]
        scene = self.scene_encoder(tokens, pad, torch.cat([agent_pose, lane_pose], dim=1), adj)
        traj, logits = self.decoder(scene, pad)
        return traj, logits

    def loss(self, batch, traj, logits):
        return v2_loss(traj, logits, batch["target_future"], batch["target_future_valid"], self.decoder.anchors,
                       assign=self.assign)

"""v2 scene transformer: token encoders, scene encoder with a relative-pose attention bias,
anchor decoder (target only, or multi-agent) and the winner-takes-all loss."""

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


class EgoPlanEncoder(nn.Module):
    """One token from the ego's planned 6 s future (block 2): [x, y, cos, sin] per step, flattened with
    validity flags (as AgentEncoder), plus a learned plan embedding. Its pose is the ego's current pose."""

    def __init__(self, d_model=128, steps=12, step_dim=4, pos_scale=1.0):
        super().__init__()
        self.pos_scale = pos_scale
        self.plan = mlp(steps * (step_dim + 1), d_model)
        self.plan_embedding = nn.Parameter(torch.zeros(d_model))
        self.out = mlp(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, batch):
        fut = batch["ego_future"]                       # (B, F, 4)
        if self.pos_scale != 1.0:
            fut = torch.cat([fut[..., :2] / self.pos_scale, fut[..., 2:]], dim=-1)
        valid = batch["ego_future_valid"].float()       # (B, F)
        x = torch.cat([fut * valid[..., None], valid[..., None]], dim=-1).flatten(1)  # (B, F*5)
        h = self.plan(x) + self.plan_embedding
        return self.norm(h + self.out(h))[:, None]      # (B, 1, d)


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
    """Pre-LN: self-attention between the K mode queries of the same agent, cross-attention to the scene, FFN.
    Queries are (B, M, K, d) for M predicted agents. Cross-attention runs on the flattened (B, M*K, d)
    sequence: each query attends to the scene independently, so this equals running every agent on its own
    without copying the scene M times."""

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
        B, M, K, d = q.shape
        h = self.norm1(q).reshape(B * M, K, d)
        q = q + self.dropout(self.self_attn(h, h, h, need_weights=False)[0]).view(B, M, K, d)
        h = self.norm2(q).reshape(B, M * K, d)
        q = q + self.dropout(self.cross_attn(h, scene, scene, key_padding_mask=pad,
                                             need_weights=False)[0]).view(B, M, K, d)
        return q + self.dropout(self.ffn(self.norm3(q)))


class AnchorDecoder(nn.Module):
    """Anchor queries q_ak = MLP(anchor_k) + emb_k + agent_token_a (eq. 5.1), for M agents and K modes.
    Each query outputs a 12-step trajectory (in the agent's own frame) and a score;
    softmax over K gives each agent's mode probabilities (eq. 5.2)."""

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

    def forward(self, scene, pad, agents):
        """scene (B, N, d), agents (B, M, d) tokens of the predicted agents -> traj (B, M, K, F, 2), logits (B, M, K)."""
        B, M = agents.shape[:2]
        K = len(self.anchors)
        q = self.anchor_mlp(self.anchors / self.pos_scale) + self.mode_embedding.weight  # (K, d)
        q = q[None, None] + agents[:, :, None]                                          # (B, M, K, d)
        for layer in self.layers:
            q = layer(q, scene, pad)
        q = self.norm(q)
        traj = self.traj_head(q).view(B, M, K, self.future_steps, 2) * self.pos_scale  # predicted in 50 m units
        logits = self.score_head(q).squeeze(-1)
        return traj, logits


def soft_score_ce(logits, ade, temperature=1.0):
    """Block 3: CE against p_k = softmax(-ADE_k / temperature) (ADE in metres, no gradient through it),
    instead of the one-hot winner. logits / ade (..., K) -> (...)."""
    p = torch.softmax(-ade.detach() / temperature, dim=-1)
    return -(p * torch.log_softmax(logits, dim=-1)).sum(-1)


def v2_loss(traj, logits, target, target_valid, anchors, score_weight=1.0, assign="predicted", soft_scores=False):
    """Eqs. 6.1-6.2: L = SmoothL1(traj_k*, Y) over valid steps + score_weight * CE(logits, k*).

    assign="anchor":    k* = anchor closest to the true endpoint
    assign="predicted": k* = predicted mode with the lowest ADE over the whole path
                        (winner-takes-all as in v1, HiVT, QCNet); anchors then only initialise the queries.
    The same k* is the probability target, or with soft_scores=True p_k = softmax(-ADE_k / 1 m)."""
    if soft_scores and assign != "predicted":
        raise ValueError("soft_scores needs assign='predicted'")
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
    cls = soft_score_ce(logits, ade).mean() if soft_scores else F.cross_entropy(logits, k_star)
    return reg + score_weight * cls, {"reg": reg.item(), "cls": cls.item()}


def multi_agent_loss(traj, logits, future, future_valid, include, score_weight=1.0, soft_scores=False,
                     target_split=False):
    """Predicted winner-takes-all per agent (as v2_loss with assign="predicted").

    target_split=False: averaged over all included agents (the target is one of ~12 per sample).
    target_split=True:  target loss (mean over samples) + mean loss of the other included agents,
                        so the target keeps the same weight as in the target-only model.
    traj (B, M, K, F, 2), logits (B, M, K), future (B, M, F, 2), future_valid (B, M, F),
    include (B, M) bool: agents that enter the loss; index 0 = target."""
    B, M, K = logits.shape
    valid = future_valid.float()
    n_valid = valid.sum(-1).clamp(min=1)                                                # (B, M)
    with torch.no_grad():
        err = (traj - future[:, :, None]).norm(dim=-1)                                  # (B, M, K, F)
        ade = (err * valid[:, :, None]).sum(-1) / n_valid[..., None]                    # (B, M, K)
        k_star = ade.argmin(-1)                                                         # (B, M)
    best = traj.gather(2, k_star[..., None, None, None].expand(-1, -1, 1, *traj.shape[3:]))[:, :, 0]  # (B, M, F, 2)
    reg = (F.smooth_l1_loss(best, future, reduction="none").sum(-1) * valid).sum(-1) / n_valid       # (B, M)
    if soft_scores:
        cls = soft_score_ce(logits, ade)                                                # (B, M)
    else:
        cls = F.cross_entropy(logits.reshape(-1, K), k_star.reshape(-1), reduction="none").view(B, M)
    def masked_mean(x, w):
        return (x * w).sum() / w.sum().clamp(min=1)

    w = include.float()
    if target_split:
        wt, wo = w.clone(), w.clone()
        wt[:, 1:], wo[:, 0] = 0, 0
        reg_t, cls_t = masked_mean(reg, wt), masked_mean(cls, wt)
        reg_o, cls_o = masked_mean(reg, wo), masked_mean(cls, wo)
        loss = reg_t + score_weight * cls_t + reg_o + score_weight * cls_o
        return loss, {"reg": (reg_t + reg_o).item(), "cls": (cls_t + cls_o).item(),
                      "reg_target": reg_t.item(), "reg_others": reg_o.item()}
    reg, cls = masked_mean(reg, w), masked_mean(cls, w)
    return reg + score_weight * cls, {"reg": reg.item(), "cls": cls.item()}


def to_agent_frames(points, pose):
    """points (B, F, 2) in the target frame -> (B, M, F, 2) in each agent's own frame; pose (B, M, 3)."""
    c, s = torch.cos(pose[..., 2])[..., None], torch.sin(pose[..., 2])[..., None]          # (B, M, 1)
    d = points[:, None] - pose[..., None, :2]                                               # (B, M, F, 2)
    return torch.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)


def to_target_frame(traj, pose):
    """traj (B, M, K, F, 2) in each agent's own frame, pose (B, M, 3) = agent (x, y, theta) in the target frame."""
    c, s = torch.cos(pose[..., 2])[..., None, None], torch.sin(pose[..., 2])[..., None, None]
    x, y = traj[..., 0], traj[..., 1]
    return torch.stack([c * x - s * y + pose[..., 0, None, None], s * x + c * y + pose[..., 1, None, None]], -1)


class V2Predictor(nn.Module):
    """Agent + lane encoders -> scene transformer -> anchor decoder.

    predict_agents=1: target only, forward returns traj (B, K, F, 2), logits (B, K) (the original model).
    predict_agents=M>1: target (slot 0) + the M-1 nearest neighbours (slots 2..M; slot 1 = ego is skipped,
    the planner controls it). Each agent's path is predicted in its own frame (origin = its current pose,
    x = its heading) and moved to the target frame; forward returns traj (B, M, K, F, 2), logits (B, M, K),
    with the target at index 0.

    ego_plan=True (block 2): the ego's planned future becomes one extra scene token at the ego's current pose.
    In training it is dropped (padded) for each sample with probability plan_drop, so the model also works
    without a plan; forward(batch, plan=True/False) forces it on/off (default: on in eval).
    soft_scores=True (block 3): score loss against softmax(-ADE_k / 1 m) instead of the one-hot winner.
    target_split=True: multi-agent loss = target loss + mean of the other agents' losses.
    plan_queries=True (needs ego_plan): the ego plan, expressed in each predicted agent's own frame (F x 2,
    in 50 m units, masked by ego_future_valid), goes through an MLP and is added to that agent's decoder
    queries, so every agent sees where the ego will be relative to itself. Zero when the plan is dropped/off."""

    MIN_FUTURE_STEPS = 6  # agents with fewer valid future steps are left out of the loss and the metrics

    def __init__(self, anchors, d_model=128, heads=8, encoder_layers=3, decoder_layers=2, dropout=0.1,
                 assign="predicted", topology=False, scale_inputs=False, predict_agents=1,
                 ego_plan=False, plan_drop=0.5, soft_scores=False, target_split=False, plan_queries=False):
        super().__init__()
        if predict_agents > 1 and assign != "predicted":
            raise ValueError("multi-agent prediction supports assign='predicted' only")
        self.assign = assign
        self.topology = topology
        self.predict_agents = predict_agents
        self.ego_plan = ego_plan
        self.plan_drop = plan_drop
        if plan_queries and not ego_plan:
            raise ValueError("plan_queries needs ego_plan=True")
        self.plan_queries = plan_queries
        self.pos_scale_plan = 50.0
        self.soft_scores = soft_scores
        self.target_split = target_split
        self.slots = [0] + list(range(2, predict_agents + 1)) if predict_agents > 1 else [0]
        pos_scale = 50.0 if scale_inputs else 1.0
        self.agent_encoder = AgentEncoder(d_model, pos_scale=pos_scale)
        self.lane_encoder = LaneEncoder(d_model, pos_scale=pos_scale)
        self.scene_encoder = SceneEncoder(d_model, heads, encoder_layers, dropout, topology=topology)
        self.decoder = AnchorDecoder(anchors, d_model, heads, decoder_layers, dropout=dropout)
        if ego_plan:
            self.plan_encoder = EgoPlanEncoder(d_model, pos_scale=pos_scale)
        if plan_queries:
            self.plan_query_mlp = mlp(2 * 12, d_model)

    def forward(self, batch, plan=None):
        agent_pose, lane_pose = token_poses(batch)
        tokens = [self.agent_encoder(batch), self.lane_encoder(batch)]
        pad = [~batch["agent_mask"], ~batch["lane_mask"]]
        poses = [agent_pose, lane_pose]
        if self.ego_plan:
            off = ~batch["ego_future_valid"].any(-1)                                    # (B,) no plan at all
            if plan is False:
                off = torch.ones_like(off)
            elif plan is None and self.training:
                off = off | (torch.rand(off.shape, device=off.device) < self.plan_drop)
            tokens.append(self.plan_encoder(batch))
            pad.append(off[:, None])
            poses.append(agent_pose[:, 1:2])                                            # ego's current pose
        tokens, pad, poses = torch.cat(tokens, 1), torch.cat(pad, 1), torch.cat(poses, 1)
        adj = None
        if self.topology:
            # Token-level successor matrix: lane-lane links from lane_adj, zero for any pair with an agent / plan
            A, L = batch["agent_mask"].shape[1], batch["lane_mask"].shape[1]
            N = tokens.shape[1]
            adj = torch.zeros(len(tokens), N, N, dtype=torch.bool, device=tokens.device)
            adj[:, A:A + L, A:A + L] = batch["lane_adj"]
        scene = self.scene_encoder(tokens, pad, poses, adj)
        agents = scene[:, self.slots]
        if self.plan_queries:
            local = to_agent_frames(batch["ego_future"][..., :2], agent_pose[:, self.slots])     # (B, M, F, 2)
            local = local * batch["ego_future_valid"][:, None, :, None] / self.pos_scale_plan
            agents = agents + self.plan_query_mlp(local.flatten(2)) * (~off)[:, None, None]
        traj, logits = self.decoder(scene, pad, agents)
        if self.predict_agents == 1:
            return traj[:, 0], logits[:, 0]          # target frame = the target's own frame
        return to_target_frame(traj, agent_pose[:, self.slots]), logits

    def agent_targets(self, batch):
        """Futures (B, M, F, 2), validity (B, M, F) and the loss/metric mask (B, M) of the predicted agents."""
        future, valid = batch["agent_future"][:, self.slots], batch["agent_future_valid"][:, self.slots]
        include = batch["agent_mask"][:, self.slots] & (valid.sum(-1) >= self.MIN_FUTURE_STEPS)
        return future, valid, include

    def loss(self, batch, traj, logits):
        if self.predict_agents == 1:
            return v2_loss(traj, logits, batch["target_future"], batch["target_future_valid"],
                           self.decoder.anchors, assign=self.assign, soft_scores=self.soft_scores)
        return multi_agent_loss(traj, logits, *self.agent_targets(batch), soft_scores=self.soft_scores,
                                target_split=self.target_split)

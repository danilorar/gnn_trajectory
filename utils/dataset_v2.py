"""v2 dataset: turns one processed_v2 sample into fixed-size agent and lane-segment tokens.

Agent order: target (0), ego (1), then neighbours sorted by current distance to the target,
so the "closest max_agents" cut never drops the target or the ego.
All coordinates are in the target frame (target at the origin, heading along +x).
"""

import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

DT = 0.5  # seconds between keyframes (2 Hz)

AGENT_TYPES = ["car", "truck_bus", "pedestrian", "cyclist", "ego", "other"]
EGO_TYPE = AGENT_TYPES.index("ego")

# nuScenes ego vehicle is a Renault Zoe; size from the manufacturer's spec
# (width 1.73 m without mirrors, length 4.08 m), as used in the nuScenes devkit.
EGO_SIZE = (1.73, 4.08)  # (width, length) in metres


def agent_type(category):
    """Map a nuScenes category name to an index in AGENT_TYPES."""
    if category == "vehicle.car" or category == "vehicle.emergency.police":
        return 0
    if category.startswith(("vehicle.truck", "vehicle.bus", "vehicle.trailer",
                            "vehicle.construction", "vehicle.emergency.ambulance")):
        return 1
    if category.startswith("human.pedestrian"):
        return 2
    if category in ("vehicle.bicycle", "vehicle.motorcycle"):
        return 3
    return 5


def add_velocity(states, valid):
    """states (..., T, 4) [x, y, cos, sin] -> (..., T, 6) with [vx, vy] appended.
    Velocity at step t uses steps t-1 and t; it is 0 where either step is invalid (and at t = 0)."""
    vel = np.zeros(states.shape[:-1] + (2,), dtype=np.float32)
    both = valid[..., 1:] & valid[..., :-1]
    vel[..., 1:, :] = (states[..., 1:, :2] - states[..., :-1, :2]) / DT * both[..., None]
    return np.concatenate([states, vel], axis=-1)


class TrajectoryDatasetV2(Dataset):
    """
    Returns per sample (A = max_agents, L = max_lanes, P = max_points, T = 5 past, F = 12 future):

    agent_hist        (A, T, 6)  [x, y, cos, sin, vx, vy]
    agent_hist_valid  (A, T)     bool
    agent_type        (A,)       index into AGENT_TYPES
    agent_size        (A, 2)     width, length
    agent_mask        (A,)       bool, real agent slot
    agent_future      (A, F, 2)  xy (ego future is not used; its slot stays invalid)
    agent_future_valid(A, F)     bool
    lane_points       (L, P, 4)  [x, y, cos, sin]
    lane_point_valid  (L, P)     bool
    lane_is_connector (L,)       bool
    lane_mask         (L,)       bool, real lane slot
    lane_adj          (L, L)     bool, lane_adj[i, j] = segment j succeeds segment i
    target_future     (F, 2)     = agent_future[0]
    target_future_valid (F,)
    """

    def __init__(self, path, max_agents=32, max_lanes=128, max_points=21):
        with open(path, "rb") as f:
            self.samples = pickle.load(f)
        self.max_agents = max_agents
        self.max_lanes = max_lanes
        self.max_points = max_points

    def __len__(self):
        return len(self.samples)

    def _agents(self, s):
        A = self.max_agents
        T, F = s["agent_past"].shape[1], s["agent_future"].shape[1]

        hist = add_velocity(s["agent_past"], s["agent_past_valid"])          # (N, T, 6)
        ego_hist = add_velocity(s["ego_past"][None], s["ego_past_valid"][None])  # (1, T, 6)
        types = np.array([agent_type(c) for c in s["agent_categories"]], dtype=np.int64)

        # Neighbours (index >= 1 in the stored arrays) sorted by current distance to the target
        neighbours = 1 + np.argsort(np.linalg.norm(s["agent_past"][1:, -1, :2], axis=-1), kind="stable")
        order = np.concatenate([[0], neighbours])[: A - 1]   # target + closest neighbours
        n = len(order) + 1                                    # + ego

        out = {
            "agent_hist": np.zeros((A, T, 6), np.float32),
            "agent_hist_valid": np.zeros((A, T), bool),
            "agent_type": np.full(A, AGENT_TYPES.index("other"), np.int64),
            "agent_size": np.zeros((A, 2), np.float32),
            "agent_mask": np.zeros(A, bool),
            "agent_future": np.zeros((A, F, 2), np.float32),
            "agent_future_valid": np.zeros((A, F), bool),
        }
        slots = np.concatenate([[0], np.arange(2, n)])        # slot 1 is reserved for the ego
        out["agent_hist"][slots] = hist[order]
        out["agent_hist_valid"][slots] = s["agent_past_valid"][order]
        out["agent_type"][slots] = types[order]
        out["agent_size"][slots] = s["agent_size"][order]
        out["agent_future"][slots] = s["agent_future"][order, :, :2]
        out["agent_future_valid"][slots] = s["agent_future_valid"][order]

        # Ego: past only; its future enters later as the ego-plan token (phase D)
        out["agent_hist"][1] = ego_hist[0]
        out["agent_hist_valid"][1] = s["ego_past_valid"]
        out["agent_type"][1] = EGO_TYPE
        out["agent_size"][1] = EGO_SIZE

        out["agent_mask"][:n] = True
        return out

    def _lanes(self, s):
        L, P = self.max_lanes, self.max_points
        segments = s["lane_points"]

        # Keep the max_lanes segments closest to the target
        dist = np.array([np.linalg.norm(seg[:, :2], axis=-1).min() for seg in segments])
        keep = np.sort(np.argsort(dist, kind="stable")[:L])
        new_index = np.full(len(segments), -1)
        new_index[keep] = np.arange(len(keep))

        out = {
            "lane_points": np.zeros((L, P, 4), np.float32),
            "lane_point_valid": np.zeros((L, P), bool),
            "lane_is_connector": np.zeros(L, bool),
            "lane_mask": np.zeros(L, bool),
            "lane_adj": np.zeros((L, L), bool),
        }
        for i, k in enumerate(keep):
            seg = segments[k][:P]
            out["lane_points"][i, : len(seg)] = seg
            out["lane_point_valid"][i, : len(seg)] = True
        out["lane_is_connector"][: len(keep)] = s["lane_is_connector"][keep]
        out["lane_mask"][: len(keep)] = True

        # Successor edges between kept segments only
        if len(s["lane_edges"]):
            a, b = new_index[s["lane_edges"][:, 0]], new_index[s["lane_edges"][:, 1]]
            ok = (a >= 0) & (b >= 0)
            out["lane_adj"][a[ok], b[ok]] = True
        return out

    def __getitem__(self, idx):
        s = self.samples[idx]
        out = {**self._agents(s), **self._lanes(s)}
        out["target_future"] = out["agent_future"][0]
        out["target_future_valid"] = out["agent_future_valid"][0]
        return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in out.items()}

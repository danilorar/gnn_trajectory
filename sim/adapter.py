"""Phase D adapter: simulator state -> v2 model inputs.

The simulator runs at 10 Hz; the v2 model wants 2 s of history at 2 Hz (5 steps) per agent, the lanes
within 50 m as <= 20 m segments, all in the target frame, plus the ego plan. This module rebuilds exactly
what scripts/build_processed_dataset.py (--lane-tokens) stores for one target, so the dataset's own
conversion (TrajectoryDatasetV2._agents/_lanes) can turn it into tensors.

    buffer = HistoryBuffer()
    every sim step:  buffer.push(snapshot_from_scenario(scenario, step))   # log replay
                     (or snapshot_from_env(env) with the live ego)
    when planning:   sample = build_sample(buffer, scenario["map_features"], target_id, ego_plan)
                     batch = to_model_inputs(sample)          # dict of tensors, as TrajectoryDatasetV2[i]

Builder rules reproduced (see get_agent_states_for_sample / build_lane_tokens):
- agents: the target first, then every vehicle.* / human.pedestrian* object present now within 35 m of it;
  history = the 5 keyframes t-2 s ... t (every 5th sim step), a missing step is invalid;
- the ego is separate (ego_past / ego_future); its future is the ego plan;
- lanes and lane connectors whose polygon intersects the 100 m x 100 m box around the target, 1 m points,
  <= 20 m segments, successor edges inside a lane and to exit lanes. ScenarioNet scenarios keep no polygon
  for lane connectors: call add_connector_polygons() once per scenario to take them from the nuScenes map
  expansion (otherwise the centreline is used, which misses connectors whose polygon only touches the box).
"""

from collections import deque
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.lane_tokens import poses_to_local, split_lane  # noqa: E402

EGO_ID = "ego"
SIM_HZ = 10
STRIDE = 5            # sim steps per model step (0.5 s)
PAST, FUTURE = 5, 12  # model steps
LANE_TYPES = {"LANE_SURFACE_STREET": False, "LANE_SURFACE_UNSTRUCTURE": True}   # type -> is connector


class HistoryBuffer:
    """The last (PAST - 1) * STRIDE + 1 snapshots (2 s at 10 Hz).
    A snapshot is {object id: (x, y, yaw, width, length, category)} of the objects present at that step."""

    def __init__(self):
        self.snapshots = deque(maxlen=(PAST - 1) * STRIDE + 1)

    def push(self, snapshot):
        self.snapshots.append(snapshot)

    def now(self):
        return self.snapshots[-1]

    def history(self, obj_id):
        """(PAST, 3) world poses at t-2 s ... t and (PAST,) validity; steps before the buffer start are invalid."""
        poses, valid = np.zeros((PAST, 3)), np.zeros(PAST, bool)
        for k in range(PAST):
            back = (PAST - 1 - k) * STRIDE                 # sim steps before now
            if back < len(self.snapshots):
                state = self.snapshots[-1 - back].get(obj_id)
                if state is not None:
                    poses[k], valid[k] = state[:3], True
        return poses, valid


def snapshot_from_scenario(scenario, step):
    """Logged states of every valid track at one sim step of a ScenarioNet scenario (ego id = EGO_ID)."""
    sdc = scenario["metadata"]["sdc_id"]
    snap = {}
    for obj_id, track in scenario["tracks"].items():
        st = track["state"]
        if step >= len(st["valid"]) or not st["valid"][step]:
            continue
        key = EGO_ID if obj_id == sdc else obj_id
        snap[key] = (float(st["position"][step][0]), float(st["position"][step][1]), float(st["heading"][step]),
                     float(np.ravel(st["width"][step])[0]), float(np.ravel(st["length"][step])[0]),
                     track["metadata"].get("type", track["type"]))
    return snap


def snapshot_from_env(env):
    """Logged states of the other agents at the env's current step + the live ego from MetaDrive.
    Non-reactive traffic: everyone except the ego replays the log, so the log is their true state."""
    scenario = env.engine.data_manager.current_scenario
    snap = snapshot_from_scenario(scenario, env.episode_step)
    ego = env.agent
    snap[EGO_ID] = (float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta),
                    float(ego.WIDTH), float(ego.LENGTH), "ego")
    return snap


def add_connector_polygons(map_features, map_name, dataroot=PROJECT_ROOT / "data/nuscenes_mini"):
    """Copy of map_features where every lane connector has its nuScenes polygon, in the features' own frame.

    The frame offset (MetaDrive shifts the map) is found from a lane that has a polygon in both."""
    from nuscenes.map_expansion.map_api import NuScenesMap

    nmap = NuScenesMap(dataroot=str(dataroot), map_name=map_name)
    lane = next(t for t, f in map_features.items()
                if f["type"] == "LANE_SURFACE_STREET" and f.get("polygon") is not None and len(f["polygon"]) >= 3)
    ref = np.asarray(nmap.extract_polygon(nmap.get("lane", lane)["polygon_token"]).boundary.xy).T
    offset = np.asarray(map_features[lane]["polygon"])[0, :2] - ref[0]
    out = dict(map_features)
    connectors = {r["token"]: r for r in nmap.lane_connector}
    for token, f in map_features.items():
        if f["type"] == "LANE_SURFACE_UNSTRUCTURE" and token in connectors:
            poly = np.asarray(nmap.extract_polygon(connectors[token]["polygon_token"]).boundary.xy).T
            out[token] = {**f, "polygon": poly + offset}
    return out


def _in_box(feature, origin, radius):
    from shapely.geometry import LineString, Polygon, box
    patch = box(origin[0] - radius, origin[1] - radius, origin[0] + radius, origin[1] + radius)
    polygon = feature.get("polygon")
    if polygon is not None and len(polygon) >= 3:
        return Polygon(np.asarray(polygon)[:, :2]).intersects(patch)
    return LineString(np.asarray(feature["polyline"])[:, :2]).intersects(patch)


class LaneSelector:
    """Which lanes / connectors intersect the box around a point: the same rule as _in_box, but the shapely
    shapes are built once per scene and kept in an STRtree, and the candidates (from a box `margin` m larger)
    are cached until the query point has moved more than `margin` m. The result is exact, not approximate:
    only the final intersects() test runs per call, on the cached candidates."""

    def __init__(self, map_features, margin=5.0):
        from shapely.geometry import LineString, Polygon
        from shapely.strtree import STRtree
        self.margin, self.center, self.candidates = margin, None, []
        self.tokens, geoms = [], []
        order = {False: 0, True: 1}
        items = [(order[LANE_TYPES[f["type"]]], n, t, f) for n, (t, f) in enumerate(map_features.items())
                 if f["type"] in LANE_TYPES and len(f["polyline"]) >= 2]
        for _, _, token, f in sorted(items, key=lambda x: (x[0], x[1])):          # lanes, then connectors
            poly = f.get("polygon")
            geoms.append(Polygon(np.asarray(poly)[:, :2]) if poly is not None and len(poly) >= 3
                         else LineString(np.asarray(f["polyline"])[:, :2]))
            self.tokens.append(token)
        self.geoms = geoms
        self.tree = STRtree(geoms)
        self.recomputed = 0

    def select(self, origin, radius):
        from shapely.geometry import box
        if self.center is None or np.hypot(*(np.asarray(origin) - self.center)) > self.margin:
            r = radius + self.margin
            hits = self.tree.query(box(origin[0] - r, origin[1] - r, origin[0] + r, origin[1] + r))
            self.candidates, self.center = sorted(int(i) for i in hits), np.asarray(origin, float)
            self.recomputed += 1
        patch = box(origin[0] - radius, origin[1] - radius, origin[0] + radius, origin[1] + radius)
        return [self.tokens[i] for i in self.candidates if self.geoms[i].intersects(patch)]


def lane_tokens(map_features, origin, yaw, radius=50.0, seg_len=20, selector=None):
    """build_lane_tokens() on ScenarioNet map features. Polylines are nuScenes discretize_lane() output
    at 1 m: (x, y, yaw) per point. Order: lanes, then connectors (as get_lanes_in_radius).
    selector: a LaneSelector for this map (cached, same result); None = test every feature."""
    if selector is not None:
        selected = selector.select(origin, radius)
    else:
        selected = []
        for is_connector in (False, True):
            for token, f in map_features.items():
                if LANE_TYPES.get(f["type"]) is is_connector and len(f["polyline"]) >= 2 and _in_box(f, origin, radius):
                    selected.append(token)
    lane_points, is_conn, edges, first, last = [], [], [], {}, {}
    for token in selected:
        f = map_features[token]
        segments = split_lane(poses_to_local(np.asarray(f["polyline"])[:, :3], origin, yaw), seg_len)
        if not segments:
            continue
        start = len(lane_points)
        lane_points.extend(segments)
        is_conn.extend([LANE_TYPES[f["type"]]] * len(segments))
        first[token], last[token] = start, len(lane_points) - 1
        edges.extend((k, k + 1) for k in range(start, last[token]))
    for token in first:
        for out in map_features[token].get("exit_lanes", []):
            if out in first:
                edges.append((last[token], first[out]))
    return lane_points, np.array(is_conn, bool), np.array(sorted(set(edges)), np.int64).reshape(-1, 2)


def build_sample(buffer, map_features, target_id, ego_plan=None, ego_plan_valid=None, agent_radius=35.0,
                 lane_radius=50.0, selector=None):
    """One processed-style sample (the fields TrajectoryDatasetV2 reads) for target_id at the buffer's last step.

    ego_plan: (12, 3) world (x, y, yaw) at t + 0.5 s ... t + 6 s, or None (then marked invalid)."""
    now = buffer.now()
    tx, ty, tyaw = now[target_id][:3]
    origin = np.array([tx, ty], np.float32)

    ids = [target_id] + [i for i, s in now.items()
                         if i not in (target_id, EGO_ID)
                         and (s[5].startswith("vehicle.") or s[5].startswith("human.pedestrian"))
                         and np.hypot(s[0] - tx, s[1] - ty) <= agent_radius]
    past = np.zeros((len(ids), PAST, 4), np.float32)
    past_valid = np.zeros((len(ids), PAST), bool)
    for n, i in enumerate(ids):
        poses, valid = buffer.history(i)
        if valid.any():
            past[n, valid] = poses_to_local(poses[valid], origin, tyaw)
        past_valid[n] = valid

    ego_poses, ego_valid = buffer.history(EGO_ID)
    ego_past = np.zeros((PAST, 4), np.float32)
    if ego_valid.any():
        ego_past[ego_valid] = poses_to_local(ego_poses[ego_valid], origin, tyaw)
    ego_future = np.zeros((FUTURE, 4), np.float32)
    if ego_plan is not None:
        ego_future = poses_to_local(np.asarray(ego_plan, np.float64), origin, tyaw)
        ego_plan_valid = np.ones(FUTURE, bool) if ego_plan_valid is None else np.asarray(ego_plan_valid, bool)
    else:
        ego_plan_valid = np.zeros(FUTURE, bool)

    lanes, is_conn, edges = lane_tokens(map_features, origin, tyaw, lane_radius, selector=selector)
    return {
        "agent_past": past, "agent_past_valid": past_valid,
        "agent_future": np.zeros((len(ids), FUTURE, 4), np.float32),          # unknown at run time
        "agent_future_valid": np.zeros((len(ids), FUTURE), bool),
        "agent_size": np.array([now[i][3:5] for i in ids], np.float32),       # (width, length)
        "agent_categories": [now[i][5] for i in ids], "agent_instance_tokens": ids,
        "ego_past": ego_past, "ego_past_valid": ego_valid,
        "ego_future": ego_future.astype(np.float32), "ego_future_valid": ego_plan_valid,
        "lane_points": lanes, "lane_is_connector": is_conn, "lane_edges": edges,
        "target_future": np.zeros((FUTURE, 2), np.float32),
    }


def ego_order(sample):
    """Neighbour indices (>= 1 in the sample arrays) sorted by current distance to the ego."""
    ego = sample["ego_past"][-1, :2] if sample["ego_past_valid"][-1] else np.zeros(2, np.float32)
    return 1 + np.argsort(np.linalg.norm(sample["agent_past"][1:, -1, :2] - ego, axis=-1), kind="stable")


def _ego_ordered_dataset():
    """TrajectoryDatasetV2 whose neighbours are sorted by distance to the ego instead of the target.
    Target stays slot 0, ego slot 1; the model has no positional encoding over agents, so the order only
    decides which agents fill the predicted slots 2..M and which are cut at max_agents."""
    from utils.dataset_v2 import EGO_SIZE, EGO_TYPE, AGENT_TYPES, TrajectoryDatasetV2, add_velocity, agent_type

    class EgoOrdered(TrajectoryDatasetV2):
        def _agents(self, s):                     # = TrajectoryDatasetV2._agents with another sort key
            A = self.max_agents
            T, F = s["agent_past"].shape[1], s["agent_future"].shape[1]
            hist = add_velocity(s["agent_past"], s["agent_past_valid"])
            ego_hist = add_velocity(s["ego_past"][None], s["ego_past_valid"][None])
            types = np.array([agent_type(c) for c in s["agent_categories"]], dtype=np.int64)
            order = np.concatenate([[0], ego_order(s)])[: A - 1]
            n = len(order) + 1
            out = {
                "agent_hist": np.zeros((A, T, 6), np.float32),
                "agent_hist_valid": np.zeros((A, T), bool),
                "agent_type": np.full(A, AGENT_TYPES.index("other"), np.int64),
                "agent_size": np.zeros((A, 2), np.float32),
                "agent_mask": np.zeros(A, bool),
                "agent_future": np.zeros((A, F, 2), np.float32),
                "agent_future_valid": np.zeros((A, F), bool),
            }
            slots = np.concatenate([[0], np.arange(2, n)])
            out["agent_hist"][slots] = hist[order]
            out["agent_hist_valid"][slots] = s["agent_past_valid"][order]
            out["agent_type"][slots] = types[order]
            out["agent_size"][slots] = s["agent_size"][order]
            out["agent_future"][slots] = s["agent_future"][order, :, :2]
            out["agent_future_valid"][slots] = s["agent_future_valid"][order]
            out["agent_hist"][1] = ego_hist[0]
            out["agent_hist_valid"][1] = s["ego_past_valid"]
            out["agent_type"][1] = EGO_TYPE
            out["agent_size"][1] = EGO_SIZE
            out["agent_mask"][:n] = True
            return out

    return EgoOrdered


def to_model_inputs(sample, max_agents=64, max_lanes=192, max_points=21, order="target"):
    """The dict of tensors TrajectoryDatasetV2.__getitem__ returns, for one adapter sample.
    order="ego": neighbours sorted by distance to the ego (for planning); "target": as in training."""
    from utils.dataset_v2 import TrajectoryDatasetV2

    cls = _ego_ordered_dataset() if order == "ego" else TrajectoryDatasetV2
    ds = cls.__new__(cls)                           # reuse the conversion code, no file needed
    ds.samples, ds.max_agents, ds.max_lanes, ds.max_points = [sample], max_agents, max_lanes, max_points
    return ds[0]

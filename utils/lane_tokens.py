import numpy as np
from nuscenes.prediction.input_representation.static_layers import get_lanes_in_radius


def poses_to_local(poses, origin, yaw):
    """poses: list of (x, y, lane_yaw) in world frame.
    Return (P, 4) array [x, y, cos(psi), sin(psi)] in the target frame.
    psi = lane_yaw - yaw (relative heading)."""
    poses = np.asarray(poses, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)

    # 1. shift by origin, rotate by -yaw (same math as world_to_local in v1)
    c, s = np.cos(-yaw), np.sin(-yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = (poses[:, :2] - origin) @ rotation.T

    # 2. psi = lane_yaw - yaw; store cos and sin (no angle wrapping needed then)
    psi = poses[:, 2] - yaw
    return np.column_stack([xy, np.cos(psi), np.sin(psi)]).astype(np.float32)


def split_lane(points, seg_len=20):
    """Cut one lane (P, 4) into pieces of about seg_len points (= metres, at 1 m spacing).
    Drop pieces with fewer than 2 points."""
    pieces = []
    for start in range(0, len(points), seg_len):
        # Take one extra point so consecutive pieces share their boundary point (no gap).
        piece = points[start:start + seg_len + 1]
        if len(piece) >= 2:
            pieces.append(piece)
    return pieces


def build_lane_tokens(nusc_map, origin, yaw, radius=50.0, seg_len=20):
    """Main function. Returns lane_points, lane_is_connector, lane_edges.

    lane_points       : list of S arrays, each (P_i, 4) [x, y, cos(psi), sin(psi)]
    lane_is_connector : (S,) bool, True if the segment comes from a lane connector
    lane_edges        : (E, 2) int64, directed (from_segment, to_segment) successor links
    """
    # 1. lanes -> {lane_token: [(x, y, yaw), ...]}  KEEP the yaw this time
    lanes = get_lanes_in_radius(
        x=float(origin[0]),
        y=float(origin[1]),
        radius=radius,
        discretization_meters=1.0,
        map_api=nusc_map,
    )

    # 2. connector tokens
    connector_tokens = {r["token"] for r in nusc_map.lane_connector}

    lane_points, lane_is_connector, edges = [], [], []
    first_seg, last_seg = {}, {}

    # 3. for each lane: to local -> split -> append segments
    for token, poses in lanes.items():
        if len(poses) < 2:
            continue
        segments = split_lane(poses_to_local(poses, origin, yaw), seg_len)
        if not segments:
            continue

        start = len(lane_points)
        lane_points.extend(segments)
        lane_is_connector.extend([token in connector_tokens] * len(segments))
        first_seg[token] = start
        last_seg[token] = len(lane_points) - 1

        # 4. edges inside a lane: (k, k+1) for consecutive segments
        edges.extend((k, k + 1) for k in range(start, last_seg[token]))

    # 5. edges between lanes (only to lanes that are inside the radius)
    for token in first_seg:
        try:
            outgoing = nusc_map.get_outgoing_lane_ids(token)
        except KeyError:  # token missing from the map's connectivity table
            continue
        for out_token in outgoing:
            if out_token in first_seg:
                edges.append((last_seg[token], first_seg[out_token]))

    lane_edges = np.array(sorted(set(edges)), dtype=np.int64).reshape(-1, 2)
    return lane_points, np.array(lane_is_connector, dtype=bool), lane_edges

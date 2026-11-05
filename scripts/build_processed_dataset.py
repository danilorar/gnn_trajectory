import numpy as np

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm
from collections import Counter
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.prediction.input_representation.static_layers import get_lanes_in_radius # For map lane extraction

from utils.lane_tokens import build_lane_tokens, poses_to_local # v2 lane-segment tokens


MAP_CACHE = {} # Cache for NuScenesMap objects to avoid reloading the same map multiple times


# ============================================================
# Coordinate transformation
# ============================================================

def world_to_local(points, origin, yaw):
    """
    Transform 2D points from global nuScenes coordinates into
    the target agent's local coordinate frame.

    In the local frame:
        - the target is located at (0, 0)
        - the target heading points approximately along +x

    Parameters
    ----------
    points : array-like, shape (N, 2)
        Global xy positions.

    origin : array-like, shape (2,)
        Current xy position of the target agent.

    yaw : float
        Current target heading in radians.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Positions expressed in the target-local frame.
    """

    points = np.asarray(points, dtype=np.float32)
    origin = np.asarray(origin, dtype=np.float32)

    # First translate so the target is at the origin.
    centered_points = points - origin

    # Rotate by -yaw so the target heading is aligned with +x.
    c = np.cos(-yaw)
    s = np.sin(-yaw)

    rotation = np.array(
        [
            [c, -s],
            [s,  c]
        ],
        dtype=np.float32
    )

    return centered_points @ rotation.T


# ============================================================
# Agent history helper
# ============================================================

def get_past_positions(nusc, ann, past_steps=4):
    """
    Retrieve the current position and previous positions of one agent.

    For past_steps=4 at approximately 2 Hz, this gives:

        5 positions = approximately 2 seconds of history

    Parameters
    ----------
    nusc : NuScenes
        Loaded nuScenes object.

    ann : dict
        Current sample_annotation of the agent.

    past_steps : int
        Number of previous annotation steps to retrieve.

    Returns
    -------
    np.ndarray, shape (<= past_steps + 1, 2)
        Positions ordered from oldest to newest.
    """

    positions = [ann["translation"][:2]]
    current_ann = ann

    # Walk backwards through this same physical agent.
    for _ in range(past_steps):

        if current_ann["prev"] == "":
            break

        current_ann = nusc.get(
            "sample_annotation",
            current_ann["prev"]
        )

        positions.append(
            current_ann["translation"][:2]
        )

    # We traversed backwards, so restore chronological order.
    positions.reverse()

    return np.asarray(
        positions,
        dtype=np.float32
    )


# ============================================================
# Neighbor extraction
# ============================================================

def get_neighbors_for_sample(
    nusc,
    sample_info,
    radius=35.0,
    past_steps=4
):
    """
    Find nearby road users around the target at the current time.

    A neighbor is kept if:
        1. it is a vehicle or pedestrian,
        2. it lies within `radius` metres of the target,
        3. it has a complete history of past_steps + 1 positions.

    All neighbor trajectories are transformed into the SAME
    coordinate frame as the target.

    Parameters
    ----------
    sample_info : dict
        Information about the current target prediction sample.

    radius : float
        Maximum target-neighbor distance in metres.

    past_steps : int
        Number of previous annotation steps required.

    Returns
    -------
    list[dict]
        One dictionary per valid neighbor.
    """

    # Current nuScenes sample / timestamp.
    current_sample = nusc.get(
        "sample",
        sample_info["sample_token"]
    )

    # Current target annotation.
    target_ann = nusc.get(
        "sample_annotation",
        sample_info["current_ann_token"]
    )

    target_position = np.asarray(
        target_ann["translation"][:2],
        dtype=np.float32
    )

    target_yaw = sample_info["yaw"]
    target_instance_token = sample_info["instance_token"]

    neighbors = []

    # Every annotation present at this timestamp.
    for ann_token in current_sample["anns"]:

        ann = nusc.get(
            "sample_annotation",
            ann_token
        )

        # Do not include the target itself.
        if ann["instance_token"] == target_instance_token:
            continue

        category = ann["category_name"]

        # Keep surrounding vehicles and pedestrians.
        if not (
            category.startswith("vehicle.")
            or category.startswith("human.pedestrian")
        ):
            continue

        neighbor_position = np.asarray(
            ann["translation"][:2],
            dtype=np.float32
        )

        # Euclidean target-neighbor distance at current time.
        distance = np.linalg.norm(
            neighbor_position - target_position
        )

        if distance > radius:
            continue

        # Retrieve approximately 2 seconds of neighbor history.
        neighbor_past_world = get_past_positions(
            nusc,
            ann,
            past_steps=past_steps
        )

        # Require complete history.
        if len(neighbor_past_world) < past_steps + 1:
            continue

        # IMPORTANT:
        # Transform neighbor using TARGET origin and TARGET yaw,
        # not the neighbor's own frame.
        neighbor_past_local = world_to_local(
            neighbor_past_world,
            target_position,
            target_yaw
        )

        neighbors.append({
            "instance_token": ann["instance_token"],
            "category": category,
            "distance": float(distance),
            "past": neighbor_past_local
        })

    return neighbors

# ============================================================
# v2 agent and ego states (only used with --lane-tokens)
# ============================================================

def get_sample_timeline(nusc, sample_record, past_steps=4, future_steps=12):
    """Sample tokens from t = -past_steps to t = +future_steps ("" where the scene ends).
    Index past_steps is the current sample."""
    past, current = [], sample_record
    for _ in range(past_steps):
        current = nusc.get("sample", current["prev"]) if current and current["prev"] else None
        past.append(current["token"] if current else "")

    future, current = [], sample_record
    for _ in range(future_steps):
        current = nusc.get("sample", current["next"]) if current and current["next"] else None
        future.append(current["token"] if current else "")

    return past[::-1] + [sample_record["token"]] + future


def get_pose(record):
    """(x, y, yaw) of an annotation or ego_pose record in world frame."""
    return (record["translation"][0], record["translation"][1],
            Quaternion(record["rotation"]).yaw_pitch_roll[0])


def get_agent_states_for_sample(
    nusc,
    sample_record,
    target_ann,
    origin,
    yaw,
    radius=35.0,
    past_steps=4,
    future_steps=12,
):
    """
    All vehicles/pedestrians within `radius` at the current time, target first (index 0),
    plus the ego vehicle. Steps are aligned by sample timestamp, so a missed annotation
    becomes an invalid step instead of a shifted one. Agents with incomplete history are kept
    (unlike v1 neighbours); their missing steps are marked invalid.

    States are [x, y, cos(theta), sin(theta)] in the target frame; invalid steps are zeros.
    """
    timeline = get_sample_timeline(nusc, sample_record, past_steps, future_steps)
    step_of = {token: t for t, token in enumerate(timeline) if token}
    T = len(timeline)

    # Target first, then neighbours
    anns = [target_ann]
    for ann_token in sample_record["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        category = ann["category_name"]
        if ann["instance_token"] == target_ann["instance_token"]:
            continue
        if not (category.startswith("vehicle.") or category.startswith("human.pedestrian")):
            continue
        if np.linalg.norm(np.asarray(ann["translation"][:2]) - origin) > radius:
            continue
        anns.append(ann)

    N = len(anns)
    states = np.zeros((N, T, 4), dtype=np.float32)
    valid = np.zeros((N, T), dtype=bool)
    sizes = np.zeros((N, 2), dtype=np.float32)

    for i, ann in enumerate(anns):
        sizes[i] = ann["size"][:2]  # nuScenes size = (width, length, height)

        # Walk this agent's annotation chain both ways, placing each one at its timestep
        for direction in ("prev", "next"):
            current = ann
            while current is not None and current["sample_token"] in step_of:
                t = step_of[current["sample_token"]]
                states[i, t] = poses_to_local([get_pose(current)], origin, yaw)[0]
                valid[i, t] = True
                current = nusc.get("sample_annotation", current[direction]) if current[direction] else None

    # Ego vehicle pose at each keyframe (from the LIDAR_TOP sample_data)
    ego = np.zeros((T, 4), dtype=np.float32)
    ego_valid = np.zeros(T, dtype=bool)
    for t, token in enumerate(timeline):
        if not token:
            continue
        sample_data = nusc.get("sample_data", nusc.get("sample", token)["data"]["LIDAR_TOP"])
        ego[t] = poses_to_local([get_pose(nusc.get("ego_pose", sample_data["ego_pose_token"]))], origin, yaw)[0]
        ego_valid[t] = True

    p = past_steps + 1
    return {
        "agent_past": states[:, :p],              # (N, 5, 4), index 0 = target
        "agent_past_valid": valid[:, :p],         # (N, 5)
        "agent_future": states[:, p:],            # (N, 12, 4)
        "agent_future_valid": valid[:, p:],       # (N, 12)
        "agent_size": sizes,                      # (N, 2) width, length
        "agent_categories": [ann["category_name"] for ann in anns],
        "agent_instance_tokens": [ann["instance_token"] for ann in anns],
        "ego_past": ego[:p],                      # (5, 4)
        "ego_past_valid": ego_valid[:p],          # (5,)
        "ego_future": ego[p:],                    # (12, 4)
        "ego_future_valid": ego_valid[p:],        # (12,)
    }


# ============================================================
# Load the map of one sample
# ============================================================

def get_map_for_sample(nusc, sample_record):
    """Return the (cached) NuScenesMap of the location this sample was recorded in."""
    scene_record = nusc.get("scene", sample_record["scene_token"])
    log_record = nusc.get("log", scene_record["log_token"])
    map_name = log_record["location"]

    # To avoid reloading the same map multiple times, we cache NuScenesMap objects in MAP_CACHE
    if map_name not in MAP_CACHE:
        MAP_CACHE[map_name] = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)

    return MAP_CACHE[map_name]


# ============================================================
# Get map lanes for one sample 
# ============================================================

def get_map_lanes_for_sample(
    nusc, 
    sample_record, 
    origin,
    yaw, 
    map_radius=50.0 # Fixed lane-centerline context radius of the finalized datasets.
): 
    """
    Find nearby lane centerlines for one prediction sample
    and express them in the target-local coordinate frame.
    """
    
    nusc_map = get_map_for_sample(nusc, sample_record)
    
    # --------------------------------------------------------
    # Find nearby lane + lane_connectors centerlines 
    # Returned structure: 
    # 
    # {
    #     lane_token_1: [(x, y, yaw), ...],
    #     lane_token_2: [(x, y, yaw), ...],
    #     ...
    # }
    # --------------------------------------------------------
    
    lanes_world = get_lanes_in_radius(
        x=float(origin[0]), # current target's x position
        y=float(origin[1]), # current target's y position
        radius=map_radius,
        discretization_meters=1.0, # 1 meter between points
        map_api=nusc_map       
    )
    
    # --------------------------------------------------------
    # Convert each lane polyline into target-local coordinates
    # --------------------------------------------------------
    
    map_lanes =[]
    
    for lane_poses in lanes_world.values(): 
        
        if len(lane_poses) == 0: # Skip empty lanes
            continue
        
        # Each pose is (x, y, lane_yaw) 
        # For now we only want x and y 
        lane_xy_world = np.array(
            [
                [pose[0], pose[1]] for pose in lane_poses
            ], 
            dtype=np.float32
        )
        
        # Convert to target-local coordinates
        lane_xy_local = world_to_local(
            lane_xy_world,
            origin,
            yaw
        )
        
        map_lanes.append(lane_xy_local)
    
    return map_lanes


# ============================================================
# Process one official nuScenes prediction target
# ============================================================

def process_prediction_target(
    nusc,
    target_string,
    radius=35.0,
    past_steps=4,
    future_steps=12,
    include_map=False,
    include_lane_tokens=False,
):
    """
    Convert one official nuScenes prediction target into one
    processed trajectory-prediction example.

    Map extraction is opt-in; existing callers need no map files.
    include_lane_tokens adds the v2 lane-segment tokens (direction + successor edges).

    Returns
    -------
    processed_sample : dict or None
        Processed sample if valid.

    skip_reason : str or None
        Reason why the sample was skipped.
    """

    # --------------------------------------------------------
    # Split official target string
    # --------------------------------------------------------

    instance_token, sample_token = target_string.split("_")

    # Get the current nuScenes sample
    sample_record = nusc.get(
        "sample",
        sample_token
    )

    # --------------------------------------------------------
    # Find target annotation at this exact timestamp
    # --------------------------------------------------------

    target_ann = None

    for ann_token in sample_record["anns"]:

        ann = nusc.get(
            "sample_annotation",
            ann_token
        )

        if ann["instance_token"] == instance_token:
            target_ann = ann
            break

    if target_ann is None:
        return None, "target_not_found"

    # --------------------------------------------------------
    # Keep only vehicle targets
    # --------------------------------------------------------

    if not target_ann["category_name"].startswith("vehicle."):
        return None, "non_vehicle"

    # --------------------------------------------------------
    # Get target past
    #
    # 4 previous steps + current point
    # = 5 points ≈ 2 seconds at 2 Hz
    # --------------------------------------------------------

    past_anns = [target_ann]
    current_ann = target_ann

    for _ in range(past_steps):

        if current_ann["prev"] == "":
            return None, "missing_past"

        current_ann = nusc.get(
            "sample_annotation",
            current_ann["prev"]
        )

        past_anns.append(current_ann)

    # We walked backwards, so restore chronological order
    past_anns.reverse()

    # --------------------------------------------------------
    # Get target future
    #
    # 12 future points ≈ 6 seconds at 2 Hz
    # --------------------------------------------------------

    future_anns = []
    current_ann = target_ann

    for _ in range(future_steps):

        if current_ann["next"] == "":
            return None, "missing_future"

        current_ann = nusc.get(
            "sample_annotation",
            current_ann["next"]
        )

        future_anns.append(current_ann)

    # --------------------------------------------------------
    # Convert annotations into xy arrays
    # --------------------------------------------------------

    past_world = np.asarray(
        [
            ann["translation"][:2]
            for ann in past_anns
        ],
        dtype=np.float32
    )

    future_world = np.asarray(
        [
            ann["translation"][:2]
            for ann in future_anns
        ],
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Current target pose
    # --------------------------------------------------------

    origin = np.asarray(
        target_ann["translation"][:2],
        dtype=np.float32
    )

    quaternion = Quaternion(
        target_ann["rotation"]
    )

    yaw = quaternion.yaw_pitch_roll[0]

    # --------------------------------------------------------
    # Transform target past/future into target-local frame
    # --------------------------------------------------------

    past_local = world_to_local(
        past_world,
        origin,
        yaw
    )

    future_local = world_to_local(
        future_world,
        origin,
        yaw
    )

    # --------------------------------------------------------
    # Temporary structure used by neighbor extractor
    # --------------------------------------------------------

    target_sample = {
        "past": past_local,
        "future": future_local,
        "instance_token": instance_token,
        "current_ann_token": target_ann["token"],
        "sample_token": sample_token,
        "yaw": yaw
    }

    # --------------------------------------------------------
    # Find nearby agents
    # --------------------------------------------------------

    neighbors = get_neighbors_for_sample(
        nusc,
        target_sample,
        radius=radius,
        past_steps=past_steps
    )
    
    # --------------------------------------------------------
    # Find nearby map lanes
    # --------------------------------------------------------
    
    if include_map:
        map_lanes = get_map_lanes_for_sample(
            nusc,
            sample_record,
            origin,
            yaw,
            map_radius=50.0
        )
    
    if include_lane_tokens:
        lane_points, lane_is_connector, lane_edges = build_lane_tokens(
            get_map_for_sample(nusc, sample_record),
            origin,
            yaw,
            radius=50.0,
            seg_len=20
        )
        agent_states = get_agent_states_for_sample(
            nusc,
            sample_record,
            target_ann,
            origin,
            yaw,
            radius=radius,
            past_steps=past_steps,
            future_steps=future_steps
        )

    # NOTE: For debugging
    # print(" | Nº of nearby lanes:", len(map_lanes))
    
    # if len(map_lanes) > 0:
    #     print("First lane shape:", map_lanes[0].shape)
    #     print("First few local points")
    #     print(map_lanes[0][:5])
    
    # --------------------------------------------------------
    # Final processed sample
    # --------------------------------------------------------

    processed_sample = {
        "target_past": past_local,
        "target_future": future_local,

        "neighbors_pasts": [
            neighbor["past"]
            for neighbor in neighbors
        ],

        "neighbors_categories": [
            neighbor["category"]
            for neighbor in neighbors
        ],

        "neighbors_distances": [
            neighbor["distance"]
            for neighbor in neighbors
        ],

        "neighbor_instance_tokens": [
            neighbor["instance_token"]
            for neighbor in neighbors
        ],
        

        "scene_token": sample_record["scene_token"],
        "sample_token": sample_token,
        "target_instance_token": instance_token,
        "current_ann_token": target_ann["token"],
        "target_category": target_ann["category_name"]
    }

    if include_map:
        processed_sample["map_lanes"] = map_lanes

    if include_lane_tokens:
        processed_sample["lane_points"] = lane_points             # list of (P_i, 4) [x, y, cos psi, sin psi]
        processed_sample["lane_is_connector"] = lane_is_connector # (S,) bool
        processed_sample["lane_edges"] = lane_edges               # (E, 2) successor links
        processed_sample.update(agent_states)

    return processed_sample, None


# ============================================================
# Build many processed prediction examples
# ============================================================

_WORKER_NUSC = None  # NuScenes object shared with forked worker processes


def _process_target_in_worker(job):
    target_string, kwargs = job
    return process_prediction_target(_WORKER_NUSC, target_string, **kwargs)


def build_prediction_dataset(
    nusc,
    target_strings,
    radius=35.0,
    past_steps=4,
    future_steps=12,
    include_map=False,
    include_lane_tokens=False,
    workers=1,
):
    """Build samples, extracting lane context only when include_map / include_lane_tokens is True.
    workers > 1 processes targets in parallel (forked processes, same output order)."""
    global _WORKER_NUSC
    dataset = []
    skip_reasons = Counter()
    kwargs = dict(
        radius=radius,
        past_steps=past_steps,
        future_steps=future_steps,
        include_map=include_map,
        include_lane_tokens=include_lane_tokens,
    )

    if workers > 1:
        import multiprocessing as mp
        _WORKER_NUSC = nusc  # inherited by the workers through fork, not pickled
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap(_process_target_in_worker, [(t, kwargs) for t in target_strings], chunksize=32)
    else:
        pool = None
        results = (process_prediction_target(nusc, t, **kwargs) for t in target_strings)

    for processed_sample, skip_reason in tqdm(
        results,
        total=len(target_strings),
        desc="Processing prediction targets"
    ):
        # Keep valid samples
        if processed_sample is not None:
            dataset.append(processed_sample)

        # Count skipped samples
        else:
            skip_reasons[skip_reason] += 1

    if pool is not None:
        pool.close()
        pool.join()

    return dataset, skip_reasons


# ============================================================
# Validate processed samples
# ============================================================

def validate_dataset(dataset, include_lane_tokens=False):
    """Assert the expected shapes; returns the number of targets with a missing keyframe."""
    n_target_gaps = 0
    for sample in dataset:

        assert sample["target_past"].shape == (5, 2)
        assert sample["target_future"].shape == (12, 2)

        for neighbor_past in sample["neighbors_pasts"]:
            assert neighbor_past.shape == (5, 2)

        # lane check
        for lane in sample.get("map_lanes", []):
            assert lane.ndim == 2
            assert lane.shape[1] == 2
            assert np.isfinite(lane).all(), "Lane contains NaN or Inf values"

        # lane-token check
        if include_lane_tokens:
            n_segments = len(sample["lane_points"])
            assert len(sample["lane_is_connector"]) == n_segments
            for segment in sample["lane_points"]:
                assert segment.ndim == 2 and segment.shape[1] == 4 and len(segment) >= 2
                assert np.isfinite(segment).all(), "Lane segment contains NaN or Inf values"
            assert sample["lane_edges"].shape[1] == 2
            assert (sample["lane_edges"] < n_segments).all()

            n_agents = len(sample["agent_instance_tokens"])
            assert sample["agent_past"].shape == (n_agents, 5, 4)
            assert sample["agent_future"].shape == (n_agents, 12, 4)
            assert sample["agent_size"].shape == (n_agents, 2)
            assert sample["agent_past_valid"][0, -1], "target must be valid at the current step"
            # A target missing from one keyframe leaves an invalid (time-aligned) step,
            # while v1's target_future follows the annotation chain; compare only gap-free targets.
            if sample["agent_past_valid"][0].all() and sample["agent_future_valid"][0].all():
                assert np.allclose(sample["agent_future"][0, :, :2], sample["target_future"], atol=1e-3)
            else:
                n_target_gaps += 1
            assert sample["ego_past"].shape == (5, 4) and sample["ego_future"].shape == (12, 4)

    return n_target_gaps


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Build the fixed 35 m prediction datasets.")
    parser.add_argument(
        "--without-map", action="store_true",
        help="Build non-map datasets for the target-only, attention and multimodal baselines",
    )
    parser.add_argument(
        "--lane-tokens", action="store_true",
        help="Also store v2 lane-segment tokens; saved to data/processed_v2 so v1 data is not overwritten",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel worker processes (each holds its own map cache; ~4 fits in 16 GB)",
    )
    args = parser.parse_args()
    include_map = not args.without_map
    include_lane_tokens = args.lane_tokens

    DATA_DIR = PROJECT_ROOT / "data"
    VERSION = "v1.0-trainval"
    
    # Load nuScenes
    nusc = NuScenes(
        version=VERSION,
        dataroot=str(DATA_DIR),
        verbose=True
    )

    train_targets = get_prediction_challenge_split(
        "train",
        dataroot=str(DATA_DIR)
    )

    val_targets = get_prediction_challenge_split(
        "train_val",
        dataroot=str(DATA_DIR)
    )

    test_targets = get_prediction_challenge_split(
        "val",
        dataroot=str(DATA_DIR)
    )

    print("Train targets:", len(train_targets))
    print("Val targets:", len(val_targets))
    print("Test targets:", len(test_targets))

    import pickle

    PROCESSED_DIR = DATA_DIR / ("processed_v2" if include_lane_tokens else "processed")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.without_map else "_map"

    # --------------------------------------------------------
    # Build, validate and save each split before starting the next one
    # --------------------------------------------------------

    for split_name, targets in [
        ("train", train_targets),
        ("val", val_targets),
        ("test", test_targets),
    ]:
        dataset, skip_reasons = build_prediction_dataset(
            nusc,
            targets,
            radius=35.0,
            past_steps=4,
            future_steps=12,
            include_map=include_map,
            include_lane_tokens=include_lane_tokens,
            workers=args.workers,
        )

        print(f"\n{split_name.capitalize()} processed:", len(dataset))
        print(f"{split_name.capitalize()} skipped:", skip_reasons)

        n_target_gaps = validate_dataset(dataset, include_lane_tokens)
        print(f"{split_name} sample shapes are valid.")
        if include_lane_tokens:
            print(f"{split_name} targets with a missing keyframe (kept, step marked invalid): {n_target_gaps}")

        path = PROCESSED_DIR / f"nuscenes_{split_name}_prediction_samples{suffix}.pkl"
        with open(path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"Saved {split_name} dataset to: {path}")

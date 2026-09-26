from matplotlib.pylab import sample
import numpy as np

from pathlib import Path
from tqdm import tqdm
from collections import Counter
from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split


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
    radius=20.0,
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
# Process one official nuScenes prediction target
# ============================================================

def process_prediction_target(
    nusc,
    target_string,
    radius=20.0,
    past_steps=4,
    future_steps=12
):
    """
    Convert one official nuScenes prediction target into one
    processed trajectory-prediction example.

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

    return processed_sample, None


# ============================================================
# Build many processed prediction examples
# ============================================================

def build_prediction_dataset(
    nusc,
    target_strings,
    radius=20.0,
    past_steps=4,
    future_steps=12
):
    dataset = []
    skip_reasons = Counter()

    for target_string in tqdm(
        target_strings,
        desc="Processing prediction targets"
    ):

        # process_prediction_target now returns TWO values
        processed_sample, skip_reason = process_prediction_target(
            nusc,
            target_string,
            radius=radius,
            past_steps=past_steps,
            future_steps=future_steps
        )

        # Keep valid samples
        if processed_sample is not None:
            dataset.append(processed_sample)

        # Count skipped samples
        else:
            skip_reasons[skip_reason] += 1

    return dataset, skip_reasons

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    DATA_DIR = (
        Path.home()
        / "Desktop"
        / "gnn_trajectory"
        / "data"
    )

    VERSION = "v1.0-trainval"

    # Load full nuScenes train/validation metadata.
    nusc = NuScenes(
        version=VERSION,
        dataroot=str(DATA_DIR),
        verbose=True
    )

    # Official nuScenes prediction target splits.
    train_targets = get_prediction_challenge_split(
        "train",
        dataroot=str(DATA_DIR)
    )

    val_targets = get_prediction_challenge_split(
        "val",
        dataroot=str(DATA_DIR)
    )

    print("Train targets:", len(train_targets))
    print("Val targets:", len(val_targets))

    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT process all 41k targets yet.
    #
    # First verify the new pipeline on a tiny subset.
    # --------------------------------------------------------

    # test_targets = train_targets[:1000] # change after to train_targets for full dataset

    # Build the full train and validation datasets
    train_dataset, train_skip_reasons = build_prediction_dataset(
    nusc,
    train_targets,
    radius=20.0,
    past_steps=4,
    future_steps=12
    )

    val_dataset, val_skip_reasons = build_prediction_dataset(
        nusc,
        val_targets,
        radius=20.0,
        past_steps=4,
        future_steps=12
    )

    print("\nTrain processed:", len(train_dataset))
    print("Train skipped:", train_skip_reasons)

    print("\nVal processed:", len(val_dataset))
    print("Val skipped:", val_skip_reasons)

    # Validate processed datasets before saving
    for dataset_name, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
    ]:
        for sample in dataset:
            assert sample["target_past"].shape == (5, 2)
            assert sample["target_future"].shape == (12, 2)

            for neighbor_past in sample["neighbors_pasts"]:
                assert neighbor_past.shape == (5, 2)

        print(f"{dataset_name} sample shapes are valid.")


    # Save the processed datasets to disk
    import pickle
    
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(        
    parents=True,
    exist_ok=True
    )
    
    train_path = PROCESSED_DIR / "nuscenes_train_prediction_samples.pkl"
    val_path = PROCESSED_DIR / "nuscenes_val_prediction_samples.pkl"
    
    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(val_path, "wb") as f:
        pickle.dump(val_dataset, f)

    print(f"Saved train dataset to: {train_path}")
    print(f"Saved val dataset to: {val_path}")
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion


def load_json_records(path):
    with open(path, "r") as file:
        return json.load(file)


def load_metadata(data_dir, sample_token):
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "v1.0-trainval"
    sample_records = load_json_records(metadata_dir / "sample.json")

    if not any(record["token"] == sample_token for record in sample_records):
        metadata_dir = data_dir / "v1.0-mini"
        sample_records = load_json_records(metadata_dir / "sample.json")

    sample_record = next(
        record for record in sample_records if record["token"] == sample_token
    )
    sample_data = {
        record["token"]: record
        for record in load_json_records(metadata_dir / "sample_data.json")
    }
    calibrated_sensors = {
        record["token"]: record
        for record in load_json_records(metadata_dir / "calibrated_sensor.json")
    }
    ego_poses = {
        record["token"]: record
        for record in load_json_records(metadata_dir / "ego_pose.json")
    }
    sensors = {
        record["token"]: record
        for record in load_json_records(metadata_dir / "sensor.json")
    }

    return (
        data_dir,
        metadata_dir,
        sample_record,
        sample_data,
        calibrated_sensors,
        ego_poses,
        sensors,
    )


def local_to_global(points, target_position, target_yaw):
    """Convert target-relative xy points to global nuScenes xy coordinates."""
    points = np.asarray(points, dtype=np.float64)
    rotation = np.array(
        [
            [np.cos(target_yaw), -np.sin(target_yaw)],
            [np.sin(target_yaw), np.cos(target_yaw)],
        ]
    )
    return points @ rotation.T + np.asarray(target_position[:2])


def plot_sample_ego_frame(sample, data_dir):
    """Plot current target and neighbor positions in the ego frame."""
    (
        data_dir,
        metadata_dir,
        sample_record,
        sample_data,
        calibrated_sensors,
        ego_poses,
        sensors,
    ) = load_metadata(data_dir, sample["sample_token"])

    annotations = load_json_records(metadata_dir / "sample_annotation.json")
    target_annotation = next(
        annotation
        for annotation in annotations
        if annotation["token"] == sample["current_ann_token"]
    )
    target_position = np.asarray(target_annotation["translation"], dtype=float)
    target_yaw = Quaternion(target_annotation["rotation"]).yaw_pitch_roll[0]

    camera_record = next(
        record
        for record in sample_data.values()
        if record["sample_token"] == sample["sample_token"]
        and sensors[
            calibrated_sensors[record["calibrated_sensor_token"]]["sensor_token"]
        ]["channel"] == "CAM_FRONT"
    )
    ego_pose = ego_poses[camera_record["ego_pose_token"]]
    ego_rotation = Quaternion(ego_pose["rotation"]).rotation_matrix
    ego_translation = np.asarray(ego_pose["translation"], dtype=float)

    local_points = np.vstack(
        [
            np.zeros(2),
            [past[-1] for past in sample["neighbors_pasts"]],
        ]
    )
    global_points = local_to_global(local_points, target_position, target_yaw)
    global_points = np.column_stack([global_points, np.zeros(len(global_points))])
    ego_points = (global_points - ego_translation) @ ego_rotation

    fig, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(
        ego_points[1:, 0],
        ego_points[1:, 1],
        label="Neighbors",
    )
    axis.scatter(
        ego_points[0, 0],
        ego_points[0, 1],
        label="Target",
    )
    axis.scatter(0, 0, marker="s", s=90, color="black", label="Ego vehicle")

    axis.set_xlabel("Ego X: forward (m)")
    axis.set_ylabel("Ego Y: left (m)")
    axis.set_title("Current positions in ego-vehicle frame")
    axis.legend()
    axis.grid()
    axis.set_aspect("equal", adjustable="box")

    all_points = np.vstack([ego_points[:, :2], np.zeros((1, 2))])
    margin = 3.0
    axis.set_xlim(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin)
    axis.set_ylim(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin)
    fig.tight_layout()
    plt.show()
    return fig, axis


def project_global_points(points, ego_pose, calibrated_sensor):
    """Project global xyz points into camera pixels."""
    points = np.asarray(points, dtype=np.float64)
    points_xyz = np.column_stack((points[:, :2], np.zeros(len(points))))

    ego_rotation = Quaternion(ego_pose["rotation"]).rotation_matrix
    ego_translation = np.asarray(ego_pose["translation"], dtype=np.float64)
    points_ego = (points_xyz - ego_translation) @ ego_rotation

    sensor_rotation = Quaternion(calibrated_sensor["rotation"]).rotation_matrix
    sensor_translation = np.asarray(calibrated_sensor["translation"], dtype=np.float64)
    points_camera = (points_ego - sensor_translation) @ sensor_rotation

    intrinsic = np.asarray(calibrated_sensor["camera_intrinsic"], dtype=np.float64)
    pixels = points_camera @ intrinsic.T
    valid_depth = points_camera[:, 2] > 0
    pixels[valid_depth, :2] /= points_camera[valid_depth, 2, None]
    return pixels[:, :2], valid_depth


def plot_sample_camera_projection(sample, data_dir, channel="CAM_FRONT"):
    """Overlay a processed sample's local trajectories on a nuScenes camera image.

    The past and future trajectories are expressed in the target's current local
    frame, so they are transformed using the target's current global pose. This
    is a geometric visualization, not a time-synchronized reprojection.
    """
    (
        data_dir,
        metadata_dir,
        sample_record,
        sample_data,
        calibrated_sensors,
        ego_poses,
        sensors,
    ) = load_metadata(data_dir, sample["sample_token"])

    camera_record = None
    for record in sample_data.values():
        if record["sample_token"] != sample["sample_token"]:
            continue
        calibrated_sensor = calibrated_sensors[record["calibrated_sensor_token"]]
        sensor = sensors[calibrated_sensor["sensor_token"]]
        if sensor["channel"] == channel:
            camera_record = record
            break

    if camera_record is None:
        raise ValueError(f"{channel} is not available for this sample.")

    calibrated_sensor = calibrated_sensors[camera_record["calibrated_sensor_token"]]
    sensor = sensors[calibrated_sensor["sensor_token"]]
    if not sensor["channel"].startswith("CAM_"):
        raise ValueError(f"{channel} is not a camera channel.")

    image_path = data_dir / camera_record["filename"]
    if not image_path.exists() or image_path.stat().st_size == 0:
        raise FileNotFoundError(f"Camera image is unavailable: {image_path}")

    target_annotation = next(
        annotation
        for annotation in load_json_records(metadata_dir / "sample_annotation.json")
        if annotation["token"] == sample["current_ann_token"]
    )
    target_position = np.asarray(target_annotation["translation"], dtype=np.float64)
    target_yaw = Quaternion(target_annotation["rotation"]).yaw_pitch_roll[0]

    trajectories = [("Target past", sample["target_past"], "tab:blue")]
    trajectories.append(("Target future", sample["target_future"], "tab:green"))
    for index, past in enumerate(sample["neighbors_pasts"]):
        trajectories.append((f"Neighbor {index}", past, "tab:orange"))

    ego_pose = ego_poses[camera_record["ego_pose_token"]]
    fig, axis = plt.subplots(figsize=(12, 10))
    with Image.open(image_path) as image:
        axis.imshow(image.copy())

    image_width, image_height = image.size
    for label, local_points, color in trajectories:
        global_points = local_to_global(local_points, target_position, target_yaw)
        pixels, valid = project_global_points(
            global_points, ego_pose, calibrated_sensor
        )
        valid &= (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < image_width)
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < image_height)
        )
        if valid.any():
            axis.plot(
                pixels[valid, 0],
                pixels[valid, 1],
                marker="o",
                markersize=4,
                linewidth=1.5,
                color=color,
                label=label,
            )

    axis.set_title(f"{channel} projection for sample {sample['sample_token']}")
    axis.axis("off")
    axis.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    plt.show()
    return fig, axis

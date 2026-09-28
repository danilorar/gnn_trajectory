import json
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image


def load_json_records(path):
    with open(path, "r") as f:
        return json.load(f)


def is_readable_image(image_path):
    if not image_path.exists() or image_path.stat().st_size == 0:
        return False
    try:
        with Image.open(image_path) as image:
            image.verify()
        return True
    except (OSError, Image.UnidentifiedImageError):
        return False


def view_sample_images(sample, data_dir):
    """Display every camera image associated with a processed sample."""
    sample_token = sample["sample_token"]
    data_dir = Path(data_dir)
    metadata_dir = data_dir / "v1.0-trainval"

    # Use the mini metadata if the sample is not present in trainval.
    sample_records = load_json_records(metadata_dir / "sample.json")
    if not any(record["token"] == sample_token for record in sample_records):
        metadata_dir = data_dir / "v1.0-mini"
        sample_records = load_json_records(metadata_dir / "sample.json")

    sample_record = next(record for record in sample_records if record["token"] == sample_token)
    calibrated_sensors = {
        record["token"]: record
        for record in load_json_records(metadata_dir / "calibrated_sensor.json")
    }
    sensors = {
        record["token"]: record
        for record in load_json_records(metadata_dir / "sensor.json")
    }

    camera_files = {}
    for record in load_json_records(metadata_dir / "sample_data.json"):
        if record["sample_token"] != sample_token:
            continue
        calibrated_sensor = calibrated_sensors[record["calibrated_sensor_token"]]
        sensor = sensors[calibrated_sensor["sensor_token"]]
        if sensor["channel"].startswith("CAM_"):
            camera_files[sensor["channel"]] = data_dir / record["filename"]

    print("Metadata split:", metadata_dir.name)
    print("Scene token:", sample_record["scene_token"])
    print("Camera images for this sample:")
    for channel, image_path in sorted(camera_files.items()):
        if not image_path.exists():
            status = "missing locally"
        elif image_path.stat().st_size == 0:
            status = "empty file"
        elif is_readable_image(image_path):
            status = "valid image"
        else:
            status = "unreadable file"
        print(f"{channel}: {image_path} ({status})")

    # Plot every identified camera in a fixed 2-by-3 layout.
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))

    axes = axes.ravel()
    for axis, channel in zip(axes, sorted(camera_files)):
        image_path = camera_files[channel]
        axis.set_title(channel)
        axis.axis("off")
        if is_readable_image(image_path):
            with Image.open(image_path) as image:
                axis.imshow(image.copy())
        else:
            axis.text(
                0.5,
                0.5,
                "Image unavailable locally",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    for axis in axes[len(camera_files):]:
        axis.axis("off")

    fig.suptitle(f"Camera images for sample {sample_token}", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    raise SystemExit("Import view_sample_images(sample, data_dir) from a notebook or script.")
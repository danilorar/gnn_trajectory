from pathlib import Path
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

import numpy as np
import pickle

from neighbors import get_neighbors_for_sample,  world_to_local


def extract_target_samples(
    nusc,
    instance_token,
    past_steps=4,      # 4 intervals = 2 s
    future_steps=12,   # 12 intervals = 6 s
    min_displacement=5.0
):
    instance = nusc.get("instance", instance_token)

    ann_token = instance["first_annotation_token"]

    anns = []

    while ann_token != "":
        ann = nusc.get("sample_annotation", ann_token)
        anns.append(ann)
        ann_token = ann["next"]

    # Need enough points at all
    min_length = past_steps + 1 + future_steps

    if len(anns) < min_length:
        return []

    positions = np.array([
        ann["translation"][:2]
        for ann in anns
    ])

    # Simple moving-target filter
    total_displacement = np.linalg.norm(
        positions[-1] - positions[0]
    )

    if total_displacement < min_displacement:
        return []

    samples = []

    # Example:
    # past_steps = 4
    # future_steps = 12
    #
    # valid current index:
    # 4 ... len(anns)-13
    for current_idx in range(
        past_steps,
        len(anns) - future_steps
    ):

        current_ann = anns[current_idx]

        past_world = positions[
            current_idx - past_steps : current_idx + 1
        ]

        future_world = positions[
            current_idx + 1 : current_idx + 1 + future_steps
        ]

        origin = positions[current_idx]

        q = Quaternion(current_ann["rotation"])
        yaw = q.yaw_pitch_roll[0]

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

        samples.append({
            "past": past_local,
            "future": future_local,
            "instance_token": instance_token,
            "current_ann_token": current_ann["token"],
            "sample_token": current_ann["sample_token"],
            "yaw": yaw
        })

    return samples



def build_processed_dataset(nusc, radius=20.0, past_steps=4, future_steps=12, min_displacement=5.0):
    
    dataset = []
    
    # Go scene by scene
    for scene in tqdm(nusc.scene, desc="Processing scenes"): 
        
        # --------------------------------------------------
        # 1. Find all unique vehicle instances in this scene 
        # --------------------------------------------------
        
        vehicle_instances = set() # Store unique vehicle instance tokens in this scene
        print(f"Processing scene: {scene['name']}")
        this_scene = scene['first_sample_token']

        
        while this_scene != "": 
            
            # Get a dictionary with info about the sample
            sample_info = nusc.get('sample', this_scene)
            
            for ann_token in sample_info["anns"]: 
                ann = nusc.get('sample_annotation', ann_token) # get the info for this annotation token (i.e. the vehicle, pedestrian, etc.)
                
                
                # A vehicle has a instance token, so for every vehicle we find, we add its instance token to the set of unique vehicle instances in this scene 
                if ann["category_name"].startswith("vehicle."): 
                    vehicle_instances.add(ann["instance_token"])    
                    
            
            this_scene = sample_info["next"] #  move to next sample in the scene
            
            
        # ----------------------------------------------------------------------
        # 2. For each unique vehicle instance, extract its samples and neighbors
        # ----------------------------------------------------------------------
        
        for current_instance_token in vehicle_instances: 

            # Extract all samples from this scene for the current vehicle instance 
            target_samples = extract_target_samples(nusc, current_instance_token, past_steps, future_steps, min_displacement)
            
            # Find neighbors for each sample and add to dataset
            for one_target_sample in target_samples:
                
                neighbors = get_neighbors_for_sample(nusc, one_target_sample, radius, past_steps)
                
                # Store information about the processed sample
                processed_sample = {
                    
                    # Target vehicle information
                    "target_past": one_target_sample["past"],
                    "target_future": one_target_sample["future"],
                    
                    # Neighbor information
                    "neighbors_pasts": [neighbor["past"] for neighbor in neighbors],
                    "neighbors_categories": [neighbor["category"] for neighbor in neighbors],
                    "neighbors_distances": [neighbor["distance"] for neighbor in neighbors],
                    
                    # Scene and instance information
                    "scene_token": scene["token"],
                    "sample_token": one_target_sample["sample_token"], # current sample containing ego and neighbors
                    "target_instance_token": current_instance_token,
                    "current_ann_token": one_target_sample["current_ann_token"],
                }
                
                dataset.append(processed_sample)
                
    return dataset

if __name__ == "__main__":
    
    # Define the path to the NuScenes dataset
    DATA_DIR = Path.home() / "Desktop" / "gnn_trajectory" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    PROCESSED_DIR = DATA_DIR / "processed"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {DATA_DIR}")
    
    VERSION = "v1.0-mini"
    
    # Process 10 mini scenes 
    nusc = NuScenes(version=VERSION, dataroot=str(DATA_DIR), verbose=True)

    mini_dataset = build_processed_dataset(nusc,radius=20.0, past_steps=4, future_steps=12, min_displacement=5.0)
    
    print("Total prediction samples:", len(mini_dataset))
    
    # ---------------------------
    # 3. Save data and sanity check
    # ---------------------------
    
    print(f"Processed directory: {PROCESSED_DIR}")
    # Save processed data
    output_path = (PROCESSED_DIR / "nuscenes_mini_trajectory_samples.pkl")

    with open(output_path, "wb") as f:
        pickle.dump(mini_dataset, f)

    print(f"Saved dataset to: {output_path}")

    # Sanity
    for sample in mini_dataset:

        assert sample["target_past"].shape == (5, 2)
        assert sample["target_future"].shape == (12, 2)

        for neighbor_past in sample["neighbors_pasts"]:
            assert neighbor_past.shape == (5, 2)

    print("All sample shapes valid.")
            
        
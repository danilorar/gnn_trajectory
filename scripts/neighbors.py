import numpy as np
from pyquaternion import Quaternion

def world_to_local(points, origin, yaw): 
    """ Transform 2D world coordinates into the target local frame."""
    
    points = np.array(points)
    centered_points = points - origin
    
    # Rotation matrix for yaw angle
    c = np.cos(-yaw)
    s = np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    
    return centered_points @ R.T  # Apply rotation


# function to get past positions
def get_past_positions(nusc, ann, past_steps=4): 
    """
    Return up to past positions + 1 position for one agent, 
    ending at the current annotation
    """
    
    positions = [ann["translation"][:2]]  # Start with the current position
    current_ann = ann
    
    # Walk backwards to get past positions
    for _ in range(past_steps):
        if current_ann["prev"] == "":
            break  # No more past annotations
        
        current_ann = nusc.get("sample_annotation", current_ann["prev"])
        positions.append(current_ann["translation"][:2])
        
    # We walked backwards, so reverse the list to have the oldest position first
    positions = positions[::-1]
    
    return np.array(positions).astype(np.float32)  # Convert to numpy array of type float32


# Find the actual neighbor function
def get_neighbors_for_sample(nusc, sample_info, radius=20.0, past_steps=4):
    """
    Given a sample_info dictionary (which contains the current annotation and its sample token),
    find all neighboring agents within a certain radius and return their past positions.
    """
    
    current_sample = nusc.get("sample", sample_info["sample_token"])
    target_ann = nusc.get("sample_annotation", sample_info["current_ann_token"])
    
    target_position = np.array(target_ann["translation"][:2])
    target_yaw = sample_info["yaw"]
    
    neighbors = []
    
    for ann_token in current_sample["anns"]: 
        
        ann = nusc.get("sample_annotation", ann_token)
        
        # Skip the target agent itself
        if ann["token"] == target_ann["token"]:
            continue
        
        # For now, only consider road users (vehicles, pedestrians, etc.)
        category = ann["category_name"]
        
        if not (
            category.startswith("vehicle.") or
            category.startswith("pedestrian.")
        ): 
            continue
        
        # Get the neighbor's position
        neighbor_position = np.array(ann["translation"][:2])
        
        # Compute distance to the target agent
        distance = np.linalg.norm(neighbor_position - target_position)
        
        if distance > radius:
            continue # Skip if outside the radius
        
        neighbor_past_world = get_past_positions(nusc, ann, past_steps=past_steps)
        
        # Require at least 2 past positions to consider it a valid neighbor
        if len(neighbor_past_world) < past_steps + 1:
            continue
        
        neighbor_past_local = world_to_local(
            neighbor_past_world,
            target_position,
            target_yaw
        )
        
        neighbors.append({
        "instance_token": ann["instance_token"],
        "category": category,
        "distance": distance,
        "past": neighbor_past_local
    })
            
    return neighbors
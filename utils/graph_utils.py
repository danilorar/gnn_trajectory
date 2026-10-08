import numpy as np
import torch
from torch_geometric.data import Data


def general_category(category):

    if category == "vehicle.car":
        return "car"

    elif (
        category.startswith("vehicle.truck")
        or category.startswith("vehicle.bus")
        or category.startswith("vehicle.trailer")
    ):
        return "large_vehicle"

    elif (
        category.startswith("vehicle.bicycle")
        or category.startswith("vehicle.motorcycle")
    ):
        return "two_wheeler"

    elif category.startswith("human.pedestrian"):
        return "pedestrian"

    elif category.startswith("vehicle."):
        return "other_vehicle"

    else:
        return "other"

# ========================================================
    
def category_one_hot(category):
    general = general_category(category)

    category_to_idx = {
        "car": 0,
        "large_vehicle": 1,
        "two_wheeler": 2,
        "pedestrian": 3,
        "other_vehicle": 4
    }

    one_hot = np.zeros(5, dtype=np.float32)
    one_hot[category_to_idx[general]] = 1.0

    return one_hot    
      
# ========================================================    

def sample_to_graph(sample):
    """Convert a single sample from the dataset into a PyG Data object."""

    # ----------------------
    # BUILD NODE FEATURES
    # ----------------------

    node_features = []

    # Target = node 0
    target_traj = sample["target_past"].flatten()
    target_type = category_one_hot(sample["target_category"])

    target_features = np.concatenate((target_traj, target_type))

    node_features.append(target_features)

    # Neighbors = nodes 1, 2, ...
    for neighbor_past, category in zip(
        sample["neighbors_pasts"],
        sample["neighbors_categories"]):
        
        neighbor_traj = neighbor_past.flatten()
        neighbor_type = category_one_hot(category)
        features = np.concatenate((neighbor_traj, neighbor_type))

        node_features.append(features)

    # Shape: [num_nodes, 15]
    # 10 trajectory values + 5 category one-hot values
    x = np.array(node_features,dtype=np.float32)
    
    # ----------------------
    # TARGET NODE MASK
    # ----------------------

    target_mask = np.zeros(len(x), dtype=bool)
    target_mask[0] = True  # Mark the target node as True

    # ----------------------
    # CURRENT POSITIONS
    # ----------------------

    current_positions = x[:, 8:10]


    # ----------------------
    # BUILD EDGES
    # ----------------------

    num_nodes = len(current_positions)

    # Fully connect every node pair; distance is passed via edge_attr
    # below instead of used to prune edges, so attention can weigh
    # relevance itself rather than a fixed cutoff.
    edges = [
        [i, j]
        for i in range(num_nodes)
        for j in range(num_nodes)
        if i != j
    ]

    # Handle graphs with no edges
    if len(edges) == 0:
        # If there are no edges, create an empty edge index
        edge_index = np.empty(
            (2, 0),
            dtype=np.int64
        )
    else:
        edge_index = np.array(
            edges,
            dtype=np.int64
        ).T


    # ----------------------
    # BUILD EDGE ATTRIBUTES
    # ----------------------

    edge_attributes = []

    for source, target in edge_index.T:

        source_pos = current_positions[source]
        target_pos = current_positions[target]

        delta = target_pos - source_pos

        delta_x = delta[0]
        delta_y = delta[1]
        distance = np.linalg.norm(delta)

        edge_attributes.append([delta_x, delta_y, distance])

    if len(edge_attributes) == 0:
        edge_attr = np.empty((0, 3),dtype=np.float32)
    else:
        edge_attr = np.array(edge_attributes,dtype=np.float32)


    # ----------------------
    # TARGET FUTURE
    # ----------------------

    y = np.array(
        sample["target_future"],
        dtype=np.float32
    )


    # ----------------------
    # PyG GRAPH
    # ----------------------

    graph = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        target_mask=torch.tensor(target_mask, dtype=torch.bool)
    )

    return graph

# ========================================================

def density_bucket(sample):
    """Assign a sample to an interaction-density bucket."""

    num_neighbors = len(sample["neighbors_pasts"])

    if num_neighbors <= 1:
        return "sparse"

    elif num_neighbors <= 4:
        return "medium"

    else:
        return "dense"

# ========================================================

def remove_neighbors(graph):
    """
    Return a graph containing only the prediction target.
    Neighbor nodes and all graph edges are removed.
    """
    
    target_idx = torch.where(graph.target_mask)[0][0]

    x_target = graph.x[target_idx].unsqueeze(0)

    target_mask = torch.tensor([True], dtype=torch.bool)

    edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = torch.empty(
        (0, graph.edge_attr.shape[1]),
        dtype=graph.edge_attr.dtype
    )

    return Data(
        x=x_target,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=graph.y,
        target_mask=target_mask
    )


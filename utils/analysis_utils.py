import numpy as np
import torch

from torch_geometric.data import Batch

from utils.graph_utils import (
    density_bucket,
    remove_neighbors,
)

def evaluate_by_density(model, dataset, graphs, device=None):
    """
    Evaluate ADE and FDE separately for sparse, medium,
    and dense interaction scenes.
    """

    if device is None:
        device = next(model.parameters()).device

    results = {
        "sparse": {"ade": [], "fde": []},
        "medium": {"ade": [], "fde": []},
        "dense": {"ade": [], "fde": []},
    }

    model.eval()

    with torch.no_grad():

        for sample, graph in zip(dataset, graphs):
            bucket = density_bucket(sample)

            graph = graph.to(device)

            batch = Batch.from_data_list([graph])

            pred = model(batch)

            target = graph.y.view(1, 12, 2)

            errors = torch.norm(pred - target, dim=2)

            ade = errors.mean().item()
            fde = errors[0, -1].item()

            results[bucket]["ade"].append(ade)
            results[bucket]["fde"].append(fde)

    summary = {}

    for bucket, values in results.items():

        summary[bucket] = {
            "ade": np.mean(values["ade"]),
            "fde": np.mean(values["fde"]),
            "n": len(values["ade"]),
        }

    return summary

# ========================================================    

def evaluate_neighbor_without_neighbors(model, graphs,device=None):
    """
    Evaluate the model after removing all neighbor nodes.
    """
    
    if device is None:
        device = next(model.parameters()).device

    ade_list = []
    fde_list = []

    model.eval()

    with torch.no_grad():

        for graph in graphs:

            ablated_graph = remove_neighbors(graph).to(device)

            batch = Batch.from_data_list([ablated_graph])

            pred = model(batch)

            target = ablated_graph.y.view(1, 12, 2)

            errors = torch.norm(pred - target, dim=2)

            ade_list.append(errors.mean().item())

            fde_list.append(errors[0, -1].item())

    return (
        np.mean(ade_list),
        np.mean(fde_list)
    )
    
# ========================================================    
    
def neighbor_effect_by_count(
    model,
    dataset,
    graphs,
    device=None
):
    """
    Compare normal-model ADE against ablated-model ADE
    for every sample.

    Returns:
        neighbor_counts
        delta_ades

    delta_ADE = normal ADE - ablated ADE

    Negative -> neighbors help
    Positive -> neighbors hurt
    """

    if device is None:
        device = next(model.parameters()).device

    neighbor_counts = []
    delta_ades = []

    model.eval()

    with torch.no_grad():

        for sample, graph in zip(dataset, graphs):

            num_neighbors = len(
                sample["neighbors_pasts"]
            )

            # Normal graph
            graph_device = graph.to(device)

            batch = Batch.from_data_list(
                [graph_device]
            )

            pred = model(batch)

            target = graph_device.y.view(
                1,
                12,
                2
            )

            normal_errors = torch.norm(
                pred - target,
                dim=2
            )

            normal_ade = (
                normal_errors.mean().item()
            )

            # Ablated graph
            ablated_graph = remove_neighbors(
                graph
            ).to(device)

            ablated_batch = Batch.from_data_list(
                [ablated_graph]
            )

            ablated_pred = model(
                ablated_batch
            )

            ablated_errors = torch.norm(
                ablated_pred - target,
                dim=2
            )

            ablated_ade = (
                ablated_errors.mean().item()
            )

            delta_ade = (
                normal_ade - ablated_ade
            )

            neighbor_counts.append(
                num_neighbors
            )

            delta_ades.append(
                delta_ade
            )

    return neighbor_counts, delta_ades

# ======================================================== 

def bin_neighbor_effect(
    neighbor_counts,
    delta_ades
):
    """
    Group delta ADE values by neighbor-count ranges.

    delta_ADE = normal ADE - ablated ADE

    Negative -> neighbors help
    Positive -> neighbors hurt
    """

    bins = {
        "0-4": [],
        "5-9": [],
        "10-19": [],
        "20-29": [],
        "30-39": [],
        "40+": [],
    }

    for n_neighbors, delta_ade in zip(
        neighbor_counts,
        delta_ades
    ):

        if n_neighbors <= 4:
            bins["0-4"].append(delta_ade)

        elif n_neighbors <= 9:
            bins["5-9"].append(delta_ade)

        elif n_neighbors <= 19:
            bins["10-19"].append(delta_ade)

        elif n_neighbors <= 29:
            bins["20-29"].append(delta_ade)

        elif n_neighbors <= 39:
            bins["30-39"].append(delta_ade)

        else:
            bins["40+"].append(delta_ade)

    summary = {}

    for bin_name, values in bins.items():

        summary[bin_name] = {
            "mean_delta_ade": np.mean(values),
            "n": len(values),
        }

    return summary

# ========================================================

def neighbor_effect_by_distance(
    model,
    dataset,
    graphs,
    device=None
):
    """
    Compare normal-model ADE against ablated-model ADE for every
    sample that has at least one neighbor, paired with the distance
    to that sample's nearest neighbor.

    Returns:
        nearest_distances
        delta_ades

    delta_ADE = normal ADE - ablated ADE

    Negative -> neighbors help
    Positive -> neighbors hurt
    """

    if device is None:
        device = next(model.parameters()).device

    nearest_distances = []
    delta_ades = []

    model.eval()

    with torch.no_grad():

        for sample, graph in zip(dataset, graphs):

            if len(sample["neighbors_distances"]) == 0:
                continue

            nearest_distance = min(sample["neighbors_distances"])

            # Normal graph
            graph_device = graph.to(device)

            batch = Batch.from_data_list(
                [graph_device]
            )

            pred = model(batch)

            target = graph_device.y.view(
                1,
                12,
                2
            )

            normal_errors = torch.norm(
                pred - target,
                dim=2
            )

            normal_ade = (
                normal_errors.mean().item()
            )

            # Ablated graph
            ablated_graph = remove_neighbors(
                graph
            ).to(device)

            ablated_batch = Batch.from_data_list(
                [ablated_graph]
            )

            ablated_pred = model(
                ablated_batch
            )

            ablated_errors = torch.norm(
                ablated_pred - target,
                dim=2
            )

            ablated_ade = (
                ablated_errors.mean().item()
            )

            delta_ade = (
                normal_ade - ablated_ade
            )

            nearest_distances.append(
                nearest_distance
            )

            delta_ades.append(
                delta_ade
            )

    return nearest_distances, delta_ades

# ========================================================

def bin_neighbor_effect_by_distance(
    nearest_distances,
    delta_ades,
    bin_edges=(10, 20, 30, 40, 50)
):
    """
    Group delta ADE values by nearest-neighbor-distance ranges.

    delta_ADE = normal ADE - ablated ADE

    Negative -> neighbors help
    Positive -> neighbors hurt
    """

    nearest_distances = np.asarray(nearest_distances)
    delta_ades = np.asarray(delta_ades)

    edges = [0] + list(bin_edges) + [np.inf]

    summary = {}

    for lo, hi in zip(edges[:-1], edges[1:]):

        label = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f"{lo:g}+"

        mask = (nearest_distances >= lo) & (nearest_distances < hi)
        values = delta_ades[mask]

        summary[label] = {
            "mean_delta_ade": float(values.mean()) if len(values) else None,
            "n": int(mask.sum()),
        }

    return summary
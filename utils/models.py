import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool


class TargetHistoryMLP(nn.Module):
    """Target history/category baseline with no interaction modeling."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

    def forward(self, batch):

        # Keep only the target node.
        # Shape: [batch_size, 15]
        x_target = batch.x[batch.target_mask]

        h = self.encoder(x_target)

        # Predict 12 future xy positions
        pred = self.decoder(h)

        return pred.view(-1, 12, 2)


class GraphNetworkAttention(nn.Module):
    """Two-layer GATv2 attention GNN for single-mode trajectory prediction."""

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU()
        )

        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        self.gat2 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 24)
        )

    def forward(self, batch):

        # Node encoding
        h0 = self.encoder(batch.x)

        # Attention layers
        h1 = self.gat1(h0, batch.edge_index, batch.edge_attr)
        h1 = torch.relu(h1)

        h2 = self.gat2(h1, batch.edge_index, batch.edge_attr)
        h2 = torch.relu(h2)

        # Keep only the prediction targets
        h_target = h2[batch.target_mask]

        # Decode 12 future xy positions
        pred = self.decoder(h_target)
        pred = pred.view(-1, 12, 2)  # Reshape to (batch_size, 12, 2)

        return pred


class GraphNetworkAttentionMultimodal(nn.Module):
    """Predict multiple 12-step xy trajectories for each target agent."""

    def __init__(self, num_modes=6):
        super().__init__()
        self.num_modes = num_modes

        self.encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU()
        )

        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )
        self.gat2 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * 12 * 2)
        )

    def forward(self, batch):
        h = self.encoder(batch.x)
        h = torch.relu(self.gat1(h, batch.edge_index, batch.edge_attr))
        h = torch.relu(self.gat2(h, batch.edge_index, batch.edge_attr))
        h_target = h[batch.target_mask]
        pred = self.decoder(h_target)
        return pred.view(-1, self.num_modes, 12, 2)


class MapEncoder(nn.Module):
    """Encodes lane-centerline map context into a fixed-size embedding for each target agent."""

    def __init__(self, hidden_dim=32):
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(
        self,
        map_points,
        map_lane_index,
        map_lane_batch,
        batch_size
    ):
        point_features = self.point_mlp(map_points)

        lane_embeddings = global_max_pool(
            point_features,
            map_lane_index,
            # Keep a zero embedding for lanes with no sampled points.
            size=map_lane_batch.numel(),
        )

        map_embeddings = global_max_pool(
            lane_embeddings,
            map_lane_batch,
            size=batch_size
        )

        return map_embeddings


class GraphNetworkAttentionMultimodalWithMap(nn.Module):
    """Multimodal GATv2 trajectory predictor with lane-centerline map context"""

    def __init__(self, num_modes=6, map_dim=32):
        super().__init__()

        self.num_modes = num_modes

        self.encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU()
        )

        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        self.gat2 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        self.map_encoder = MapEncoder(
            hidden_dim=map_dim
        )

        # 64-D target embedding + 32-D map embedding = 96-D input to decoder
        self.decoder = nn.Sequential(
            nn.Linear(64 + map_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * 12 * 2)
        )

    def forward(self, batch):

        h = self.encoder(batch.x)

        h = torch.relu(
            self.gat1(h, batch.edge_index, batch.edge_attr)
        )

        h = torch.relu(
            self.gat2(h, batch.edge_index, batch.edge_attr)
        )

        # One 64-D embedding for each prediction target
        h_target = h[batch.target_mask]

        num_map_lanes = batch.num_map_lanes

        map_lane_batch = torch.repeat_interleave(
            torch.arange(
                len(num_map_lanes),
                device=batch.map_points.device
            ),
            num_map_lanes
        )

        batch_size = h_target.shape[0]

        map_embedding = self.map_encoder(
            batch.map_points,
            batch.map_lane_index,
            map_lane_batch,
            batch_size
        )

        combined = torch.cat(
            [h_target, map_embedding],
            dim=1
        )

        pred = self.decoder(combined)

        # Return shape: (batch_size, num_modes, 12, 2)
        return pred.view(
            -1,
            self.num_modes,
            12,
            2
        )


class GRUFutureDecoder(nn.Module):
    """Autoregressive displacement rollout with a context/mode-conditioned h0.

    A single GRUCell and linear head are shared across all modes and steps.
    The only recurrent input is the previous predicted displacement (zero at
    step one). Positions are cumulative displacements from the current origin.
    Ground-truth futures are never inputs to this decoder.
    """

    def __init__(
        self,
        context_dim=96,
        num_modes=6,
        mode_dim=16,
        hidden_dim=64,
        steps=12,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.steps = steps
        self.mode_embedding = nn.Embedding(num_modes, mode_dim)
        self.initial_hidden = nn.Linear(context_dim + mode_dim, hidden_dim)
        self.gru = nn.GRUCell(input_size=2, hidden_size=hidden_dim)
        self.displacement_head = nn.Linear(hidden_dim, 2)

    def forward(self, context):
        batch_size = context.shape[0]
        modes = self.mode_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        contexts = context.unsqueeze(1).expand(-1, self.num_modes, -1)
        hidden = torch.tanh(self.initial_hidden(torch.cat((contexts, modes), dim=-1)))
        hidden = hidden.reshape(batch_size * self.num_modes, -1)
        previous_delta = context.new_zeros(batch_size * self.num_modes, 2)
        deltas = []
        for _ in range(self.steps):
            hidden = self.gru(previous_delta, hidden)
            previous_delta = self.displacement_head(hidden)
            deltas.append(previous_delta.reshape(batch_size, self.num_modes, 2))
        return torch.stack(deltas, dim=2).cumsum(dim=2)


class GraphNetworkAttentionMultimodalWithMapGRU(GraphNetworkAttentionMultimodalWithMap):
    """Final map-aware model with the tested recurrent displacement decoder."""

    def __init__(self, num_modes=6, map_dim=32):
        super().__init__(num_modes=num_modes, map_dim=map_dim)
        self.decoder = GRUFutureDecoder(context_dim=64 + map_dim, num_modes=num_modes)

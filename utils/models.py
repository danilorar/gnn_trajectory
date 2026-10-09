import torch
import torch.nn as nn

from torch_geometric.nn import GATv2Conv


class GraphNetworkAttention(nn.Module):
    """Two-layer GATv2 attention GNN for single-mode trajectory prediction."""

    def __init__(self):
        super().__init__()

        # -------------------
        # ENCODER
        # -------------------
        self.encoder = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU()
        )

        # -------------------
        # ATTENTION LAYER 1
        # -------------------
        self.gat1 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        # -------------------
        # ATTENTION LAYER 2
        # -------------------
        self.gat2 = GATv2Conv(
            in_channels=64,
            out_channels=64,
            heads=1,
            concat=False,
            edge_dim=3
        )

        # -------------------
        # DECODER
        # -------------------
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

        # Keep only the prediction targets for the ego agent
        h_target = h2[batch.target_mask]

        # Decode 12 future xy positions
        pred = self.decoder(h_target)
        pred = pred.view(-1, 12, 2)  # Reshape to (batch_size, 12, 2)

        return pred

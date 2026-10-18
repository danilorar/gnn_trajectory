import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_max_pool


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
    """Encodes lane-map context into a fixed-size embedding for each target agent."""
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
    """Multimodal GATv2 trajectory predictor with lane-map context"""
    
    def __init__(self, num_modes=6, map_dim=32): 
        super().__init__()
        
        
        self.num_modes = num_modes
                
        # --------------------
        # AGENT ENCODER 
        # --------------------
        self.encoder = nn.Sequential(
            nn.Linear(15, 64),  
            nn.ReLU()
        )
        
        # --------------------
        # ATTENTION LAYERS 
        # --------------------
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
        
        # --------------------
        # MAP ENCODER
        # --------------------
        self.map_encoder = MapEncoder(
            hidden_dim=map_dim
        )
        
        # --------------------
        # DECODER
        # --------------------
        # 64-D target embedding + 32-D map embedding = 96-D input to decoder
        self.decoder = nn.Sequential(
            nn.Linear(64 + map_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * 12 * 2)
        )
    
    def forward(self, batch):
        
        # ==========================================
        # 1. AGENT / INTERACTION ENCODING
        # ==========================================
        
        h = self.encoder(batch.x)
        
        h = torch.relu(
            self.gat1(h, batch.edge_index, batch.edge_attr)
        )
        
        h = torch.relu(
            self.gat2(h, batch.edge_index, batch.edge_attr)
        )
        
        # One 64-D embedding for each prediction target 
        h_target = h[batch.target_mask]
        
        # ==========================================
        # 2. BUILD LANE -> SAMPLE ASSIGNMENT
        # ==========================================
        
        num_map_lanes = batch.num_map_lanes 
        
        map_lane_batch = torch.repeat_interleave(
            torch.arange(
                len(num_map_lanes),
                device = batch.map_points.device
            ),
            num_map_lanes
        )
        
        # ==========================================
        # 3. MAP ENCODING
        # ==========================================
        
        batch_size = h_target.shape[0]        
        
        map_embedding = self.map_encoder(
            batch.map_points,
            batch.map_lane_index,
            map_lane_batch,
            batch_size
        )
        
        # ==========================================
        # 4. CONCATENATE AGENT + MAP EMBEDDINGS
        # ==========================================

        combined = torch.cat(
            [h_target, map_embedding],
            dim=1
        )
        
        # ==========================================
        # 5. MULTIMODAL TRAJECTORY DECODER
        # ==========================================
        
        pred = self.decoder(combined)
        
        
        # Return shape: (batch_size, num_modes, 12, 2) 
        return pred.view(
            -1,
            self.num_modes,
            12,
            2
        )

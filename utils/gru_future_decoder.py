"""Experimental future decoder; the baseline map/GAT model is unchanged."""

import torch
from torch import nn

from utils.models import GraphNetworkAttentionMultimodalWithMap


class GRUFutureDecoder(nn.Module):
    """Autoregressive displacement rollout with a context/mode-conditioned h0.

    A single GRUCell and linear head are shared across all modes and steps.
    The only recurrent input is the previous predicted displacement (zero at
    step one). Positions are cumulative displacements from the current origin.
    Ground-truth futures are never inputs to this decoder.
    """

    def __init__(self, context_dim=96, num_modes=6, mode_dim=16, hidden_dim=64, steps=12):
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
    """Keep the inherited encoders, fused context and forward path; replace only decoder."""

    def __init__(self, num_modes=6, map_dim=32):
        super().__init__(num_modes=num_modes, map_dim=map_dim)
        self.decoder = GRUFutureDecoder(context_dim=64 + map_dim, num_modes=num_modes)

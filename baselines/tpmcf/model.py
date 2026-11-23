"""TPMCF spatial encoders and temporal prediction network."""
import torch
from torch import nn
from torch.nn import functional as F

class CosineAutoencoder(nn.Module):
    """Released-code stacked AE: user 120/80/50, service 1024/256/50."""

    def __init__(self, input_dim, hidden_a, hidden_b, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_a), nn.Tanh(), nn.Dropout(0.4),
            nn.Linear(hidden_a, hidden_b), nn.Tanh(), nn.Dropout(0.4),
            nn.Linear(hidden_b, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_b), nn.Tanh(), nn.Dropout(0.4),
            nn.Linear(hidden_b, hidden_a), nn.Tanh(),
            nn.Linear(hidden_a, input_dim),
        )

    def forward(self, values):
        return self.decoder(self.encoder(values))


class GraphConvolutionalMF(nn.Module):
    """Two GConv layers, concatenate outputs, then pairwise dot reconstruction."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.first = nn.Linear(input_dim, hidden_dim)
        self.second = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features, adjacency):
        first = F.relu(self.first(torch.sparse.mm(adjacency, features)))
        second = F.relu(self.second(torch.sparse.mm(adjacency, first)))
        return torch.cat((first, second), dim=1)


class PredictiveBlock(nn.Module):
    """MHA + residual/LN + the source-release Conv1D pair + residual/LN."""

    def __init__(self, time_steps, model_dim, conv_filters, dropout):
        super().__init__()
        self.norm_attention = nn.LayerNorm(model_dim)
        self.conv_first = nn.Conv1d(time_steps, conv_filters, 3, padding=1)
        self.conv_second = nn.Conv1d(conv_filters, time_steps, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm_conv = nn.LayerNorm(model_dim)

    def forward(self, values, shared_attention):
        attended, _ = shared_attention(values, values, values, need_weights=False)
        values = self.norm_attention(values + attended)
        convolved = self.conv_second(self.dropout(F.relu(self.conv_first(values))))
        return self.norm_conv(values + convolved)


class TPMCFPTE(nn.Module):
    def __init__(self, time_steps, graph_dim, blocks, heads, conv_filters, dropout):
        super().__init__()
        model_dim = graph_dim * 4
        if model_dim % heads:
            raise ValueError("PTE feature dimension must be divisible by head count")
        self.attention = nn.MultiheadAttention(model_dim, heads, dropout=dropout,
                                               batch_first=True)
        self.blocks = nn.ModuleList([
            PredictiveBlock(time_steps, model_dim, conv_filters, dropout)
            for _ in range(blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(model_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, users, services, user_bank, service_bank):
        # [T,B,2d] + [T,B,2d] -> [B,T,4d]
        values = torch.cat((user_bank[:, users, :], service_bank[:, services, :]), dim=2)
        values = values.permute(1, 0, 2)
        for block in self.blocks:
            values = block(values, self.attention)
        return self.head(values.max(dim=1).values).squeeze(1)


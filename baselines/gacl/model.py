"""Pair-conditioned, two-hop GACL reproduction from paper Eqs. (2)-(29).

The second hop uses sampled neighbors because the shared historical graphs are
dense. Sampling and normalization choices are documented by the runner.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(length, dim):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    even = torch.arange(0, dim, 2, dtype=torch.float32)
    rate = torch.exp(-math.log(10000.0) * even / dim)
    result = torch.zeros(length, dim, dtype=torch.float32)
    result[:, 0::2] = torch.sin(positions * rate)
    result[:, 1::2] = torch.cos(positions * rate[:result[:, 1::2].shape[1]])
    return result


class TargetPromptLayer(nn.Module):
    """Eqs. (3), (5)-(10), with L2 norm as the paper's Norm operator."""

    def __init__(self, dim):
        super(TargetPromptLayer, self).__init__()
        self.semantic = nn.Parameter(torch.empty(dim))
        self.value_weight = nn.Parameter(torch.empty(dim))
        self.value_bias = nn.Parameter(torch.empty(dim))
        self.scale = nn.Linear(2 * dim, 1)
        self.offset = nn.Linear(2 * dim, 1)
        self.message = nn.Linear(dim, dim, bias=False)
        self.activation = nn.PReLU()
        nn.init.normal_(self.semantic, std=0.02)
        nn.init.normal_(self.value_weight, std=0.02)
        # A zero bias makes L2-normalized w * W identical for every positive
        # QoS value, so initialize b_w away from zero as in the trainable cue.
        nn.init.normal_(self.value_bias, std=0.02)

    def forward(self, center, neighbors, prompt, weights, mask):
        semantic = torch.sigmoid(
            ((center.unsqueeze(-2) + neighbors) * self.semantic).sum(dim=-1))
        collaborative = F.normalize(prompt.unsqueeze(-2) + neighbors,
                                    p=2, dim=-1, eps=1e-6)
        historical = F.normalize(
            weights.unsqueeze(-1) * self.value_weight + self.value_bias,
            p=2, dim=-1, eps=1e-6)
        cue = torch.cat((collaborative, historical), dim=-1)
        alpha = torch.tanh(self.scale(cue).squeeze(-1))
        beta = torch.tanh(self.offset(cue).squeeze(-1))
        score = alpha * semantic + beta
        masked_score = score.masked_fill(~mask, -1e4)
        attention = torch.softmax(masked_score, dim=-1) * mask.to(score.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        aggregate = (attention.unsqueeze(-1) * self.message(neighbors)).sum(dim=-2)
        return self.activation(center + aggregate)


class GACLPaperReproduction(nn.Module):
    """Target-prompt bipartite GAT, shared Transformer and ReLU QoS head."""

    def __init__(self, n_users, n_services, dim, window, graph_layers,
                 transformer_layers, heads, edge_scale):
        super(GACLPaperReproduction, self).__init__()
        if graph_layers != 2:
            raise ValueError("this reproduction implements exactly two graph layers")
        if dim % heads:
            raise ValueError("dimension must be divisible by attention heads")
        if edge_scale <= 0:
            raise ValueError("edge_scale must be positive")
        self.n_users = n_users
        self.n_services = n_services
        self.window = window
        self.edge_scale = float(edge_scale)
        self.embedding = nn.Embedding(n_users + n_services, dim)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.graph_layers = nn.ModuleList([TargetPromptLayer(dim) for _ in range(2)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=4 * dim, dropout=0.0,
            activation="relu")
        self.temporal = nn.TransformerEncoder(encoder_layer,
                                              num_layers=transformer_layers)
        self.register_buffer("position", positional_encoding(window, dim))
        self.invocation = nn.Linear(2 * dim, dim)
        self.invocation_activation = nn.PReLU()
        self.output = nn.Linear(dim, 1)

    def _embed_users(self, ids):
        return self.embedding(ids)

    def _embed_services(self, ids):
        return self.embedding(ids + self.n_users)

    def _snapshot(self, users, services, graph, time_id):
        user_first_ids = graph["user_first_ids"][time_id, users]
        user_first_w = graph["user_first_weight"][time_id, users] / self.edge_scale
        user_first_mask = graph["user_first_mask"][time_id, users]
        service_first_ids = graph["service_first_ids"][time_id, services]
        service_first_w = graph["service_first_weight"][time_id, services] / self.edge_scale
        service_first_mask = graph["service_first_mask"][time_id, services]

        service_second_ids = graph["service_second_ids"][time_id, user_first_ids]
        service_second_w = graph["service_second_weight"][time_id, user_first_ids] / self.edge_scale
        service_second_mask = graph["service_second_mask"][time_id, user_first_ids]
        user_second_ids = graph["user_second_ids"][time_id, service_first_ids]
        user_second_w = graph["user_second_weight"][time_id, service_first_ids] / self.edge_scale
        user_second_mask = graph["user_second_mask"][time_id, service_first_ids]

        target_user = self._embed_users(users)
        target_service = self._embed_services(services)
        first_services = self._embed_services(user_first_ids)
        first_users = self._embed_users(service_first_ids)
        second_users = self._embed_users(service_second_ids)
        second_services = self._embed_services(user_second_ids)
        layer1, layer2 = self.graph_layers
        user_1 = layer1(target_user, first_services, target_service,
                        user_first_w, user_first_mask)
        service_1 = layer1(target_service, first_users, target_user,
                           service_first_w, service_first_mask)
        first_services_1 = layer1(first_services, second_users,
                                  target_user.unsqueeze(1), service_second_w,
                                  service_second_mask)
        first_users_1 = layer1(first_users, second_services,
                               target_service.unsqueeze(1), user_second_w,
                               user_second_mask)
        user_2 = layer2(user_1, first_services_1, service_1,
                        user_first_w, user_first_mask)
        service_2 = layer2(service_1, first_users_1, user_1,
                           service_first_w, service_first_mask)
        return user_2, service_2

    def forward(self, users, services, graph):
        user_frames, service_frames = [], []
        for time_id in range(self.window):
            u, s = self._snapshot(users, services, graph, time_id)
            user_frames.append(u)
            service_frames.append(s)
        sequence = torch.cat((torch.stack(user_frames),
                              torch.stack(service_frames)), dim=1)
        sequence = sequence + self.position.unsqueeze(1)
        evolved = self.temporal(sequence)[-1]
        batch_size = users.shape[0]
        joined = torch.cat((evolved[:batch_size], evolved[batch_size:]), dim=-1)
        invocation = self.invocation_activation(self.invocation(joined))
        return F.relu(self.output(invocation).squeeze(-1))

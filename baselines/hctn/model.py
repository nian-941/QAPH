"""Five-module, history-only HCTN paper reproduction.

GPAM is a frozen history-only NMF input; HCFM, GMM, TGEM, and CQPM remain
trainable end to end. Hypergraph products are factored through the incidence
matrix and do not materialize a dense 4500-by-4500 service graph.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def positional_encoding(length, width):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, width, 2,
                                         dtype=torch.float32) *
                            (-math.log(10000.0) / width))
    encoded = torch.zeros(length, width, dtype=torch.float32)
    encoded[:, 0::2] = torch.sin(positions * frequencies)
    encoded[:, 1::2] = torch.cos(positions * frequencies[:
        encoded[:, 1::2].shape[1]])
    return encoded


def graph_operators(incidence):
    """Paper FIG normalization and Eq. (1) incidence-factor operators."""
    incidence = incidence.float()
    user_degree = incidence.sum(dim=1)
    service_degree = incidence.sum(dim=0)
    # D_u and D_s in Eq. (1) come from binary co-invocation adjacency,
    # not from first-order invocation counts. Only their diagonals are kept.
    user_shared = incidence @ incidence.T
    service_shared = incidence.T @ incidence
    user_shared.fill_diagonal_(0)
    service_shared.fill_diagonal_(0)
    user_second_degree = (user_shared > 0).sum(dim=1).clamp(min=1).float()
    service_second_degree = (service_shared > 0).sum(dim=1).clamp(min=1).float()
    user_inverse = user_second_degree.reciprocal()
    service_inverse = service_second_degree.reciprocal()
    fig_user_inverse = user_degree.clamp(min=1).reciprocal()
    fig_service_inverse = service_degree.clamp(min=1).reciprocal()
    return {
        "h": incidence,
        "fig_user_self": fig_user_inverse,
        "fig_service_self": fig_service_inverse,
        "fig_cross": (incidence *
                      fig_user_inverse.sqrt().unsqueeze(1) *
                      fig_service_inverse.sqrt().unsqueeze(0)),
        "user_inv_sqrt": user_inverse.sqrt(),
        "service_inv_sqrt": service_inverse.sqrt(),
        "user_inv": user_inverse,
        "service_inv": service_inverse,
    }


class HypergraphConvolutionUnit(nn.Module):
    """FCU, SUCU, SSCU with shared per-slice weights."""

    def __init__(self, width):
        super(HypergraphConvolutionUnit, self).__init__()
        self.fig_linear = nn.Linear(width, width)
        self.user_linear = nn.Linear(width, width)
        self.service_linear = nn.Linear(width, width)

    def forward(self, fig, user, service, graph, n_users):
        h = graph["h"]
        cross = graph["fig_cross"]
        fig_user = (fig[:n_users] * graph["fig_user_self"].unsqueeze(1) +
                    cross @ fig[n_users:])
        fig_service = (fig[n_users:] *
                       graph["fig_service_self"].unsqueeze(1) +
                       cross.T @ fig[:n_users])
        fig_out = F.relu(self.fig_linear(torch.cat((fig_user, fig_service))))

        # D_u^-1/2 H D_s^-1 H.T D_u^-1/2 X_u
        user_part = user * graph["user_inv_sqrt"].unsqueeze(1)
        user_part = h.T @ user_part
        user_part = user_part * graph["service_inv"].unsqueeze(1)
        user_part = h @ user_part
        user_part = user_part * graph["user_inv_sqrt"].unsqueeze(1)
        user_out = F.relu(self.user_linear(user_part))

        # D_s^-1/2 H.T D_u^-1 H D_s^-1/2 X_s
        service_part = service * graph["service_inv_sqrt"].unsqueeze(1)
        service_part = h @ service_part
        service_part = service_part * graph["user_inv"].unsqueeze(1)
        service_part = h.T @ service_part
        service_part = service_part * graph["service_inv_sqrt"].unsqueeze(1)
        service_out = F.relu(self.service_linear(service_part))
        return fig_out, user_out, service_out


class HCTNPaperReproduction(nn.Module):
    """GPAM -> HCFM -> GMM -> TGEM -> CQPM on all historical nodes."""

    def __init__(self, n_users, n_services, rank, width, window, layers,
                 heads, output_width, dropout):
        super(HCTNPaperReproduction, self).__init__()
        reduced = width // 4
        if (width % 4 or reduced % heads or window < 4 or
                layers < 1 or output_width < 1):
            raise ValueError("invalid HCTN model widths/window/layers")
        self.n_users = n_users
        self.n_services = n_services
        self.window = window
        self.initial = nn.Linear(rank, width)
        self.hcus = nn.ModuleList(
            HypergraphConvolutionUnit(width) for _ in range(layers))
        self.spatial_attention = nn.Linear(2 * width, 2)
        self.gmm_fusion = nn.Linear(width + 14, width)
        self.temporal_bn = nn.BatchNorm1d(width)
        self.reduce = nn.Linear(width, reduced)
        self.reduced_bn = nn.BatchNorm1d(reduced)
        self.mha = nn.MultiheadAttention(reduced, heads, dropout=dropout,
                                         batch_first=True)
        self.restore = nn.Linear(reduced, width)
        self.multiscale_bn = nn.BatchNorm1d(width)
        self.temporal_down = nn.Conv1d(width, width, kernel_size=4, stride=4)
        self.temporal_up = nn.ConvTranspose1d(width, width, kernel_size=4,
                                               stride=4)
        self.feature_down = nn.Linear(width, reduced)
        self.feature_up = nn.Linear(reduced, width)
        self.dropout = nn.Dropout(dropout)
        self.scale_attention = nn.Linear(2 * width, 2)
        self.output_bn = nn.BatchNorm1d(width)
        self.output_fc1 = nn.Linear(width, width)
        self.output_fc2 = nn.Linear(width, output_width)
        self.register_buffer("position", positional_encoding(window, width))

    def _hcfm(self, gpam, graphs):
        frames = []
        for t in range(self.window):
            original = F.relu(self.initial(gpam[t]))
            fig = original
            user = original[:self.n_users]
            service = original[self.n_users:]
            fig_sum = fig
            user_sum = user
            service_sum = service
            for level, hcu in enumerate(self.hcus, 1):
                fig, user, service = hcu(fig, user, service,
                                         graphs[t], self.n_users)
                weight = 1.0 / (level + 1)
                fig_sum = fig_sum + weight * fig
                user_sum = user_sum + weight * user
                service_sum = service_sum + weight * service
            hyper = torch.cat((user_sum, service_sum), dim=0)
            pair = torch.cat((fig_sum, hyper), dim=1)
            attention = torch.softmax(self.spatial_attention(pair), dim=1)
            combined = (attention[:, :1] * fig_sum +
                        attention[:, 1:] * hyper)
            frames.append(combined + original)  # GPAM skip connection
        return torch.stack(frames, dim=1)  # N x tau x f2

    def _tgem(self, z3):
        # BatchNorm1d expects nodes x channels x time.
        z4 = self.temporal_bn(z3.transpose(1, 2)).transpose(1, 2)
        z4 = z4 + self.position.unsqueeze(0)
        z5 = self.reduce(z4)
        z5 = self.reduced_bn(z5.transpose(1, 2)).transpose(1, 2)
        attended, _ = self.mha(z5, z5, z5, need_weights=False)
        z7 = self.restore(attended + z5)
        z8 = self.multiscale_bn(z7.transpose(1, 2)).transpose(1, 2)

        temporal = z8.transpose(1, 2)
        temporal = F.relu(self.temporal_down(temporal))
        temporal = self.dropout(temporal)
        temporal = F.relu(self.temporal_up(temporal))
        if temporal.shape[-1] != self.window:
            temporal = F.interpolate(temporal, size=self.window,
                                     mode="linear", align_corners=False)
        temporal = temporal.transpose(1, 2) + z8

        feature = F.relu(self.feature_down(z8))
        feature = self.dropout(feature)
        feature = F.relu(self.feature_up(feature)) + z8
        weights = torch.softmax(self.scale_attention(
            torch.cat((temporal, feature), dim=2)), dim=2)
        return weights[:, :, :1] * temporal + weights[:, :, 1:] * feature

    def forward(self, gpam, graphs, statistics, greysheep):
        y1 = self._hcfm(gpam, graphs)
        y3 = F.relu(self.gmm_fusion(torch.cat((y1, statistics), dim=2)))
        z3 = greysheep * y3 + (1.0 - greysheep) * y1
        x3 = self._tgem(z3)
        fused = self.output_bn((z3 + x3).transpose(1, 2))
        pooled = fused.mean(dim=2)
        embedding = self.output_fc2(
            self.dropout(F.relu(self.output_fc1(pooled))))
        user = embedding[:self.n_users]
        service = embedding[self.n_users:]
        return user @ service.T

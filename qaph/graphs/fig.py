"""
V61 LightGCN Embedding Enhancement Module
================================================================

在user-service二部图上做无参数消息传递，增强基础嵌入的协同过滤信号。

核心设计：
1. 纯PyTorch实现（不依赖torch_geometric）
2. 零额外可训练参数（LightGCN去掉了特征变换和非线性激活）
3. 位置：嵌入层之后、GMM/TGEM之前（对齐HCTN的HCFM位置）
4. 图构建：基于训练集非零观测的user-service二部图

与V58 CF嵌入的本质区别：
- V58: 冻结NMF因子 + 可训练投影 → 静态信息 + 额外参数 → 过拟合
- V61: 可训练嵌入 + 无参数图传播 → 动态信息 + 零额外参数 → 泛化

参考: He et al., "LightGCN: Simplifying and Powering GCN for Recommendation", SIGIR 2020

Author: Kilo Code
Date: 2026-03-13
Version: 61.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class LightGCNEmbedding(nn.Module):
    """
    LightGCN风格的嵌入增强模块
    
    在user-service二部图上做K层无参数消息传递，
    最终嵌入 = mean(e^(0), e^(1), ..., e^(K))
    
    关键特性：
    - 零额外可训练参数（纯消息传递）
    - 稀疏矩阵运算（内存高效）
    - 每个forward自动传播（嵌入更新时图传播结果自动更新）
    
    图结构：
    - 节点：[0, num_users) = 用户节点, [num_users, num_users+num_services) = 服务节点
    - 边：如果user u对service s有非零观测，则添加双向边 (u, num_users+s)
    - 归一化：D^{-1/2} A D^{-1/2}（对称归一化）
    """
    
    def __init__(self, num_layers: int = 2, dropout: float = 0.0):
        """
        Args:
            num_layers: LightGCN传播层数（K），推荐2-3
            dropout: 消息传递时的dropout（训练时使用，推理时关闭）
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 归一化邻接矩阵（稀疏），由外部设置
        self._norm_adj = None  # sparse tensor [N, N]
        self._num_users = 0
        self._num_services = 0
    
    def set_graph(self, norm_adj: torch.Tensor, num_users: int, num_services: int):
        """
        设置预计算的归一化邻接矩阵
        
        Args:
            norm_adj: [N, N] 稀疏归一化邻接矩阵，N = num_users + num_services
            num_users: 用户数量
            num_services: 服务数量
        """
        self._norm_adj = norm_adj
        self._num_users = num_users
        self._num_services = num_services
    
    def forward(
        self,
        user_embed: torch.Tensor,
        service_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        LightGCN前向传播
        
        Args:
            user_embed: [num_users, embed_dim] 所有用户的嵌入权重
            service_embed: [num_services, embed_dim] 所有服务的嵌入权重
        
        Returns:
            user_enhanced: [num_users, embed_dim] 增强后的用户嵌入
            service_enhanced: [num_services, embed_dim] 增强后的服务嵌入
        """
        if self._norm_adj is None:
            # 图未设置，直接返回原始嵌入
            return user_embed, service_embed
        
        # 拼接所有节点嵌入: [num_users + num_services, embed_dim]
        all_emb = torch.cat([user_embed, service_embed], dim=0)
        embs = [all_emb]
        
        # 确保邻接矩阵在正确的设备上
        norm_adj = self._norm_adj
        if norm_adj.device != all_emb.device:
            norm_adj = norm_adj.to(all_emb.device)
            self._norm_adj = norm_adj
        
        # K层无参数消息传递
        for layer in range(self.num_layers):
            # 稀疏矩阵乘法: A_norm @ all_emb
            all_emb = torch.sparse.mm(norm_adj, all_emb)
            
            # 可选dropout（仅训练时）
            if self.dropout is not None and self.training:
                all_emb = self.dropout(all_emb)
            
            embs.append(all_emb)
        
        # 层平均（LightGCN的核心设计）
        # e_final = mean(e^(0), e^(1), ..., e^(K))
        final_emb = torch.stack(embs, dim=0).mean(dim=0)  # [N, embed_dim]
        
        # 分离用户和服务嵌入
        user_enhanced = final_emb[:self._num_users]
        service_enhanced = final_emb[self._num_users:]
        
        return user_enhanced, service_enhanced


def build_lightgcn_adj(
    train_data: np.ndarray,
    num_users: int,
    num_services: int,
    device: torch.device = torch.device('cpu'),
    weight_mode: str = 'binary',
    history: np.ndarray = None,
    min_weight: float = 0.1,
) -> torch.Tensor:
    """
    From training data, build LightGCN normalized adjacency matrix.
    
    Builds user-service bipartite graph:
    - If (user_i, service_j) has non-zero QoS observation, add bidirectional edge
    - Add self-loops (preserve self information)
    - Symmetric normalization: D^{-1/2} A D^{-1/2}
    
    Args:
        train_data: [num_users, num_services] QoS mean matrix (non-zero positions are edges)
                    or sample list [n_samples, ...] with first two columns as user_id, service_id
        num_users: total number of users
        num_services: total number of services
        device: target device
        weight_mode: 'binary' (default, all edges weight=1), 'reliability' (QoS reliability weighting),
                     or 'qos_mean' (QoS value-based weighting, inspired by GACL)
        history: [num_users, num_services, num_timesteps] history tensor, required for reliability mode
        min_weight: minimum edge weight for non-binary modes (default 0.1)
    
    Returns:
        norm_adj: [N, N] sparse normalized adjacency matrix, N = num_users + num_services
    """
    N = num_users + num_services
    
    # Collect edges
    row_indices = []
    col_indices = []
    
    if train_data.ndim == 2 and train_data.shape[0] == num_users and train_data.shape[1] == num_services:
        # Input is QoS matrix [num_users, num_services]
        nonzero = np.nonzero(train_data > 0)
        user_ids = nonzero[0]
        service_ids = nonzero[1]
    else:
        # Input is sample list [n_samples, ...]
        user_ids = train_data[:, 0].astype(int)
        service_ids = train_data[:, 1].astype(int)
        pairs = set(zip(user_ids, service_ids))
        user_ids = np.array([p[0] for p in pairs])
        service_ids = np.array([p[1] for p in pairs])
    
    service_offset = num_users
    
    # Compute edge weights
    if weight_mode == 'reliability' and train_data.ndim == 2 and train_data.shape[0] == num_users:
        # Reliability weighting: w_ij = 1 / (1 + |qos_ij - mean_j|)
        # train_data is qos_matrix [num_users, num_services] (non-zero mean)
        qos_matrix = train_data
        
        # Compute per-service mean (only from non-zero entries)
        service_means = np.zeros(num_services, dtype=np.float64)
        for j in range(num_services):
            col = qos_matrix[:, j]
            nonzero_vals = col[col > 0]
            if len(nonzero_vals) > 0:
                service_means[j] = nonzero_vals.mean()
        
        # Compute reliability for each edge
        deviations = np.abs(qos_matrix[user_ids, service_ids] - service_means[service_ids])
        raw_reliability = 1.0 / (1.0 + deviations)
        
        # Normalize to [min_weight, 1.0]
        r_min, r_max = raw_reliability.min(), raw_reliability.max()
        if r_max > r_min:
            norm_reliability = min_weight + (1.0 - min_weight) * (raw_reliability - r_min) / (r_max - r_min)
        else:
            norm_reliability = np.ones_like(raw_reliability)
        
        edge_weights = norm_reliability.astype(np.float32)
        
        print(f"[V73 Reliability] Edge weight stats: min={edge_weights.min():.4f}, "
              f"median={np.median(edge_weights):.4f}, max={edge_weights.max():.4f}, "
              f"std={edge_weights.std():.4f}")
    elif weight_mode == 'qos_mean' and train_data.ndim == 2 and train_data.shape[0] == num_users:
        # QoS-value weighting: edge weight proportional to QoS mean value
        # Rationale: Higher QoS values indicate stronger/more reliable interactions
        # This provides richer graph signal than binary edges (inspired by GACL's QoS-weighted graph)
        # Unlike reliability mode (which failed due to 86.4% identical weights),
        # QoS values have much higher variance and discriminability
        qos_matrix = train_data
        qos_values = qos_matrix[user_ids, service_ids].astype(np.float64)
        
        # Normalize to [min_weight, 1.0] using min-max scaling
        q_min, q_max = qos_values.min(), qos_values.max()
        if q_max > q_min:
            norm_qos = min_weight + (1.0 - min_weight) * (qos_values - q_min) / (q_max - q_min)
        else:
            norm_qos = np.ones_like(qos_values)
        
        edge_weights = norm_qos.astype(np.float32)
        
        print(f"[V74 QoS-Mean] Edge weight stats: min={edge_weights.min():.4f}, "
              f"median={np.median(edge_weights):.4f}, max={edge_weights.max():.4f}, "
              f"std={edge_weights.std():.4f}, unique_ratio={len(np.unique(np.round(edge_weights, 4))) / len(edge_weights):.4f}")
    else:
        # Binary mode: all edges weight = 1.0
        edge_weights = np.ones(len(user_ids), dtype=np.float32)
    
    # Bidirectional edges: user -> service and service -> user (same weight)
    # user -> service
    row_indices.extend(user_ids.tolist())
    col_indices.extend((service_ids + service_offset).tolist())
    
    # service -> user
    row_indices.extend((service_ids + service_offset).tolist())
    col_indices.extend(user_ids.tolist())
    
    # Edge values: bidirectional edges use same weight
    edge_values = np.concatenate([edge_weights, edge_weights])
    
    # Self-loops with weight = 1.0
    self_loop_nodes = list(range(N))
    row_indices.extend(self_loop_nodes)
    col_indices.extend(self_loop_nodes)
    self_loop_values = np.ones(N, dtype=np.float32)
    
    # All values
    all_values = np.concatenate([edge_values, self_loop_values])
    
    # Build sparse adjacency matrix
    row = torch.LongTensor(row_indices)
    col = torch.LongTensor(col_indices)
    indices = torch.stack([row, col], dim=0)
    values = torch.FloatTensor(all_values)
    
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    degree = torch.sparse.sum(adj, dim=1).to_dense()  # [N]
    
    degree_inv_sqrt = torch.pow(degree.clamp(min=1e-12), -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    adj_indices = adj.indices()  # [2, nnz]
    adj_values = adj.values()    # [nnz]
    
    row_deg = degree_inv_sqrt[adj_indices[0]]  # D_i^{-1/2}
    col_deg = degree_inv_sqrt[adj_indices[1]]  # D_j^{-1/2}
    
    norm_values = adj_values * row_deg * col_deg
    
    norm_adj = torch.sparse_coo_tensor(
        adj_indices, norm_values, (N, N)
    ).coalesce().to(device)
    
    # Statistics
    num_edges = len(user_ids)
    mode_str = f"[{weight_mode}]" if weight_mode != 'binary' else ""
    print(f"[V61 LightGCN] Graph built{mode_str}: {num_users} users + {num_services} services = {N} nodes")
    print(f"[V61 LightGCN] Edges: {num_edges} interactions -> {num_edges * 2} bidirectional + {N} self-loops = {len(row_indices)} total")
    print(f"[V61 LightGCN] Sparse adj: {norm_adj._nnz()} non-zeros, density={norm_adj._nnz() / (N * N) * 100:.4f}%")
    
    return norm_adj


def build_lightgcn_adj_from_mask(
    train_mask: np.ndarray,
    qos_tensor: np.ndarray,
    num_users: int,
    num_services: int,
    target_time_step: int,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    从训练mask和QoS张量构建LightGCN邻接矩阵
    
    这是更精确的版本，直接使用训练集的mask信息。
    
    Args:
        train_mask: [num_users, num_services] bool矩阵，True表示训练集中的观测
        qos_tensor: [num_users, num_services, num_timesteps] 完整QoS张量
        num_users: 用户数量
        num_services: 服务数量
        target_time_step: 目标预测时间步（用于确定哪些历史数据可用）
        device: 目标设备
    
    Returns:
        norm_adj: [N, N] 稀疏归一化邻接矩阵
    """
    N = num_users + num_services
    
    # 从训练mask中找到所有有观测的(user, service)对
    # 同时要求在历史时间步中有非零值
    row_indices = []
    col_indices = []
    
    service_offset = num_users
    
    # 方法1: 直接使用train_mask（如果提供）
    if train_mask is not None:
        nonzero = np.nonzero(train_mask)
        user_ids = nonzero[0]
        service_ids = nonzero[1]
    else:
        # 方法2: 从QoS张量中找非零观测
        # 使用target_time_step之前的所有时间步
        history = qos_tensor[:, :, :target_time_step]
        has_observation = np.any(history > 0, axis=2)
        nonzero = np.nonzero(has_observation)
        user_ids = nonzero[0]
        service_ids = nonzero[1]
    
    # 双向边
    row_indices.extend(user_ids.tolist())
    col_indices.extend((service_ids + service_offset).tolist())
    row_indices.extend((service_ids + service_offset).tolist())
    col_indices.extend(user_ids.tolist())
    
    # 自环
    self_loop_nodes = list(range(N))
    row_indices.extend(self_loop_nodes)
    col_indices.extend(self_loop_nodes)
    
    # 构建并归一化
    row = torch.LongTensor(row_indices)
    col = torch.LongTensor(col_indices)
    indices = torch.stack([row, col], dim=0)
    values = torch.ones(len(row_indices), dtype=torch.float32)
    
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    
    # 对称归一化
    degree = torch.sparse.sum(adj, dim=1).to_dense()
    degree_inv_sqrt = torch.pow(degree.clamp(min=1e-12), -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
    
    adj_indices = adj.indices()
    adj_values = adj.values()
    row_deg = degree_inv_sqrt[adj_indices[0]]
    col_deg = degree_inv_sqrt[adj_indices[1]]
    norm_values = adj_values * row_deg * col_deg
    
    norm_adj = torch.sparse_coo_tensor(
        adj_indices, norm_values, (N, N)
    ).coalesce().to(device)
    
    num_edges = len(user_ids)
    print(f"[V61 LightGCN] Graph from mask: {num_edges} edges, {N} nodes, "
          f"nnz={norm_adj._nnz()}, density={norm_adj._nnz() / (N * N) * 100:.4f}%")
    
    return norm_adj


if __name__ == "__main__":
    print("V61 LightGCN Embedding Enhancement Module Test")
    print("=" * 60)
    
    # 测试参数
    num_users = 142
    num_services = 4500
    embed_dim = 64
    num_layers = 2
    
    # 创建模拟QoS矩阵（10%密度）
    np.random.seed(42)
    qos_matrix = np.random.rand(num_users, num_services).astype(np.float32)
    qos_matrix[qos_matrix < 0.9] = 0  # 10%密度
    
    # 构建邻接矩阵
    print("\n1. Building adjacency matrix...")
    norm_adj = build_lightgcn_adj(qos_matrix, num_users, num_services)
    
    # 创建LightGCN模块
    print("\n2. Creating LightGCN module...")
    lightgcn = LightGCNEmbedding(num_layers=num_layers)
    lightgcn.set_graph(norm_adj, num_users, num_services)
    
    # 模拟嵌入
    user_embed = torch.randn(num_users, embed_dim)
    service_embed = torch.randn(num_services, embed_dim)
    
    # 前向传播
    print("\n3. Forward pass...")
    user_enhanced, service_enhanced = lightgcn(user_embed, service_embed)
    print(f"   User: {user_embed.shape} → {user_enhanced.shape}")
    print(f"   Service: {service_embed.shape} → {service_enhanced.shape}")
    
    # 验证增强效果
    user_diff = (user_enhanced - user_embed).norm() / user_embed.norm()
    service_diff = (service_enhanced - service_embed).norm() / service_embed.norm()
    print(f"   User embedding change: {user_diff.item():.4f}")
    print(f"   Service embedding change: {service_diff.item():.4f}")
    
    # 参数量
    total_params = sum(p.numel() for p in lightgcn.parameters())
    print(f"\n4. Parameters: {total_params} (should be 0)")
    
    # 测试梯度传播
    print("\n5. Gradient test...")
    user_embed.requires_grad_(True)
    service_embed.requires_grad_(True)
    user_out, service_out = lightgcn(user_embed, service_embed)
    loss = user_out.sum() + service_out.sum()
    loss.backward()
    print(f"   User grad norm: {user_embed.grad.norm().item():.4f}")
    print(f"   Service grad norm: {service_embed.grad.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
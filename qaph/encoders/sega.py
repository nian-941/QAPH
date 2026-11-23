"""
V77 Parametric GCN with KNN SUIG (Direction H)
================================================================

V77新增: KNN SUIG (QoS-Similarity K-Nearest-Neighbor User Graph)
- 基于QoS向量余弦相似度构建用户相似度图
- KNN稀疏化到~12%密度（解决V71的99.4%过稠密问题）
- 单门控融合（避免V71的双重门控复杂性）

核心设计：
1. 带可学习权重矩阵的图卷积（W1, W2, ..., Wn）
2. LayerNorm + ReLU + Dropout
3. 残差门控：enhanced = raw + sigmoid(α) * gcn(raw)，α初始化使gate≈0.12
4. 纯PyTorch实现（不依赖torch_geometric），使用稀疏矩阵乘法
5. V70: 可配置层数（2→3→4），支持更深的图信息传播
6. V77: KNN SUIG支持（可选）

与V61 LightGCN的区别：
- LightGCN: 零参数，层平均，适合推荐系统
- ParametricGCN: 带参数变换，残差门控，适合QoS预测的异构嵌入增强

参考:
- Kipf & Welling (ICLR 2017) "Semi-Supervised Classification with GCN"
- He et al. (SIGIR 2020) "LightGCN" (图构建策略)
- Direction H Feasibility Analysis (2026-04-04)

Author: Kilo Code
Date: 2026-04-04
Version: 77.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from torch_scatter import scatter_add


def build_knn_suig(qos_matrix: torch.Tensor, k: int = 10,
                   device: str = 'cuda') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建KNN稀疏化的用户相似度图 (Direction H)
    
    基于QoS向量余弦相似度，使用KNN稀疏化到~12%密度。
    解决V71的99.4%过稠密问题。
    
    Args:
        qos_matrix: [n_users, n_services, n_timesteps] QoS张量
        k: KNN邻居数量（默认10，对应~12%密度）
        device: 'cuda' or 'cpu'
    
    Returns:
        edge_index: [2, nnz] COO格式边索引
        edge_weight: [nnz] 归一化后的边权重
    
    算法流程:
        1. 展平QoS向量: [n_users, n_services * n_timesteps]
        2. 计算余弦相似度矩阵: [n_users, n_users]
        3. KNN稀疏化: 每个用户保留top-K最相似邻居
        4. 对称化: u1→u2 和 u2→u1都保留
        5. 度归一化: D^{-1/2} A D^{-1/2}
    
    参考:
        - plans/DIRECTION_H_FEASIBILITY_COMPREHENSIVE_ANALYSIS.md
        - 可行性分析: K=10 → 11.0-12.7%密度，Pearson r=0.661
    """
    n_users, n_services, n_timesteps = qos_matrix.shape
    
    # 1. 展平QoS向量: [n_users, n_services * n_timesteps]
    qos_vectors = qos_matrix.reshape(n_users, -1).to(device)
    
    # 2. 处理缺失值 (mask=0的位置)
    mask = (qos_vectors > 0).float()
    qos_vectors_masked = qos_vectors * mask
    
    # 3. 计算余弦相似度矩阵 [n_users, n_users]
    # 使用F.normalize + matmul实现高效余弦相似度
    qos_norm = F.normalize(qos_vectors_masked, p=2, dim=1)
    similarity_matrix = torch.mm(qos_norm, qos_norm.t())
    
    # 4. KNN稀疏化
    # 对每个用户，保留top-K最相似的邻居
    topk_values, topk_indices = torch.topk(similarity_matrix, k=k+1, dim=1)
    # k+1是因为包含自己，需要去除
    topk_values = topk_values[:, 1:]  # 去除自己
    topk_indices = topk_indices[:, 1:]
    
    # 5. 构建COO格式稀疏矩阵
    row_indices = torch.arange(n_users, device=device).unsqueeze(1).expand(-1, k).flatten()
    col_indices = topk_indices.flatten()
    edge_weights = topk_values.flatten()
    
    # 6. 对称化 (u1->u2 和 u2->u1都保留)
    row_sym = torch.cat([row_indices, col_indices])
    col_sym = torch.cat([col_indices, row_indices])
    weight_sym = torch.cat([edge_weights, edge_weights])
    
    # 7. 去重 (同一条边可能被双向添加)
    edge_index = torch.stack([row_sym, col_sym], dim=0)
    # 使用torch.unique去重
    edge_index_unique, inverse_indices = torch.unique(
        edge_index, dim=1, return_inverse=True
    )
    # 对重复边的权重求平均
    edge_weights_unique = scatter_add(
        weight_sym, inverse_indices, dim=0, dim_size=edge_index_unique.size(1)
    )
    # 计算每条边的重复次数
    edge_counts = scatter_add(
        torch.ones_like(weight_sym), inverse_indices, dim=0,
        dim_size=edge_index_unique.size(1)
    )
    edge_weights_unique = edge_weights_unique / edge_counts
    
    # 8. 度归一化 (D^{-1/2} A D^{-1/2})
    deg = scatter_add(edge_weights_unique, edge_index_unique[0], dim=0, dim_size=n_users)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    edge_weights_norm = (deg_inv_sqrt[edge_index_unique[0]] *
                         edge_weights_unique *
                         deg_inv_sqrt[edge_index_unique[1]])
    
    return edge_index_unique, edge_weights_norm


class ParametricGCN(nn.Module):
    """
    带参数的N层图卷积模块
    
    在user-service二部图上做带参数的消息传递：
        Layer i: Hi = ReLU(LN(A_norm @ (Wi @ H_{i-1})))  (i < num_layers)
        Layer N: HN = LN(A_norm @ (WN @ H_{N-1}))        (最后一层无ReLU)
        Output:  X + sigmoid(α) * HN
    
    关键特性：
    - 可学习的线性变换（W1, ..., WN）
    - LayerNorm稳定训练
    - 残差门控（α初始化为-2.0，sigmoid(-2)≈0.12）
    - 稀疏矩阵运算（内存高效）
    - V70: 可配置层数（2→3→4），更深的图传播捕获更远邻居信息
    
    图结构（复用LightGCN的构建逻辑）：
    - 节点：[0, num_users) = 用户, [num_users, N) = 服务
    - 边：训练集非零观测的双向边 + 自环
    - 归一化：D^{-1/2} A D^{-1/2}
    """
    
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1,
                 init_gate: float = -2.0, num_layers: int = 2,
                 use_knn_suig: bool = False, suig_k: int = 10):
        """
        Args:
            embed_dim: 嵌入维度
            dropout: Dropout率
            init_gate: 残差门控α的初始值（sigmoid(init_gate)为初始门控强度）
                       -2.0 → sigmoid≈0.12, -1.0 → 0.27, 0.0 → 0.5
            num_layers: GCN层数（V70新增，默认2保持向后兼容）
            use_knn_suig: V77新增，是否使用KNN SUIG（Direction H）
            suig_k: KNN邻居数量（默认10，对应~12%密度）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_knn_suig = use_knn_suig
        self.suig_k = suig_k
        
        # N层线性变换（无bias，对齐GCN原论文）
        self.W_layers = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)
        ])
        
        # N层LayerNorm
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 残差门控参数
        self.gate_alpha = nn.Parameter(torch.tensor(init_gate))
        
        # V77: KNN SUIG支持
        if use_knn_suig:
            # SUIG用户图GCN层
            self.suig_W_layers = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)
            ])
            self.suig_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(num_layers)
            ])
            # SUIG融合门控（初始化为弱门控）
            self.suig_gate = nn.Parameter(torch.tensor(-2.0))
        
        # 预计算的归一化邻接矩阵
        self._norm_adj = None
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
        qos_matrix: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ParametricGCN前向传播（V77: 支持KNN SUIG）
        
        Args:
            user_embed: [num_users, embed_dim] 所有用户的嵌入权重
            service_embed: [num_services, embed_dim] 所有服务的嵌入权重
            qos_matrix: [num_users, num_services, n_timesteps] QoS张量（V77新增，用于构建KNN SUIG）
        
        Returns:
            user_enhanced: [num_users, embed_dim] 增强后的用户嵌入
            service_enhanced: [num_services, embed_dim] 增强后的服务嵌入
        """
        if self._norm_adj is None:
            return user_embed, service_embed
        
        # 拼接所有节点嵌入: [N, embed_dim]
        all_emb = torch.cat([user_embed, service_embed], dim=0)
        
        # 确保邻接矩阵在正确的设备上
        norm_adj = self._norm_adj
        if norm_adj.device != all_emb.device:
            norm_adj = norm_adj.to(all_emb.device)
            self._norm_adj = norm_adj
        
        # N层FIG图卷积
        h = all_emb
        for i in range(self.num_layers):
            h = self.W_layers[i](h)                   # [N, d] 线性变换
            h = torch.sparse.mm(norm_adj, h)           # [N, d] 图传播
            h = self.norms[i](h)                       # LayerNorm
            if i < self.num_layers - 1:
                # 中间层: ReLU + Dropout
                h = F.relu(h)
                h = self.dropout(h)
            # 最后一层: 无ReLU（保留线性表达能力）
        
        # 残差门控: enhanced = raw + sigmoid(α) * gcn(raw)
        gate = torch.sigmoid(self.gate_alpha)
        enhanced = all_emb + gate * h
        
        # 分离用户和服务嵌入
        user_enhanced = enhanced[:self._num_users]
        service_enhanced = enhanced[self._num_users:]
        
        # V77: KNN SUIG融合
        if self.use_knn_suig and qos_matrix is not None:
            # 构建KNN SUIG
            suig_edge_index, suig_edge_weight = build_knn_suig(
                qos_matrix, k=self.suig_k, device=user_embed.device
            )
            
            # 构建稀疏邻接矩阵
            suig_adj = torch.sparse_coo_tensor(
                suig_edge_index, suig_edge_weight,
                size=(self._num_users, self._num_users),
                device=user_embed.device
            ).coalesce()
            
            # SUIG GCN（仅用户图）
            h_suig = user_embed
            for i in range(self.num_layers):
                h_suig = self.suig_W_layers[i](h_suig)
                h_suig = torch.sparse.mm(suig_adj, h_suig)
                h_suig = self.suig_norms[i](h_suig)
                if i < self.num_layers - 1:
                    h_suig = F.relu(h_suig)
                    h_suig = self.dropout(h_suig)
            
            # SUIG门控融合
            suig_gate = torch.sigmoid(self.suig_gate)
            user_enhanced = user_enhanced + suig_gate * h_suig
        
        return user_enhanced, service_enhanced
    
    def extra_repr(self):
        gate_val = torch.sigmoid(self.gate_alpha).item()
        return (f'embed_dim={self.embed_dim}, num_layers={self.num_layers}, '
                f'gate={gate_val:.3f} (alpha={self.gate_alpha.item():.3f})')


class MultiGraphGCN(nn.Module):
    """
    V71 三路图卷积网络：FIG + SUIG + SSIG
    
    对齐HCTN的HCU设计：
    - FIG路径: 在完整二部图上做GCN（复用现有ParametricGCN）
    - SUIG路径: 在user-user二阶图上做GCN
    - SSIG路径: 在service-service二阶图上做GCN
    - 融合: enhanced = raw + gate_fig * h_fig + gate_second * (h_suig ⊕ h_ssig)
    
    接口与ParametricGCN完全一致：forward(user_embed, service_embed) → (user_enhanced, service_enhanced)
    """
    
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1,
                 init_gate: float = -2.0, num_layers: int = 2,
                 use_suig: bool = True, use_ssig: bool = True,
                 second_gate_init: float = -2.0):
        """
        Args:
            embed_dim: 嵌入维度
            dropout: Dropout率
            init_gate: FIG路径的残差门控初始值
            num_layers: 每条路径的GCN层数
            use_suig: 是否启用SUIG路径
            use_ssig: 是否启用SSIG路径
            second_gate_init: 二阶路径的门控初始值
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_suig = use_suig
        self.use_ssig = use_ssig
        
        # FIG路径（复用现有ParametricGCN）
        self.fig_gcn = ParametricGCN(
            embed_dim=embed_dim, dropout=dropout,
            init_gate=init_gate, num_layers=num_layers
        )
        
        # SUIG路径（user-user, [142, 142]）
        if use_suig:
            self.suig_W_layers = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)
            ])
            self.suig_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(num_layers)
            ])
        
        # SSIG路径（service-service, [4500, 4500]）
        if use_ssig:
            self.ssig_W_layers = nn.ModuleList([
                nn.Linear(embed_dim, embed_dim, bias=False) for _ in range(num_layers)
            ])
            self.ssig_norms = nn.ModuleList([
                nn.LayerNorm(embed_dim) for _ in range(num_layers)
            ])
        
        # 二阶融合门控（独立于FIG的gate）
        self.second_order_gate = nn.Parameter(torch.tensor(second_gate_init))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 二阶图存储
        self._suig_adj = None  # [num_users, num_users] sparse
        self._ssig_adj = None  # [num_services, num_services] sparse
        self._num_users = 0
        self._num_services = 0
    
    def set_graph(self, fig_adj: torch.Tensor, num_users: int, num_services: int,
                  suig_adj: Optional[torch.Tensor] = None,
                  ssig_adj: Optional[torch.Tensor] = None):
        """
        设置三种图的邻接矩阵
        
        Args:
            fig_adj: [N, N] FIG稀疏邻接矩阵（N = num_users + num_services）
            num_users: 用户数量
            num_services: 服务数量
            suig_adj: [num_users, num_users] SUIG稀疏邻接矩阵（可选）
            ssig_adj: [num_services, num_services] SSIG稀疏邻接矩阵（可选）
        """
        self.fig_gcn.set_graph(fig_adj, num_users, num_services)
        self._suig_adj = suig_adj
        self._ssig_adj = ssig_adj
        self._num_users = num_users
        self._num_services = num_services
    
    def _second_order_gcn(self, h: torch.Tensor, adj: torch.Tensor,
                          W_layers: nn.ModuleList, norms: nn.ModuleList) -> torch.Tensor:
        """
        二阶图上的N层GCN前向传播
        
        Args:
            h: [M, embed_dim] 节点嵌入（M=num_users或num_services）
            adj: [M, M] 稀疏邻接矩阵
        
        Returns:
            h: [M, embed_dim] GCN输出
        """
        # 确保邻接矩阵在正确设备上
        if adj.device != h.device:
            adj = adj.to(h.device)
        
        for i in range(self.num_layers):
            h = W_layers[i](h)                    # 线性变换
            h = torch.sparse.mm(adj, h)            # 图传播
            h = norms[i](h)                        # LayerNorm
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout_layer(h)
        return h
    
    def forward(
        self,
        user_embed: torch.Tensor,
        service_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三路图卷积前向传播
        
        Args:
            user_embed: [num_users, embed_dim]
            service_embed: [num_services, embed_dim]
        
        Returns:
            user_enhanced: [num_users, embed_dim]
            service_enhanced: [num_services, embed_dim]
        """
        # FIG路径（现有ParametricGCN，已包含残差门控）
        user_fig, service_fig = self.fig_gcn(user_embed, service_embed)
        
        # 二阶路径
        second_gate = torch.sigmoid(self.second_order_gate)
        
        user_enhanced = user_fig
        service_enhanced = service_fig
        
        if self.use_suig and self._suig_adj is not None:
            h_suig = self._second_order_gcn(
                user_embed, self._suig_adj,
                self.suig_W_layers, self.suig_norms
            )  # [num_users, embed_dim]
            user_enhanced = user_enhanced + second_gate * h_suig
        
        if self.use_ssig and self._ssig_adj is not None:
            h_ssig = self._second_order_gcn(
                service_embed, self._ssig_adj,
                self.ssig_W_layers, self.ssig_norms
            )  # [num_services, embed_dim]
            service_enhanced = service_enhanced + second_gate * h_ssig
        
        return user_enhanced, service_enhanced
    
    @property
    def gate_alpha(self):
        """兼容性：返回FIG的gate_alpha（用于日志打印）"""
        return self.fig_gcn.gate_alpha
    
    def extra_repr(self):
        fig_gate = torch.sigmoid(self.fig_gcn.gate_alpha).item()
        second_gate = torch.sigmoid(self.second_order_gate).item()
        return (f'embed_dim={self.embed_dim}, num_layers={self.num_layers}, '
                f'fig_gate={fig_gate:.3f}, second_gate={second_gate:.3f}, '
                f'use_suig={self.use_suig}, use_ssig={self.use_ssig}')


class ProjectionHead(nn.Module):
    """
    轻量投影头：将共享嵌入投影到各模块的输入空间
    
    使用 Linear + LayerNorm（无ReLU，保留线性可逆性）
    """
    
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


if __name__ == "__main__":
    print("V70 ParametricGCN Test")
    print("=" * 60)
    
    from lightgcn_embedding import build_lightgcn_adj
    
    num_users = 142
    num_services = 4500
    embed_dim = 64
    num_layers = 4  # V70: 测试4层GCN
    
    # 创建模拟QoS矩阵（10%密度）
    np.random.seed(42)
    qos_matrix = np.random.rand(num_users, num_services).astype(np.float32)
    qos_matrix[qos_matrix < 0.9] = 0
    
    # 构建邻接矩阵（复用LightGCN的构建逻辑）
    print("\n1. Building adjacency matrix...")
    norm_adj = build_lightgcn_adj(qos_matrix, num_users, num_services)
    
    # 创建ParametricGCN
    print(f"\n2. Creating ParametricGCN (num_layers={num_layers})...")
    pgcn = ParametricGCN(embed_dim=embed_dim, dropout=0.1, init_gate=-2.0,
                         num_layers=num_layers)
    pgcn.set_graph(norm_adj, num_users, num_services)
    
    # 模拟嵌入
    user_embed = torch.randn(num_users, embed_dim)
    service_embed = torch.randn(num_services, embed_dim)
    
    # 前向传播
    print("\n3. Forward pass...")
    user_enhanced, service_enhanced = pgcn(user_embed, service_embed)
    print(f"   User: {user_embed.shape} → {user_enhanced.shape}")
    print(f"   Service: {service_embed.shape} → {service_enhanced.shape}")
    
    # 验证增强效果
    user_diff = (user_enhanced - user_embed).norm() / user_embed.norm()
    service_diff = (service_enhanced - service_embed).norm() / service_embed.norm()
    gate_val = torch.sigmoid(pgcn.gate_alpha).item()
    print(f"   Gate value: {gate_val:.4f}")
    print(f"   User embedding change: {user_diff.item():.4f}")
    print(f"   Service embedding change: {service_diff.item():.4f}")
    
    # 参数量
    total_params = sum(p.numel() for p in pgcn.parameters())
    print(f"\n4. Parameters: {total_params:,} ({num_layers} layers × W={embed_dim*embed_dim}, "
          f"LN={num_layers}×{2*embed_dim}, gate=1)")
    
    # 测试投影头
    print("\n5. ProjectionHead test...")
    proj = ProjectionHead(embed_dim)
    proj_params = sum(p.numel() for p in proj.parameters())
    print(f"   ProjectionHead params: {proj_params:,}")
    out = proj(user_enhanced[:10])
    print(f"   Input: [10, {embed_dim}] → Output: {out.shape}")
    
    # 测试梯度传播
    print("\n6. Gradient test...")
    user_embed.requires_grad_(True)
    service_embed.requires_grad_(True)
    user_out, service_out = pgcn(user_embed, service_embed)
    loss = user_out.sum() + service_out.sum()
    loss.backward()
    print(f"   User grad norm: {user_embed.grad.norm().item():.4f}")
    print(f"   Service grad norm: {service_embed.grad.norm().item():.4f}")
    print(f"   W_layers[0] grad norm: {pgcn.W_layers[0].weight.grad.norm().item():.4f}")
    print(f"   gate_alpha grad: {pgcn.gate_alpha.grad.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
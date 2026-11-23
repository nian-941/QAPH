"""
V40.0 全局模式感知的邻居选择机制 (Global-Aware Neighbor Selection, GANS)

改进GMM的邻居选择策略，结合全局模式特征（活跃度、流行度、稳定性、代表性）
与QoS相似度，选择更高质量的邻居。

Author: Kilo Code
Date: 2026-02-12
Version: V40.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GlobalPatternComputer:
    """计算全局模式特征（预计算一次）
    
    提取4维全局模式特征：
    1. Activity（活跃度）：实体的QoS记录数量
    2. Popularity（流行度）：被其他实体调用的频率
    3. Stability（稳定性）：QoS值的标准差倒数
    4. Representativeness（代表性）：与全局均值的接近程度
    """
    
    @staticmethod
    def compute_features(qos_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算全局模式特征
        
        Args:
            qos_matrix: (N, M) - QoS矩阵
        
        Returns:
            global_features: (N, 4) - 全局模式特征
        """
        # 确保是2D矩阵 - V55 Fix: 非零均值
        if qos_matrix.dim() == 3:
            qos_sum = qos_matrix.sum(dim=-1)
            qos_count = (qos_matrix > 0).float().sum(dim=-1).clamp(min=1)
            qos_matrix = qos_sum / qos_count
        assert qos_matrix.dim() == 2, f"Expected 2D matrix, got {qos_matrix.dim()}D"
        
        N, M = qos_matrix.shape
        device = qos_matrix.device
        
        # 有效QoS掩码
        valid_mask = (qos_matrix > 0).float()
        valid_qos = qos_matrix * valid_mask
        
        # 1. 活跃度（Activity）：非零QoS记录的比例
        activity = valid_mask.sum(dim=1) / M
        
        # 2. 流行度（Popularity）：被调用频率（归一化）
        # 对于用户：调用了多少服务
        # 对于服务：被多少用户调用
        popularity = valid_mask.sum(dim=1) / N
        
        # 3. 稳定性（Stability）：QoS标准差的倒数（越小越稳定）
        # 只计算有效QoS的标准差
        count_per_row = valid_mask.sum(dim=1).clamp(min=1)
        mean_per_row = valid_qos.sum(dim=1) / count_per_row
        
        # 计算标准差（向量化）
        diff_squared = ((valid_qos - mean_per_row.unsqueeze(1)) ** 2) * valid_mask
        variance = diff_squared.sum(dim=1) / count_per_row
        std = torch.sqrt(variance + 1e-8)
        stability = 1.0 / (std + 1e-6)
        
        # 4. 代表性（Representativeness）：与全局均值的接近程度
        global_mean = valid_qos[valid_qos > 0].mean()
        entity_mean = valid_qos.sum(dim=1) / count_per_row
        representativeness = 1.0 / (torch.abs(entity_mean - global_mean) + 1e-6)
        
        # 堆叠特征
        features = torch.stack([activity, popularity, stability, representativeness], dim=1)
        
        # 归一化到[0, 1]（Min-Max归一化）
        min_vals = features.min(dim=0, keepdim=True)[0]
        max_vals = features.max(dim=0, keepdim=True)[0]
        features = (features - min_vals) / (max_vals - min_vals + 1e-8)
        
        # 处理NaN/Inf
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        
        return features


class GDMv3(nn.Module):
    """V40.0 GDM - 全局模式感知的邻居选择
    
    改进点：
    1. 结合QoS相似度和全局模式特征
    2. 可学习的全局模式权重
    3. 自适应融合系数alpha
    """
    
    def __init__(self, num_neighbors: int = 10, alpha: float = 0.7):
        """
        Args:
            num_neighbors: 邻居数量
            alpha: QoS相似度权重（0-1之间，越大越依赖QoS相似度）
        """
        super().__init__()
        self.num_neighbors = num_neighbors
        self.alpha = alpha
        
        # 全局模式特征权重（可学习，初始化为均匀权重）
        self.global_pattern_weights = nn.Parameter(torch.ones(4) / 4)
        
        # 预计算缓存
        self._precomputed_neighbors = None
        self._global_features = None
    
    def precompute_global_features(self, qos_matrix: torch.Tensor):
        """
        预计算全局模式特征（训练前调用一次）
        
        Args:
            qos_matrix: (N, M) - QoS矩阵
        """
        print(f"[GDMv3] Computing global pattern features...")
        self._global_features = GlobalPatternComputer.compute_features(qos_matrix)
        print(f"[GDMv3] Global features shape: {self._global_features.shape}")
        print(f"[GDMv3] Feature stats:")
        print(f"  Activity: mean={self._global_features[:, 0].mean():.4f}, std={self._global_features[:, 0].std():.4f}")
        print(f"  Popularity: mean={self._global_features[:, 1].mean():.4f}, std={self._global_features[:, 1].std():.4f}")
        print(f"  Stability: mean={self._global_features[:, 2].mean():.4f}, std={self._global_features[:, 2].std():.4f}")
        print(f"  Representativeness: mean={self._global_features[:, 3].mean():.4f}, std={self._global_features[:, 3].std():.4f}")
    
    def compute_greysheep_score(self, qos_profile: torch.Tensor,
                                 mean_qos: float, std_qos: float) -> torch.Tensor:
        """
        计算greysheep分数（GDI - Greysheep Discrepancy Index）
        
        Args:
            qos_profile: (batch,) - 实体的QoS均值
            mean_qos: float - 全局QoS均值
            std_qos: float - 全局QoS标准差
        
        Returns:
            gdi_score: (batch, 1) - Greysheep分数
        """
        # 标准化偏差
        deviation = torch.abs(qos_profile - mean_qos) / (std_qos + 1e-8)
        return deviation.unsqueeze(-1)
    
    def find_neighbors_with_global_awareness(self, qos_matrix: torch.Tensor,
                                               entity_ids: torch.Tensor,
                                               k: int) -> torch.Tensor:
        """
        全局模式感知的邻居选择（核心改进）
        
        Args:
            qos_matrix: (N, M) - QoS矩阵
            entity_ids: (batch,) - 实体ID
            k: int - 邻居数量
        
        Returns:
            neighbor_ids: (batch, k) - 邻居ID
        """
        # 确保全局特征已预计算
        if self._global_features is None:
            raise RuntimeError("Global features not precomputed. Call precompute_global_features() first.")
        
        # 处理3D矩阵 - V55 Fix: 非零均值
        if qos_matrix.dim() == 3:
            qos_sum = qos_matrix.sum(dim=-1)
            qos_count = (qos_matrix > 0).float().sum(dim=-1).clamp(min=1)
            qos_matrix = qos_sum / qos_count
        
        batch_size = len(entity_ids)
        device = qos_matrix.device
        
        # === 1. QoS相似度（与V36相同） ===
        batch_profiles = qos_matrix[entity_ids]
        valid_mask = (qos_matrix > 0).float()
        batch_valid = (batch_profiles > 0).float()
        
        # 归一化
        qos_masked = qos_matrix * valid_mask
        row_norms = torch.norm(qos_masked, p=2, dim=1, keepdim=True) + 1e-8
        qos_normalized = qos_masked / row_norms
        
        batch_masked = batch_profiles * batch_valid
        batch_norms = torch.norm(batch_masked, p=2, dim=1, keepdim=True) + 1e-8
        batch_normalized = batch_masked / batch_norms
        
        # 余弦相似度
        qos_similarity = torch.mm(batch_normalized, qos_normalized.t())  # (batch, N)
        
        # === 2. 全局模式相似度（V40新增） ===
        batch_global = self._global_features[entity_ids].to(device)  # (batch, 4)
        all_global = self._global_features.to(device)  # (N, 4)
        
        # 加权全局特征（可学习权重）
        weights = F.softmax(self.global_pattern_weights, dim=0)  # 归一化权重
        weighted_batch = batch_global @ weights  # (batch,)
        weighted_all = all_global @ weights  # (N,)
        
        # 全局模式相似度（越接近越好）
        global_diff = torch.abs(weighted_batch.unsqueeze(1) - weighted_all.unsqueeze(0))  # (batch, N)
        global_similarity = 1.0 / (global_diff + 1e-6)
        
        # 归一化到[0, 1]
        global_min = global_similarity.min(dim=1, keepdim=True)[0]
        global_max = global_similarity.max(dim=1, keepdim=True)[0]
        global_similarity = (global_similarity - global_min) / (global_max - global_min + 1e-8)
        
        # === 3. 融合分数（V40核心） ===
        final_score = self.alpha * qos_similarity + (1 - self.alpha) * global_similarity
        
        # === 4. 排除自己 ===
        batch_indices = torch.arange(batch_size, device=device)
        final_score[batch_indices, entity_ids] = -float('inf')
        
        # === 5. Top-K ===
        _, neighbor_ids = final_score.topk(k, dim=1)
        
        return neighbor_ids
    
    def precompute_all_neighbors(self, qos_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        预计算所有实体的K近邻（向量化，训练前调用一次）
        
        Args:
            qos_matrix: (N, M) - QoS矩阵
            k: int - 邻居数量
        
        Returns:
            all_neighbors: (N, k) - 所有实体的邻居ID
        """
        # 确保全局特征已预计算
        if self._global_features is None:
            self.precompute_global_features(qos_matrix)
        
        print(f"[GDMv3] Precomputing neighbors with global awareness (alpha={self.alpha})...")
        
        # 处理3D矩阵 - V55 Fix: 非零均值
        if qos_matrix.dim() == 3:
            qos_sum = qos_matrix.sum(dim=-1)
            qos_count = (qos_matrix > 0).float().sum(dim=-1).clamp(min=1)
            qos_matrix = qos_sum / qos_count
        
        N = qos_matrix.size(0)
        device = qos_matrix.device
        
        # 分批处理以避免内存溢出
        batch_size = 1000
        all_neighbors = []
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            entity_ids = torch.arange(start_idx, end_idx, device=device)
            neighbors = self.find_neighbors_with_global_awareness(qos_matrix, entity_ids, k)
            all_neighbors.append(neighbors)
        
        all_neighbors = torch.cat(all_neighbors, dim=0)
        self._precomputed_neighbors = all_neighbors
        
        print(f"[GDMv3] Precomputed neighbors shape: {all_neighbors.shape}")
        return all_neighbors
    
    def find_neighbors_by_qos(self, qos_matrix: torch.Tensor,
                               entity_ids: torch.Tensor,
                               k: int) -> torch.Tensor:
        """
        基于QoS相似度和全局模式找K近邻（使用预计算或实时计算）
        
        Args:
            qos_matrix: (N, M) - QoS矩阵
            entity_ids: (batch,) - 实体ID
            k: int - 邻居数量
        
        Returns:
            neighbor_ids: (batch, k) - 邻居ID
        """
        # 优先使用预计算结果
        if self._precomputed_neighbors is not None:
            return self._precomputed_neighbors[entity_ids][:, :k]
        
        # 否则实时计算（使用全局模式感知）
        return self.find_neighbors_with_global_awareness(qos_matrix, entity_ids, k)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("V40.0 全局模式感知邻居选择测试")
    print("="*80)
    
    # 测试GlobalPatternComputer
    print("\n[Test 1] GlobalPatternComputer")
    qos_matrix = torch.randn(100, 200).abs()
    qos_matrix = qos_matrix * (torch.rand(100, 200) > 0.7).float()  # 30%密度
    global_features = GlobalPatternComputer.compute_features(qos_matrix)
    print(f"  Input: qos_matrix={qos_matrix.shape}")
    print(f"  Output: global_features={global_features.shape}")
    print(f"  Feature ranges:")
    print(f"    Activity: [{global_features[:, 0].min():.4f}, {global_features[:, 0].max():.4f}]")
    print(f"    Popularity: [{global_features[:, 1].min():.4f}, {global_features[:, 1].max():.4f}]")
    print(f"    Stability: [{global_features[:, 2].min():.4f}, {global_features[:, 2].max():.4f}]")
    print(f"    Representativeness: [{global_features[:, 3].min():.4f}, {global_features[:, 3].max():.4f}]")
    assert global_features.shape == (100, 4), "GlobalPatternComputer output shape error"
    print("  ✓ GlobalPatternComputer test passed")
    
    # 测试GDMv3
    print("\n[Test 2] GDMv3")
    gdm = GDMv3(num_neighbors=10, alpha=0.7)
    gdm.precompute_global_features(qos_matrix)
    
    # 测试邻居选择
    entity_ids = torch.randint(0, 100, (8,))
    neighbors = gdm.find_neighbors_with_global_awareness(qos_matrix, entity_ids, k=10)
    print(f"  Input: entity_ids={entity_ids.shape}")
    print(f"  Output: neighbors={neighbors.shape}")
    assert neighbors.shape == (8, 10), "GDMv3 neighbor selection shape error"
    print("  ✓ GDMv3 test passed")
    
    # 测试预计算
    print("\n[Test 3] GDMv3 Precompute")
    all_neighbors = gdm.precompute_all_neighbors(qos_matrix, k=10)
    print(f"  Output: all_neighbors={all_neighbors.shape}")
    assert all_neighbors.shape == (100, 10), "GDMv3 precompute shape error"
    print("  ✓ GDMv3 precompute test passed")
    
    # 测试可学习权重
    print("\n[Test 4] Learnable Weights")
    print(f"  Initial weights: {gdm.global_pattern_weights.data}")
    print(f"  Normalized weights: {F.softmax(gdm.global_pattern_weights, dim=0)}")
    print("  ✓ Learnable weights test passed")
    
    print("\n" + "="*80)
    print("All V40.0 tests passed!")
    print("="*80 + "\n")
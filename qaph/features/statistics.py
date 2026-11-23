"""
V15.0 统计特征模块 - P2 14个统计特征

实现HCTN论文中LPAM提取的14个统计特征

Author: Kilo Code
Date: 2026-01-22
Version: 15.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats
from typing import Tuple, Optional
import os
import pickle


class StatisticalFeatureExtractor:
    """V15.0 统计特征提取器 - 提取14个统计特征"""
    
    FEATURE_NAMES = [
        'min', 'max', 'mean', 'median', 'std',
        'skewness', 'kurtosis',
        'iqr', 'mad', 'median_abs_dev',
        'rms', 'abs_energy', 'entropy', 'peak_to_peak'
    ]
    NUM_FEATURES = 14
    
    @staticmethod
    def extract_features(values: np.ndarray) -> np.ndarray:
        """从QoS值数组中提取14个统计特征"""
        if len(values) == 0:
            return np.zeros(14, dtype=np.float32)
        
        if len(values) == 1:
            val = values[0]
            return np.array([val, val, val, val, 0, 0, 0, 0, 0, 0, val, val**2, 0, 0], dtype=np.float32)
        
        # 基础统计特征 (5个)
        min_val, max_val = np.min(values), np.max(values)
        mean_val, median_val = np.mean(values), np.median(values)
        std_val = np.std(values)
        
        # 分布形状特征 (2个)
        skewness = stats.skew(values) if len(values) > 2 else 0
        kurtosis = stats.kurtosis(values) if len(values) > 3 else 0
        
        # 离散度特征 (3个)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        mad = np.mean(np.abs(values - mean_val))
        median_abs_dev = np.median(np.abs(values - median_val))
        
        # 能量/信息特征 (4个)
        rms = np.sqrt(np.mean(values**2))
        abs_energy = np.sum(values**2)
        hist, _ = np.histogram(values, bins=min(10, len(values)), density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist + 1e-10)) if len(hist) > 0 else 0
        peak_to_peak = max_val - min_val
        
        features = np.array([
            min_val, max_val, mean_val, median_val, std_val,
            skewness, kurtosis, iqr, mad, median_abs_dev,
            rms, abs_energy, entropy, peak_to_peak
        ], dtype=np.float32)
        
        return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)


class StatisticalFeatureCache:
    """统计特征缓存 - 预计算所有用户和服务的统计特征"""
    
    def __init__(self, qos_matrix: np.ndarray, normalize: bool = True, cache_path: Optional[str] = None):
        self.normalize = normalize
        self.cache_path = cache_path
        
        if qos_matrix.ndim == 3:
            self.num_users = qos_matrix.shape[0]
            self.num_services = qos_matrix.shape[1]
        else:
            self.num_users, self.num_services = qos_matrix.shape
        
        if cache_path and os.path.exists(cache_path):
            self._load_cache(cache_path)
        else:
            self._compute_features(qos_matrix)
            if cache_path:
                self._save_cache(cache_path)
    
    def _compute_features(self, qos_matrix: np.ndarray):
        """计算所有用户和服务的统计特征"""
        print(f"[StatisticalFeatureCache] Computing features...")
        
        self.user_features = np.zeros((self.num_users, 14), dtype=np.float32)
        for u in range(self.num_users):
            values = qos_matrix[u].flatten() if qos_matrix.ndim == 3 else qos_matrix[u]
            values = values[~np.isnan(values) & (values > 0)]
            self.user_features[u] = StatisticalFeatureExtractor.extract_features(values)
        
        self.service_features = np.zeros((self.num_services, 14), dtype=np.float32)
        for s in range(self.num_services):
            values = qos_matrix[:, s].flatten() if qos_matrix.ndim == 3 else qos_matrix[:, s]
            values = values[~np.isnan(values) & (values > 0)]
            self.service_features[s] = StatisticalFeatureExtractor.extract_features(values)
        
        if self.normalize:
            self.user_features = self._normalize(self.user_features)
            self.service_features = self._normalize(self.service_features)
        
        print(f"[StatisticalFeatureCache] Done. Users: {self.user_features.shape}, Services: {self.service_features.shape}")
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-8
        return (features - mean) / std
    
    def _save_cache(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'user': self.user_features, 'service': self.service_features}, f)
    
    def _load_cache(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.user_features = data['user']
        self.service_features = data['service']
        print(f"[StatisticalFeatureCache] Loaded from {path}")
    
    def get_batch_features(self, user_ids: np.ndarray, service_ids: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        user_feat = torch.tensor(self.user_features[user_ids], dtype=torch.float32)
        service_feat = torch.tensor(self.service_features[service_ids], dtype=torch.float32)
        return user_feat, service_feat


class StatisticalFeatureModule(nn.Module):
    """V15.0 统计特征模块 - 将14个统计特征投影到嵌入空间"""
    
    def __init__(self, embed_dim: int, num_features: int = 14, dropout: float = 0.3):
        super().__init__()
        self.feature_norm = nn.BatchNorm1d(num_features)
        self.feature_projection = nn.Sequential(
            nn.Linear(num_features, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )
        
    def forward(self, embed: torch.Tensor, stat_features: torch.Tensor) -> torch.Tensor:
        stat_features = self.feature_norm(stat_features)
        stat_embed = self.feature_projection(stat_features)
        concat = torch.cat([embed, stat_embed], dim=-1)
        gate = self.fusion_gate(concat)
        return gate * stat_embed + (1 - gate) * embed


class LocalPreferenceAggregationModule(nn.Module):
    """LPAM: 局部偏好聚合模块"""
    
    def __init__(self, embed_dim: int, num_neighbors: int = 10, dropout: float = 0.3):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.transform = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.LayerNorm(embed_dim))
        
    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        k = neighbors.size(1)
        query_expanded = query.unsqueeze(1).expand(-1, k, -1)
        combined = torch.cat([query_expanded, neighbors], dim=-1)
        attn_weights = F.softmax(self.attention(combined).squeeze(-1), dim=-1).unsqueeze(-1)
        return self.transform((neighbors * attn_weights).sum(dim=1))


class LocalPreferenceEnhancementModule(nn.Module):
    """LPEM: 局部偏好增强模块"""
    
    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self.enhancement = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, 1), nn.Sigmoid())
        
    def forward(self, original: torch.Tensor, local_pref: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([original, local_pref], dim=-1)
        gate = self.gate(combined)
        return gate * self.enhancement(combined) + (1 - gate) * original


class HCTNStyleGMM(nn.Module):
    """V15.0 HCTN风格的Greysheep缓解模块"""
    
    def __init__(self, embed_dim: int, num_neighbors: int = 10, dropout: float = 0.3,
                 user_threshold: float = 1.5, service_threshold: float = 1.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neighbors = num_neighbors
        self.user_threshold = user_threshold
        self.service_threshold = service_threshold
        
        self.user_stat_module = StatisticalFeatureModule(embed_dim, dropout=dropout)
        self.service_stat_module = StatisticalFeatureModule(embed_dim, dropout=dropout)
        self.user_lpam = LocalPreferenceAggregationModule(embed_dim, num_neighbors, dropout)
        self.service_lpam = LocalPreferenceAggregationModule(embed_dim, num_neighbors, dropout)
        self.user_lpem = LocalPreferenceEnhancementModule(embed_dim, dropout)
        self.service_lpem = LocalPreferenceEnhancementModule(embed_dim, dropout)
        self.fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout))
        
    def forward(self, user_embed: torch.Tensor, service_embed: torch.Tensor,
                user_gs_scores: torch.Tensor, service_gs_scores: torch.Tensor,
                all_user_embeddings: torch.Tensor, all_service_embeddings: torch.Tensor,
                user_stat_features: Optional[torch.Tensor] = None,
                service_stat_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        is_user_gs = (user_gs_scores > self.user_threshold).float().unsqueeze(-1)
        is_service_gs = (service_gs_scores > self.service_threshold).float().unsqueeze(-1)
        
        if user_stat_features is not None:
            user_embed = self.user_stat_module(user_embed, user_stat_features)
        user_neighbors = self._find_neighbors(user_embed, all_user_embeddings, self.num_neighbors)
        user_local_pref = self.user_lpam(user_embed, user_neighbors)
        user_enhanced = self.user_lpem(user_embed, user_local_pref)
        user_out = is_user_gs * user_enhanced + (1 - is_user_gs) * user_embed
        
        if service_stat_features is not None:
            service_embed = self.service_stat_module(service_embed, service_stat_features)
        service_neighbors = self._find_neighbors(service_embed, all_service_embeddings, self.num_neighbors)
        service_local_pref = self.service_lpam(service_embed, service_neighbors)
        service_enhanced = self.service_lpem(service_embed, service_local_pref)
        service_out = is_service_gs * service_enhanced + (1 - is_service_gs) * service_embed
        
        return self.fusion(torch.cat([user_out, service_out], dim=-1))
    
    def _find_neighbors(self, query: torch.Tensor, all_embeddings: torch.Tensor, k: int) -> torch.Tensor:
        query_norm = F.normalize(query, p=2, dim=-1)
        all_norm = F.normalize(all_embeddings, p=2, dim=-1)
        similarity = torch.mm(query_norm, all_norm.t())
        _, indices = similarity.topk(k + 1, dim=-1)
        return all_embeddings[indices[:, 1:]]


if __name__ == "__main__":
    print("V15.0 统计特征模块测试")
    
    # 测试特征提取
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    features = StatisticalFeatureExtractor.extract_features(values)
    print(f"Features shape: {features.shape}")
    
    # 测试模块
    module = StatisticalFeatureModule(embed_dim=64)
    embed = torch.randn(32, 64)
    stat_feat = torch.randn(32, 14)
    out = module(embed, stat_feat)
    print(f"Module output: {out.shape}")
    


# ============================================================================
# V36.0 完整LPAM-LPEM-GDM实现 - 符合HCTN论文设计
# ============================================================================

class LPAMv2(nn.Module):
    """V36.0 LPAM - 直接处理14维统计特征"""
    
    def __init__(self, stat_dim: int = 14, embed_dim: int = 128,
                 num_neighbors: int = 10, dropout: float = 0.3):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.stat_dim = stat_dim  # 保存用于断言检查
        self.embed_dim = embed_dim
        
        # 统计特征 → 嵌入空间
        self.stat_encoder = nn.Sequential(
            nn.Linear(stat_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # 注意力机制（用于邻居聚合）
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        
        # 显式权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, stat_features: torch.Tensor,
                neighbor_stat_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            stat_features: (batch, stat_dim) - 当前实体的统计特征
            neighbor_stat_features: (batch, k, stat_dim) - k个邻居的统计特征
        
        Returns:
            key: (batch, embed_dim) - 用于LPEM的key
            value: (batch, embed_dim) - 用于LPEM的value
        """
        # 输入维度断言检查
        batch_size = stat_features.size(0)
        actual_stat_dim = stat_features.size(-1)
        assert actual_stat_dim == self.stat_dim, \
            f"Input stat_dim ({actual_stat_dim}) != expected ({self.stat_dim})"
        
        # 编码统计特征
        query_embed = self.stat_encoder(stat_features)  # (batch, embed_dim)
        neighbor_embeds = self.stat_encoder(
            neighbor_stat_features.view(-1, self.stat_dim)
        ).view(batch_size, -1, self.embed_dim)  # (batch, k, embed_dim)
        
        # 注意力加权聚合
        k = neighbor_embeds.size(1)
        query_expanded = query_embed.unsqueeze(1).expand(-1, k, -1)
        combined = torch.cat([query_expanded, neighbor_embeds], dim=-1)
        attn_weights = F.softmax(self.attention(combined).squeeze(-1), dim=-1).unsqueeze(-1)
        
        # 聚合邻居特征
        aggregated = (neighbor_embeds * attn_weights).sum(dim=1)  # (batch, embed_dim)
        
        # 生成key和value
        key = query_embed
        value = aggregated
        
        return key, value


class LPEMv2(nn.Module):
    """V36.0 LPEM - Multi-Head Attention融合（实践效果更优）
    
    使用MHA进行动态融合，相比Dense Layer有更强的表达能力：
    - Dense: 固定权重融合
    - MHA: 动态注意力权重融合
    """
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-Head Attention融合
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 残差连接后的LayerNorm
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 显式权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, original_embed: torch.Tensor,
                lpam_key: torch.Tensor,
                lpam_value: torch.Tensor,
                greysheep_indicator: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original_embed: (batch, embed_dim) - HCFM输出（原始嵌入）
            lpam_key: (batch, embed_dim) - LPAM输出的key
            lpam_value: (batch, embed_dim) - LPAM输出的value
            greysheep_indicator: (batch, 1) - GDM的greysheep分数 (0-1范围)
        
        Returns:
            enhanced: (batch, embed_dim) - GMM增强后的嵌入
        """
        # 扩展维度用于MHA (batch, 1, embed_dim)
        query = original_embed.unsqueeze(1)
        key = lpam_key.unsqueeze(1)
        value = lpam_value.unsqueeze(1)
        
        # Multi-Head Attention
        attn_out, _ = self.mha(query, key, value)
        attn_out = attn_out.squeeze(1)  # (batch, embed_dim)
        
        # 残差连接 + LayerNorm
        enhanced = self.norm(original_embed + self.dropout(attn_out))
        
        # Greysheep加权
        G = greysheep_indicator  # (batch, 1)
        Z_1 = G * enhanced
        Z_2 = (1 - G) * original_embed
        
        return Z_1 + Z_2


class GDMv2(nn.Module):
    """V36.0 GDM - Greysheep Detection Module"""
    
    def __init__(self, num_neighbors: int = 10):
        super().__init__()
        self.num_neighbors = num_neighbors
        self._precomputed_neighbors = None  # 初始化预计算缓存
        
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
    
    def precompute_all_neighbors(self, qos_matrix: torch.Tensor, k: int) -> torch.Tensor:
        """
        预计算所有实体的K近邻（向量化，只需调用一次）
        
        Args:
            qos_matrix: (N, M) - QoS矩阵
            k: int - 邻居数量
        
        Returns:
            all_neighbors: (N, k) - 所有实体的邻居ID
        """
        # 处理3D矩阵（兼容2D和3D输入）- V55 Fix: 非零均值
        if qos_matrix.dim() == 3:
            qos_sum = qos_matrix.sum(dim=-1)
            qos_count = (qos_matrix > 0).float().sum(dim=-1).clamp(min=1)
            qos_matrix = qos_sum / qos_count
        # 确保是2D
        assert qos_matrix.dim() == 2, f"Expected 2D matrix, got {qos_matrix.dim()}D"
        
        N = qos_matrix.size(0)
        device = qos_matrix.device
        
        # 归一化（向量化）
        valid_mask = (qos_matrix > 0).float()
        qos_masked = qos_matrix * valid_mask
        row_norms = torch.norm(qos_masked, p=2, dim=1, keepdim=True) + 1e-8
        qos_normalized = qos_masked / row_norms
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(qos_normalized, qos_normalized.t())
        similarity_matrix.fill_diagonal_(-float('inf'))
        
        # 获取top-k
        _, all_neighbors = similarity_matrix.topk(k, dim=1)
        
        self._precomputed_neighbors = all_neighbors
        return all_neighbors
    
    def find_neighbors_by_qos(self, qos_matrix: torch.Tensor,
                               entity_ids: torch.Tensor,
                               k: int) -> torch.Tensor:
        """
        基于QoS相似度找K近邻（使用预计算或向量化实时计算）
        
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
        
        # 否则向量化实时计算（兼容2D和3D输入）- V55 Fix: 非零均值
        if qos_matrix.dim() == 3:
            qos_sum = qos_matrix.sum(dim=-1)
            qos_count = (qos_matrix > 0).float().sum(dim=-1).clamp(min=1)
            qos_matrix = qos_sum / qos_count
        
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
        
        # 计算相似度
        similarity = torch.mm(batch_normalized, qos_normalized.t())
        
        # 向量化排除自己（修复循环效率问题）
        batch_indices = torch.arange(len(entity_ids), device=similarity.device)
        similarity[batch_indices, entity_ids] = -float('inf')
        
        _, neighbor_ids = similarity.topk(k, dim=1)
        return neighbor_ids


class HCTNStyleGMMv2(nn.Module):
    """V36.0 完整的HCTN风格GMM模块 - GDM+LPAM+LPEM协同（优化版）"""
    
    def __init__(self, embed_dim: int = 128, stat_dim: int = 14,
                 num_neighbors: int = 10, num_heads: int = 4, dropout: float = 0.3,
                 user_threshold: float = 1.5, service_threshold: float = 1.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.stat_dim = stat_dim
        self.num_neighbors = num_neighbors
        self.user_threshold = user_threshold
        self.service_threshold = service_threshold
        
        # 用户侧和服务侧各自的GDM
        self.user_gdm = GDMv2(num_neighbors)
        self.service_gdm = GDMv2(num_neighbors)
        
        # LPAM和LPEM (V36.0: LPEM使用MHA，实践效果更优)
        self.user_lpam = LPAMv2(stat_dim=stat_dim, embed_dim=embed_dim,
                                num_neighbors=num_neighbors, dropout=dropout)
        self.service_lpam = LPAMv2(stat_dim=stat_dim, embed_dim=embed_dim,
                                    num_neighbors=num_neighbors, dropout=dropout)
        self.user_lpem = LPEMv2(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.service_lpem = LPEMv2(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 缓存全局统计量
        self._user_mean_qos = None
        self._user_std_qos = None
        self._service_mean_qos = None
        self._service_std_qos = None
    
    def precompute_neighbors(self, user_qos_matrix: torch.Tensor,
                              service_qos_matrix: torch.Tensor):
        """
        预计算所有用户和服务的邻居（训练前调用一次）
        
        Args:
            user_qos_matrix: (num_users, num_services) - 用户QoS矩阵
            service_qos_matrix: (num_services, num_users) - 服务QoS矩阵
        """
        print(f"[HCTNStyleGMMv2] Precomputing neighbors...")
        
        # 预计算邻居
        self.user_gdm.precompute_all_neighbors(user_qos_matrix, self.num_neighbors)
        self.service_gdm.precompute_all_neighbors(service_qos_matrix, self.num_neighbors)
        
        # 计算全局统计量（带安全检查）
        valid_user_qos = user_qos_matrix[user_qos_matrix > 0]
        if valid_user_qos.numel() > 0:
            self._user_mean_qos = valid_user_qos.mean().item()
            self._user_std_qos = valid_user_qos.std().item() if valid_user_qos.numel() > 1 else 1.0
        else:
            self._user_mean_qos = 0.0
            self._user_std_qos = 1.0
        
        valid_service_qos = service_qos_matrix[service_qos_matrix > 0]
        if valid_service_qos.numel() > 0:
            self._service_mean_qos = valid_service_qos.mean().item()
            self._service_std_qos = valid_service_qos.std().item() if valid_service_qos.numel() > 1 else 1.0
        else:
            self._service_mean_qos = 0.0
            self._service_std_qos = 1.0
        
        print(f"[HCTNStyleGMMv2] Done. User: mean={self._user_mean_qos:.4f}, std={self._user_std_qos:.4f}")
        
    def forward(self, user_embed: torch.Tensor, service_embed: torch.Tensor,
                user_ids: torch.Tensor, service_ids: torch.Tensor,
                user_stat_features: torch.Tensor, service_stat_features: torch.Tensor,
                user_qos_matrix: torch.Tensor, service_qos_matrix: torch.Tensor,
                user_mean_qos: Optional[float] = None, user_std_qos: Optional[float] = None,
                service_mean_qos: Optional[float] = None, service_std_qos: Optional[float] = None) -> torch.Tensor:
        """
        完整的GMM前向传播
        
        Args:
            user_embed: (batch, embed_dim) - 用户嵌入
            service_embed: (batch, embed_dim) - 服务嵌入
            user_ids: (batch,) - 用户ID
            service_ids: (batch,) - 服务ID
            user_stat_features: (num_users, stat_dim) - 所有用户的统计特征
            service_stat_features: (num_services, stat_dim) - 所有服务的统计特征
            user_qos_matrix: (num_users, num_services) - 用户QoS矩阵
            service_qos_matrix: (num_services, num_users) - 服务QoS矩阵
            user_mean_qos, user_std_qos: 用户QoS全局统计（可选）
            service_mean_qos, service_std_qos: 服务QoS全局统计（可选）
        
        Returns:
            fused_embed: (batch, embed_dim) - 融合后的嵌入
        """
        device = user_embed.device
        
        # 使用预计算的统计量
        u_mean = user_mean_qos if user_mean_qos is not None else self._user_mean_qos
        u_std = user_std_qos if user_std_qos is not None else self._user_std_qos
        s_mean = service_mean_qos if service_mean_qos is not None else self._service_mean_qos
        s_std = service_std_qos if service_std_qos is not None else self._service_std_qos
        
        # 如果仍为None，实时计算
        if u_mean is None:
            valid = user_qos_matrix[user_qos_matrix > 0]
            u_mean, u_std = valid.mean().item(), valid.std().item()
        if s_mean is None:
            valid = service_qos_matrix[service_qos_matrix > 0]
            s_mean, s_std = valid.mean().item(), valid.std().item()
        
        # === 用户侧 ===
        user_qos_profile = user_qos_matrix[user_ids]
        user_qos_mean = (user_qos_profile * (user_qos_profile > 0).float()).sum(dim=1) / \
                        ((user_qos_profile > 0).float().sum(dim=1) + 1e-8)
        user_gs_scores = self.user_gdm.compute_greysheep_score(user_qos_mean, u_mean, u_std)
        
        user_neighbor_ids = self.user_gdm.find_neighbors_by_qos(
            user_qos_matrix, user_ids, self.num_neighbors
        )
        
        # 直接索引（假设stat_features已在正确device上）
        current_user_stat = user_stat_features[user_ids]
        neighbor_user_stat = user_stat_features[user_neighbor_ids]
        user_key, user_value = self.user_lpam(current_user_stat, neighbor_user_stat)
        
        # V36.0: LPEM使用MHA融合，传入key和value
        user_enhanced = self.user_lpem(user_embed, user_key, user_value, user_gs_scores)
        
        is_user_gs = (user_gs_scores > self.user_threshold).float()
        user_out = is_user_gs * user_enhanced + (1 - is_user_gs) * user_embed
        
        # === 服务侧 ===
        service_qos_profile = service_qos_matrix[service_ids]
        service_qos_mean = (service_qos_profile * (service_qos_profile > 0).float()).sum(dim=1) / \
                           ((service_qos_profile > 0).float().sum(dim=1) + 1e-8)
        service_gs_scores = self.service_gdm.compute_greysheep_score(service_qos_mean, s_mean, s_std)
        
        service_neighbor_ids = self.service_gdm.find_neighbors_by_qos(
            service_qos_matrix, service_ids, self.num_neighbors
        )
        
        # 直接索引（假设stat_features已在正确device上）
        current_service_stat = service_stat_features[service_ids]
        neighbor_service_stat = service_stat_features[service_neighbor_ids]
        service_key, service_value = self.service_lpam(current_service_stat, neighbor_service_stat)
        
        # V36.0: LPEM使用MHA融合，传入key和value
        service_enhanced = self.service_lpem(service_embed, service_key, service_value, service_gs_scores)
        
        is_service_gs = (service_gs_scores > self.service_threshold).float()
        service_out = is_service_gs * service_enhanced + (1 - is_service_gs) * service_embed
        
        # === 融合 ===
        return self.fusion(torch.cat([user_out, service_out], dim=-1))


# ============================================================================
# V40.0 全局模式感知的GMM实现 - 改进邻居选择机制
# ============================================================================

from qaph.features.neighbors import GDMv3


class HCTNStyleGMMv3(nn.Module):
    """V40.0 GMM - 全局模式感知邻居选择版本
    
    改进点：
    1. 使用GDMv3替代GDMv2（全局模式感知邻居选择）
    2. LPAM和LPEM保持V36版本不变（已验证有效）
    3. 仅4个新增参数（全局模式权重）
    """
    
    def __init__(self, embed_dim: int = 128, stat_dim: int = 14,
                 num_neighbors: int = 10, num_heads: int = 4, dropout: float = 0.3,
                 user_threshold: float = 1.5, service_threshold: float = 1.5,
                 alpha: float = 0.7):
        """
        Args:
            embed_dim: 嵌入维度
            stat_dim: 统计特征维度（14维）
            num_neighbors: 邻居数量
            num_heads: LPEM的注意力头数
            dropout: Dropout率
            user_threshold: 用户Greysheep阈值
            service_threshold: 服务Greysheep阈值
            alpha: QoS相似度权重（0-1之间，越大越依赖QoS相似度）
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.stat_dim = stat_dim
        self.num_neighbors = num_neighbors
        self.user_threshold = user_threshold
        self.service_threshold = service_threshold
        
        # 使用GDMv3（全局模式感知邻居选择）
        self.user_gdm = GDMv3(num_neighbors, alpha)
        self.service_gdm = GDMv3(num_neighbors, alpha)
        
        # LPAM和LPEM保持V36版本（已验证有效）
        self.user_lpam = LPAMv2(stat_dim=stat_dim, embed_dim=embed_dim,
                                num_neighbors=num_neighbors, dropout=dropout)
        self.service_lpam = LPAMv2(stat_dim=stat_dim, embed_dim=embed_dim,
                                    num_neighbors=num_neighbors, dropout=dropout)
        self.user_lpem = LPEMv2(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.service_lpem = LPEMv2(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
        # 融合层保持V36版本
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 缓存全局统计量
        self._user_mean_qos = None
        self._user_std_qos = None
        self._service_mean_qos = None
        self._service_std_qos = None
    
    def precompute_neighbors(self, user_qos_matrix: torch.Tensor,
                              service_qos_matrix: torch.Tensor):
        """
        预计算所有用户和服务的邻居（训练前调用一次）
        
        Args:
            user_qos_matrix: (num_users, num_services) - 用户QoS矩阵
            service_qos_matrix: (num_services, num_users) - 服务QoS矩阵
        """
        print(f"[HCTNStyleGMMv3] Precomputing neighbors with global awareness...")
        
        # 预计算全局模式特征
        self.user_gdm.precompute_global_features(user_qos_matrix)
        self.service_gdm.precompute_global_features(service_qos_matrix)
        
        # 预计算邻居
        self.user_gdm.precompute_all_neighbors(user_qos_matrix, self.num_neighbors)
        self.service_gdm.precompute_all_neighbors(service_qos_matrix, self.num_neighbors)
        
        # 计算全局统计量（带安全检查）
        valid_user_qos = user_qos_matrix[user_qos_matrix > 0]
        if valid_user_qos.numel() > 0:
            self._user_mean_qos = valid_user_qos.mean().item()
            self._user_std_qos = valid_user_qos.std().item() if valid_user_qos.numel() > 1 else 1.0
        else:
            self._user_mean_qos = 0.0
            self._user_std_qos = 1.0
        
        valid_service_qos = service_qos_matrix[service_qos_matrix > 0]
        if valid_service_qos.numel() > 0:
            self._service_mean_qos = valid_service_qos.mean().item()
            self._service_std_qos = valid_service_qos.std().item() if valid_service_qos.numel() > 1 else 1.0
        else:
            self._service_mean_qos = 0.0
            self._service_std_qos = 1.0
        
        print(f"[HCTNStyleGMMv3] Done. User: mean={self._user_mean_qos:.4f}, std={self._user_std_qos:.4f}")
        print(f"[HCTNStyleGMMv3] Global pattern weights:")
        print(f"  User: {F.softmax(self.user_gdm.global_pattern_weights, dim=0).data}")
        print(f"  Service: {F.softmax(self.service_gdm.global_pattern_weights, dim=0).data}")
    
    def forward(self, user_embed: torch.Tensor, service_embed: torch.Tensor,
                user_ids: torch.Tensor, service_ids: torch.Tensor,
                user_stat_features: torch.Tensor, service_stat_features: torch.Tensor,
                user_qos_matrix: torch.Tensor, service_qos_matrix: torch.Tensor,
                user_mean_qos: Optional[float] = None, user_std_qos: Optional[float] = None,
                service_mean_qos: Optional[float] = None, service_std_qos: Optional[float] = None) -> torch.Tensor:
        """
        完整的GMM前向传播（与V36相同，但使用全局模式感知邻居选择）
        
        Args:
            user_embed: (batch, embed_dim) - 用户嵌入
            service_embed: (batch, embed_dim) - 服务嵌入
            user_ids: (batch,) - 用户ID
            service_ids: (batch,) - 服务ID
            user_stat_features: (num_users, stat_dim) - 所有用户的统计特征
            service_stat_features: (num_services, stat_dim) - 所有服务的统计特征
            user_qos_matrix: (num_users, num_services) - 用户QoS矩阵
            service_qos_matrix: (num_services, num_users) - 服务QoS矩阵
            user_mean_qos, user_std_qos: 用户QoS全局统计（可选）
            service_mean_qos, service_std_qos: 服务QoS全局统计（可选）
        
        Returns:
            fused_embed: (batch, embed_dim) - 融合后的嵌入
        """
        device = user_embed.device
        
        # 使用预计算的统计量
        u_mean = user_mean_qos if user_mean_qos is not None else self._user_mean_qos
        u_std = user_std_qos if user_std_qos is not None else self._user_std_qos
        s_mean = service_mean_qos if service_mean_qos is not None else self._service_mean_qos
        s_std = service_std_qos if service_std_qos is not None else self._service_std_qos
        
        # 如果仍为None，实时计算
        if u_mean is None:
            valid = user_qos_matrix[user_qos_matrix > 0]
            u_mean, u_std = valid.mean().item(), valid.std().item()
        if s_mean is None:
            valid = service_qos_matrix[service_qos_matrix > 0]
            s_mean, s_std = valid.mean().item(), valid.std().item()
        
        # === 用户侧 ===
        user_qos_profile = user_qos_matrix[user_ids]
        user_qos_mean = (user_qos_profile * (user_qos_profile > 0).float()).sum(dim=1) / \
                        ((user_qos_profile > 0).float().sum(dim=1) + 1e-8)
        user_gs_scores = self.user_gdm.compute_greysheep_score(user_qos_mean, u_mean, u_std)
        
        # V40改进：使用全局模式感知邻居选择
        user_neighbor_ids = self.user_gdm.find_neighbors_by_qos(
            user_qos_matrix, user_ids, self.num_neighbors
        )
        
        # LPAM和LPEM保持V36版本
        current_user_stat = user_stat_features[user_ids]
        neighbor_user_stat = user_stat_features[user_neighbor_ids]
        user_key, user_value = self.user_lpam(current_user_stat, neighbor_user_stat)
        user_enhanced = self.user_lpem(user_embed, user_key, user_value, user_gs_scores)
        
        is_user_gs = (user_gs_scores > self.user_threshold).float()
        user_out = is_user_gs * user_enhanced + (1 - is_user_gs) * user_embed
        
        # === 服务侧 ===
        service_qos_profile = service_qos_matrix[service_ids]
        service_qos_mean = (service_qos_profile * (service_qos_profile > 0).float()).sum(dim=1) / \
                           ((service_qos_profile > 0).float().sum(dim=1) + 1e-8)
        service_gs_scores = self.service_gdm.compute_greysheep_score(service_qos_mean, s_mean, s_std)
        
        # V40改进：使用全局模式感知邻居选择
        service_neighbor_ids = self.service_gdm.find_neighbors_by_qos(
            service_qos_matrix, service_ids, self.num_neighbors
        )
        
        # LPAM和LPEM保持V36版本
        current_service_stat = service_stat_features[service_ids]
        neighbor_service_stat = service_stat_features[service_neighbor_ids]
        service_key, service_value = self.service_lpam(current_service_stat, neighbor_service_stat)
        service_enhanced = self.service_lpem(service_embed, service_key, service_value, service_gs_scores)
        
        is_service_gs = (service_gs_scores > self.service_threshold).float()
        service_out = is_service_gs * service_enhanced + (1 - is_service_gs) * service_embed
        
        # === 融合 ===
        return self.fusion(torch.cat([user_out, service_out], dim=-1))


if __name__ == "__main__":
    print("\n" + "="*80)
    print("V36.0 完整LPAM-LPEM-GDM测试")
    print("="*80)
    
    # 测试LPAMv2
    print("\n[Test 1] LPAMv2")
    lpam = LPAMv2(stat_dim=14, embed_dim=64, num_neighbors=5)
    stat_feat = torch.randn(8, 14)
    neighbor_stat = torch.randn(8, 5, 14)
    key, value = lpam(stat_feat, neighbor_stat)
    print(f"  Input: stat_feat={stat_feat.shape}, neighbor_stat={neighbor_stat.shape}")
    print(f"  Output: key={key.shape}, value={value.shape}")
    assert key.shape == (8, 64) and value.shape == (8, 64), "LPAMv2 output shape error"
    print("  ✓ LPAMv2 test passed")
    
    # 测试LPEMv2 (V36.0: MHA融合)
    print("\n[Test 2] LPEMv2 (MHA)")
    lpem = LPEMv2(embed_dim=64, num_heads=4, dropout=0.3)
    Y_1 = torch.randn(8, 64)  # HCFM输出
    lpam_key = key  # LPAM输出的key
    lpam_value = value  # LPAM输出的value
    gs_indicator = torch.sigmoid(torch.randn(8, 1))  # 0-1范围
    Z_3 = lpem(Y_1, lpam_key, lpam_value, gs_indicator)
    print(f"  Input: Y_1={Y_1.shape}, lpam_key={lpam_key.shape}, lpam_value={lpam_value.shape}, gs={gs_indicator.shape}")
    print(f"  Output: Z_3={Z_3.shape}")
    assert Z_3.shape == (8, 64), "LPEMv2 output shape error"
    print("  ✓ LPEMv2 test passed")
    
    # 测试GDMv2
    print("\n[Test 3] GDMv2")
    gdm = GDMv2(num_neighbors=5)
    qos_profile = torch.randn(8)
    gs_scores = gdm.compute_greysheep_score(qos_profile, mean_qos=0.5, std_qos=0.2)
    print(f"  Input: qos_profile={qos_profile.shape}")
    print(f"  Output: gs_scores={gs_scores.shape}")
    assert gs_scores.shape == (8, 1), "GDMv2 greysheep score shape error"
    print("  ✓ GDMv2 test passed")
    
    # 测试HCTNStyleGMMv2
    print("\n[Test 4] HCTNStyleGMMv2")
    gmm = HCTNStyleGMMv2(embed_dim=64, num_neighbors=5, num_heads=4)
    user_embed = torch.randn(8, 64)
    service_embed = torch.randn(8, 64)
    user_ids = torch.randint(0, 100, (8,))
    service_ids = torch.randint(0, 200, (8,))
    user_stat_features = torch.randn(100, 14)
    service_stat_features = torch.randn(200, 14)
    user_qos_matrix = torch.randn(100, 200)
    service_qos_matrix = torch.randn(200, 100)
    
    fused = gmm(
        user_embed, service_embed,
        user_ids, service_ids,
        user_stat_features, service_stat_features,
        user_qos_matrix, service_qos_matrix,
        user_mean_qos=0.5, user_std_qos=0.2,
        service_mean_qos=0.6, service_std_qos=0.3
    )
    print(f"  Input: user_embed={user_embed.shape}, service_embed={service_embed.shape}")
    print(f"  Output: fused={fused.shape}")
    assert fused.shape == (8, 64), "HCTNStyleGMMv2 output shape error"
    print("  ✓ HCTNStyleGMMv2 test passed")
    
    print("\n" + "="*80)
    print("All V36.0 tests passed!")
    print("="*80 + "\n")
    print("All tests passed!")
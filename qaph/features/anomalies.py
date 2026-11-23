"""
v6.9 异常处理 (Isolation Forest + Greysheep Detection)

参考: HCTN论文的异常值处理方法
- Isolation Forest: 检测异常的QoS值
- Greysheep Detection: 检测行为异常的用户和服务
- GDI (Greysheep Detection Index): 量化异常程度

目标:
- RT MAE: 0.80 → 0.70 (-12.5%)
- TP MAE: 3.20 → 2.50 (-22%)

Author: Kilo Code
Version: v6.9
Date: 2026-01-10
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, Dict, List
import warnings


class QoSAnomalyDetector:
    """QoS异常值检测器 - 使用Isolation Forest检测异常的用户-服务交互"""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Args:
            contamination: 异常值比例 (λ参数)，HCTN论文使用0.1
            random_state: 随机种子
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.is_fitted = False
        
    def fit(self, qos_matrix: np.ndarray):
        """训练异常检测模型"""
        valid_mask = ~np.isnan(qos_matrix)
        valid_values = qos_matrix[valid_mask].reshape(-1, 1)
        valid_values_scaled = self.scaler.fit_transform(valid_values)
        self.model.fit(valid_values_scaled)
        self.threshold = np.percentile(
            self.model.score_samples(valid_values_scaled),
            self.contamination * 100
        )
        self.is_fitted = True
        
    def predict(self, qos_values: np.ndarray) -> np.ndarray:
        """预测是否为异常值，返回布尔数组"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        values = np.array(qos_values).reshape(-1, 1)
        values_scaled = self.scaler.transform(values)
        scores = self.model.score_samples(values_scaled)
        return scores < self.threshold
    
    def get_anomaly_scores(self, qos_values: np.ndarray) -> np.ndarray:
        """获取异常分数（越低越异常）"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        values = np.array(qos_values).reshape(-1, 1)
        values_scaled = self.scaler.transform(values)
        return self.model.score_samples(values_scaled)
    
    def get_sample_weights(self, qos_values: np.ndarray) -> np.ndarray:
        """获取样本权重（异常样本权重低）"""
        scores = self.get_anomaly_scores(qos_values)
        # 将分数转换为0-1的权重
        min_score, max_score = scores.min(), scores.max()
        weights = (scores - min_score) / (max_score - min_score + 1e-8)
        return weights


class GreysheepDetector:
    """
    Greysheep检测器 - 检测行为异常的用户和服务
    
    HCTN论文公式 (Eq. 3):
    G(i,1,t) = 1 if G^t(u_i) > μ^t_gu + c_1 * σ^t_gu
    
    其中c_1 = c_2 = 1.0是最优值（论文Table 3）
    """
    
    def __init__(self, c: float = 1.0, use_dynamic_threshold: bool = True):
        """
        Args:
            c: HCTN论文中的c参数，用于动态阈值计算
               threshold = mean(GDI) + c * std(GDI)
               HCTN论文Table 3显示c=1.0是最优值
            use_dynamic_threshold: 是否使用动态阈值（HCTN论文方法）
        """
        self.c = c
        self.use_dynamic_threshold = use_dynamic_threshold
        self.user_gdi = None
        self.service_gdi = None
        self.global_mean = None
        self.global_std = None
        self.user_threshold = None
        self.service_threshold = None
        
    def fit(self, qos_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算所有用户和服务的GDI (Greysheep Detection Index)
        
        HCTN论文 Eq. 2:
        G^t(u_i) = (1/|S_i^t|) Σ |Q^t(i,j) - μ^t_j| / σ^t_j
        
        Args:
            qos_matrix: [num_users, num_services] QoS矩阵
        
        Returns:
            user_gdi: [num_users] 用户GDI分数
            service_gdi: [num_services] 服务GDI分数
        """
        self.global_mean = np.nanmean(qos_matrix)
        self.global_std = np.nanstd(qos_matrix)
        
        # 计算每个服务的均值和标准差（用于用户GDI）
        service_means = np.nanmean(qos_matrix, axis=0)
        service_stds = np.nanstd(qos_matrix, axis=0)
        service_stds = np.where(service_stds < 1e-8, 1e-8, service_stds)
        
        # 计算每个用户的均值和标准差（用于服务GDI）
        user_means = np.nanmean(qos_matrix, axis=1)
        user_stds = np.nanstd(qos_matrix, axis=1)
        user_stds = np.where(user_stds < 1e-8, 1e-8, user_stds)
        
        # 计算用户GDI (HCTN Eq. 2)
        num_users = qos_matrix.shape[0]
        self.user_gdi = np.zeros(num_users, dtype=np.float32)
        for u in range(num_users):
            user_ratings = qos_matrix[u, :]
            valid_mask = ~np.isnan(user_ratings)
            if valid_mask.sum() > 0:
                valid_ratings = user_ratings[valid_mask]
                valid_service_means = service_means[valid_mask]
                valid_service_stds = service_stds[valid_mask]
                # 标准化偏差
                normalized_deviation = np.abs(valid_ratings - valid_service_means) / valid_service_stds
                self.user_gdi[u] = normalized_deviation.mean()
        
        # 计算服务GDI
        num_services = qos_matrix.shape[1]
        self.service_gdi = np.zeros(num_services, dtype=np.float32)
        for s in range(num_services):
            service_ratings = qos_matrix[:, s]
            valid_mask = ~np.isnan(service_ratings)
            if valid_mask.sum() > 0:
                valid_ratings = service_ratings[valid_mask]
                valid_user_means = user_means[valid_mask]
                valid_user_stds = user_stds[valid_mask]
                # 标准化偏差
                normalized_deviation = np.abs(valid_ratings - valid_user_means) / valid_user_stds
                self.service_gdi[s] = normalized_deviation.mean()
        
        # 计算动态阈值 (HCTN Eq. 3)
        if self.use_dynamic_threshold:
            user_gdi_mean = np.mean(self.user_gdi)
            user_gdi_std = np.std(self.user_gdi)
            self.user_threshold = user_gdi_mean + self.c * user_gdi_std
            
            service_gdi_mean = np.mean(self.service_gdi)
            service_gdi_std = np.std(self.service_gdi)
            self.service_threshold = service_gdi_mean + self.c * service_gdi_std
        else:
            # 使用固定阈值（向后兼容）
            self.user_threshold = self.c
            self.service_threshold = self.c
        
        return self.user_gdi, self.service_gdi
    
    def is_greysheep_user(self, user_id: int) -> bool:
        """判断用户是否为Greysheep"""
        return self.user_gdi[user_id] > self.user_threshold
    
    def is_greysheep_service(self, service_id: int) -> bool:
        """判断服务是否为Greysheep"""
        return self.service_gdi[service_id] > self.service_threshold
    
    def get_greysheep_users(self) -> np.ndarray:
        """获取所有Greysheep用户的ID"""
        return np.where(self.user_gdi > self.user_threshold)[0]
    
    def get_greysheep_services(self) -> np.ndarray:
        """获取所有Greysheep服务的ID"""
        return np.where(self.service_gdi > self.service_threshold)[0]
    
    def get_greysheep_ratio(self) -> Tuple[float, float]:
        """获取Greysheep用户和服务的比例"""
        user_ratio = len(self.get_greysheep_users()) / len(self.user_gdi)
        service_ratio = len(self.get_greysheep_services()) / len(self.service_gdi)
        return user_ratio, service_ratio


class LocalPreferenceAggregationModule(nn.Module):
    """LPAM: 局部偏好聚合模块 - 为Greysheep用户/服务聚合局部偏好"""
    
    def __init__(self, embed_dim: int, num_neighbors: int = 10, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neighbors = num_neighbors
        
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 1)
        )
        self.transform = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, embed_dim] 查询实体的嵌入
            neighbors: [batch, num_neighbors, embed_dim] 邻居实体的嵌入
        Returns:
            aggregated: [batch, embed_dim] 聚合后的局部偏好
        """
        batch_size = query.size(0)
        query_expanded = query.unsqueeze(1).expand(-1, neighbors.size(1), -1)
        combined = torch.cat([query_expanded, neighbors], dim=-1)
        attn_scores = self.attention(combined).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        aggregated = torch.bmm(attn_weights.unsqueeze(1), neighbors).squeeze(1)
        return self.transform(aggregated)


class LocalPreferenceEnhancementModule(nn.Module):
    """LPEM: 局部偏好增强模块 - 增强Greysheep的特征表示"""
    
    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super().__init__()
        self.enhancement = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, original: torch.Tensor, local_pref: torch.Tensor) -> torch.Tensor:
        """
        Args:
            original: [batch, embed_dim] 原始特征
            local_pref: [batch, embed_dim] 局部偏好特征
        Returns:
            enhanced: [batch, embed_dim] 增强后的特征
        """
        combined = torch.cat([original, local_pref], dim=-1)
        gate = self.gate(combined)
        enhanced = self.enhancement(combined)
        return gate * enhanced + (1 - gate) * original


class GreysheepMitigationModule(nn.Module):
    """GMM: Greysheep缓解模块 - 完整的Greysheep处理流程
    
    支持两种调用方式:
    1. forward(features, neighbors, is_greysheep) - 原始接口
    2. forward(user_embed, service_embed, user_gs_scores, service_gs_scores,
               all_user_embeddings, all_service_embeddings) - v7.0统一模型接口
    
    v7.24更新: 使用HCTN论文的动态阈值机制
    """
    
    def __init__(self, embed_dim: int, num_neighbors: int = 10, dropout: float = 0.3,
                 greysheep_c: float = 1.0,  # v7.24: HCTN论文最优c值
                 user_threshold: float = None,  # 动态阈值（由GreysheepDetector计算）
                 service_threshold: float = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_neighbors = num_neighbors
        self.greysheep_c = greysheep_c
        # 如果没有提供动态阈值，使用默认值（向后兼容）
        self.user_threshold = user_threshold if user_threshold is not None else 1.5
        self.service_threshold = service_threshold if service_threshold is not None else 1.5
        
        # 用户和服务的LPAM
        self.user_lpam = LocalPreferenceAggregationModule(embed_dim, num_neighbors, dropout)
        self.service_lpam = LocalPreferenceAggregationModule(embed_dim, num_neighbors, dropout)
        
        # 用户和服务的LPEM
        self.user_lpem = LocalPreferenceEnhancementModule(embed_dim, dropout)
        self.service_lpem = LocalPreferenceEnhancementModule(embed_dim, dropout)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 兼容旧接口
        self.lpam = self.user_lpam
        self.lpem = self.user_lpem
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        支持两种调用方式的前向传播
        """
        if len(args) == 3:
            # 原始接口: (features, neighbors, is_greysheep)
            return self._forward_legacy(*args)
        elif len(args) == 6:
            # v7.0接口: (user_embed, service_embed, user_gs_scores, service_gs_scores,
            #           all_user_embeddings, all_service_embeddings)
            return self._forward_v7(*args)
        else:
            raise ValueError(f"Expected 3 or 6 arguments, got {len(args)}")
    
    def _forward_legacy(self, features: torch.Tensor, neighbors: torch.Tensor,
                        is_greysheep: torch.Tensor) -> torch.Tensor:
        """
        原始接口
        
        Args:
            features: [batch, embed_dim] 实体特征
            neighbors: [batch, num_neighbors, embed_dim] 邻居特征
            is_greysheep: [batch] 布尔张量，标记是否为Greysheep
        Returns:
            output: [batch, embed_dim] 处理后的特征
        """
        local_pref = self.lpam(features, neighbors)
        enhanced = self.lpem(features, local_pref)
        # 只对Greysheep应用增强
        is_gs = is_greysheep.float().unsqueeze(-1)
        return is_gs * enhanced + (1 - is_gs) * features
    
    def _forward_v7(
        self,
        user_embed: torch.Tensor,
        service_embed: torch.Tensor,
        user_gs_scores: torch.Tensor,
        service_gs_scores: torch.Tensor,
        all_user_embeddings: torch.Tensor,
        all_service_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        v7.0统一模型接口
        
        Args:
            user_embed: [batch, embed_dim] 当前batch的用户嵌入
            service_embed: [batch, embed_dim] 当前batch的服务嵌入
            user_gs_scores: [batch] 用户Greysheep分数
            service_gs_scores: [batch] 服务Greysheep分数
            all_user_embeddings: [num_users, embed_dim] 所有用户嵌入
            all_service_embeddings: [num_services, embed_dim] 所有服务嵌入
        Returns:
            output: [batch, embed_dim] 处理后的特征
        """
        batch_size = user_embed.size(0)
        device = user_embed.device
        
        # 判断是否为Greysheep（使用动态阈值）
        is_user_gs = (user_gs_scores > self.user_threshold).float().unsqueeze(-1)
        is_service_gs = (service_gs_scores > self.service_threshold).float().unsqueeze(-1)
        
        # 为用户找邻居（基于嵌入相似度）
        user_neighbors = self._find_neighbors(
            user_embed, all_user_embeddings, self.num_neighbors
        )  # [batch, num_neighbors, embed_dim]
        
        # 为服务找邻居
        service_neighbors = self._find_neighbors(
            service_embed, all_service_embeddings, self.num_neighbors
        )  # [batch, num_neighbors, embed_dim]
        
        # 用户LPAM + LPEM
        user_local_pref = self.user_lpam(user_embed, user_neighbors)
        user_enhanced = self.user_lpem(user_embed, user_local_pref)
        user_out = is_user_gs * user_enhanced + (1 - is_user_gs) * user_embed
        
        # 服务LPAM + LPEM
        service_local_pref = self.service_lpam(service_embed, service_neighbors)
        service_enhanced = self.service_lpem(service_embed, service_local_pref)
        service_out = is_service_gs * service_enhanced + (1 - is_service_gs) * service_embed
        
        # 融合用户和服务特征
        combined = torch.cat([user_out, service_out], dim=-1)
        output = self.fusion(combined)
        
        return output
    
    def _find_neighbors(
        self,
        query: torch.Tensor,
        all_embeddings: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        基于嵌入相似度找k个最近邻
        
        Args:
            query: [batch, embed_dim] 查询嵌入
            all_embeddings: [num_entities, embed_dim] 所有实体嵌入
            k: 邻居数量
        Returns:
            neighbors: [batch, k, embed_dim] 邻居嵌入
        """
        batch_size = query.size(0)
        
        # 计算相似度
        # query: [batch, embed_dim], all_embeddings: [num_entities, embed_dim]
        similarity = torch.mm(query, all_embeddings.t())  # [batch, num_entities]
        
        # 找top-k邻居（排除自己，通过设置对角线为负无穷）
        # 注意：这里简化处理，实际上应该排除query对应的实体
        _, indices = similarity.topk(k + 1, dim=-1)  # [batch, k+1]
        indices = indices[:, 1:]  # 排除最相似的（可能是自己）
        
        # 获取邻居嵌入
        neighbors = all_embeddings[indices]  # [batch, k, embed_dim]
        
        return neighbors


class AnomalyAwareTrainer:
    """异常感知训练器 - 在训练中考虑异常值"""
    
    def __init__(self, anomaly_detector: QoSAnomalyDetector, 
                 greysheep_detector: GreysheepDetector,
                 anomaly_weight: float = 0.5):
        """
        Args:
            anomaly_detector: QoS异常检测器
            greysheep_detector: Greysheep检测器
            anomaly_weight: 异常样本的权重（0-1）
        """
        self.anomaly_detector = anomaly_detector
        self.greysheep_detector = greysheep_detector
        self.anomaly_weight = anomaly_weight
        
    def get_sample_weights(self, user_ids: np.ndarray, service_ids: np.ndarray,
                          qos_values: np.ndarray) -> np.ndarray:
        """获取样本权重"""
        weights = np.ones(len(qos_values))
        
        # 基于QoS异常的权重
        if self.anomaly_detector.is_fitted:
            anomaly_weights = self.anomaly_detector.get_sample_weights(qos_values)
            weights *= anomaly_weights
        
        # 基于Greysheep的权重
        if self.greysheep_detector.user_gdi is not None:
            for i, (u, s) in enumerate(zip(user_ids, service_ids)):
                if self.greysheep_detector.is_greysheep_user(u):
                    weights[i] *= self.anomaly_weight
                if self.greysheep_detector.is_greysheep_service(s):
                    weights[i] *= self.anomaly_weight
        
        return weights


    def set_thresholds(self, user_threshold: float, service_threshold: float):
        """设置动态阈值（由GreysheepDetector计算后传入）"""
        self.user_threshold = user_threshold
        self.service_threshold = service_threshold


if __name__ == "__main__":
    print("v7.24 异常处理模块测试 (HCTN合规)")
    
    # 测试数据
    qos_matrix = np.random.rand(100, 200).astype(np.float32)
    qos_matrix[qos_matrix < 0.1] = np.nan  # 添加缺失值
    
    # 测试QoS异常检测器
    detector = QoSAnomalyDetector(contamination=0.1)
    detector.fit(qos_matrix)
    test_values = np.array([0.5, 0.1, 0.9, 10.0, -5.0])
    is_anomaly = detector.predict(test_values)
    print(f"异常检测: {is_anomaly}")
    
    # 测试Greysheep检测器 (HCTN风格，c=1.0)
    gs_detector = GreysheepDetector(c=1.0, use_dynamic_threshold=True)
    gs_detector.fit(qos_matrix)
    gs_users = gs_detector.get_greysheep_users()
    gs_services = gs_detector.get_greysheep_services()
    user_ratio, service_ratio = gs_detector.get_greysheep_ratio()
    print(f"Greysheep用户数: {len(gs_users)} ({user_ratio*100:.1f}%)")
    print(f"Greysheep服务数: {len(gs_services)} ({service_ratio*100:.1f}%)")
    print(f"用户阈值: {gs_detector.user_threshold:.4f}")
    print(f"服务阈值: {gs_detector.service_threshold:.4f}")
    
    # 测试GMM模块（使用动态阈值）
    gmm = GreysheepMitigationModule(
        embed_dim=128,
        user_threshold=gs_detector.user_threshold,
        service_threshold=gs_detector.service_threshold
    )
    features = torch.randn(32, 128)
    neighbors = torch.randn(32, 10, 128)
    is_gs = torch.randint(0, 2, (32,)).bool()
    output = gmm(features, neighbors, is_gs)
    print(f"GMM输出: {output.shape}")
    
    print("\n测试通过!")
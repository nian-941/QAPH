"""Frozen-v0 QAPH implementation, organized by responsibility. Original source credit: Kilo Code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple

class MoEPredictionHead(nn.Module):
    """
    V56 RAPP: Range-Adaptive Partitioned Prediction via Mixture of Experts
    
    多专家预测头，每个专家擅长不同数量级的QoS值预测。
    软路由器根据user+service表示自动分配样本到对应专家。
    
    学术依据:
    - AMEND (Mercurius et al., 2024): MoE处理长尾轨迹预测
    - Moirai-MoE (Liu et al., 2024, cited 107): 稀疏MoE时序基础模型
    - "A time series is worth five experts" (Wang et al., 2024)
    """
    
    def __init__(self, embed_dim: int = 64, num_experts: int = 3, dropout: float = 0.2):
        super().__init__()
        self.num_experts = num_experts
        
        # Router: 基于user+service表示决定专家权重
        self.router = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_experts),
        )
        
        # 每个专家是独立的MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 1)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, user_repr: torch.Tensor, service_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            user_repr: [batch, embed_dim]
            service_repr: [batch, embed_dim]
        Returns:
            dict with 'prediction' [batch, 1] and 'router_weights' [batch, num_experts]
        """
        combined = torch.cat([user_repr, service_repr], dim=-1)  # [batch, 2d]
        
        # Router logits → softmax weights
        router_logits = self.router(combined)  # [batch, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)  # [batch, num_experts]
        
        # 每个专家的预测
        expert_outputs = torch.stack([
            expert(combined) for expert in self.experts
        ], dim=1)  # [batch, num_experts, 1]
        
        # 加权融合
        pred = (router_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [batch, 1]
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        
        return {'prediction': pred, 'router_weights': router_weights}
    
    def load_balance_loss(self, router_weights: torch.Tensor) -> torch.Tensor:
        """
        负载均衡损失：防止所有样本路由到同一专家
        
        L_balance = N × Σ(f_k × p_k)
        其中 f_k = 分配到专家k的样本比例, p_k = 路由到专家k的平均概率
        """
        # 平均路由概率
        avg_weights = router_weights.mean(dim=0)  # [num_experts]
        # 负载均衡: 鼓励均匀分布
        balance_loss = self.num_experts * (avg_weights * avg_weights).sum()
        return balance_loss


class HAPHead(nn.Module):
    """
    V56 HAP: History-Anchored Prediction
    
    模型预测相对于历史锚点的缩放因子和偏移量：
        pred = anchor × exp(log_scale) + offset
    
    其中anchor = 该(user, service)对的历史非零均值（预计算）。
    
    学术依据:
    - AnIO (Stentoumi et al., Neural Computing and Applications, 2024):
      锚定输入输出学习，MAPE改善22.7%
    - TRCF (Zou et al., IEEE TSC 2023):
      时序平均QoS偏差迁移
    
    数据支撑 (WS-DREAM2 TP, density=10%):
    - 85.1%样本的target/anchor ratio ∈ [0.5, 2.0)
    - ratio中位数 = 1.035, P99 = 2.85
    - anchor对[1, ∞)范围改善69-73% MAE
    """
    
    def __init__(self, embed_dim: int = 64, dropout: float = 0.2,
                 log_scale_min: float = -1.5, log_scale_max: float = 2.0):
        super().__init__()
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        
        # Scale head: 预测log_scale，exp后得到乘性因子
        self.scale_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
        # Offset head: 预测加性偏移
        self.offset_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
        # 初始化：使scale_head输出接近0（exp(0)=1），offset_head输出接近0
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
    
    def forward(self, user_repr: torch.Tensor, service_repr: torch.Tensor,
                anchor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_repr: [batch, embed_dim]
            service_repr: [batch, embed_dim]
            anchor: [batch] 历史非零均值
        Returns:
            prediction: [batch, 1]
        """
        combined = torch.cat([user_repr, service_repr], dim=-1)  # [batch, 2d]
        
        # Scale: exp(clamp(log_scale))
        log_scale = self.scale_head(combined)  # [batch, 1]
        log_scale = log_scale.clamp(self.log_scale_min, self.log_scale_max)
        scale = torch.exp(log_scale)  # [batch, 1], range [exp(-2), exp(3)] ≈ [0.135, 20.09]
        
        # Offset
        offset = self.offset_head(combined)  # [batch, 1]
        
        # Anchored prediction
        anchor_2d = anchor.unsqueeze(-1)  # [batch, 1]
        pred = anchor_2d * scale + offset
        
        # 数值安全
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        
        return pred


class QAPHHead(nn.Module):
    """V65/V69 QAPH: QoS-Adaptive Prediction Head
    
    V69改进:
    - gate_net输出NaN guard（防止GCN极端值导致gate=nan传播）
    - gate_init_bias可配置（TP用-2.0更偏向HAP，RT用-1.0更偏向CAPH）
    - combined特征NaN guard（在gate_net输入前清理）
    """
    
    def __init__(self, embed_dim: int = 64, bilinear_rank: int = 16, dropout: float = 0.2,
                 hap_log_scale_min: float = -2.0, hap_log_scale_max: float = 3.0,
                 gate_init_bias: float = -1.0, anchor_usage: str = "hap",
                 qaph_gate_mode: str = "learnable"):
        super().__init__()
        self.embed_dim = embed_dim
        self.hap_log_scale_min = hap_log_scale_min
        self.hap_log_scale_max = hap_log_scale_max
        assert anchor_usage in ["hap", "feature"]
        self.anchor_usage = anchor_usage
        assert qaph_gate_mode in ["learnable", "fixed_0_5"]
        self.qaph_gate_mode = qaph_gate_mode
        self.user_to_service_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True)
        self.service_to_user_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True)
        self.bilinear_user_proj = nn.Linear(embed_dim, bilinear_rank, bias=False)
        self.bilinear_service_proj = nn.Linear(embed_dim, bilinear_rank, bias=False)
        interaction_dim = embed_dim * 4 + bilinear_rank
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_dim, embed_dim * 2), nn.LayerNorm(embed_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
        self.anchor_scale_head = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        self.anchor_offset_head = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embed_dim, 1))
        self.anchor_feature_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
        nn.init.zeros_(self.anchor_scale_head[-1].weight); nn.init.zeros_(self.anchor_scale_head[-1].bias)
        nn.init.zeros_(self.anchor_offset_head[-1].weight); nn.init.zeros_(self.anchor_offset_head[-1].bias)
        self.gate_net = nn.Sequential(nn.Linear(embed_dim * 2 + 1, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1), nn.Sigmoid())
        nn.init.constant_(self.gate_net[2].bias, gate_init_bias)
    
    def forward(self, user_repr, service_repr, anchor=None):
        batch_size = user_repr.size(0)
        user_3d, service_3d = user_repr.unsqueeze(1), service_repr.unsqueeze(1)
        u2s, _ = self.user_to_service_attn(user_3d, service_3d, service_3d); u2s = u2s.squeeze(1)
        s2u, _ = self.service_to_user_attn(service_3d, user_3d, user_3d); s2u = s2u.squeeze(1)
        bilinear = self.bilinear_user_proj(user_repr) * self.bilinear_service_proj(service_repr)
        interaction = torch.cat([u2s, s2u, user_repr * service_repr, user_repr - service_repr, bilinear], dim=-1)
        mlp_pred = self.interaction_mlp(interaction)
        inner_product = torch.sum(user_repr * service_repr, dim=-1, keepdim=True)
        w = torch.sigmoid(self.residual_weight)
        direct_pred = ((1 - w) * mlp_pred + w * inner_product).clamp(min=0.0)
        # V69: NaN guard on direct_pred
        direct_pred = torch.nan_to_num(direct_pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        combined = torch.cat([user_repr, service_repr], dim=-1)
        combined = torch.nan_to_num(combined, nan=0.0, posinf=10.0, neginf=-10.0)
        if anchor is None:
            raise ValueError("QAPHHead requires anchor for both hap and feature anchor_usage modes.")
        anchor_2d = anchor.unsqueeze(-1) if anchor.dim() == 1 else anchor
        anchor_feat = torch.log1p(anchor_2d.clamp(min=0.0)) / 10.0
        anchor_feat = torch.nan_to_num(anchor_feat, nan=0.0, posinf=10.0, neginf=0.0)
        if self.anchor_usage == "hap":
            log_scale = self.anchor_scale_head(combined)
            log_scale = torch.nan_to_num(
                log_scale,
                nan=0.0,
                posinf=self.hap_log_scale_max,
                neginf=self.hap_log_scale_min,
            ).clamp(self.hap_log_scale_min, self.hap_log_scale_max)
            offset = self.anchor_offset_head(combined)
            offset = torch.nan_to_num(offset, nan=0.0, posinf=10000.0, neginf=-10000.0)
            anchor_pred = (anchor_2d * torch.exp(log_scale) + offset).clamp(min=0.0)
            # V69: NaN guard on anchor_pred before gate computation
            anchor_pred = torch.nan_to_num(anchor_pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        else:
            af_input = torch.cat([combined, anchor_feat], dim=-1)
            anchor_pred = self.anchor_feature_mlp(af_input)
            anchor_pred = torch.nan_to_num(anchor_pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        if self.qaph_gate_mode == "learnable":
            gate_input = torch.cat([combined, anchor_feat], dim=-1)
            gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=10.0, neginf=-10.0)
            gate = self.gate_net(gate_input)
            # V69: NaN guard on gate output (critical fix for s456-type crashes)
            gate = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        else:
            gate = torch.full_like(direct_pred, 0.5)
        prediction = ((1 - gate) * direct_pred + gate * anchor_pred).clamp(min=0.0)
        # V69: Final NaN guard
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        return {
            'prediction': prediction,
            'gate': gate,
            'direct_pred': direct_pred,
            'anchor_pred': anchor_pred,
            'anchor_usage': self.anchor_usage,
            'gate_mode': self.qaph_gate_mode
        }


class FiLMLayer(nn.Module):
    """
    V75 FiLM: Target-Aware Embedding Conditioning
    
    将target-agnostic的GCN嵌入转化为target-aware的。
    用目标节点的嵌入生成逐维度的缩放(γ)和偏移(β)参数，
    对源节点嵌入进行仿射变换。
    
    FiLM(x | c) = γ(c) ⊙ x + β(c)
    
    零初始化：γ初始=1, β初始=0，训练初期等价于恒等变换。
    
    参考: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (AAAI 2018)
    
    参数量: 2 × (d × d + d) = 2 × (64×64 + 64) = 8,320
    """
    
    def __init__(self, embed_dim: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        
        # γ生成器：条件嵌入 → 缩放因子
        self.gamma_net = nn.Linear(embed_dim, embed_dim)
        # β生成器：条件嵌入 → 偏移量
        self.beta_net = nn.Linear(embed_dim, embed_dim)
        
        # 零初始化：训练初期 γ=1, β=0，等价于无FiLM
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)    # γ初始=1
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)    # β初始=0
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d] 被调制的嵌入（源节点GCN输出）
            condition: [batch, d] 条件信号（目标节点GCN输出）
        Returns:
            modulated: [batch, d] 调制后的嵌入
        """
        gamma = self.gamma_net(condition)   # [batch, d]
        beta = self.beta_net(condition)     # [batch, d]
        return gamma * x + beta            # [batch, d]
    
    def extra_repr(self):
        return f'embed_dim={self.embed_dim}'


def extract_user_neighbors(qos_matrix, max_neighbors=None):
    """
    从QoS矩阵提取每个用户的邻居服务索引（非零位置=有调用关系）
    
    Args:
        qos_matrix: np.ndarray [num_users, num_services], 非零位置表示有调用关系
        max_neighbors: 最大邻居数（None=不限制，超过则随机采样）
    
    Returns:
        neighbor_indices: torch.LongTensor [num_users, max_K], padding=-1
        neighbor_counts: torch.LongTensor [num_users] 每个用户的实际邻居数
    """
    import numpy as np
    num_users = qos_matrix.shape[0]
    neighbors_list = []
    counts = []
    
    for u in range(num_users):
        nonzero = np.nonzero(qos_matrix[u] > 0)[0]
        if max_neighbors is not None and len(nonzero) > max_neighbors:
            nonzero = np.sort(np.random.choice(nonzero, max_neighbors, replace=False))
        neighbors_list.append(nonzero)
        counts.append(len(nonzero))
    
    max_K = max(counts) if not max_neighbors else min(max(counts), max_neighbors)
    padded = np.full((num_users, max_K), -1, dtype=np.int64)
    for u, nb in enumerate(neighbors_list):
        k = min(len(nb), max_K)
        padded[u, :k] = nb[:k]
    
    return torch.LongTensor(padded), torch.LongTensor(counts)


def get_neighbor_service_embeds(user_ids, service_cache, neighbor_indices, neighbor_counts, max_K):
    """
    从service缓存中查表获取邻居服务嵌入
    
    Args:
        user_ids: [batch] 用户ID
        service_cache: [num_services, d] 服务GCN缓存嵌入
        neighbor_indices: [num_users, max_stored_K] 预计算的邻居索引，padding=-1
        neighbor_counts: [num_users] 每个用户的实际邻居数
        max_K: 本次使用的最大邻居数
    
    Returns:
        neighbor_embeds: [batch, max_K, d]
        neighbor_mask: [batch, max_K] bool, True=有效邻居
    """
    device = user_ids.device
    
    # 取出每个用户的邻居索引
    nb_indices = neighbor_indices[user_ids]  # [batch, max_stored_K]
    nb_counts = neighbor_counts[user_ids]    # [batch]
    
    # 截断到max_K
    nb_indices = nb_indices[:, :max_K]  # [batch, max_K]
    
    # 构建mask
    positions = torch.arange(max_K, device=device).unsqueeze(0)  # [1, max_K]
    neighbor_mask = positions < nb_counts.unsqueeze(1).clamp(max=max_K)  # [batch, max_K]
    
    # 将padding索引(-1)替换为0（查表安全，mask会忽略）
    safe_indices = nb_indices.clamp(min=0)  # [batch, max_K]
    
    # 从service缓存查表
    neighbor_embeds = service_cache[safe_indices]  # [batch, max_K, d]
    
    # 将padding位置的嵌入置零（防止信息泄漏）
    neighbor_embeds = neighbor_embeds * neighbor_mask.unsqueeze(-1).float()
    
    return neighbor_embeds, neighbor_mask


class TargetPromptAttention(nn.Module):
    """
    V76 TPA: Target-Prompt Attention on Cached GCN Embeddings
    
    核心思想：用target service嵌入作为query，在user的邻居服务嵌入上做注意力聚合，
    生成target-aware的user嵌入增强。解决ParametricGCN的target-agnostic问题。
    
    数学形式：
        attn_out = MHA(Q=service, K=V=neighbor_services)
        user_enhanced = user_embed + sigmoid(gate) * LayerNorm(attn_out)
    
    参数量: MHA(d=64, heads=4) ≈ 16,513 (+2.7% vs V70的619K)
    """
    
    def __init__(self, embed_dim: int = 64, num_heads: int = 4,
                 dropout: float = 0.1, init_gate: float = -2.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Parameter(torch.tensor(init_gate))  # sigmoid(-2)≈0.12
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, user_embed, service_embed, neighbor_embeds, neighbor_mask):
        """
        Args:
            user_embed: [batch, d] 用户GCN嵌入
            service_embed: [batch, d] 目标服务GCN嵌入
            neighbor_embeds: [batch, max_K, d] 邻居服务GCN嵌入
            neighbor_mask: [batch, max_K] bool, True=有效邻居
        
        Returns:
            user_enhanced: [batch, d] target-aware用户嵌入
        """
        query = service_embed.unsqueeze(1)  # [batch, 1, d]
        key = value = neighbor_embeds        # [batch, max_K, d]
        
        # key_padding_mask: True表示需要忽略的位置
        attn_out, _ = self.attn(
            query, key, value,
            key_padding_mask=~neighbor_mask  # ~mask: padding位置为True
        )  # [batch, 1, d]
        
        attn_out = attn_out.squeeze(1)  # [batch, d]
        
        gate = torch.sigmoid(self.gate)
        return user_embed + gate * self.norm(attn_out)
    
    def extra_repr(self):
        return f'gate_init={self.gate.item():.2f}, gate_value={torch.sigmoid(self.gate).item():.4f}'


class AACAPHHead(nn.Module):
    """
    V66 AACAPH: Anchor-Augmented Cross-Attention Prediction Head
    
    统一预测头，在CAPH基础上将anchor信息作为辅助特征注入交互建模。
    单一预测路径，不需要gate来区分RT/TP — 模型通过注意力机制
    自动学习anchor信息的使用程度。
    
    anchor_info: 3维特征向量
      - log1p(anchor): anchor值的对数（压缩值域）
      - nonzero_ratio: 历史非零观测比例（anchor可靠性）
      - cv: 历史变异系数 std/mean（anchor稳定性）
    
    anchor_encoder零初始化，训练初期等价于原CAPH。
    
    学术依据:
    - Side Information Injection (He et al., WWW 2017): 辅助特征注入
    - HCTN (IEEE TNSM 2025): RT和TP使用统一架构
    - Conditional Prediction: anchor作为条件信号调制预测
    
    数据支撑 (anchor_analysis):
    - RT: ratio std=3.06, [0.8,1.2)=38.9% → anchor不可靠，模型应学习忽略
    - TP: ratio std=0.68, [0.8,1.2)=55.8% → anchor有价值，模型应学习利用
    """
    
    ANCHOR_INFO_DIM = 3  # [log1p(anchor), nonzero_ratio, cv]
    
    def __init__(self, embed_dim: int = 64, bilinear_rank: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Anchor Encoder: 3维 → embed_dim
        # 零初始化最后一层，训练初期输出全零，等价于原CAPH
        self.anchor_encoder = nn.Sequential(
            nn.Linear(self.ANCHOR_INFO_DIM, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        nn.init.zeros_(self.anchor_encoder[-1].weight)
        nn.init.zeros_(self.anchor_encoder[-1].bias)
        
        # Cross-Attention (单头，轻量)
        self.user_to_service_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.service_to_user_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        
        # 低秩双线性
        self.bilinear_user_proj = nn.Linear(embed_dim, bilinear_rank, bias=False)
        self.bilinear_service_proj = nn.Linear(embed_dim, bilinear_rank, bias=False)
        
        # 交互特征融合 MLP
        # 原CAPH: 4d + r = embed_dim*4 + bilinear_rank
        # AACAPH: 4d + r + 2d(anchor×user, anchor×service) = embed_dim*6 + bilinear_rank
        interaction_dim = embed_dim * 6 + bilinear_rank
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
        # 残差内积权重 (可学习)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, user_repr: torch.Tensor, service_repr: torch.Tensor,
                anchor_info: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            user_repr: [batch, embed_dim]
            service_repr: [batch, embed_dim]
            anchor_info: [batch, 3] anchor辅助特征 (log1p_anchor, nonzero_ratio, cv)
                         如果为None，使用全零（等价于原CAPH）
        Returns:
            dict with 'prediction' [batch, 1]
        """
        # Cross-Attention
        user_3d = user_repr.unsqueeze(1)
        service_3d = service_repr.unsqueeze(1)
        
        u2s, _ = self.user_to_service_attn(user_3d, service_3d, service_3d)
        u2s = u2s.squeeze(1)
        s2u, _ = self.service_to_user_attn(service_3d, user_3d, user_3d)
        s2u = s2u.squeeze(1)
        
        # 基础交互特征
        elem_product = user_repr * service_repr
        elem_diff = user_repr - service_repr
        bilinear = self.bilinear_user_proj(user_repr) * self.bilinear_service_proj(service_repr)
        
        # Anchor交互特征
        if anchor_info is not None:
            anchor_feat = self.anchor_encoder(anchor_info)  # [batch, embed_dim]
        else:
            anchor_feat = torch.zeros_like(user_repr)  # [batch, embed_dim]
        
        anchor_user = anchor_feat * user_repr       # [batch, embed_dim]
        anchor_service = anchor_feat * service_repr  # [batch, embed_dim]
        
        # 拼接所有交互特征
        interaction = torch.cat([
            u2s, s2u, elem_product, elem_diff, bilinear,
            anchor_user, anchor_service
        ], dim=-1)
        
        # MLP预测
        mlp_pred = self.interaction_mlp(interaction)
        
        # 残差内积
        inner_product = torch.sum(user_repr * service_repr, dim=-1, keepdim=True)
        
        # 融合
        w = torch.sigmoid(self.residual_weight)
        prediction = (1 - w) * mlp_pred + w * inner_product
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        
        return {'prediction': prediction}


class CrossAttentionPredictionHead(nn.Module):
    """
    交叉注意力预测头
    
    替换V41的简单内积预测，使用更丰富的交互建模：
    1. Cross-Attention: 用户作为Query，服务作为Key/Value（及反向）
    2. 多模态交互: 元素积 + 元素差 + 低秩双线性
    3. 残差内积: 保留原始内积信号，可学习权重融合
    
    参考: Multi-Head Bilinear Fusion (2025)
    """
    
    def __init__(self, embed_dim: int = 64, bilinear_rank: int = 16, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.bilinear_rank = bilinear_rank
        
        # Cross-Attention (单头，轻量)
        self.user_to_service_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.service_to_user_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        
        # 低秩双线性: user_proj(d→r) × service_proj(d→r) → r维
        self.bilinear_user_proj = nn.Linear(embed_dim, bilinear_rank, bias=False)
        self.bilinear_service_proj = nn.Linear(embed_dim, bilinear_rank, bias=False)
        
        # 交互特征融合 MLP
        # 输入: cross_attn_user(d) + cross_attn_service(d) + elem_product(d) + elem_diff(d) + bilinear(r)
        interaction_dim = embed_dim * 4 + bilinear_rank
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1)
        )
        
        # 残差内积权重 (可学习)
        self.residual_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, user_repr: torch.Tensor, service_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_repr: [batch, embed_dim] 用户表示
            service_repr: [batch, embed_dim] 服务表示
        Returns:
            prediction: [batch, 1] 预测值
        """
        batch_size = user_repr.size(0)
        
        # Cross-Attention需要3D输入 [batch, seq_len=1, embed_dim]
        user_3d = user_repr.unsqueeze(1)
        service_3d = service_repr.unsqueeze(1)
        
        # User→Service attention
        u2s, _ = self.user_to_service_attn(user_3d, service_3d, service_3d)
        u2s = u2s.squeeze(1)  # [batch, embed_dim]
        
        # Service→User attention
        s2u, _ = self.service_to_user_attn(service_3d, user_3d, user_3d)
        s2u = s2u.squeeze(1)  # [batch, embed_dim]
        
        # 元素积
        elem_product = user_repr * service_repr  # [batch, embed_dim]
        
        # 元素差
        elem_diff = user_repr - service_repr  # [batch, embed_dim]
        
        # 低秩双线性
        bilinear = self.bilinear_user_proj(user_repr) * self.bilinear_service_proj(service_repr)  # [batch, rank]
        
        # 拼接所有交互特征
        interaction = torch.cat([u2s, s2u, elem_product, elem_diff, bilinear], dim=-1)
        
        # MLP预测
        mlp_pred = self.interaction_mlp(interaction)  # [batch, 1]
        
        # 残差内积
        inner_product = torch.sum(user_repr * service_repr, dim=-1, keepdim=True)  # [batch, 1]
        
        # 融合
        w = torch.sigmoid(self.residual_weight)
        prediction = (1 - w) * mlp_pred + w * inner_product
        
        # 数值安全
        prediction = torch.nan_to_num(prediction, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        
        return prediction


class CQPMV42(nn.Module):
    """V42 CQPM - 支持CAPH/内积/HAP/RAPP/QAPH/AACAPH切换
    
    V47改进点4: fuse-first-then-split架构（对齐HCTN论文）
    V56改进: HAP (History-Anchored Prediction) + RAPP (MoE Prediction)
    V60改进: NACC (Neighbor-Aggregated Collaborative Context) 注入
    V65改进: QAPH (QoS-Adaptive Prediction Head) — 统一预测头
    V66改进: AACAPH (Anchor-Augmented CAPH) — 统一预测头（推荐）
    """
    
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1, bilinear_rank: int = 16,
                 use_caph: bool = True, use_fuse_first: bool = False,
                 use_hap: bool = False, use_rapp: bool = False,
                 rapp_num_experts: int = 3,
                 hap_log_scale_min: float = -1.5, hap_log_scale_max: float = 2.0,
                 nacc_dim: int = 0,
                 use_qaph: bool = False,
                 use_aacaph: bool = False,
                 qaph_gate_init_bias: float = -1.0,
                 qaph_anchor_usage: str = "hap",
                 qaph_gate_mode: str = "learnable"):
        super().__init__()
        self.use_caph = use_caph
        self.use_fuse_first = use_fuse_first
        self.use_hap = use_hap
        self.use_rapp = use_rapp
        self.use_qaph = use_qaph
        self.use_aacaph = use_aacaph
        self.nacc_dim = nacc_dim
        
        # V60: NACC投影层（6维→embed_dim）
        if nacc_dim > 0:
            self.nacc_projection = nn.Sequential(
                nn.Linear(nacc_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, embed_dim)
            )
            # 可学习的NACC融合权重，初始化为小值使NACC渐进引入
            self.nacc_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            nn.init.constant_(self.nacc_gate[0].bias, -1.0)  # 初始gate≈0.27
        
        if use_fuse_first:
            # V47: fuse-first-then-split (对齐HCTN)
            # 输入: user_embed(d) + service_embed(d) + gmm_output(d) + tgem_output(d) = 4d
            self.unified_bn = nn.BatchNorm1d(embed_dim * 4)
            self.unified_fusion = nn.Sequential(
                nn.Linear(embed_dim * 4, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim * 2),
                nn.ReLU(),
            )
            # 分裂为user和service表示
            self.user_head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            self.service_head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
        else:
            # 原V42架构: 分离融合（保留用于消融对比）
            self.user_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            self.service_fusion = nn.Sequential(
                nn.Linear(embed_dim * 3, embed_dim * 2),
                nn.LayerNorm(embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim)
            )
            self.user_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            self.service_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
        
        # V66 AACAPH预测头（最高优先级）
        if use_aacaph:
            self.aacaph_head = AACAPHHead(
                embed_dim=embed_dim, bilinear_rank=bilinear_rank, dropout=dropout
            )
        # V65 QAPH预测头
        elif use_qaph:
            self.qaph_head = QAPHHead(
                embed_dim=embed_dim, bilinear_rank=bilinear_rank, dropout=dropout,
                hap_log_scale_min=hap_log_scale_min, hap_log_scale_max=hap_log_scale_max,
                gate_init_bias=qaph_gate_init_bias, anchor_usage=qaph_anchor_usage,
                qaph_gate_mode=qaph_gate_mode
            )
        else:
            # V56 HAP预测头（可选）
            if use_hap:
                self.hap_head = HAPHead(
                    embed_dim=embed_dim, dropout=dropout,
                    log_scale_min=hap_log_scale_min, log_scale_max=hap_log_scale_max
                )
            
            # V56 RAPP预测头（可选）
            if use_rapp:
                self.rapp_head = MoEPredictionHead(
                    embed_dim=embed_dim, num_experts=rapp_num_experts, dropout=dropout
                )
            
            # CAPH预测头（HAP/RAPP未启用时使用）
            if use_caph and not use_hap and not use_rapp:
                self.prediction_head = CrossAttentionPredictionHead(
                    embed_dim=embed_dim, bilinear_rank=bilinear_rank, dropout=dropout
                )
    
    def forward(self, user_embed: torch.Tensor, service_embed: torch.Tensor,
                gmm_output: torch.Tensor, tgem_output: torch.Tensor,
                anchor: Optional[torch.Tensor] = None,
                nacc_features: Optional[torch.Tensor] = None,
                anchor_info: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        if self.use_fuse_first:
            # V47: fuse-first-then-split
            fused_input = torch.cat([user_embed, service_embed, gmm_output, tgem_output], dim=-1)
            fused_input = self.unified_bn(fused_input)
            shared = self.unified_fusion(fused_input)  # [batch, embed_dim*2]
            user_final = self.user_head(shared)         # [batch, embed_dim]
            service_final = self.service_head(shared)   # [batch, embed_dim]
        else:
            # 原V42: 分离融合
            user_fused = self.user_fusion(torch.cat([user_embed, gmm_output, tgem_output], dim=-1))
            service_fused = self.service_fusion(torch.cat([service_embed, gmm_output, tgem_output], dim=-1))
            user_final = self.user_proj(user_fused)
            service_final = self.service_proj(service_fused)
        
        # V60: NACC融合（在预测头之前注入邻居协同上下文）
        if self.nacc_dim > 0 and nacc_features is not None:
            nacc_repr = self.nacc_projection(nacc_features)  # [batch, embed_dim]
            # 门控融合到user_final和service_final
            gate_u = self.nacc_gate(torch.cat([user_final, nacc_repr], dim=-1))  # [batch, embed_dim]
            gate_s = self.nacc_gate(torch.cat([service_final, nacc_repr], dim=-1))  # [batch, embed_dim]
            user_final = user_final + gate_u * nacc_repr
            service_final = service_final + gate_s * nacc_repr
        
        # 预测（V66优先级: AACAPH > QAPH > HAP+RAPP > HAP > RAPP > CAPH > 内积）
        result = {'user_repr': user_final, 'service_repr': service_final}
        
        if self.use_aacaph:
            aacaph_out = self.aacaph_head(user_final, service_final, anchor_info=anchor_info)
            pred = aacaph_out['prediction']
        elif self.use_qaph:
            qaph_out = self.qaph_head(user_final, service_final, anchor=anchor)
            pred = qaph_out['prediction']
            result['gate'] = qaph_out['gate']
            result['direct_pred'] = qaph_out['direct_pred']
            result['anchor_pred'] = qaph_out['anchor_pred']
            result['anchor_usage'] = qaph_out['anchor_usage']
        elif self.use_hap and self.use_rapp:
            # HAP + RAPP组合
            rapp_out = self.rapp_head(user_final, service_final)
            result['router_weights'] = rapp_out['router_weights']
            if anchor is not None:
                hap_pred = self.hap_head(user_final, service_final, anchor)
                pred = hap_pred
                result['hap_pred'] = hap_pred
                result['rapp_pred'] = rapp_out['prediction']
            else:
                pred = rapp_out['prediction']
        elif self.use_hap:
            if anchor is not None:
                pred = self.hap_head(user_final, service_final, anchor)
            else:
                # anchor不可用时fallback
                pred = torch.sum(user_final * service_final, dim=-1, keepdim=True)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        elif self.use_rapp:
            rapp_out = self.rapp_head(user_final, service_final)
            pred = rapp_out['prediction']
            result['router_weights'] = rapp_out['router_weights']
        elif self.use_caph:
            pred = self.prediction_head(user_final, service_final)
        else:
            pred = torch.sum(user_final * service_final, dim=-1, keepdim=True)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=10000.0, neginf=0.0).clamp(min=0.0)
        
        result['prediction'] = pred
        return result


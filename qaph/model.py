"""Frozen-v0 QAPH implementation, organized by responsibility. Original source credit: Kilo Code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple
from qaph.features.statistics import HCTNStyleGMMv2
from qaph.encoders.temporal import ParallelTGEMAdapter, ParallelTGEMAdapterV42
from qaph.graphs.fig import LightGCNEmbedding
from qaph.encoders.sega import ParametricGCN, MultiGraphGCN, ProjectionHead
from torch_geometric.nn import GCNConv
from qaph.heads import MoEPredictionHead, HAPHead, QAPHHead, FiLMLayer, extract_user_neighbors, get_neighbor_service_embeds, TargetPromptAttention, AACAPHHead, CrossAttentionPredictionHead, CQPMV42

class LightweightGCN(nn.Module):
    """
    轻量级2层GCN，用于从协同过滤图中提取邻域特征。
    
    输入: 节点嵌入 + 邻接矩阵(预计算的稀疏图)
    输出: 增强后的节点嵌入
    """
    
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, embed_dim] 节点嵌入
            edge_index: [2, num_edges] COO格式边索引
        Returns:
            out: [num_nodes, embed_dim] 增强嵌入
        """
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.norm2(h)
        return h


class LGCFEModule(nn.Module):
    """
    LGCFE模块：对用户和服务分别构建GCN，提取协同过滤特征。
    
    通过可学习的残差系数α将GCN特征与TGEM特征融合：
        final = tgem_output + α * gcn_output
    
    α初始化为0.1，确保GCN特征以小权重渐进引入，不破坏已有的TGEM预测能力。
    """
    
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1, init_alpha: float = 0.1):
        super().__init__()
        self.user_gcn = LightweightGCN(embed_dim, dropout)
        self.service_gcn = LightweightGCN(embed_dim, dropout)
        
        # 可学习融合系数，初始化为小值
        self.user_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.service_alpha = nn.Parameter(torch.tensor(init_alpha))
        
        # 预计算的边索引（由训练脚本设置）
        self.register_buffer('user_edge_index', torch.zeros(2, 1, dtype=torch.long))
        self.register_buffer('service_edge_index', torch.zeros(2, 1, dtype=torch.long))
    
    def set_graph(self, user_edge_index: torch.Tensor, service_edge_index: torch.Tensor):
        """设置预计算的图结构"""
        self.user_edge_index = user_edge_index
        self.service_edge_index = service_edge_index
    
    def forward(self, user_embed_all: torch.Tensor, service_embed_all: torch.Tensor,
                user_ids: torch.Tensor, service_ids: torch.Tensor,
                user_tgem: torch.Tensor, service_tgem: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            user_embed_all: [num_users, embed_dim] 全部用户嵌入
            service_embed_all: [num_services, embed_dim] 全部服务嵌入
            user_ids: [batch] 当前batch的用户ID
            service_ids: [batch] 当前batch的服务ID
            user_tgem: [batch, embed_dim] 用户侧TGEM输出
            service_tgem: [batch, embed_dim] 服务侧TGEM输出（与user_tgem相同，因为TGEM是共享的）
        Returns:
            user_enhanced: [batch, embed_dim] 增强后的用户特征
            service_enhanced: [batch, embed_dim] 增强后的服务特征
        """
        # GCN前向：对全部节点计算
        user_gcn_all = self.user_gcn(user_embed_all, self.user_edge_index)    # [num_users, d]
        service_gcn_all = self.service_gcn(service_embed_all, self.service_edge_index)  # [num_services, d]
        
        # 按batch索引取出
        user_gcn = user_gcn_all[user_ids]      # [batch, d]
        service_gcn = service_gcn_all[service_ids]  # [batch, d]
        
        # 残差融合: tgem + α * gcn
        alpha_u = torch.sigmoid(self.user_alpha)    # 限制在(0,1)
        alpha_s = torch.sigmoid(self.service_alpha)
        
        user_enhanced = user_tgem + alpha_u * user_gcn
        service_enhanced = service_tgem + alpha_s * service_gcn
        
        return user_enhanced, service_enhanced


class GMMAdapterV42(nn.Module):
    """V42 GMM适配器（与V41相同）"""
    def __init__(self, embed_dim: int = 64, num_neighbors: int = 10, num_heads: int = 4,
                 dropout: float = 0.3, user_threshold: float = 1.5, service_threshold: float = 1.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.gmm = HCTNStyleGMMv2(
            embed_dim=embed_dim, num_neighbors=num_neighbors, num_heads=num_heads,
            dropout=dropout, user_threshold=user_threshold, service_threshold=service_threshold
        )
        self.user_qos_matrix = None
        self.service_qos_matrix = None
        self.user_stat_features = None
        self.service_stat_features = None
        self.user_mean_qos, self.user_std_qos = 0.0, 1.0
        self.service_mean_qos, self.service_std_qos = 0.0, 1.0
    
    def set_qos_data(self, user_qos_matrix, service_qos_matrix, user_stat_features,
                     service_stat_features, user_mean_qos, user_std_qos,
                     service_mean_qos, service_std_qos):
        self.user_qos_matrix = user_qos_matrix
        self.service_qos_matrix = service_qos_matrix
        self.user_stat_features = user_stat_features
        self.service_stat_features = service_stat_features
        self.user_mean_qos, self.user_std_qos = user_mean_qos, user_std_qos
        self.service_mean_qos, self.service_std_qos = service_mean_qos, service_std_qos
        self.gmm.precompute_neighbors(user_qos_matrix, service_qos_matrix)
    
    def forward(self, user_embed, service_embed, user_ids, service_ids):
        if self.user_qos_matrix is None:
            return torch.zeros(user_embed.size(0), self.embed_dim, device=user_embed.device)
        return self.gmm(
            user_embed, service_embed, user_ids, service_ids,
            self.user_stat_features, self.service_stat_features,
            self.user_qos_matrix, self.service_qos_matrix,
            self.user_mean_qos, self.user_std_qos,
            self.service_mean_qos, self.service_std_qos
        )


class TGEMAdapterV42(nn.Module):
    """V42 TGEM适配器 - 支持MS-TGEM/原版切换（消融实验用）
    
    V50: 支持GPAM-Aligned Input (gpam_input_dim参数)
    V52: 支持MFVP (mfvp_input_dim参数)
    V60: 支持GMM→TGEM串行化 (gmm_context_dim参数)
    """
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1, use_ms_tgem: bool = True,
                 e_block_ratio: int = 4, tgem_num_branches: int = 4,
                 use_missing_token: bool = False,
                 gpam_input_dim: int = 0, mfvp_input_dim: int = 0,
                 gmm_context_dim: int = 0,
                 disable_tblock: bool = False, disable_fblock: bool = False):
        super().__init__()
        if use_ms_tgem:
            self.tgem = ParallelTGEMAdapterV42(
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                dropout=dropout, use_positional_encoding=True,
                e_block_ratio=e_block_ratio,
                num_branches=tgem_num_branches,
                use_missing_token=use_missing_token,
                gpam_input_dim=gpam_input_dim,
                mfvp_input_dim=mfvp_input_dim,
                gmm_context_dim=gmm_context_dim,
                disable_tblock=disable_tblock,
                disable_fblock=disable_fblock,
            )
        else:
            # 消融: 使用原版ParallelTGEMAdapter (固定k=3 Conv1D)
            self.tgem = ParallelTGEMAdapter(
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                dropout=dropout, use_positional_encoding=True
            )
    
    def forward(self, history_sequence, attention_mask=None, user_embed=None, service_embed=None,
                gpam_features=None, mfvp_features=None, gmm_context=None):
        if mfvp_features is not None and hasattr(self.tgem, 'mfvp_input_dim'):
            return self.tgem(history_sequence, attention_mask, user_embed, service_embed,
                           mfvp_features=mfvp_features, gmm_context=gmm_context)
        if gpam_features is not None and hasattr(self.tgem, 'gpam_input_dim'):
            return self.tgem(history_sequence, attention_mask, user_embed, service_embed,
                           gpam_features=gpam_features, gmm_context=gmm_context)
        return self.tgem(history_sequence, attention_mask, user_embed, service_embed,
                        gmm_context=gmm_context)


class UnifiedQoSModelV42(nn.Module):
    """
    V42 统一QoS预测模型
    
    改进：
    1. MS-TGEM: 多尺度膨胀卷积T-block
    2. CAPH: 交叉注意力预测头
    
    损失函数改进（在训练脚本中实现）：
    3. BAL: Barron自适应损失
    4. AGH: 自适应梯度调和
    """
    
    def __init__(self, num_users: int, num_services: int, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1,
                 qos_type: str = 'rt', use_gmm: bool = True, use_tgem: bool = True,
                 num_neighbors: int = 10, gmm_num_heads: int = 4, gmm_dropout: float = 0.3,
                 user_threshold: float = 1.5, service_threshold: float = 1.5,
                 bilinear_rank: int = 16, use_ms_tgem: bool = True, use_caph: bool = True,
                 use_lgcfe: bool = False, lgcfe_init_alpha: float = 0.1,
                 e_block_ratio: int = 4, tgem_num_branches: int = 4,
                 use_missing_token: bool = False,
                 use_fuse_first: bool = False,
                 gpam_input_dim: int = 0,
                 mfvp_input_dim: int = 0,
                 use_hap: bool = False, use_rapp: bool = False,
                 rapp_num_experts: int = 3,
                 hap_log_scale_min: float = -2.0, hap_log_scale_max: float = 3.0,
                 use_cf_feature: bool = False, cf_factor_dim: int = 64,
                 nacc_dim: int = 0, use_serial_gmm: bool = False,
                 use_lightgcn: bool = False, lightgcn_layers: int = 2,
                 lightgcn_dropout: float = 0.0,
                 use_sega: bool = False, sega_gcn_dropout: float = 0.1,
                 sega_init_gate: float = -2.0,
                 pgcn_num_layers: int = 2,
                 use_multi_graph: bool = False,
                 multi_graph_use_suig: bool = True,
                 multi_graph_use_ssig: bool = True,
                 multi_graph_second_gate_init: float = -2.0,
                 use_indep_gcn: bool = False, indep_gcn_dropout: float = 0.1,
                 indep_gcn_init_gate: float = 0.0,
                 use_qaph: bool = False,
                 use_aacaph: bool = False,
                 qaph_gate_init_bias: float = -1.0,
                 qaph_anchor_usage: str = "hap",
                 qaph_gate_mode: str = "learnable",
                 use_film: bool = False,
                 use_tpa: bool = False,
                 tpa_num_heads: int = 4,
                 tpa_max_neighbors: int = 256,
                 tpa_init_gate: float = -2.0,
                 tpa_dropout: float = 0.1,
                 disable_tblock: bool = False,
                 disable_fblock: bool = False,
                 **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_services = num_services
        self.embed_dim = embed_dim
        self.use_gmm = use_gmm
        self.use_tgem = use_tgem
        self.use_ms_tgem = use_ms_tgem
        self.use_caph = use_caph
        self.use_lgcfe = use_lgcfe
        self.gpam_input_dim = gpam_input_dim
        self.mfvp_input_dim = mfvp_input_dim
        self.use_hap = use_hap
        self.use_rapp = use_rapp
        self.use_cf_feature = use_cf_feature
        self.nacc_dim = nacc_dim
        self.use_serial_gmm = use_serial_gmm
        self.use_lightgcn = use_lightgcn
        self.use_sega = use_sega
        self.use_multi_graph = use_multi_graph
        self.use_indep_gcn = use_indep_gcn
        self.use_qaph = use_qaph
        self.use_aacaph = use_aacaph
        self.use_film = use_film
        self.use_tpa = use_tpa
        self.tpa_max_neighbors = tpa_max_neighbors
        
        # 基础嵌入层（SEGA模式下作为唯一共享嵌入）
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.service_embedding = nn.Embedding(num_services, embed_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.service_embedding.weight)
        
        # V63 SEGA: 共享嵌入 + ParametricGCN/MultiGraphGCN + 投影头
        if use_sega:
            if use_multi_graph:
                # V71: 三路图卷积（FIG + SUIG + SSIG）
                self.pgcn = MultiGraphGCN(
                    embed_dim=embed_dim, dropout=sega_gcn_dropout,
                    init_gate=sega_init_gate,
                    num_layers=pgcn_num_layers,
                    use_suig=multi_graph_use_suig,
                    use_ssig=multi_graph_use_ssig,
                    second_gate_init=multi_graph_second_gate_init
                )
                print(f"[V71] MultiGraphGCN created: {pgcn_num_layers}L, "
                      f"SUIG={multi_graph_use_suig}, SSIG={multi_graph_use_ssig}, "
                      f"second_gate_init={multi_graph_second_gate_init}")
            else:
                self.pgcn = ParametricGCN(
                    embed_dim=embed_dim, dropout=sega_gcn_dropout,
                    init_gate=sega_init_gate,
                    num_layers=pgcn_num_layers
                )
            # 3个投影头：base(→CQPM), gmm(→GMM), tgem(→TGEM)
            self.proj_base = ProjectionHead(embed_dim)
            self.proj_gmm = ProjectionHead(embed_dim) if use_gmm else None
            self.proj_tgem = ProjectionHead(embed_dim) if use_tgem else None
            
            # V75 FiLM: Target-Aware Embedding Conditioning
            if use_film:
                self.film_user = FiLMLayer(embed_dim)    # 用service调制user
                self.film_service = FiLMLayer(embed_dim)  # 用user调制service
                print(f"[V75] FiLM layers created: 2 × FiLMLayer(d={embed_dim}), "
                      f"params={2 * (embed_dim * embed_dim + embed_dim) * 2}")
            else:
                self.film_user = None
                self.film_service = None
            
            # V76 TPA: Target-Prompt Attention on Cached GCN Embeddings
            if use_tpa:
                self.tpa = TargetPromptAttention(
                    embed_dim=embed_dim, num_heads=tpa_num_heads,
                    dropout=tpa_dropout, init_gate=tpa_init_gate
                )
                tpa_params = sum(p.numel() for p in self.tpa.parameters())
                print(f"[V76] TPA created: heads={tpa_num_heads}, max_nb={tpa_max_neighbors}, "
                      f"gate_init={tpa_init_gate}, params={tpa_params}")
            else:
                self.tpa = None
            
            # TPA邻居索引缓存（由set_user_neighbors()设置）
            self._tpa_neighbor_indices = None
            self._tpa_neighbor_counts = None
        else:
            self.pgcn = None
            self.proj_base = None
            self.proj_gmm = None
            self.proj_tgem = None
        
        # V64 Independent-GCN: 对每套独立嵌入分别应用ParametricGCN
        # 与SEGA互斥：SEGA统一嵌入+GCN，Independent-GCN保持独立嵌入+各自GCN
        if use_indep_gcn and not use_sega:
            self.base_gcn = ParametricGCN(
                embed_dim=embed_dim, dropout=indep_gcn_dropout,
                init_gate=indep_gcn_init_gate,
                num_layers=pgcn_num_layers
            )
            self.gmm_gcn = ParametricGCN(
                embed_dim=embed_dim, dropout=indep_gcn_dropout,
                init_gate=indep_gcn_init_gate,
                num_layers=pgcn_num_layers
            ) if use_gmm else None
            self.tgem_gcn = ParametricGCN(
                embed_dim=embed_dim, dropout=indep_gcn_dropout,
                init_gate=indep_gcn_init_gate,
                num_layers=pgcn_num_layers
            ) if use_tgem else None
        else:
            self.base_gcn = None
            self.gmm_gcn = None
            self.tgem_gcn = None
        
        # Independent-GCN缓存（每epoch预计算，detach）
        self._indep_gcn_cache = {}
        
        # GMM模块
        if use_gmm:
            self.gmm = GMMAdapterV42(
                embed_dim=embed_dim, num_neighbors=num_neighbors,
                num_heads=gmm_num_heads, dropout=gmm_dropout,
                user_threshold=user_threshold, service_threshold=service_threshold
            )
            # SEGA模式下不创建独立GMM嵌入（使用共享嵌入+投影头）
            if not use_sega:
                self.gmm_user_embed = nn.Embedding(num_users, embed_dim)
                self.gmm_service_embed = nn.Embedding(num_services, embed_dim)
                nn.init.xavier_uniform_(self.gmm_user_embed.weight)
                nn.init.xavier_uniform_(self.gmm_service_embed.weight)
            else:
                self.gmm_user_embed = None
                self.gmm_service_embed = None
        else:
            self.gmm = None
            self.gmm_user_embed = None
            self.gmm_service_embed = None
        
        # TGEM模块 (V42: MS-TGEM; V50: GPAM; V52: MFVP; V60: serial GMM)
        if use_tgem:
            gmm_ctx_dim = embed_dim if (use_serial_gmm and use_gmm) else 0
            self.tgem = TGEMAdapterV42(
                embed_dim=embed_dim, num_heads=num_heads,
                num_layers=num_layers, dropout=dropout,
                use_ms_tgem=use_ms_tgem,
                e_block_ratio=e_block_ratio,
                tgem_num_branches=tgem_num_branches,
                use_missing_token=use_missing_token,
                gpam_input_dim=gpam_input_dim,
                mfvp_input_dim=mfvp_input_dim,
                gmm_context_dim=gmm_ctx_dim,
                disable_tblock=disable_tblock,
                disable_fblock=disable_fblock,
            )
            # SEGA模式下不创建独立TGEM嵌入
            if not use_sega:
                self.tgem_user_embed = nn.Embedding(num_users, embed_dim)
                self.tgem_service_embed = nn.Embedding(num_services, embed_dim)
                nn.init.xavier_uniform_(self.tgem_user_embed.weight)
                nn.init.xavier_uniform_(self.tgem_service_embed.weight)
            else:
                self.tgem_user_embed = None
                self.tgem_service_embed = None
        else:
            self.tgem = None
            self.tgem_user_embed = None
            self.tgem_service_embed = None
        
        # LGCFE模块（可选）
        if use_lgcfe:
            self.lgcfe = LGCFEModule(embed_dim=embed_dim, dropout=dropout, init_alpha=lgcfe_init_alpha)
        else:
            self.lgcfe = None
        
        # V61: LightGCN嵌入增强（零参数消息传递）
        if use_lightgcn:
            self.lightgcn = LightGCNEmbedding(
                num_layers=lightgcn_layers,
                dropout=lightgcn_dropout
            )
        else:
            self.lightgcn = None
        
        # V58: CF-Enhanced Embedding (持久化NMF因子 + 投影 + 门控融合)
        if use_cf_feature:
            # 冻结的NMF因子作为register_buffer（不参与梯度更新）
            self.register_buffer('cf_user_factors', torch.zeros(num_users, cf_factor_dim))
            self.register_buffer('cf_service_factors', torch.zeros(num_services, cf_factor_dim))
            # 可训练的投影层：将NMF因子投影到embed_dim
            self.cf_user_proj = nn.Sequential(
                nn.Linear(cf_factor_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            self.cf_service_proj = nn.Sequential(
                nn.Linear(cf_factor_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
            # 门控网络：user和service分别使用独立的门控（分布不同，不应共享）
            self.cf_user_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            self.cf_service_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            # 初始化门控偏置为负值，使初始gate≈0.27，CF信号以小权重引入
            nn.init.constant_(self.cf_user_gate[0].bias, -1.0)
            nn.init.constant_(self.cf_service_gate[0].bias, -1.0)
        
        # V60: GMM→TGEM串行化 — GMM输出投影层（作为TGEM的额外上下文）
        if use_serial_gmm and use_gmm and use_tgem:
            self.gmm_to_tgem_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
        
        # V42 CQPM (V47: fuse-first + CAPH; V56: HAP/RAPP; V60: NACC; V65: QAPH; V66: AACAPH)
        self.cqpm = CQPMV42(embed_dim=embed_dim, dropout=dropout, bilinear_rank=bilinear_rank,
                            use_caph=use_caph, use_fuse_first=use_fuse_first,
                            use_hap=use_hap, use_rapp=use_rapp,
                            rapp_num_experts=rapp_num_experts,
                            hap_log_scale_min=hap_log_scale_min,
                            hap_log_scale_max=hap_log_scale_max,
                            nacc_dim=nacc_dim,
                            use_qaph=use_qaph,
                            use_aacaph=use_aacaph,
                            qaph_gate_init_bias=qaph_gate_init_bias,
                            qaph_anchor_usage=qaph_anchor_usage,
                            qaph_gate_mode=qaph_gate_mode)
    
    def set_qos_data(self, user_qos_matrix, service_qos_matrix, user_stat_features,
                     service_stat_features, user_mean_qos, user_std_qos,
                     service_mean_qos, service_std_qos):
        if self.gmm is not None:
            self.gmm.set_qos_data(
                user_qos_matrix, service_qos_matrix,
                user_stat_features, service_stat_features,
                user_mean_qos, user_std_qos, service_mean_qos, service_std_qos
            )
    
    def set_lgcfe_graph(self, user_edge_index: torch.Tensor, service_edge_index: torch.Tensor):
        """设置LGCFE的图结构"""
        if self.lgcfe is not None:
            self.lgcfe.set_graph(user_edge_index, service_edge_index)
    
    def set_lightgcn_graph(self, norm_adj: torch.Tensor):
        """
        V61: 设置LightGCN的归一化邻接矩阵
        
        Args:
            norm_adj: [N, N] 稀疏归一化邻接矩阵，N = num_users + num_services
        """
        if self.lightgcn is not None:
            self.lightgcn.set_graph(norm_adj, self.num_users, self.num_services)
            print(f"[V61] LightGCN graph set: {self.num_users} users + {self.num_services} services")
    
    def set_indep_gcn_graph(self, norm_adj: torch.Tensor):
        """
        V64 Independent-GCN: 设置所有独立GCN的归一化邻接矩阵
        
        Args:
            norm_adj: [N, N] 稀疏归一化邻接矩阵，N = num_users + num_services
        """
        gcn_names = []
        if self.base_gcn is not None:
            self.base_gcn.set_graph(norm_adj, self.num_users, self.num_services)
            gcn_names.append('base')
        if self.gmm_gcn is not None:
            self.gmm_gcn.set_graph(norm_adj, self.num_users, self.num_services)
            gcn_names.append('gmm')
        if self.tgem_gcn is not None:
            self.tgem_gcn.set_graph(norm_adj, self.num_users, self.num_services)
            gcn_names.append('tgem')
        self._indep_gcn_cache = {}
        print(f"[V64] Independent-GCN graph set for {gcn_names}: "
              f"{self.num_users} users + {self.num_services} services")
    
    def precompute_indep_gcn_embeddings(self):
        """
        V64 Independent-GCN: 每epoch开始时预计算所有GCN增强嵌入的增量并缓存（detach版）
        
        缓存GCN增量 delta = GCN(embed) - embed，而非完整增强嵌入。
        这样forward中可以用 embed(user_ids) + delta[user_ids] 保持嵌入层梯度。
        
        同时缓存完整增强嵌入（用于GCN更新步的临时替换）。
        """
        if not self.use_indep_gcn:
            return
        
        with torch.no_grad():
            if self.base_gcn is not None:
                raw_u = self.user_embedding.weight
                raw_s = self.service_embedding.weight
                u_enh, s_enh = self.base_gcn(raw_u, raw_s)
                self._indep_gcn_cache['base_user_delta'] = (u_enh - raw_u).detach()
                self._indep_gcn_cache['base_service_delta'] = (s_enh - raw_s).detach()
                self._indep_gcn_cache['base_user'] = u_enh.detach()
                self._indep_gcn_cache['base_service'] = s_enh.detach()
            
            if self.gmm_gcn is not None and self.gmm_user_embed is not None:
                raw_u = self.gmm_user_embed.weight
                raw_s = self.gmm_service_embed.weight
                u_enh, s_enh = self.gmm_gcn(raw_u, raw_s)
                self._indep_gcn_cache['gmm_user_delta'] = (u_enh - raw_u).detach()
                self._indep_gcn_cache['gmm_service_delta'] = (s_enh - raw_s).detach()
                self._indep_gcn_cache['gmm_user'] = u_enh.detach()
                self._indep_gcn_cache['gmm_service'] = s_enh.detach()
            
            if self.tgem_gcn is not None and self.tgem_user_embed is not None:
                raw_u = self.tgem_user_embed.weight
                raw_s = self.tgem_service_embed.weight
                u_enh, s_enh = self.tgem_gcn(raw_u, raw_s)
                self._indep_gcn_cache['tgem_user_delta'] = (u_enh - raw_u).detach()
                self._indep_gcn_cache['tgem_service_delta'] = (s_enh - raw_s).detach()
                self._indep_gcn_cache['tgem_user'] = u_enh.detach()
                self._indep_gcn_cache['tgem_service'] = s_enh.detach()
    
    def indep_gcn_forward_with_grad(self):
        """
        V64 Independent-GCN: 带梯度的全图GCN传播（用于GCN参数更新步）
        
        对所有独立GCN做带梯度的forward，返回增强嵌入字典。
        同时更新detach缓存。
        
        Returns:
            dict: 各GCN的增强嵌入（保留梯度图）
        """
        if not self.use_indep_gcn:
            return {}
        
        results = {}
        
        if self.base_gcn is not None:
            u_in = self.user_embedding.weight.detach()
            s_in = self.service_embedding.weight.detach()
            u_enh, s_enh = self.base_gcn(u_in, s_in)
            results['base_user'] = u_enh
            results['base_service'] = s_enh
            self._indep_gcn_cache['base_user'] = u_enh.detach()
            self._indep_gcn_cache['base_service'] = s_enh.detach()
        
        if self.gmm_gcn is not None and self.gmm_user_embed is not None:
            u_in = self.gmm_user_embed.weight.detach()
            s_in = self.gmm_service_embed.weight.detach()
            u_enh, s_enh = self.gmm_gcn(u_in, s_in)
            results['gmm_user'] = u_enh
            results['gmm_service'] = s_enh
            self._indep_gcn_cache['gmm_user'] = u_enh.detach()
            self._indep_gcn_cache['gmm_service'] = s_enh.detach()
        
        if self.tgem_gcn is not None and self.tgem_user_embed is not None:
            u_in = self.tgem_user_embed.weight.detach()
            s_in = self.tgem_service_embed.weight.detach()
            u_enh, s_enh = self.tgem_gcn(u_in, s_in)
            results['tgem_user'] = u_enh
            results['tgem_service'] = s_enh
            self._indep_gcn_cache['tgem_user'] = u_enh.detach()
            self._indep_gcn_cache['tgem_service'] = s_enh.detach()
        
        return results
    
    def set_sega_graph(self, norm_adj: torch.Tensor,
                       suig_adj: torch.Tensor = None,
                       ssig_adj: torch.Tensor = None):
        """
        V63/V71 SEGA: 设置GCN的归一化邻接矩阵
        
        Args:
            norm_adj: [N, N] FIG稀疏归一化邻接矩阵，N = num_users + num_services
            suig_adj: [num_users, num_users] SUIG稀疏邻接矩阵（V71 MultiGraph）
            ssig_adj: [num_services, num_services] SSIG稀疏邻接矩阵（V71 MultiGraph）
        """
        if self.pgcn is not None:
            if self.use_multi_graph and isinstance(self.pgcn, MultiGraphGCN):
                self.pgcn.set_graph(norm_adj, self.num_users, self.num_services,
                                    suig_adj=suig_adj, ssig_adj=ssig_adj)
                graph_info = f"FIG + "
                if suig_adj is not None:
                    graph_info += f"SUIG({suig_adj._nnz()}nnz) + "
                if ssig_adj is not None:
                    graph_info += f"SSIG({ssig_adj._nnz()}nnz)"
                print(f"[V71 SEGA] MultiGraphGCN graphs set: {graph_info}")
            else:
                self.pgcn.set_graph(norm_adj, self.num_users, self.num_services)
                print(f"[V63 SEGA] ParametricGCN graph set: {self.num_users} users + {self.num_services} services")
        # 初始化缓存为None
        self._sega_user_cache = None
        self._sega_service_cache = None
    
    def set_user_neighbors(self, neighbor_indices: torch.LongTensor, neighbor_counts: torch.LongTensor):
        """
        V76 TPA: 设置每个用户的邻居服务索引（预计算，随模型移动到GPU）
        
        Args:
            neighbor_indices: [num_users, max_K] LongTensor, padding=-1
            neighbor_counts: [num_users] 每个用户的实际邻居数
        """
        device = next(self.parameters()).device
        self._tpa_neighbor_indices = neighbor_indices.to(device)
        self._tpa_neighbor_counts = neighbor_counts.to(device)
        print(f"[V76 TPA] User neighbors set: {neighbor_indices.shape}, "
              f"avg_neighbors={neighbor_counts.float().mean():.1f}, "
              f"max={neighbor_counts.max().item()}, min={neighbor_counts.min().item()}")
    
    def precompute_sega_embeddings(self):
        """
        SEGA v0：预计算全图增强表示并保存不带梯度的缓存。

        训练入口在每个epoch开始时调用本函数，并在周期性的GCN参数更新后
        再次调用。普通batch使用此缓存，梯度仅到达缓存之后的模块，不会
        回传到GCN或共享的原始user/service embedding。周期性更新步通过
        sega_gcn_forward_with_grad()让GCN参数接收梯度；该步的原始
        embedding输入仍被detach。更新间隔由训练入口配置，默认5个epoch。
        """
        if self.pgcn is None or not self.use_sega:
            return
        
        with torch.no_grad():
            all_user_embed = self.user_embedding.weight
            all_service_embed = self.service_embedding.weight
            enhanced_user, enhanced_service = self.pgcn(all_user_embed, all_service_embed)
            self._sega_user_cache = enhanced_user.detach()
            self._sega_service_cache = enhanced_service.detach()
    
    def sega_gcn_forward_with_grad(self):
        """
        V63 SEGA: 带梯度的全图GCN传播（用于GCN参数更新步）
        
        返回增强嵌入（保留梯度图），供外部训练循环计算主任务损失并更新GCN参数。
        调用后会同时更新detach缓存。
        
        Returns:
            enhanced_user: [num_users, embed_dim] 带梯度的增强用户嵌入
            enhanced_service: [num_services, embed_dim] 带梯度的增强服务嵌入
        """
        if self.pgcn is None or not self.use_sega:
            return None, None
        
        # Embedding权重detach：不对全部4642行计算梯度
        # 但GCN参数（W1, W2, gate_alpha）保留梯度图
        all_user_embed = self.user_embedding.weight.detach()
        all_service_embed = self.service_embedding.weight.detach()
        enhanced_user, enhanced_service = self.pgcn(all_user_embed, all_service_embed)
        
        # 同时更新detach缓存（供后续普通batch使用）
        self._sega_user_cache = enhanced_user.detach()
        self._sega_service_cache = enhanced_service.detach()
        
        return enhanced_user, enhanced_service
    
    def init_embeddings_from_nmd(self, user_factors: np.ndarray, service_factors: np.ndarray):
        """
        用NMD(NMF)分解结果初始化所有Embedding层
        
        将NMF因子矩阵L2归一化后缩放到Xavier初始化量级，替换随机Embedding权重。
        
        学术依据: He et al. (WWW 2017) NCF中使用预训练嵌入初始化
        
        Args:
            user_factors: [num_users, embed_dim] NMF用户因子矩阵
            service_factors: [num_services, embed_dim] NMF服务因子矩阵
        """
        # L2归一化 + 缩放到Xavier量级: std ≈ sqrt(2/(fan_in+fan_out))
        target_std = np.sqrt(2.0 / (1 + self.embed_dim))
        
        user_norm = np.linalg.norm(user_factors, axis=1, keepdims=True).clip(min=1e-8)
        service_norm = np.linalg.norm(service_factors, axis=1, keepdims=True).clip(min=1e-8)
        user_normalized = user_factors / user_norm * target_std
        service_normalized = service_factors / service_norm * target_std
        
        user_tensor = torch.tensor(user_normalized, dtype=torch.float32)
        service_tensor = torch.tensor(service_normalized, dtype=torch.float32)
        
        # 初始化所有Embedding层
        with torch.no_grad():
            self.user_embedding.weight.copy_(user_tensor)
            self.service_embedding.weight.copy_(service_tensor)
            if self.gmm_user_embed is not None:
                self.gmm_user_embed.weight.copy_(user_tensor)
            if self.gmm_service_embed is not None:
                self.gmm_service_embed.weight.copy_(service_tensor)
            if self.tgem_user_embed is not None:
                self.tgem_user_embed.weight.copy_(user_tensor)
            if self.tgem_service_embed is not None:
                self.tgem_service_embed.weight.copy_(service_tensor)
    
    def set_cf_factors(self, user_factors: np.ndarray, service_factors: np.ndarray):
        """
        V58: 设置CF-Enhanced Embedding的NMF因子（冻结buffer）
        
        将NMF因子L2归一化后存入register_buffer，训练期间不更新。
        可训练的projection层和gate网络负责学习如何利用这些因子。
        
        Args:
            user_factors: [num_users, cf_factor_dim] NMF用户因子
            service_factors: [num_services, cf_factor_dim] NMF服务因子
        """
        if not self.use_cf_feature:
            print("[V58] Warning: CF feature not enabled, skipping set_cf_factors")
            return
        
        # L2归一化（保持方向信息，消除量级差异）
        user_norm = np.linalg.norm(user_factors, axis=1, keepdims=True).clip(min=1e-8)
        service_norm = np.linalg.norm(service_factors, axis=1, keepdims=True).clip(min=1e-8)
        user_normalized = user_factors / user_norm
        service_normalized = service_factors / service_norm
        
        self.cf_user_factors.copy_(torch.tensor(user_normalized, dtype=torch.float32))
        self.cf_service_factors.copy_(torch.tensor(service_normalized, dtype=torch.float32))
        
        print(f"[V58] CF factors set: user={user_factors.shape}, service={service_factors.shape}")
    
    def init_gmm_from_cf(self, user_cf_embed: np.ndarray, service_cf_embed: np.ndarray):
        """
        V54: 用预计算的CF嵌入初始化GMM的user/service Embedding
        
        只初始化gmm_user_embed和gmm_service_embed，不影响其他Embedding层。
        CF嵌入来自HCFM_Light（LightGCN风格无参数图卷积），包含协同过滤信号。
        
        处理流程：L2归一化 → 缩放到Xavier量级 → 替换GMM Embedding权重
        
        学术依据: HCTN论文中HCFM输出作为GMM的original_embed输入
        
        Args:
            user_cf_embed: [num_users, embed_dim] CF用户嵌入
            service_cf_embed: [num_services, embed_dim] CF服务嵌入
        """
        if self.gmm_user_embed is None or self.gmm_service_embed is None:
            print("[V54] Warning: GMM not enabled, skipping CF embedding init")
            return
        
        target_std = np.sqrt(2.0 / (1 + self.embed_dim))
        
        user_norm = np.linalg.norm(user_cf_embed, axis=1, keepdims=True).clip(min=1e-8)
        service_norm = np.linalg.norm(service_cf_embed, axis=1, keepdims=True).clip(min=1e-8)
        user_scaled = user_cf_embed / user_norm * target_std
        service_scaled = service_cf_embed / service_norm * target_std
        
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32)
        service_tensor = torch.tensor(service_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            self.gmm_user_embed.weight.copy_(user_tensor)
            self.gmm_service_embed.weight.copy_(service_tensor)
        
        print(f"[V54] GMM embeddings initialized from CF: "
              f"user={user_cf_embed.shape}, service={service_cf_embed.shape}, "
              f"target_std={target_std:.4f}")
    
    # 兼容性方法
    def set_hypergraph(self, H): pass
    def set_stat_features(self, user_stat_features, service_stat_features): pass
    def set_gdi_scores(self, user_gdi, service_gdi, user_threshold=1.5, service_threshold=1.5): pass

    @staticmethod
    def _finite_feature(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    
    def forward(self, user_ids: torch.Tensor, service_ids: torch.Tensor,
                user_nmf_seq=None, service_nmf_seq=None,
                qos_history=None, attention_mask=None,
                gpam_features=None, mfvp_features=None,
                anchor=None, nacc_features=None,
                anchor_info=None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_size = user_ids.size(0)
        device = user_ids.device
        
        # ============================================================
        # V63 SEGA: 共享嵌入 + ParametricGCN + 投影头
        # 每个epoch刷新增强嵌入缓存；GCN参数按配置的间隔更新（默认每5个epoch）
        # ============================================================
        if self.use_sega and self.pgcn is not None:
            # 普通batch读取无梯度缓存；周期性GCN更新步临时使用带梯度缓存
            if self._sega_user_cache is not None and self._sega_service_cache is not None:
                # 按batch索引取出预计算的增强嵌入
                user_enhanced = self._sega_user_cache[user_ids]         # [batch, embed_dim]
                service_enhanced = self._sega_service_cache[service_ids] # [batch, embed_dim]
            else:
                # 缓存未初始化时fallback到原始嵌入
                user_enhanced = self.user_embedding(user_ids)
                service_enhanced = self.service_embedding(service_ids)
            
            # V75 FiLM: Target-Aware Conditioning
            # 用目标节点嵌入调制源节点嵌入，使GCN输出变为target-aware
            if self.use_film and self.film_user is not None:
                user_enhanced = self.film_user(user_enhanced, service_enhanced)
                service_enhanced = self.film_service(service_enhanced, user_enhanced)
            
            # V76 TPA: Target-Prompt Attention on Cached GCN Embeddings
            # 用target service嵌入在user的邻居服务嵌入上做注意力聚合，增强user嵌入的target-awareness
            if self.use_tpa and self.tpa is not None and self._tpa_neighbor_indices is not None:
                neighbor_embeds, neighbor_mask = get_neighbor_service_embeds(
                    user_ids, self._sega_service_cache, 
                    self._tpa_neighbor_indices, self._tpa_neighbor_counts,
                    max_K=self.tpa_max_neighbors
                )
                user_enhanced = self.tpa(user_enhanced, service_enhanced, neighbor_embeds, neighbor_mask)
            
            # 投影到各模块输入空间（投影头有梯度，正常训练）
            user_embed = self.proj_base(user_enhanced)          # → CQPM
            service_embed = self.proj_base(service_enhanced)
            
            # GMM（使用投影后的嵌入）
            if self.use_gmm and self.gmm is not None:
                gmm_user_in = self.proj_gmm(user_enhanced)
                gmm_service_in = self.proj_gmm(service_enhanced)
                gmm_output = self.gmm(gmm_user_in, gmm_service_in, user_ids, service_ids)
            else:
                gmm_output = torch.zeros(batch_size, self.embed_dim, device=device)
            
            # TGEM（使用投影后的嵌入）
            if self.use_tgem and self.tgem is not None and qos_history is not None:
                tgem_user_in = self.proj_tgem(user_enhanced)
                tgem_service_in = self.proj_tgem(service_enhanced)
                
                gmm_context = None
                if self.use_serial_gmm and self.use_gmm and hasattr(self, 'gmm_to_tgem_proj'):
                    gmm_context = self.gmm_to_tgem_proj(gmm_output)
                
                tgem_output = self.tgem(qos_history, attention_mask, tgem_user_in, tgem_service_in,
                                        gpam_features=gpam_features, mfvp_features=mfvp_features,
                                        gmm_context=gmm_context)
            else:
                tgem_output = torch.zeros(batch_size, self.embed_dim, device=device)
            
            # CQPM融合 + 预测
            user_embed = self._finite_feature(user_embed)
            service_embed = self._finite_feature(service_embed)
            gmm_output = self._finite_feature(gmm_output)
            tgem_output = self._finite_feature(tgem_output)
            return self.cqpm(user_embed, service_embed, gmm_output, tgem_output,
                            anchor=anchor, nacc_features=nacc_features,
                            anchor_info=anchor_info)
        
        # ============================================================
        # 原始路径（非SEGA模式，保持完全兼容）
        # V64: 支持Independent-GCN增强
        # ============================================================
        # 基础嵌入（V64: 原始嵌入 + GCN增量delta，保持嵌入层梯度）
        if self.use_indep_gcn and 'base_user_delta' in self._indep_gcn_cache:
            user_embed = self.user_embedding(user_ids) + self._indep_gcn_cache['base_user_delta'][user_ids]
            service_embed = self.service_embedding(service_ids) + self._indep_gcn_cache['base_service_delta'][service_ids]
        else:
            user_embed = self.user_embedding(user_ids)
            service_embed = self.service_embedding(service_ids)
        
        # V61: LightGCN嵌入增强（零参数图传播，在所有其他处理之前）
        if self.use_lightgcn and self.lightgcn is not None:
            all_user_embed = self.user_embedding.weight
            all_service_embed = self.service_embedding.weight
            enhanced_user_all, enhanced_service_all = self.lightgcn(all_user_embed, all_service_embed)
            user_embed = enhanced_user_all[user_ids]
            service_embed = enhanced_service_all[service_ids]
        
        # V58: CF-Enhanced Embedding（门控融合NMF因子到基础嵌入）
        if self.use_cf_feature:
            cf_user_raw = self.cf_user_factors[user_ids]
            cf_service_raw = self.cf_service_factors[service_ids]
            cf_user_proj = self.cf_user_proj(cf_user_raw)
            cf_service_proj = self.cf_service_proj(cf_service_raw)
            gate_u = self.cf_user_gate(torch.cat([user_embed, cf_user_proj], dim=-1))
            gate_s = self.cf_service_gate(torch.cat([service_embed, cf_service_proj], dim=-1))
            user_embed = user_embed + gate_u * cf_user_proj
            service_embed = service_embed + gate_s * cf_service_proj
        
        # GMM（V64: 原始嵌入 + GCN增量delta，保持嵌入层梯度）
        if self.use_gmm and self.gmm is not None:
            if self.use_indep_gcn and 'gmm_user_delta' in self._indep_gcn_cache:
                gmm_user_in = self.gmm_user_embed(user_ids) + self._indep_gcn_cache['gmm_user_delta'][user_ids]
                gmm_service_in = self.gmm_service_embed(service_ids) + self._indep_gcn_cache['gmm_service_delta'][service_ids]
            else:
                gmm_user_in = self.gmm_user_embed(user_ids)
                gmm_service_in = self.gmm_service_embed(service_ids)
            gmm_output = self.gmm(gmm_user_in, gmm_service_in, user_ids, service_ids)
        else:
            gmm_output = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # TGEM (MS-TGEM; V50: GPAM; V52: MFVP; V60: serial GMM context)
        # V64: 原始嵌入 + GCN增量delta，保持嵌入层梯度
        if self.use_tgem and self.tgem is not None and qos_history is not None:
            if self.use_indep_gcn and 'tgem_user_delta' in self._indep_gcn_cache:
                tgem_user_in = self.tgem_user_embed(user_ids) + self._indep_gcn_cache['tgem_user_delta'][user_ids]
                tgem_service_in = self.tgem_service_embed(service_ids) + self._indep_gcn_cache['tgem_service_delta'][service_ids]
            else:
                tgem_user_in = self.tgem_user_embed(user_ids)
                tgem_service_in = self.tgem_service_embed(service_ids)
            
            gmm_context = None
            if self.use_serial_gmm and self.use_gmm and hasattr(self, 'gmm_to_tgem_proj'):
                gmm_context = self.gmm_to_tgem_proj(gmm_output)
            
            tgem_output = self.tgem(qos_history, attention_mask, tgem_user_in, tgem_service_in,
                                    gpam_features=gpam_features, mfvp_features=mfvp_features,
                                    gmm_context=gmm_context)
        else:
            tgem_output = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # LGCFE: GCN协同特征增强
        if self.use_lgcfe and self.lgcfe is not None:
            all_user_embed = self.user_embedding.weight
            all_service_embed = self.service_embedding.weight
            user_tgem_enhanced, service_tgem_enhanced = self.lgcfe(
                all_user_embed, all_service_embed,
                user_ids, service_ids,
                tgem_output, tgem_output
            )
            tgem_output = user_tgem_enhanced
        
        # CQPM融合 + 预测（V56: anchor给HAP; V60: nacc_features给NACC; V66: anchor_info给AACAPH）
        user_embed = self._finite_feature(user_embed)
        service_embed = self._finite_feature(service_embed)
        gmm_output = self._finite_feature(gmm_output)
        tgem_output = self._finite_feature(tgem_output)
        return self.cqpm(user_embed, service_embed, gmm_output, tgem_output,
                        anchor=anchor, nacc_features=nacc_features,
                        anchor_info=anchor_info)


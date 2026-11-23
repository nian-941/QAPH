"""
ParallelTGEM - HCTN完全对齐版时序图编码模块
================================================================

核心改进：
1. E-block和T-block、F-block并行处理（HCTN论文架构）
2. T-block和F-block独立处理E-block输出（Z8）
3. 使用soft-attention融合ZT和ZF
4. T-block简化为HCTN原版的两层Conv1D

参考：HCTN论文 Section III-D (TGEM)
- 论文第1855-2117行描述的TGEM架构
- E-block: Transformer Encoder with MHA
- T-block: 两层Conv1D (τ/4 → τ) + 残差
- F-block: 两层Dense (f2/4 → f2) + 残差
- 融合: soft-attention(ZT, ZF)

Author: Kilo Code
Date: 2026-02-03
Version: 34.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ParallelTGEM(nn.Module):
    """
    HCTN对齐版TGEM：E+T+F并行处理
    
    论文流程：
    1. E-block: Z3 → BN → TPE → Dense → BN → MHA → Dense+残差 → BN → Z8
    2. T-block: Z8 → Conv1D(τ/4) → Dropout → Conv1D(τ) → 残差 → ZT
    3. F-block: Z8 → Dense(f2/4) → Dropout → Dense(f2) → 残差 → ZF
    4. 融合: soft-attention(ZT, ZF) → X3
    
    关键差异：
    - T-block和F-block **并行处理Z8**，而不是串行
    - F-block输入是Z8，而不是concat(e_out, t_out)
    - 使用soft-attention融合，而不是门控融合
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_positional_encoding: bool = True
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
            use_positional_encoding: 是否使用位置编码
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        
        # === E-block: Transformer Encoder ===
        # BN1
        self.bn1 = nn.BatchNorm1d(embed_dim)
        
        # 降维 (HCTN论文: 降到1/4维度)
        self.dim_reduce = nn.Linear(embed_dim, embed_dim // 4)
        self.bn2 = nn.BatchNorm1d(embed_dim // 4)
        
        # 位置编码（可选）
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 100, embed_dim // 4) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim // 4,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.e_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 升维 + 残差
        self.dim_restore = nn.Linear(embed_dim // 4, embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        
        # === T-block: 两层Conv1D (HCTN原版) ===
        # Conv1D(embed_dim → embed_dim/4)
        self.t_conv1 = nn.Conv1d(embed_dim, embed_dim // 4, kernel_size=3, padding=1)
        self.t_bn1 = nn.BatchNorm1d(embed_dim // 4)
        self.t_dropout = nn.Dropout(dropout)
        
        # Conv1D(embed_dim/4 → embed_dim)
        self.t_conv2 = nn.Conv1d(embed_dim // 4, embed_dim, kernel_size=3, padding=1)
        self.t_bn2 = nn.BatchNorm1d(embed_dim)
        
        # === F-block: 两层Dense (HCTN原版) ===
        # Dense(embed_dim → embed_dim/4)
        self.f_dense1 = nn.Linear(embed_dim, embed_dim // 4)
        self.f_dropout = nn.Dropout(dropout)
        
        # Dense(embed_dim/4 → embed_dim)
        self.f_dense2 = nn.Linear(embed_dim // 4, embed_dim)
        
        # === Soft-attention融合 ===
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        HCTN对齐版TGEM前向传播
        
        Args:
            x: [batch, seq_len, embed_dim] 输入序列
            mask: Optional attention mask [batch, seq_len]
                  语义: True = 有效位置, False = padding位置
        
        Returns:
            output: [batch, seq_len, embed_dim] 输出序列
        """
        batch_size, seq_len, _ = x.shape
        residual = x  # 保存输入用于最终残差连接
        
        # ========== E-block: Transformer Encoder ==========
        # Step 1: BN1
        x_bn = self.bn1(x.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, embed_dim]
        
        # Step 2: 降维
        x_reduced = self.dim_reduce(x_bn)  # [batch, seq_len, embed_dim/4]
        x_reduced = self.bn2(x_reduced.transpose(1, 2)).transpose(1, 2)
        
        # Step 3: 位置编码（可选）
        if self.use_positional_encoding:
            x_reduced = x_reduced + self.pos_encoding[:, :seq_len, :]
        
        # Step 4: MHA
        # PyTorch的src_key_padding_mask语义: True = 忽略, False = 保留
        # 我们的mask语义: True = 有效, False = padding
        # 所以需要取反
        padding_mask = None
        if mask is not None:
            padding_mask = ~mask
            # Transformer attention cannot handle rows where every time step is masked.
            # For all-missing histories, let the row attend to its zero-valued sequence.
            padding_mask = padding_mask & (~padding_mask.all(dim=1, keepdim=True))
        e_out = self.e_block(x_reduced, src_key_padding_mask=padding_mask)  # [batch, seq_len, embed_dim/4]
        
        # Step 5: 升维 + 残差
        e_out = self.dim_restore(e_out)  # [batch, seq_len, embed_dim]
        z8 = e_out + residual  # 残差连接
        z8 = self.bn3(z8.transpose(1, 2)).transpose(1, 2)  # [batch, seq_len, embed_dim]
        
        # ========== T-block: 两层Conv1D (并行) ==========
        z8_conv = z8.transpose(1, 2)  # [batch, embed_dim, seq_len]
        
        # Conv1D层1
        t_out = self.t_conv1(z8_conv)  # [batch, embed_dim/4, seq_len]
        t_out = self.t_bn1(t_out)
        t_out = F.relu(t_out)
        t_out = self.t_dropout(t_out)
        
        # Conv1D层2
        t_out = self.t_conv2(t_out)  # [batch, embed_dim, seq_len]
        t_out = self.t_bn2(t_out)
        t_out = t_out.transpose(1, 2)  # [batch, seq_len, embed_dim]
        
        # 残差连接
        z_t = t_out + z8  # [batch, seq_len, embed_dim]
        
        # ========== F-block: 两层Dense (并行) ==========
        # Dense层1
        f_out = self.f_dense1(z8)  # [batch, seq_len, embed_dim/4]
        f_out = F.relu(f_out)
        f_out = self.f_dropout(f_out)
        
        # Dense层2
        f_out = self.f_dense2(f_out)  # [batch, seq_len, embed_dim]
        
        # 残差连接
        z_f = f_out + z8  # [batch, seq_len, embed_dim]
        
        # ========== Soft-attention融合 ==========
        concat = torch.cat([z_t, z_f], dim=-1)  # [batch, seq_len, embed_dim * 2]
        attn_weights = self.attention_weights(concat)  # [batch, seq_len, 2]
        
        # 加权融合
        output = attn_weights[:, :, 0:1] * z_t + attn_weights[:, :, 1:2] * z_f
        
        return output


class ParallelTGEMAdapter(nn.Module):
    """
    ParallelTGEM适配器 - 用于V34模型
    
    提供与V33 TGEMAdapter相同的接口，但使用ParallelTGEM实现
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_positional_encoding: bool = True
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            dropout: Dropout比例
            use_positional_encoding: 是否使用位置编码
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # 核心TGEM模块
        self.tgem = ParallelTGEM(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding
        )
        
        # 序列值投影层 (将标量序列转换为embed_dim维度)
        self.value_projection = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        
        # 用户/服务上下文融合
        self.context_fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 序列聚合层 (将seq_len维度聚合为1)
        self.seq_aggregation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.seq_attention = nn.Linear(embed_dim, 1)
    
    def forward(
        self,
        history_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        user_embed: Optional[torch.Tensor] = None,
        service_embed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            history_sequence: [batch, seq_len] 历史QoS序列
            attention_mask: [batch, seq_len] 注意力掩码
            user_embed: [batch, embed_dim] 用户嵌入
            service_embed: [batch, embed_dim] 服务嵌入
        
        Returns:
            output: [batch, embed_dim] 聚合后的时序表示
        """
        batch_size = history_sequence.size(0)
        seq_len = history_sequence.size(1)
        
        # 将标量序列转换为嵌入 [batch, seq_len] -> [batch, seq_len, embed_dim]
        x = history_sequence.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.value_projection(x)  # [batch, seq_len, embed_dim]
        
        # 融合用户/服务上下文
        if user_embed is not None and service_embed is not None:
            # 扩展到序列维度
            user_ctx = user_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, embed_dim]
            service_ctx = service_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, embed_dim]
            
            # 融合
            ctx_input = torch.cat([x, user_ctx, service_ctx], dim=-1)  # [batch, seq_len, embed_dim*3]
            x = self.context_fusion(ctx_input)  # [batch, seq_len, embed_dim]
        
        # ParallelTGEM处理
        tgem_out = self.tgem(x, attention_mask)  # [batch, seq_len, embed_dim]
        
        # 序列聚合 (使用注意力机制)
        attn_scores = self.seq_attention(self.seq_aggregation(tgem_out))  # [batch, seq_len, 1]
        
        # 数值稳定性处理
        attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]
        
        if attention_mask is not None:
            # 将mask为0的位置设为-inf
            mask3 = attention_mask.unsqueeze(-1).bool()
            all_masked = ~mask3.any(dim=1, keepdim=True)
            masked_scores = attn_scores.masked_fill(~mask3, -1e4)
            # Avoid softmax over all-missing histories; short windows can contain
            # pairs with no observed historical QoS values.
            safe_scores = torch.where(all_masked, torch.zeros_like(masked_scores), masked_scores)
            attn_weights = F.softmax(safe_scores, dim=1)
            uniform_weights = torch.ones_like(attn_weights) / attn_weights.size(1)
            attn_weights = torch.where(all_masked, uniform_weights, attn_weights)
        else:
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len, 1]
        
        # 处理可能的NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=1.0/attn_weights.size(1))
        
        output = (tgem_out * attn_weights).sum(dim=1)  # [batch, embed_dim]
        
        # 确保输出无NaN
        output = torch.nan_to_num(output, nan=0.0)
        
        return output


# =============================================================================
# V42 Multi-Scale T-block (MS-TGEM)
# =============================================================================

class MultiScaleTBlock(nn.Module):
    """
    多尺度膨胀卷积T-block - 替换原始固定k=3 Conv1D
    
    1. Conv1d(k=3, dilation=1) - 局部模式 (感受野=3)
    2. Conv1d(k=3, dilation=2) - 中程模式 (感受野=5)
    3. Conv1d(k=3, dilation=4) - 远程模式 (感受野=9)
    4. AdaptiveAvgPool1d    - 全局统计
    
    每个分支输出 embed_dim//4，concat后Linear→LayerNorm→ReLU→残差
    
    参考: SFTCN (IEEE 2025) - Scale Fusion Temporal Convolutional Network
    """
    
    def __init__(self, embed_dim: int, dropout: float = 0.3, num_branches: int = 4):
        super().__init__()
        self.num_branches = num_branches
        
        if num_branches == 5:
            # V73 5分支: 4个卷积分支(各12维) + 全局池化(16维) = 64
            conv_branch_dim = (embed_dim - embed_dim // 4) // 4  # 12 for embed_dim=64
            global_branch_dim = embed_dim - conv_branch_dim * 4  # 16 for embed_dim=64
        else:
            # 原始4分支: 每分支embed_dim//4
            conv_branch_dim = embed_dim // 4
            global_branch_dim = embed_dim // 4
        
        branch_dim = conv_branch_dim  # 兼容后续代码
        
        # Branch 1: dilation=1 (局部)
        self.branch1 = nn.Sequential(
            nn.Conv1d(embed_dim, branch_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
        )
        
        # Branch 2: dilation=2 (中程)
        self.branch2 = nn.Sequential(
            nn.Conv1d(embed_dim, branch_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
        )
        
        # Branch 3: dilation=4 (远程)
        self.branch3 = nn.Sequential(
            nn.Conv1d(embed_dim, branch_dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(branch_dim),
            nn.ReLU(),
        )
        
        # V73: Branch 4 dilation=8 (长程, RF=17) — 仅5分支模式
        if num_branches == 5:
            self.branch4_conv = nn.Sequential(
                nn.Conv1d(embed_dim, conv_branch_dim, kernel_size=3, padding=8, dilation=8),
                nn.BatchNorm1d(conv_branch_dim),
                nn.ReLU(),
            )
        
        # 全局池化分支
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, global_branch_dim),
            nn.ReLU(),
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, z8: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z8: [batch, seq_len, embed_dim] E-block输出
        Returns:
            z_t: [batch, seq_len, embed_dim] 含残差的T-block输出
        """
        batch_size, seq_len, embed_dim = z8.shape
        z8 = torch.nan_to_num(z8, nan=0.0, posinf=1e4, neginf=-1e4)
        x = z8.transpose(1, 2)  # [batch, embed_dim, seq_len]
        
        # 卷积分支
        b1 = torch.nan_to_num(self.branch1(x), nan=0.0, posinf=1e4, neginf=-1e4)  # [batch, branch_dim, seq_len]
        b2 = torch.nan_to_num(self.branch2(x), nan=0.0, posinf=1e4, neginf=-1e4)
        b3 = torch.nan_to_num(self.branch3(x), nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 全局池化分支
        g_pooled = self.global_pool(x).squeeze(-1)  # [batch, embed_dim]
        g = self.global_proj(g_pooled)  # [batch, global_branch_dim]
        g = torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4)
        g = g.unsqueeze(-1).expand(-1, -1, seq_len)  # [batch, global_branch_dim, seq_len]
        
        if self.num_branches == 5:
            b4 = torch.nan_to_num(self.branch4_conv(x), nan=0.0, posinf=1e4, neginf=-1e4)  # [batch, conv_branch_dim, seq_len]
            concat = torch.cat([b1, b2, b3, b4, g], dim=1)  # [batch, embed_dim, seq_len]
        else:
            concat = torch.cat([b1, b2, b3, g], dim=1)  # [batch, embed_dim, seq_len]
        concat = concat.transpose(1, 2)  # [batch, seq_len, embed_dim]
        concat = torch.nan_to_num(concat, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 融合 + 残差
        t_out = self.fusion(concat)
        t_out = torch.nan_to_num(t_out, nan=0.0, posinf=1e4, neginf=-1e4)
        z_t = t_out + z8  # 残差连接
        z_t = torch.nan_to_num(z_t, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return z_t


class ParallelTGEMv42(nn.Module):
    """
    V42版ParallelTGEM - 使用MultiScaleTBlock替换原T-block
    
    V47改进：
    1. E-block维度压缩比从1/4调整为1/2 (64→32替代64→16)
       - 每个注意力头从4维提升到8维，捕获更丰富的时序依赖
    2. 与ParallelTGEM的区别：T-block使用多尺度膨胀卷积
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_positional_encoding: bool = True,
        e_block_ratio: int = 4,  # E-block压缩比，默认1/4
        num_branches: int = 4,  # V73: T-block分支数(4或5)
        disable_tblock: bool = False,  # TGEM消融: 禁用T-block
        disable_fblock: bool = False,  # TGEM消融: 禁用F-block
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_positional_encoding = use_positional_encoding
        self.disable_tblock = disable_tblock
        self.disable_fblock = disable_fblock
        self.e_block_dim = embed_dim // e_block_ratio  # V47: 可配置的E-block维度
        
        # === E-block: Transformer Encoder (V47: 64→32替代64→16) ===
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.dim_reduce = nn.Linear(embed_dim, self.e_block_dim)
        self.bn2 = nn.BatchNorm1d(self.e_block_dim)
        
        if use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 100, self.e_block_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.e_block_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False
        )
        self.e_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dim_restore = nn.Linear(self.e_block_dim, embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim)
        
        # === T-block: 多尺度膨胀卷积 (V42改进) ===
        self.t_block = MultiScaleTBlock(embed_dim=embed_dim, dropout=dropout, num_branches=num_branches)
        
        # === F-block: 两层Dense (与原版相同) ===
        self.f_dense1 = nn.Linear(embed_dim, embed_dim // 4)
        self.f_dropout = nn.Dropout(dropout)
        self.f_dense2 = nn.Linear(embed_dim // 4, embed_dim)
        
        # === Soft-attention融合 (与原版相同) ===
        self.attention_weights = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Tanh(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional [batch, seq_len], True=有效, False=padding
        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        residual = x
        
        # E-block
        x_bn = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x_bn = torch.nan_to_num(x_bn, nan=0.0, posinf=1e4, neginf=-1e4)
        x_reduced = self.dim_reduce(x_bn)
        x_reduced = torch.nan_to_num(x_reduced, nan=0.0, posinf=1e4, neginf=-1e4)
        x_reduced = self.bn2(x_reduced.transpose(1, 2)).transpose(1, 2)
        x_reduced = torch.nan_to_num(x_reduced, nan=0.0, posinf=1e4, neginf=-1e4)
        
        if self.use_positional_encoding:
            x_reduced = x_reduced + self.pos_encoding[:, :seq_len, :]
            x_reduced = torch.nan_to_num(x_reduced, nan=0.0, posinf=1e4, neginf=-1e4)
        
        padding_mask = None
        if mask is not None:
            padding_mask = ~mask
            # Transformer attention cannot handle a sample whose every time step is masked:
            # softmax over all -inf produces NaN gradients. Short history windows can create
            # all-missing histories, so let those rows attend to their zero-valued sequence.
            padding_mask = padding_mask & (~padding_mask.all(dim=1, keepdim=True))
        e_out = self.e_block(x_reduced, src_key_padding_mask=padding_mask)
        e_out = torch.nan_to_num(e_out, nan=0.0, posinf=1e4, neginf=-1e4)
        e_out = self.dim_restore(e_out)
        e_out = torch.nan_to_num(e_out, nan=0.0, posinf=1e4, neginf=-1e4)
        z8 = e_out + residual
        z8 = torch.nan_to_num(z8, nan=0.0, posinf=1e4, neginf=-1e4)
        z8 = self.bn3(z8.transpose(1, 2)).transpose(1, 2)
        z8 = torch.nan_to_num(z8, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # T-block (多尺度膨胀卷积) - 可禁用用于消融实验
        if not self.disable_tblock:
            z_t = self.t_block(z8)
        else:
            z_t = z8  # 跳过T-block，直接传递E-block输出
        
        # F-block - 可禁用用于消融实验
        if not self.disable_fblock:
            f_out = self.f_dense1(z8)
            f_out = F.relu(f_out)
            f_out = self.f_dropout(f_out)
            f_out = self.f_dense2(f_out)
            z_f = f_out + z8
        else:
            z_f = z8  # 跳过F-block，直接传递E-block输出
        
        # Soft-attention融合 (即使某个分支被禁用，仍通过融合层保持维度一致)
        concat = torch.cat([z_t, z_f], dim=-1)
        attn_weights = self.attention_weights(concat)
        output = attn_weights[:, :, 0:1] * z_t + attn_weights[:, :, 1:2] * z_f
        
        return output


class ParallelTGEMAdapterV42(nn.Module):
    """V42版TGEM适配器 - 使用ParallelTGEMv42
    
    V50改进: 支持GPAM-Aligned Input
    V52改进: 支持MFVP (Multi-Feature Value Projection)
    - mfvp_input_dim > 0时，value_projection输入维度从1变为mfvp_input_dim
    - MFVP为每个时间步构造多维特征向量（原始值+log1p+局部统计+变化率+分位数+非零指示）
    """
    
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_positional_encoding: bool = True,
        e_block_ratio: int = 4,  # E-block压缩比，默认1/4
        use_missing_token: bool = False,  # 是否使用可学习缺失值token
        gpam_input_dim: int = 0,  # V50: GPAM因子输入维度 (0=禁用, >0=2*f1)
        mfvp_input_dim: int = 0,  # V52: MFVP特征输入维度 (0=禁用, >0=特征数)
        num_branches: int = 4,  # V73: T-block分支数(4或5)
        gmm_context_dim: int = 0,  # V60: GMM→TGEM串行化上下文维度 (0=禁用)
        disable_tblock: bool = False,  # TGEM ablation: disable T-block
        disable_fblock: bool = False,  # TGEM ablation: disable F-block
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_missing_token = use_missing_token
        self.gpam_input_dim = gpam_input_dim
        self.mfvp_input_dim = mfvp_input_dim
        self.gmm_context_dim = gmm_context_dim
        
        # 使用V42版TGEM
        self.tgem = ParallelTGEMv42(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_positional_encoding=use_positional_encoding,
            e_block_ratio=e_block_ratio,
            num_branches=num_branches,
            disable_tblock=disable_tblock,
            disable_fblock=disable_fblock,
        )
        
        # 序列值投影层 - 输入维度根据模式决定
        if mfvp_input_dim > 0:
            # V52 MFVP模式: 输入为 [batch, seq_len, mfvp_input_dim] (多维特征)
            self.value_projection = nn.Sequential(
                nn.Linear(mfvp_input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        elif gpam_input_dim > 0:
            # V50 GPAM模式: 输入为 [batch, seq_len, 2*f1+1] (NMF因子 + 原始QoS值)
            vp_input_dim = gpam_input_dim + 1  # +1 for original QoS scalar
            self.value_projection = nn.Sequential(
                nn.Linear(vp_input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim),
            )
        else:
            # 原始模式: 输入为 [batch, seq_len, 1] (单标量QoS值)
            self.value_projection = nn.Sequential(
                nn.Linear(1, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, embed_dim),
            )
        
        # 可学习的缺失值token
        if use_missing_token:
            self.missing_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)
        
        # 用户/服务上下文融合 (V60: +gmm_context时为4d输入)
        ctx_input_dim = embed_dim * 3  # x + user + service
        if gmm_context_dim > 0:
            ctx_input_dim += gmm_context_dim  # + gmm_context
        self.context_fusion = nn.Sequential(
            nn.Linear(ctx_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 序列聚合层
        self.seq_aggregation = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh(),
        )
        self.seq_attention = nn.Linear(embed_dim, 1)
    
    def forward(
        self,
        history_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        user_embed: Optional[torch.Tensor] = None,
        service_embed: Optional[torch.Tensor] = None,
        gpam_features: Optional[torch.Tensor] = None,
        mfvp_features: Optional[torch.Tensor] = None,
        gmm_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            history_sequence: [batch, seq_len] 原始QoS历史序列
            attention_mask: [batch, seq_len]
            user_embed: [batch, embed_dim]
            service_embed: [batch, embed_dim]
            gpam_features: [batch, seq_len, 2*f1] V50 GPAM NMF因子 (可选)
            mfvp_features: [batch, seq_len, mfvp_dim] V52 MFVP多维特征 (可选)
            gmm_context: [batch, embed_dim] V60 GMM串行化上下文 (可选)
        Returns:
            output: [batch, embed_dim]
        """
        batch_size = history_sequence.size(0)
        seq_len = history_sequence.size(1)
        history_sequence = torch.nan_to_num(history_sequence, nan=0.0, posinf=1e4, neginf=0.0)
        if gpam_features is not None:
            gpam_features = torch.nan_to_num(gpam_features, nan=0.0, posinf=1e4, neginf=-1e4)
        if mfvp_features is not None:
            mfvp_features = torch.nan_to_num(mfvp_features, nan=0.0, posinf=1e4, neginf=-1e4)
        if user_embed is not None:
            user_embed = torch.nan_to_num(user_embed, nan=0.0, posinf=1e4, neginf=-1e4)
        if service_embed is not None:
            service_embed = torch.nan_to_num(service_embed, nan=0.0, posinf=1e4, neginf=-1e4)
        if gmm_context is not None:
            gmm_context = torch.nan_to_num(gmm_context, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 值投影 - 根据模式决定输入
        if self.mfvp_input_dim > 0 and mfvp_features is not None:
            # V52 MFVP模式: 直接使用预计算的多维特征
            x = self.value_projection(mfvp_features)  # [batch, seq_len, embed_dim]
        elif self.gpam_input_dim > 0 and gpam_features is not None:
            # V50 GPAM模式: 拼接原始QoS值和NMF因子
            qos_scalar = history_sequence.unsqueeze(-1)  # [batch, seq_len, 1]
            x = torch.cat([qos_scalar, gpam_features], dim=-1)  # [batch, seq_len, 2*f1+1]
            x = self.value_projection(x)  # [batch, seq_len, embed_dim]
        else:
            # 原始模式: 单标量
            x = history_sequence.unsqueeze(-1)  # [batch, seq_len, 1]
            x = self.value_projection(x)  # [batch, seq_len, embed_dim]
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # V47改进点3: 用missing_token替换0值位置
        if self.use_missing_token and attention_mask is not None:
            missing_mask = (~attention_mask).unsqueeze(-1)  # [batch, seq_len, 1], True=缺失
            x = torch.where(missing_mask, self.missing_token.expand(batch_size, seq_len, -1), x)
        
        # 上下文融合 (V60: 可选GMM上下文)
        if user_embed is not None and service_embed is not None:
            user_ctx = user_embed.unsqueeze(1).expand(-1, seq_len, -1)
            service_ctx = service_embed.unsqueeze(1).expand(-1, seq_len, -1)
            ctx_parts = [x, user_ctx, service_ctx]
            if self.gmm_context_dim > 0 and gmm_context is not None:
                gmm_ctx = gmm_context.unsqueeze(1).expand(-1, seq_len, -1)
                ctx_parts.append(gmm_ctx)
            ctx_input = torch.cat(ctx_parts, dim=-1)
            x = self.context_fusion(ctx_input)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # ParallelTGEM处理
        tgem_out = self.tgem(x, mask=attention_mask)
        tgem_out = torch.nan_to_num(tgem_out, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # 序列聚合（仍使用mask区分有效/缺失位置）
        attn_scores = self.seq_attention(self.seq_aggregation(tgem_out))
        attn_scores = attn_scores - attn_scores.max(dim=1, keepdim=True)[0]
        
        if attention_mask is not None:
            mask3 = attention_mask.unsqueeze(-1).bool()
            all_masked = ~mask3.any(dim=1, keepdim=True)
            masked_scores = attn_scores.masked_fill(~mask3, -1e4)
            # Avoid softmax over all-missing histories; short windows can contain
            # pairs with no observed historical QoS values.
            safe_scores = torch.where(all_masked, torch.zeros_like(masked_scores), masked_scores)
            attn_weights = F.softmax(safe_scores, dim=1)
            uniform_weights = torch.ones_like(attn_weights) / attn_weights.size(1)
            attn_weights = torch.where(all_masked, uniform_weights, attn_weights)
        else:
            attn_weights = F.softmax(attn_scores, dim=1)
        
        attn_weights = torch.nan_to_num(attn_weights, nan=1.0/attn_weights.size(1))
        output = (tgem_out * attn_weights).sum(dim=1)
        output = torch.nan_to_num(output, nan=0.0)
        
        return output


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ParallelTGEM 测试")
    print("=" * 60)
    
    # 测试参数
    batch_size = 32
    seq_len = 64
    embed_dim = 128
    
    # 1. 测试ParallelTGEM
    print("\n1. ParallelTGEM:")
    model = ParallelTGEM(embed_dim=embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    mask = torch.ones(batch_size, seq_len).bool()
    
    output = model(x, mask)
    print(f"   Input:  {x.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. 测试ParallelTGEMAdapter
    print("\n2. ParallelTGEMAdapter:")
    adapter = ParallelTGEMAdapter(embed_dim=embed_dim)
    history_seq = torch.randn(batch_size, seq_len)
    user_embed = torch.randn(batch_size, embed_dim)
    service_embed = torch.randn(batch_size, embed_dim)
    attention_mask = torch.ones(batch_size, seq_len).bool()
    
    output = adapter(history_seq, attention_mask, user_embed, service_embed)
    print(f"   Input history_sequence: {history_seq.shape}")
    print(f"   Input user_embed: {user_embed.shape}")
    print(f"   Input service_embed: {service_embed.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)

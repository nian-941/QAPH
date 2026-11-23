"""Frozen-v0 QAPH implementation, organized by responsibility. Original source credit: Kilo Code."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional, Tuple

class BarronAdaptiveLoss(nn.Module):
    """
    Barron自适应损失函数
    
    ρ(x, α, c) = |α-2|/α × ((x/c)²/|α-2| + 1)^(α/2) - 1)
    
    特殊情况：
    - α=2: L2 loss
    - α=1: Charbonnier/pseudo-Huber loss  
    - α=0: Cauchy/Lorentzian loss
    - α=-2: Geman-McClure loss
    - α=-∞: Welsch/Leclerc loss
    
    α和scale(c)均可学习，通过梯度下降自动适应数据分布。
    
    参考: Barron (CVPR 2019) "A General and Adaptive Robust Loss Function"
    """
    
    def __init__(self, init_alpha: float = 0.0, init_scale: float = 1.0, 
                 alpha_lo: float = -10.0, alpha_hi: float = 3.0,
                 scale_lo: float = 0.001, scale_hi: float = 100.0):
        super().__init__()
        # 可学习参数
        self.log_alpha = nn.Parameter(torch.tensor(float(init_alpha)))  # α直接学习
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_scale)))  # log(c)
        
        # 裁剪范围
        self.alpha_lo = alpha_lo
        self.alpha_hi = alpha_hi
        self.scale_lo = scale_lo
        self.scale_hi = scale_hi
    
    @property
    def alpha(self):
        return self.log_alpha.clamp(self.alpha_lo, self.alpha_hi)
    
    @property
    def scale(self):
        return self.log_scale.exp().clamp(self.scale_lo, self.scale_hi)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算Barron损失
        
        Args:
            pred: 预测值 [batch]
            target: 目标值 [batch]
        Returns:
            loss: 标量损失
        """
        diff = pred.squeeze() - target
        alpha = self.alpha
        c = self.scale
        
        # 归一化残差
        x = diff / c
        x_sq = x ** 2
        
        # 数值稳定的Barron损失计算
        # 对不同α范围使用不同公式避免数值问题
        abs_alpha_minus_2 = torch.abs(alpha - 2.0).clamp(min=1e-8)
        
        # 通用公式: |α-2|/α × ((x²/|α-2| + 1)^(α/2) - 1)
        # 当α接近0时特殊处理
        if alpha.abs() < 1e-4:
            # α≈0: Cauchy loss = log(x²/2 + 1)
            loss = torch.log(x_sq / 2.0 + 1.0)
        elif (alpha - 2.0).abs() < 1e-4:
            # α≈2: L2 loss = x²/2
            loss = x_sq / 2.0
        else:
            # 通用公式
            base = x_sq / abs_alpha_minus_2 + 1.0
            exponent = alpha / 2.0
            loss = abs_alpha_minus_2 / alpha * (base.pow(exponent) - 1.0)
        
        # Barron原论文的partition function正则化: log(c)
        # 使用max(log(c), 0)防止c<1时loss变负导致训练不稳定
        loss = loss + torch.log(c).clamp(min=0.0)
        
        return loss.mean()
    
    def extra_repr(self):
        return f'alpha={self.alpha.item():.3f}, scale={self.scale.item():.3f}'


class SegmentedCauchyLoss(nn.Module):
    """
    分段Cauchy损失函数
    
    对不同目标值范围使用不同的Cauchy γ参数：
    - 小值样本（target < threshold）：用较小γ，提高精度
    - 大值样本（target ≥ threshold）：用较大γ，容忍更大绝对误差
    
    学术依据：分段损失函数在回归任务中有广泛应用（如Huber Loss本身就是分段的），
    SCL是Cauchy Loss针对长尾分布数据的自然扩展。
    
    参考: Huber (1964) "Robust Estimation of a Location Parameter"
    """
    
    def __init__(self, gamma_small: float = 2.0, gamma_large: float = 10.0,
                 threshold: float = 10.0):
        super().__init__()
        self.gamma_small = gamma_small
        self.gamma_large = gamma_large
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.squeeze()
        diff = pred_flat - target
        
        # 根据目标值大小选择γ
        small_mask = (target.abs() < self.threshold)
        
        # 分段计算Cauchy loss
        loss = torch.zeros_like(diff)
        if small_mask.any():
            x_small = (diff[small_mask] / self.gamma_small).clamp(-1e6, 1e6)
            loss[small_mask] = torch.log1p(x_small ** 2)
        if (~small_mask).any():
            x_large = (diff[~small_mask] / self.gamma_large).clamp(-1e6, 1e6)
            loss[~small_mask] = torch.log1p(x_large ** 2)
        
        return loss.mean()
    
    def extra_repr(self):
        return f'gamma_small={self.gamma_small}, gamma_large={self.gamma_large}, threshold={self.threshold}'


class LogSpaceCauchyLoss(nn.Module):
    """
    对数空间Cauchy损失 (Log-Space Cauchy Loss, LSCL)
    
    核心思想：在对数空间计算Cauchy Loss，模型输入输出仍在原始空间。
    
    公式: L_LSCL = log(1 + ((log(1+pred) - log(1+target)) / γ)²)
    
    解决的问题：
    - TP值域跨8个数量级，标准Cauchy Loss的梯度 2x/(γ²+x²) 在 |x|>>γ 时趋近于0
    - 对数空间将8个数量级压缩到~4个数量级，使梯度在所有尺度上保持有效
    - log1p保证小值区域近似线性，不影响小值样本精度
    
    学术依据：
    - 这是损失函数设计，不是数据预处理
    - 类似Focal Loss改变梯度分布的思路
    - 参考: Barron (CVPR 2019) 自适应损失函数框架
    
    Args:
        gamma: Cauchy尺度参数（在对数空间中）
    """
    
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch] 预测值（原始空间）
            target: [batch] 目标值（原始空间）
        Returns:
            loss: 标量损失
        """
        pred_flat = pred.squeeze()
        # log1p: log(1+x)，保证x=0时输出0，小值近似线性
        pred_log = torch.log1p(pred_flat.clamp(min=0))
        target_log = torch.log1p(target.clamp(min=0))
        diff = (pred_log - target_log) / self.gamma
        diff = diff.clamp(-1e6, 1e6)
        loss = torch.log1p(diff ** 2)
        return loss.mean()
    
    def extra_repr(self):
        return f'gamma={self.gamma}'


class AdaptiveGradientHarmonization(nn.Module):
    """
    自适应梯度调和损失 - Focal Loss回归变体
    
    核心思想：根据样本预测难度动态调整梯度权重
    - 难样本（大误差）：增大权重，加强学习
    - 易样本（小误差）：减小权重，避免过拟合
    
    公式: L_AGH = w(e) × L_base(e)
    其中 w(e) = (|e|/μ)^γ, e=pred-target, μ=batch均值误差
    γ为聚焦参数（可学习或固定）
    
    与Focal Loss的区别：
    - Focal Loss用于分类，基于预测概率
    - AGH用于回归，基于预测误差相对大小
    """
    
    def __init__(self, base_loss: nn.Module, init_gamma: float = 1.0,
                 gamma_lo: float = 0.1, gamma_hi: float = 5.0,
                 learnable: bool = True):
        super().__init__()
        self.base_loss = base_loss
        self.learnable = learnable
        self.gamma_lo = gamma_lo
        self.gamma_hi = gamma_hi
        
        if learnable:
            self.log_gamma = nn.Parameter(torch.tensor(math.log(init_gamma)))
        else:
            # 固定γ模式：不创建Parameter，不参与梯度更新
            self.register_buffer('_fixed_gamma', torch.tensor(float(init_gamma)))
    
    @property
    def gamma(self):
        if self.learnable:
            return self.log_gamma.exp().clamp(self.gamma_lo, self.gamma_hi)
        else:
            return self._fixed_gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [batch] 预测值
            target: [batch] 目标值
        Returns:
            loss: 标量
        """
        pred_flat = pred.squeeze()
        abs_error = (pred_flat - target).abs().detach()
        
        # 计算batch内的均值误差作为归一化基准
        mean_error = abs_error.mean().clamp(min=1e-6)
        
        # 计算样本权重: (|e|/μ)^γ
        gamma = self.gamma
        relative_error = (abs_error / mean_error).clamp(max=10.0)  # 防止极端值
        weights = relative_error.pow(gamma)
        
        # 归一化权重使均值为1（保持loss量级不变）
        weights = weights / weights.mean().clamp(min=1e-6)
        weights = weights.detach()  # 权重不参与梯度计算
        
        # 逐样本计算base loss
        base_loss = self._per_sample_loss(pred_flat, target)
        
        # 加权
        loss = (weights * base_loss).mean()
        
        return loss
    
    def _per_sample_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算逐样本的base loss"""
        if isinstance(self.base_loss, BarronAdaptiveLoss):
            diff = pred - target
            alpha = self.base_loss.alpha
            c = self.base_loss.scale
            x = diff / c
            x_sq = x ** 2
            abs_alpha_minus_2 = torch.abs(alpha - 2.0).clamp(min=1e-8)
            
            if alpha.abs() < 1e-4:
                loss = torch.log(x_sq / 2.0 + 1.0)
            elif (alpha - 2.0).abs() < 1e-4:
                loss = x_sq / 2.0
            else:
                base = x_sq / abs_alpha_minus_2 + 1.0
                exponent = alpha / 2.0
                loss = abs_alpha_minus_2 / alpha * (base.pow(exponent) - 1.0)
            
            loss = loss + torch.log(c).clamp(min=0.0)
            return loss
        elif isinstance(self.base_loss, LogSpaceCauchyLoss):
            # LSCL逐样本计算（对数空间Cauchy）
            pred_log = torch.log1p(pred.clamp(min=0))
            target_log = torch.log1p(target.clamp(min=0))
            diff = (pred_log - target_log) / self.base_loss.gamma
            diff = diff.clamp(-1e6, 1e6)
            return torch.log1p(diff ** 2)
        elif isinstance(self.base_loss, SegmentedCauchyLoss):
            # SCL逐样本计算
            diff = pred - target
            small_mask = (target.abs() < self.base_loss.threshold)
            loss = torch.zeros_like(diff)
            if small_mask.any():
                x_small = (diff[small_mask] / self.base_loss.gamma_small).clamp(-1e6, 1e6)
                loss[small_mask] = torch.log1p(x_small ** 2)
            if (~small_mask).any():
                x_large = (diff[~small_mask] / self.base_loss.gamma_large).clamp(-1e6, 1e6)
                loss[~small_mask] = torch.log1p(x_large ** 2)
            return loss
        else:
            # 通用fallback: 逐样本Cauchy/MSE
            diff = pred - target
            if hasattr(self.base_loss, 'gamma'):
                g = self.base_loss.gamma if isinstance(self.base_loss.gamma, (int, float)) else 1.0
                x = (diff / g).clamp(-1e6, 1e6)
                return torch.log1p(x ** 2)
            return diff ** 2
    
    def extra_repr(self):
        return f'gamma={self.gamma.item():.3f}'



class CauchyLoss(nn.Module):
    """Fixed Cauchy loss."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred.squeeze() - target
        x = (diff / self.gamma).clamp(-1e6, 1e6)
        loss = torch.log1p(x ** 2)
        return loss.mean()


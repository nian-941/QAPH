"""
HCTN数据划分器 - 完全复现HCTN论文的数据划分方式
================================================================

HCTN论文原文 (第2196-2212行):
"Since the datasets are temporal, we retain the first τ time-steps
exactly as they appear in the original QoS tensor to serve as the
historical input sequence. For the final (τ + 1)th time-step, we
randomly sample ψ% of the valid QoS entries for training and use
the remaining (100−ψ)% entries for testing; additionally, 20% of
the sampled training entries are reserved for validation."

关键差异：
- HCTN: 预测单个时间步，在同一时间步上随机采样
- 我们之前: 预测多个时间步，按时间顺序划分

这是达到SOTA的最关键修复！

Author: Kilo Code
Date: 2026-01-12
Version: 7.13.0
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class HCTNDataSplitter:
    """
    按照HCTN论文的方式划分数据
    
    任务定义:
    - 输入: 前τ个时间步的历史QoS数据
    - 输出: 预测第(τ+1)个时间步的QoS值
    
    数据划分:
    - 前τ个时间步 → 历史输入序列（保持原样）
    - 第(τ+1)个时间步 → 用于训练/验证/测试
        - 随机采样ψ%的有效QoS条目 → 训练集
        - 其中20%保留为验证集
        - 剩余(100-ψ)% → 测试集
    """
    
    def __init__(
        self,
        tau: int = 63,  # 历史时间步数量（WS-DREAM2有64个时间步，使用前63个作为历史）
        psi: float = 0.10,  # 训练密度（10%）
        val_ratio: float = 0.20,  # 验证集比例（训练集的20%）
        seed: int = 42,
    ):
        """
        Args:
            tau: 历史时间步数量（默认63，使用前63个时间步作为历史）
            psi: 训练密度（默认10%）
            val_ratio: 验证集比例（默认20%的训练集）
            seed: 随机种子
        """
        self.tau = tau
        self.psi = psi
        self.val_ratio = val_ratio
        self.seed = seed
        
        print(f"[HCTN Splitter] Initialized with tau={tau}, psi={psi:.0%}, val_ratio={val_ratio:.0%}")
    
    def split(self, qos_tensor: np.ndarray) -> Dict:
        """
        划分数据
        
        Args:
            qos_tensor: [n_users, n_services, n_timesteps] QoS张量
        
        Returns:
            dict: {
                'history': [n_users, n_services, tau] 历史数据
                'target_matrix': [n_users, n_services] 第(tau+1)个时间步的QoS矩阵
                'train_indices': 训练集索引 [(user_id, service_id), ...]
                'val_indices': 验证集索引
                'test_indices': 测试集索引
            }
        """
        n_users, n_services, n_timesteps = qos_tensor.shape
        
        print(f"[HCTN Splitter] QoS tensor shape: {qos_tensor.shape}")
        print(f"[HCTN Splitter] Using first {self.tau} time steps as history")
        print(f"[HCTN Splitter] Target time step: {self.tau} (0-indexed)")
        
        # 1. 提取历史数据 (前tau个时间步)
        # History-window sensitivity is handled after this split in run_v42_experiments.py.
        # Keeping tau=63 here preserves the fixed target-slice-64 protocol and makes
        # train/val/test indices independent of history_length.
        history = qos_tensor[:, :, :self.tau]
        
        # 2. 提取目标时间步 (第tau+1个时间步，即索引tau)
        target_matrix = qos_tensor[:, :, self.tau]
        
        # 3. 找出所有有效的QoS条目（非零值）
        valid_mask = target_matrix > 0
        valid_indices = np.argwhere(valid_mask)  # [(user_id, service_id), ...]
        n_valid = len(valid_indices)
        
        print(f"[HCTN Splitter] Valid entries in target time step: {n_valid:,}")
        print(f"[HCTN Splitter] Valid ratio: {n_valid / (n_users * n_services):.2%}")
        
        # 4. 随机采样psi%作为训练集
        np.random.seed(self.seed)
        n_train_total = int(n_valid * self.psi)
        
        shuffled_indices = np.random.permutation(n_valid)
        train_val_indices = valid_indices[shuffled_indices[:n_train_total]]
        test_indices = valid_indices[shuffled_indices[n_train_total:]]
        
        # 5. 从训练集中划分20%作为验证集
        n_val = int(len(train_val_indices) * self.val_ratio)
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]
        
        print(f"[HCTN Splitter] Train samples: {len(train_indices):,} ({len(train_indices)/n_valid:.2%})")
        print(f"[HCTN Splitter] Val samples: {len(val_indices):,} ({len(val_indices)/n_valid:.2%})")
        print(f"[HCTN Splitter] Test samples: {len(test_indices):,} ({len(test_indices)/n_valid:.2%})")
        
        return {
            'history': history,
            'target_matrix': target_matrix,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': test_indices,
            'n_users': n_users,
            'n_services': n_services,
            'n_timesteps': n_timesteps,
        }


class HCTNDataset(Dataset):
    """
    HCTN风格的数据集
    
    每个样本包含:
    - user_id: 用户ID
    - service_id: 服务ID
    - history_sequence: 该用户-服务对的历史QoS序列 [seq_len]
    - target_value: 目标时间步的QoS值
    - density: 数据密度
    - history_completeness: 历史完整度
    
    ✅ V7.22 消融实验支持：
    - use_log_transform参数可显式控制是否使用log变换
    - None: 自动模式（TP用log变换，RT不用）
    - True: 强制使用log变换
    - False: 强制不使用log变换
    """
    
    def __init__(
        self,
        history: np.ndarray,  # [n_users, n_services, tau]
        target_matrix: np.ndarray,  # [n_users, n_services]
        indices: np.ndarray,  # [(user_id, service_id), ...]
        seq_len: int = 63,  # Bug 57修复: 使用全部τ=63个时间步 (HCTN论文一致)
        normalize: bool = True,
        scaler: Optional[MinMaxScaler] = None,
        qos_type: str = 'rt',
        cf_extractor = None,  # CF特征提取器
        nmf_matrix: Optional[np.ndarray] = None,  # NMF矩阵
        density: float = 0.10,  # 数据密度
        qos_min: Optional[float] = None,  # 归一化最小值（用于共享scaler）
        qos_max: Optional[float] = None,  # 归一化最大值（用于共享scaler）
        use_log_transform: Optional[bool] = None,  # ✅ V7.22: 显式控制log变换
    ):
        self.history = history
        self.target_matrix = target_matrix
        self.indices = indices
        self.seq_len = seq_len
        self.normalize = normalize
        self.qos_type = qos_type
        self.cf_extractor = cf_extractor
        self.nmf_matrix = nmf_matrix
        self.density = density
        
        n_users, n_services, tau = history.shape
        self.n_users = n_users
        self.n_services = n_services
        self.tau = tau
        
        # ✅ V7.24: 按照HCTN论文，RT和TP都不使用log变换
        # HCTN论文明确在原始空间计算MAE/RMSE，没有任何log变换
        # 参考: HCTN论文 Table I 显示原始数据范围，Eq.5 Cauchy Loss直接使用原始值
        if use_log_transform is None:
            # ✅ V7.24修复: 默认不使用log变换（与HCTN论文一致）
            self._use_log_transform_param = None
            self.use_log_transform = False  # ✅ 关键修复：始终不使用log变换
        else:
            # 显式指定（用于消融实验对比）
            self._use_log_transform_param = use_log_transform
            self.use_log_transform = use_log_transform
        
        print(f"[HCTN Dataset V7.24] Log transform: {self.use_log_transform} (param={use_log_transform}, qos_type={qos_type})")
        if not self.use_log_transform:
            print(f"[HCTN Dataset V7.24] ✅ Using raw QoS values (HCTN paper compliant)")
        
        # 归一化
        if normalize:
            if scaler is not None:
                self.scaler = scaler
                # 使用传入的qos_min和qos_max
                if qos_min is not None and qos_max is not None:
                    self.qos_min = qos_min
                    self.qos_max = qos_max
                else:
                    # 从scaler中提取
                    self.qos_min = float(scaler.data_min_[0])
                    self.qos_max = float(scaler.data_max_[0])
                # ✅ V7.22: 不再根据qos_type自动设置，使用已确定的值
            else:
                self.scaler = self._fit_scaler()
            
            # 归一化历史数据和目标数据
            self.history_normalized = self._normalize_tensor(self.history)
            self.target_normalized = self._normalize_matrix(self.target_matrix)
        else:
            # ✅ V7.28修复: normalize=False模式下也需要设置qos_min和qos_max
            # 用于denormalize方法（虽然不会真正改变值，但需要避免AttributeError）
            self.scaler = None
            self.history_normalized = self.history.copy().astype(np.float32)  # 使用copy避免修改原数据
            self.target_normalized = self.target_matrix.copy().astype(np.float32)
            
            # 设置qos_min和qos_max为恒等变换参数（denormalize时不改变值）
            self.qos_min = 0.0
            self.qos_max = 1.0
            
            print(f"[HCTN Dataset V7.28] normalize=False: Using raw QoS values (no transformation)")
        
        # 预计算所有样本
        self.samples = self._precompute_samples()
        
        print(f"[HCTN Dataset] Created with {len(self.samples)} samples")
    
    def _fit_scaler(self) -> MinMaxScaler:
        """
        拟合归一化器
        
        ✅ V7.22: 使用self.use_log_transform控制是否应用log变换
        """
        # 使用历史数据拟合（避免数据泄漏）
        valid_values = self.history[self.history > 0]
        
        # ✅ V7.22: 根据use_log_transform决定是否应用log变换
        if self.use_log_transform:
            valid_values = np.log1p(valid_values)
            print(f"[HCTN Dataset] Applied log1p transform to valid values")
        else:
            print(f"[HCTN Dataset] Skipping log transform (use_log_transform=False)")
        
        # 使用1%-99%分位数避免异常值
        qos_min = np.percentile(valid_values, 1)
        qos_max = np.percentile(valid_values, 99)
        
        # 添加5% margin
        margin = (qos_max - qos_min) * 0.05
        qos_min = max(0, qos_min - margin)
        qos_max = qos_max + margin
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.data_min_ = np.array([qos_min])
        scaler.data_max_ = np.array([qos_max])
        scaler.data_range_ = np.array([qos_max - qos_min])
        scaler.scale_ = 1.0 / scaler.data_range_
        scaler.min_ = -scaler.data_min_ * scaler.scale_
        
        self.qos_min = qos_min
        self.qos_max = qos_max
        
        print(f"[HCTN Dataset] Normalization range: [{qos_min:.4f}, {qos_max:.4f}]")
        
        return scaler
    
    def _normalize_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """
        归一化3D张量
        
        ✅ V7.22: 使用self.use_log_transform控制是否应用log变换
        """
        result = tensor.copy().astype(np.float32)
        
        # ✅ V7.22: 根据use_log_transform决定是否应用log变换
        if self.use_log_transform:
            result = np.log1p(result)
        
        # MinMax归一化
        result = (result - self.qos_min) / (self.qos_max - self.qos_min)
        result = np.clip(result, 0, 1)
        
        return result
    
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        归一化2D矩阵
        
        ✅ V7.22: 使用self.use_log_transform控制是否应用log变换
        """
        result = matrix.copy().astype(np.float32)
        
        # ✅ V7.22: 根据use_log_transform决定是否应用log变换
        if self.use_log_transform:
            result = np.log1p(result)
        
        # MinMax归一化
        result = (result - self.qos_min) / (self.qos_max - self.qos_min)
        result = np.clip(result, 0, 1)
        
        return result
    
    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """
        反归一化
        
        ✅ V7.22: 使用self.use_log_transform控制是否应用反log变换
        ✅ V7.28: 支持normalize=False模式（直接返回原值）
        """
        # ✅ V7.28修复: 如果未使用归一化，直接返回原值
        if not self.normalize:
            return values
        
        result = values * (self.qos_max - self.qos_min) + self.qos_min
        
        # ✅ V7.22: 根据use_log_transform决定是否应用反log变换
        if self.use_log_transform:
            result = np.expm1(result)
        
        return result
    
    def _precompute_samples(self) -> List[Dict]:
        """预计算所有样本"""
        samples = []
        
        for user_id, service_id in tqdm(self.indices, desc="Building samples"):
            # 获取历史序列（最近seq_len个时间步）
            history_seq = self.history_normalized[user_id, service_id, -self.seq_len:]
            
            # 修复：使用原始数据计算历史完整度（归一化后的值可能是0或接近0）
            # 原始数据中 > 0 表示有效的QoS调用记录
            raw_history = self.history[user_id, service_id, -self.seq_len:]
            valid_count = np.sum(raw_history > 0)
            completeness = valid_count / self.seq_len
            
            # 获取目标值
            target_value = self.target_normalized[user_id, service_id]
            
            # 获取NMF值
            nmf_value = 0.0
            if self.nmf_matrix is not None:
                try:
                    nmf_value = self.nmf_matrix[user_id, service_id]
                except IndexError:
                    nmf_value = 0.0
            
            sample = {
                'user_id': np.int64(user_id),
                'service_id': np.int64(service_id),
                'history_sequence': history_seq.astype(np.float32),
                'history_completeness': np.float32(completeness),
                'target_value': np.float32(target_value),
                'density': np.float32(self.density),
                'nmf_value': np.float32(nmf_value),
            }
            
            # 提取CF特征
            if self.cf_extractor is not None:
                cf_features = self.cf_extractor.extract_features(
                    user_idx=user_id,
                    service_idx=service_id,
                    time_idx=self.tau  # 使用目标时间步
                )
                if self.nmf_matrix is not None:
                    cf_features = np.append(cf_features, nmf_value)
                sample['cf_features'] = cf_features.astype(np.float32)
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {k: torch.tensor(v) for k, v in sample.items()}


def load_wsdream2_tensor(
    data_dir: str = 'datasets/wsdream_dataset2/dataset2',
    qos_type: str = 'rt',
) -> np.ndarray:
    """
    加载WS-DREAM2数据集为3D张量
    
    Args:
        data_dir: 数据目录
        qos_type: 'rt' 或 'tp'
    
    Returns:
        qos_tensor: [n_users, n_services, n_timesteps] QoS张量
    """
    if qos_type == 'rt':
        file_path = os.path.join(data_dir, 'rtdata.txt')
    else:
        file_path = os.path.join(data_dir, 'tpdata.txt')
    
    print(f"[Load] Loading {qos_type.upper()} data from {file_path}...")
    
    # 加载数据
    data = pd.read_csv(
        file_path,
        sep=r'\s+',
        names=['user_id', 'service_id', 'time_slice', 'qos_value'],
        dtype={'user_id': np.int32, 'service_id': np.int32,
               'time_slice': np.int32, 'qos_value': np.float32}
    )
    
    print(f"[Load] Loaded {len(data):,} records")
    
    # 获取维度
    n_users = data['user_id'].max() + 1
    n_services = data['service_id'].max() + 1
    n_timesteps = data['time_slice'].max() + 1
    
    print(f"[Load] Dimensions: {n_users} users × {n_services} services × {n_timesteps} time steps")
    
    # 构建3D张量 - 使用向量化操作（比逐行遍历快100倍以上）
    qos_tensor = np.zeros((n_users, n_services, n_timesteps), dtype=np.float32)
    
    print(f"[Load] Building tensor using vectorized indexing...")
    qos_tensor[
        data['user_id'].values,
        data['service_id'].values,
        data['time_slice'].values
    ] = data['qos_value'].values
    
    # 统计
    valid_count = np.sum(qos_tensor > 0)
    total_count = n_users * n_services * n_timesteps
    density = valid_count / total_count
    
    print(f"[Load] Tensor density: {density:.2%}")
    
    return qos_tensor


def create_hctn_dataloaders(
    data_dir: str = 'datasets/wsdream_dataset2/dataset2',
    qos_type: str = 'rt',
    tau: int = 63,
    psi: float = 0.10,
    val_ratio: float = 0.20,
    seq_len: int = 63,  # Bug 57修复: 使用全部τ=63个时间步 (HCTN论文一致)
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    cf_extractor = None,
    nmf_matrix: Optional[np.ndarray] = None,
    use_log_transform: Optional[bool] = None,  # ✅ V7.22: 显式控制log变换
) -> Tuple[DataLoader, DataLoader, DataLoader, HCTNDataset]:
    """
    创建HCTN风格的DataLoader
    
    Args:
        data_dir: 数据目录
        qos_type: 'rt' 或 'tp'
        tau: 历史时间步数量
        psi: 训练密度
        val_ratio: 验证集比例
        seq_len: 输入序列长度
        batch_size: 批次大小
        num_workers: 数据加载进程数
        seed: 随机种子
        cf_extractor: CF特征提取器
        nmf_matrix: NMF矩阵
        use_log_transform: 是否使用log变换（None=自动，True=强制使用，False=强制不使用）
    
    Returns:
        train_loader, val_loader, test_loader, train_dataset
    """
    print(f"\n{'='*70}")
    print(f"Creating HCTN-style DataLoaders")
    print(f"  use_log_transform: {use_log_transform}")  # ✅ V7.22
    print(f"{'='*70}")
    
    # 1. 加载QoS张量
    qos_tensor = load_wsdream2_tensor(data_dir, qos_type)
    
    # 2. 划分数据
    splitter = HCTNDataSplitter(tau=tau, psi=psi, val_ratio=val_ratio, seed=seed)
    split_data = splitter.split(qos_tensor)
    
    # 3. 创建数据集
    print(f"\n[HCTN] Creating train dataset...")
    train_dataset = HCTNDataset(
        history=split_data['history'],
        target_matrix=split_data['target_matrix'],
        indices=split_data['train_indices'],
        seq_len=seq_len,
        normalize=True,
        qos_type=qos_type,
        cf_extractor=cf_extractor,
        nmf_matrix=nmf_matrix,
        density=psi,
        use_log_transform=use_log_transform,  # ✅ V7.22
    )
    
    print(f"\n[HCTN] Creating val dataset...")
    val_dataset = HCTNDataset(
        history=split_data['history'],
        target_matrix=split_data['target_matrix'],
        indices=split_data['val_indices'],
        seq_len=seq_len,
        normalize=True,
        scaler=train_dataset.scaler,  # 使用训练集的scaler
        qos_type=qos_type,
        cf_extractor=cf_extractor,
        nmf_matrix=nmf_matrix,
        density=psi,
        qos_min=train_dataset.qos_min,  # 传入归一化参数
        qos_max=train_dataset.qos_max,
        use_log_transform=train_dataset.use_log_transform,  # ✅ V7.22: 使用训练集确定的值
    )
    
    print(f"\n[HCTN] Creating test dataset...")
    test_dataset = HCTNDataset(
        history=split_data['history'],
        target_matrix=split_data['target_matrix'],
        indices=split_data['test_indices'],
        seq_len=seq_len,
        normalize=True,
        scaler=train_dataset.scaler,  # 使用训练集的scaler
        qos_type=qos_type,
        cf_extractor=cf_extractor,
        nmf_matrix=nmf_matrix,
        density=psi,
        qos_min=train_dataset.qos_min,  # 传入归一化参数
        qos_max=train_dataset.qos_max,
        use_log_transform=train_dataset.use_log_transform,  # ✅ V7.22: 使用训练集确定的值
    )
    
    # 4. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    print(f"\n{'='*70}")
    print(f"HCTN DataLoaders Created Successfully")
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset):,} samples, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader, train_dataset


# 导出
__all__ = [
    'HCTNDataSplitter',
    'HCTNDataset',
    'load_wsdream2_tensor',
    'create_hctn_dataloaders',
]

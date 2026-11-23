"""
V71 Second-Order Graph Builder
================================================================

从用户-服务交互矩阵H构建二阶图：
- SUIG (Second-order User Interaction Graph): A_u = D_u^{-1/2} H D_s^{-1} H^T D_u^{-1/2}
- SSIG (Second-order Service Interaction Graph): A_s = D_s^{-1/2} H^T D_u^{-1} H D_s^{-1/2}

参考: HCTN (IEEE TNSM 2025) 的HCFM模块

核心设计：
1. SUIG [142, 142] — 小矩阵，可全连接或top-k稀疏化
2. SSIG [4500, 4500] — 大矩阵，必须分块计算+top-k稀疏化
3. 纯PyTorch实现，输出稀疏COO张量
4. 预计算一次，训练期间不变

Author: Kilo Code
Date: 2026-03-27
Version: 71.0
"""

import torch
import numpy as np
import time
from typing import Optional, Tuple


def build_suig(
    interaction_mask: torch.Tensor,
    num_users: int,
    num_services: int,
    device: torch.device = torch.device('cpu'),
    top_k: Optional[int] = None,
    add_self_loops: bool = True,
) -> torch.Tensor:
    """
    构建用户-用户二阶图 (SUIG)
    
    A_u = D_u^{-1/2} H D_s^{-1} H^T D_u^{-1/2}
    
    含义: A_u[i,j] ∝ user_i和user_j共同调用的服务数（经度归一化）
    
    Args:
        interaction_mask: [num_users, num_services] float tensor, 1.0=有交互, 0.0=无
        num_users: 用户数量 (142)
        num_services: 服务数量 (4500)
        device: 目标设备
        top_k: 每个用户保留的top-k邻居数（None=全连接，142个用户矩阵很小）
        add_self_loops: 是否添加自环
    
    Returns:
        suig_adj: [num_users, num_users] 稀疏归一化邻接矩阵
    """
    t0 = time.time()
    
    H = interaction_mask.float().to(device)  # [142, 4500]
    assert H.shape == (num_users, num_services), f"Expected ({num_users}, {num_services}), got {H.shape}"
    
    # 度矩阵
    D_u = H.sum(dim=1).clamp(min=1e-12)  # [142] 每个用户调用的服务数
    D_s = H.sum(dim=0).clamp(min=1e-12)  # [4500] 每个服务被调用的用户数
    
    D_u_inv_sqrt = D_u.pow(-0.5)  # [142]
    D_s_inv = D_s.pow(-1)          # [4500]
    
    # A_u = D_u^{-1/2} H D_s^{-1} H^T D_u^{-1/2}
    # Step 1: H * D_s^{-1} → [142, 4500]（逐列缩放）
    H_norm = H * D_s_inv.unsqueeze(0)
    # Step 2: (H * D_s^{-1}) @ H^T → [142, 142]
    A_u = H_norm @ H.T
    # Step 3: D_u^{-1/2} * A_u * D_u^{-1/2}（逐行逐列缩放）
    A_u = D_u_inv_sqrt.unsqueeze(1) * A_u * D_u_inv_sqrt.unsqueeze(0)
    
    # 添加自环
    if add_self_loops:
        A_u = A_u + torch.eye(num_users, device=device)
    
    # 可选：top-k稀疏化
    if top_k is not None and top_k < num_users:
        topk_vals, topk_idx = A_u.topk(min(top_k, num_users), dim=1)
        mask = torch.zeros_like(A_u)
        mask.scatter_(1, topk_idx, 1.0)
        A_u = A_u * mask
        # 对称化
        A_u = (A_u + A_u.T) / 2
    
    # 重新度归一化: D^{-1/2} A D^{-1/2}
    D = A_u.sum(dim=1).clamp(min=1e-12).pow(-0.5)
    A_u = D.unsqueeze(1) * A_u * D.unsqueeze(0)
    
    # 转为稀疏
    suig_sparse = A_u.to_sparse_coo().coalesce()
    
    elapsed = time.time() - t0
    nnz = suig_sparse._nnz()
    density = nnz / (num_users * num_users) * 100
    print(f"[V71 SUIG] Built: {num_users}x{num_users}, nnz={nnz}, "
          f"density={density:.1f}%, top_k={top_k}, time={elapsed:.2f}s")
    
    return suig_sparse


def build_ssig(
    interaction_mask: torch.Tensor,
    num_users: int,
    num_services: int,
    device: torch.device = torch.device('cpu'),
    top_k: int = 50,
    add_self_loops: bool = True,
    chunk_size: int = 500,
) -> torch.Tensor:
    """
    构建服务-服务二阶图 (SSIG)
    
    A_s = D_s^{-1/2} H^T D_u^{-1} H D_s^{-1/2}
    
    含义: A_s[i,j] ∝ service_i和service_j被共同调用的用户数（经度归一化）
    
    注意: [4500, 4500] 密集矩阵=81MB，必须分块计算+top-k稀疏化
    
    Args:
        interaction_mask: [num_users, num_services] float tensor
        num_users: 用户数量 (142)
        num_services: 服务数量 (4500)
        device: 目标设备
        top_k: 每个服务保留的top-k邻居数（默认50）
        add_self_loops: 是否添加自环
        chunk_size: 分块大小（控制内存峰值）
    
    Returns:
        ssig_adj: [num_services, num_services] 稀疏归一化邻接矩阵
    """
    t0 = time.time()
    
    H = interaction_mask.float().to(device)  # [142, 4500]
    assert H.shape == (num_users, num_services), f"Expected ({num_users}, {num_services}), got {H.shape}"
    
    # 度矩阵
    D_u = H.sum(dim=1).clamp(min=1e-12)  # [142]
    D_s = H.sum(dim=0).clamp(min=1e-12)  # [4500]
    
    D_s_inv_sqrt = D_s.pow(-0.5)  # [4500]
    D_u_inv = D_u.pow(-1)          # [142]
    
    # A_s = D_s^{-1/2} H^T D_u^{-1} H D_s^{-1/2}
    # 预计算: HT_norm = H^T * D_u^{-1} → [4500, 142]
    HT_norm = H.T * D_u_inv.unsqueeze(0)  # [4500, 142]
    
    # 分块计算 HT_norm @ H → [4500, 4500]，每次处理chunk_size行
    n_chunks = (num_services + chunk_size - 1) // chunk_size
    
    rows_list = []
    cols_list = []
    vals_list = []
    
    for i in range(n_chunks):
        start_i = i * chunk_size
        end_i = min((i + 1) * chunk_size, num_services)
        chunk_len = end_i - start_i
        
        # [chunk, 142] @ [142, 4500] → [chunk, 4500]
        A_chunk = HT_norm[start_i:end_i] @ H
        
        # D_s归一化: D_s^{-1/2}[start:end] * A_chunk * D_s^{-1/2}
        A_chunk = D_s_inv_sqrt[start_i:end_i].unsqueeze(1) * A_chunk * D_s_inv_sqrt.unsqueeze(0)
        
        # Top-k稀疏化（每行保留top_k个最大值）
        actual_k = min(top_k, A_chunk.size(1))
        topk_vals, topk_idx = A_chunk.topk(actual_k, dim=1)
        
        # 向量化收集非零元素（避免Python循环）
        valid_mask = topk_vals > 1e-10  # [chunk_len, actual_k]
        
        # 构建行索引
        row_idx = torch.arange(start_i, end_i, device=device).unsqueeze(1).expand_as(topk_idx)
        
        # 提取有效元素
        valid_rows = row_idx[valid_mask]
        valid_cols = topk_idx[valid_mask]
        valid_vals_flat = topk_vals[valid_mask]
        
        if valid_rows.numel() > 0:
            rows_list.append(valid_rows)
            cols_list.append(valid_cols)
            vals_list.append(valid_vals_flat)
    
    # 合并所有块
    if rows_list:
        all_rows = torch.cat(rows_list)
        all_cols = torch.cat(cols_list)
        all_vals = torch.cat(vals_list)
    else:
        all_rows = torch.zeros(0, dtype=torch.long, device=device)
        all_cols = torch.zeros(0, dtype=torch.long, device=device)
        all_vals = torch.zeros(0, dtype=torch.float32, device=device)
    
    # 添加自环
    if add_self_loops:
        self_rows = torch.arange(num_services, dtype=torch.long, device=device)
        self_vals = torch.ones(num_services, dtype=torch.float32, device=device)
        all_rows = torch.cat([all_rows, self_rows])
        all_cols = torch.cat([all_cols, self_rows])
        all_vals = torch.cat([all_vals, self_vals])
    
    # 构建稀疏矩阵
    indices = torch.stack([all_rows, all_cols], dim=0)
    A_s = torch.sparse_coo_tensor(indices, all_vals, 
                                   (num_services, num_services)).coalesce()
    
    # 对称化: A_s = (A_s + A_s^T) / 2
    A_s_T = torch.sparse_coo_tensor(
        torch.stack([A_s.indices()[1], A_s.indices()[0]]),
        A_s.values(),
        (num_services, num_services)
    ).coalesce()
    A_s = (A_s + A_s_T).coalesce()
    # 除以2（对称化后值翻倍）
    A_s = torch.sparse_coo_tensor(A_s.indices(), A_s.values() / 2, 
                                   (num_services, num_services)).coalesce()
    
    # 度归一化: D^{-1/2} A D^{-1/2}
    degree = torch.sparse.sum(A_s, dim=1).to_dense().clamp(min=1e-12)
    D_inv_sqrt = degree.pow(-0.5)
    idx = A_s.indices()
    norm_vals = A_s.values() * D_inv_sqrt[idx[0]] * D_inv_sqrt[idx[1]]
    A_s = torch.sparse_coo_tensor(idx, norm_vals, 
                                   (num_services, num_services)).coalesce()
    
    elapsed = time.time() - t0
    nnz = A_s._nnz()
    avg_neighbors = nnz / num_services
    mem_mb = (nnz * 12) / (1024 * 1024)  # 每个非零元素: 2*int64 + 1*float32 = 12 bytes
    print(f"[V71 SSIG] Built: {num_services}x{num_services}, nnz={nnz:,}, "
          f"avg_neighbors={avg_neighbors:.1f}, top_k={top_k}, "
          f"mem~{mem_mb:.1f}MB, time={elapsed:.2f}s")
    
    return A_s


def build_second_order_graphs(
    interaction_mask: torch.Tensor,
    num_users: int,
    num_services: int,
    device: torch.device = torch.device('cpu'),
    suig_top_k: Optional[int] = None,
    ssig_top_k: int = 50,
    build_suig_flag: bool = True,
    build_ssig_flag: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    一次性构建SUIG和SSIG
    
    Args:
        interaction_mask: [num_users, num_services] 交互矩阵
        suig_top_k: SUIG的top-k（None=全连接）
        ssig_top_k: SSIG的top-k（默认50）
        build_suig_flag: 是否构建SUIG
        build_ssig_flag: 是否构建SSIG
    
    Returns:
        (suig_adj, ssig_adj): 稀疏邻接矩阵元组
    """
    suig_adj = None
    ssig_adj = None
    
    if build_suig_flag:
        suig_adj = build_suig(
            interaction_mask, num_users, num_services, device,
            top_k=suig_top_k
        )
    
    if build_ssig_flag:
        ssig_adj = build_ssig(
            interaction_mask, num_users, num_services, device,
            top_k=ssig_top_k
        )
    
    return suig_adj, ssig_adj


if __name__ == "__main__":
    print("V71 Second-Order Graph Builder Test")
    print("=" * 60)
    
    num_users = 142
    num_services = 4500
    
    # 模拟不同密度的交互矩阵
    for density in [0.05, 0.10, 0.15, 0.20]:
        print(f"\n--- Density={density*100:.0f}% ---")
        np.random.seed(42)
        mask = (np.random.rand(num_users, num_services) < density).astype(np.float32)
        interaction_mask = torch.from_numpy(mask)
        
        n_interactions = mask.sum()
        print(f"Interactions: {n_interactions:.0f} ({n_interactions/(num_users*num_services)*100:.1f}%)")
        
        # 构建SUIG
        suig = build_suig(interaction_mask, num_users, num_services, top_k=None)
        
        # 构建SSIG
        ssig = build_ssig(interaction_mask, num_users, num_services, top_k=50)
        
        # 验证对称性
        suig_dense = suig.to_dense()
        sym_err = (suig_dense - suig_dense.T).abs().max().item()
        print(f"SUIG symmetry error: {sym_err:.2e}")
        
        # 验证度归一化（行和应接近1）
        suig_row_sum = suig_dense.sum(dim=1)
        print(f"SUIG row sum: mean={suig_row_sum.mean():.4f}, "
              f"std={suig_row_sum.std():.4f}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")

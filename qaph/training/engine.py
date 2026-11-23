#!/usr/bin/env python3
"""QAPH frozen-v0 training, feature construction and evaluation engine. Original source credit: Kilo Code."""

import os
import sys
import json
import csv
import argparse
import logging
import math
import pickle
import hashlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR
from datetime import datetime
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qaph.model import UnifiedQoSModelV42
from qaph.losses import CauchyLoss, BarronAdaptiveLoss, AdaptiveGradientHarmonization, SegmentedCauchyLoss, LogSpaceCauchyLoss
from qaph.heads import MoEPredictionHead
from qaph.graphs.fig import build_lightgcn_adj
from qaph.graphs.second_order import build_second_order_graphs
from qaph.data.loader import load_wsdream2_tensor
from shared.protocol import (
    history_sha256, load_or_compute_rolling_statistics,
    prepare_rolling_split, save_split_indices,
)
from qaph.features.anomalies import GreysheepDetector
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from scipy import stats as scipy_stats


# ============================================================
# V70 鏂规?A: Anchor璁＄畻澧炲己
# ============================================================

def compute_ema_anchor(history_seq: np.ndarray, alpha: float = 0.1) -> float:
    """
    V70-A1: 鎸囨暟鍔犳潈鍧囧€糀nchor
    
    瀵归潪闆跺巻鍙插€兼寜鏃堕棿椤哄簭璧嬩簣鎸囨暟閫掑?鏉冮噸锛岃秺杩戞湡鐨勮?娴嬫潈閲嶈秺澶с€?
    鎹曟崏QoS鏃跺簭婕傜Щ锛坉rift锛夛紝浣縜nchor鏇磋创杩戞渶鏂扮姸鎬併€?
    
    Args:
        history_seq: [seq_len] 鍗曚釜(user, service)鐨勫巻鍙睶oS搴忓垪
        alpha: 鎸囨暟琛板噺鐜囷紝瓒婂ぇ瓒婂亸鍚戣繎鏈熴€?.1=娓╁拰鍋忓悜锛?.3=寮哄亸鍚?
    Returns:
        anchor: float, EMA鍔犳潈鍧囧€?
    """
    nonzero_mask = history_seq > 0
    nonzero_vals = history_seq[nonzero_mask]
    if len(nonzero_vals) == 0:
        return 0.0
    # 闈為浂鍊间繚鎸佸師濮嬫椂闂撮『搴忥紙瓒婂悗闈?瓒婅繎鏈燂級锛岃祴浜堟寚鏁伴€掑?鏉冮噸
    weights = np.exp(alpha * np.arange(len(nonzero_vals)))
    weights /= weights.sum()
    return float((nonzero_vals * weights).sum())


def compute_trimmed_anchor(history_seq: np.ndarray, trim_ratio: float = 0.1) -> float:
    """
    V70-A2: 鎴?柇鍧囧€糀nchor
    
    鍘绘帀top/bottom trim_ratio姣斾緥鐨勬瀬绔?€煎悗璁＄畻鍧囧€笺€?
    TP鍊煎煙璺?涓?暟閲忕骇锛屾埅鏂?潎鍊兼瘮绠€鍗曞潎鍊兼洿椴佹?銆?
    
    Args:
        history_seq: [seq_len] 鍗曚釜(user, service)鐨勫巻鍙睶oS搴忓垪
        trim_ratio: 鎴?柇姣斾緥锛屼袱绔?悇鍘绘帀璇ユ瘮渚嬬殑鍊?
    Returns:
        anchor: float, 鎴?柇鍧囧€?
    """
    nonzero_vals = np.sort(history_seq[history_seq > 0])
    if len(nonzero_vals) == 0:
        return 0.0
    n = len(nonzero_vals)
    # 鑷?劧鎴?柇锛歯<5鏃秚rim_n=0锛屼笉鎴?柇锛坱rim_ratio=0.1鏃跺疄闄呴槇鍊间负n鈮?锛?
    trim_n = int(n * trim_ratio)
    if trim_n > 0 and 2 * trim_n < n:
        trimmed = nonzero_vals[trim_n:-trim_n]
    else:
        trimmed = nonzero_vals  # 鏍锋湰澶?皯鎴杢rim_n=0锛屼笉鎴?柇
    return float(trimmed.mean()) if len(trimmed) > 0 else float(nonzero_vals.mean())


def compute_neighbor_weighted_anchor(
    u: int, s: int, history_seq: np.ndarray,
    full_history: np.ndarray,
    user_neighbors: np.ndarray, service_neighbors: np.ndarray,
    neighbor_weight: float = 0.3,
    ema_alpha: float = 0.1
) -> float:
    """
    V70-A3 + V73鏀硅繘: 閭诲眳鍔犳潈Anchor锛圗MA鐗堟湰锛?
    
    铻嶅悎鑷?韩EMA anchor鍜岀浉浼肩敤鎴?鏈嶅姟鐨凟MA anchor锛屽?鍘嗗彶绋€鐤忕殑pair鏇撮瞾妫掋€?
    V73鏀硅繘: 鑷?韩anchor鍜岄偦灞卆nchor閮戒娇鐢‥MA锛堣€岄潪绠€鍗曞潎鍊硷級锛?
    涓嶸70-A1 EMA anchor淇濇寔涓€鑷达紝鎹曟崏QoS鏃跺簭婕傜Щ銆?
    
    Args:
        u, s: 鐢ㄦ埛鍜屾湇鍔＄储寮?
        history_seq: [seq_len] 褰撳墠(u,s)鐨勫巻鍙插簭鍒?
        full_history: [n_users, n_services, tau] 瀹屾暣鍘嗗彶寮犻噺
        user_neighbors: [n_users, top_k] 姣忎釜鐢ㄦ埛鐨凾op-K鐩镐技鐢ㄦ埛
        service_neighbors: [n_services, top_k] 姣忎釜鏈嶅姟鐨凾op-K鐩镐技鏈嶅姟
        neighbor_weight: 閭诲眳anchor鐨勮瀺鍚堟潈閲?
        ema_alpha: EMA琛板噺鐜囷紙涓巃nchor_ema_alpha涓€鑷达級
    Returns:
        anchor: float, 閭诲眳鍔犳潈anchor
    """
    # V73: 鑷?韩anchor浣跨敤EMA锛堣€岄潪绠€鍗曞潎鍊硷級
    self_anchor = compute_ema_anchor(history_seq, alpha=ema_alpha)
    
    # 鐩镐技鐢ㄦ埛鍦ㄥ悓涓€鏈嶅姟涓婄殑EMA anchor
    neighbor_anchors = []
    for nb_uid in user_neighbors[u]:
        nb_hist = full_history[nb_uid, s, :]
        nb_anchor = compute_ema_anchor(nb_hist, alpha=ema_alpha)
        if nb_anchor > 0:
            neighbor_anchors.append(nb_anchor)
    
    # 鐩镐技鏈嶅姟鍦ㄥ悓涓€鐢ㄦ埛涓婄殑EMA anchor
    for nb_sid in service_neighbors[s]:
        nb_hist = full_history[u, nb_sid, :]
        nb_anchor = compute_ema_anchor(nb_hist, alpha=ema_alpha)
        if nb_anchor > 0:
            neighbor_anchors.append(nb_anchor)
    
    if len(neighbor_anchors) > 0 and self_anchor > 0:
        nb_anchor = float(np.mean(neighbor_anchors))
        return (1 - neighbor_weight) * self_anchor + neighbor_weight * nb_anchor
    elif len(neighbor_anchors) > 0:
        # 鑷?韩鏃犲巻鍙诧紝瀹屽叏渚濊禆閭诲眳
        return float(np.mean(neighbor_anchors))
    return self_anchor


def compute_mfvp_features(history: np.ndarray, seq_len: int = 63, window: int = 5) -> np.ndarray:
    """
    璁＄畻MFVP (Multi-Feature Value Projection) 鐗瑰緛
    
    涓烘瘡涓?椂闂存?鏋勯€?缁寸壒寰佸悜閲忥紝鏇夸唬鍘熷?鐨?缁存爣閲忚緭鍏ワ細
    1. 鍘熷?QoS鍊?(褰掍竴鍖?
    2. log1p(QoS鍊? - 鍘嬬缉闀垮熬鍒嗗竷
    3. 灞€閮ㄥ潎鍊?(婊戝姩绐楀彛)
    4. 灞€閮ㄦ爣鍑嗗樊 (婊戝姩绐楀彛)
    5. 鍙樺寲鐜?(涓€闃跺樊鍒?
    6. 闈為浂鎸囩ず (0/1)
    7. 灞€閮ㄥ彉寮傜郴鏁?(std/mean)
    
    Args:
        history: [n_users, n_services, tau] 鍘嗗彶QoS寮犻噺
        seq_len: 搴忓垪闀垮害
        window: 婊戝姩绐楀彛澶у皬
    Returns:
        mfvp_cache: dict with 'features' key 鈫?[n_users, n_services, seq_len, 7]
                    瀛樺偍涓虹█鐤忔牸寮忎互鑺傜渷鍐呭瓨
    """
    n_users, n_services, tau = history.shape
    mfvp_dim = 7
    
    # 棰勮?绠楀叏灞€缁熻?閲忕敤浜庡綊涓€鍖?
    nonzero_vals = history[history > 0]
    if len(nonzero_vals) == 0:
        global_mean, global_std = 0.0, 1.0
    else:
        global_mean = nonzero_vals.mean()
        global_std = nonzero_vals.std() + 1e-8
    
    return {
        'global_mean': global_mean,
        'global_std': global_std,
        'window': window,
        'mfvp_dim': mfvp_dim,
    }


def compute_mfvp_for_sample(h: np.ndarray, global_mean: float, global_std: float,
                             window: int = 5,
                             ptcs_user_signal: np.ndarray = None,
                             ptcs_service_signal: np.ndarray = None) -> np.ndarray:
    """
    涓哄崟涓?牱鏈?殑鍘嗗彶搴忓垪璁＄畻MFVP鐗瑰緛
    
    鍩虹?7缁?+ 鍙?€塒TCS 2缁?= 鏈€澶?缁?
    
    Args:
        h: [seq_len] 鍗曚釜(user, service)鐨勫巻鍙睶oS搴忓垪
        global_mean: 鍏ㄥ眬闈為浂鍧囧€?
        global_std: 鍏ㄥ眬闈為浂鏍囧噯宸?
        window: 婊戝姩绐楀彛澶у皬
        ptcs_user_signal: [seq_len] 鐢ㄦ埛閭诲眳鍦ㄦ瘡涓?椂闂存?鐨凲oS鍧囧€硷紙褰掍竴鍖栧悗锛?
        ptcs_service_signal: [seq_len] 鏈嶅姟閭诲眳鍦ㄦ瘡涓?椂闂存?鐨凲oS鍧囧€硷紙褰掍竴鍖栧悗锛?
    Returns:
        features: [seq_len, 7 or 9] MFVP鐗瑰緛
    """
    seq_len = len(h)
    use_ptcs = (ptcs_user_signal is not None and ptcs_service_signal is not None)
    feat_dim = 9 if use_ptcs else 7
    features = np.zeros((seq_len, feat_dim), dtype=np.float32)
    
    # Feature 1: 褰掍竴鍖朡oS鍊?
    features[:, 0] = (h - global_mean) / global_std
    
    # Feature 2: log1p(QoS鍊? - 褰掍竴鍖?
    log_vals = np.log1p(np.clip(h, 0, None))
    log_mean = np.log1p(global_mean)
    log_std = np.log1p(global_std)
    features[:, 1] = (log_vals - log_mean) / (log_std + 1e-8)
    
    # Feature 3 & 4: 灞€閮ㄥ潎鍊煎拰鏍囧噯宸?(婊戝姩绐楀彛)
    half_w = window // 2
    for t in range(seq_len):
        start = max(0, t - half_w)
        end = min(seq_len, t + half_w + 1)
        local_vals = h[start:end]
        local_nonzero = local_vals[local_vals > 0]
        if len(local_nonzero) > 0:
            local_mean = local_nonzero.mean()
            local_std = local_nonzero.std() if len(local_nonzero) > 1 else 0.0
            features[t, 2] = (local_mean - global_mean) / global_std
            features[t, 3] = local_std / global_std
            # Feature 7: 灞€閮ㄥ彉寮傜郴鏁?
            features[t, 6] = local_std / (local_mean + 1e-8)
    
    # Feature 5: 鍙樺寲鐜?(涓€闃跺樊鍒嗭紝褰掍竴鍖?
    diff = np.zeros(seq_len, dtype=np.float32)
    diff[1:] = h[1:] - h[:-1]
    features[:, 4] = diff / global_std
    
    # Feature 6: 闈為浂鎸囩ず
    features[:, 5] = (h > 0).astype(np.float32)
    
    # Feature 8-9: PTCS鍗忓悓淇″彿锛堝凡褰掍竴鍖栵級
    if use_ptcs:
        features[:, 7] = ptcs_user_signal[:seq_len]
        features[:, 8] = ptcs_service_signal[:seq_len]
    
    # Clip鏋佺?鍊?
    features = np.clip(features, -10, 10)
    
    return features


def compute_ptcs_neighbors(history: np.ndarray, top_k: int = 20, logger=None):
    """
    V55 PTCS: 棰勮?绠楃敤鎴峰拰鏈嶅姟鐨凾op-K鐩镐技閭诲眳
    
    浣跨敤NMF鍒嗚В鍘嗗彶鍧囧€肩煩闃碉紝鐒跺悗鍩轰簬浣欏鸡鐩镐技搴︽壘Top-K閭诲眳銆?
    
    Args:
        history: [n_users, n_services, tau] 鍘嗗彶QoS寮犻噺
        top_k: 姣忎釜鐢ㄦ埛/鏈嶅姟鐨勯偦灞呮暟
        logger: 鏃ュ織鍣?
    Returns:
        ptcs_data: dict with:
            'user_neighbors': [n_users, top_k] 姣忎釜鐢ㄦ埛鐨凾op-K鐩镐技鐢ㄦ埛绱㈠紩
            'service_neighbors': [n_services, top_k] 姣忎釜鏈嶅姟鐨凾op-K鐩镐技鏈嶅姟绱㈠紩
            'history': reference to history array
            'global_mean': 鍏ㄥ眬闈為浂鍧囧€?
            'global_std': 鍏ㄥ眬闈為浂鏍囧噯宸?
    """
    n_users, n_services, tau = history.shape
    
    # 璁＄畻闈為浂鍧囧€肩煩闃?
    qos_sum = history.sum(axis=2)
    qos_count = (history > 0).sum(axis=2).clip(min=1)
    qos_mean = (qos_sum / qos_count).clip(min=0)
    qos_mean = np.nan_to_num(qos_mean, nan=0.0)
    if logger:
        logger.info(f"PTCS: starting NMF on qos_mean={qos_mean.shape}, top_k={top_k}")
    
    # NMF鍒嗚В
    n_components = min(32, min(n_users, n_services) - 1)
    nmf = NMF(n_components=n_components, init='nndsvda', max_iter=300, random_state=42)
    W = nmf.fit_transform(qos_mean)   # [n_users, n_components]
    H = nmf.components_                # [n_components, n_services]
    
    if logger:
        logger.info(f"PTCS: NMF done, W={W.shape}, H={H.shape}, error={nmf.reconstruction_err_:.2f}")
    
    # 鐢ㄦ埛鐩镐技搴?鈫?Top-K閭诲眳
    if logger:
        logger.info("PTCS: computing user/service cosine similarity matrices")
    user_sim = sk_cosine_similarity(W)
    np.fill_diagonal(user_sim, -1)
    user_neighbors = np.zeros((n_users, top_k), dtype=np.int32)
    for i in range(n_users):
        topk = np.argsort(user_sim[i])[-top_k:][::-1]
        user_neighbors[i] = topk
    
    # 鏈嶅姟鐩镐技搴?鈫?Top-K閭诲眳
    service_sim = sk_cosine_similarity(H.T)
    np.fill_diagonal(service_sim, -1)
    service_neighbors = np.zeros((n_services, top_k), dtype=np.int32)
    for i in range(n_services):
        topk = np.argsort(service_sim[i])[-top_k:][::-1]
        service_neighbors[i] = topk
    
    # 鍏ㄥ眬缁熻?閲?
    nonzero_vals = history[history > 0]
    global_mean = float(nonzero_vals.mean()) if len(nonzero_vals) > 0 else 0.0
    global_std = float(nonzero_vals.std()) + 1e-8 if len(nonzero_vals) > 0 else 1.0
    
    if logger:
        logger.info(f"PTCS: user_neighbors={user_neighbors.shape}, service_neighbors={service_neighbors.shape}, top_k={top_k}")
    
    return {
        'user_neighbors': user_neighbors,
        'service_neighbors': service_neighbors,
        'history': history,
        'global_mean': global_mean,
        'global_std': global_std,
    }


def load_or_compute_ptcs_neighbors(history: np.ndarray, top_k: int, logger=None,
                                   cache_path: str = '', cache_label: str = 'PTCS'):
    """
    Load cached PTCS/NACC neighbors when available; otherwise compute and persist them.

    The cache is keyed at the caller level by qos_type/tau/top_k. This does not change
    the experiment semantics because the history tensor is identical for a fixed dataset
    and tau, and the cached payload stores only deterministic neighbor indices/statistics.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            required_keys = {'user_neighbors', 'service_neighbors', 'global_mean', 'global_std'}
            if isinstance(cached, dict) and required_keys.issubset(cached.keys()):
                cached['history'] = history
                if logger:
                    logger.info(f"{cache_label}: loaded cached neighbors from {cache_path}")
                return cached
            if logger:
                logger.warning(f"{cache_label}: cache missing required keys, recomputing ({cache_path})")
        except Exception as exc:
            if logger:
                logger.warning(f"{cache_label}: failed to load cache {cache_path}, recomputing ({exc})")

    ptcs_data = compute_ptcs_neighbors(history, top_k=top_k, logger=logger)

    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            serializable = {
                'user_neighbors': ptcs_data['user_neighbors'],
                'service_neighbors': ptcs_data['service_neighbors'],
                'global_mean': ptcs_data['global_mean'],
                'global_std': ptcs_data['global_std'],
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(serializable, f, protocol=pickle.HIGHEST_PROTOCOL)
            if logger:
                logger.info(f"{cache_label}: cached neighbors saved to {cache_path}")
        except Exception as exc:
            if logger:
                logger.warning(f"{cache_label}: failed to save cache {cache_path} ({exc})")

    return ptcs_data


def compute_ptcs_signals(u: int, s: int, history: np.ndarray,
                          user_neighbors: np.ndarray, service_neighbors: np.ndarray,
                          global_mean: float, global_std: float,
                          seq_len: int = 63) -> tuple:
    """
    涓哄崟涓?user, service)瀵硅?绠桺TCS per-timestep鍗忓悓淇″彿
    
    鐢ㄦ埛淇″彿: Top-K鐩镐技鐢ㄦ埛鍦╯ervice s涓婃瘡涓?椂闂存?鐨勯潪闆禥oS鍧囧€?
    鏈嶅姟淇″彿: Top-K鐩镐技鏈嶅姟鍦╱ser u涓婃瘡涓?椂闂存?鐨勯潪闆禥oS鍧囧€?
    
    Args:
        u, s: 鐢ㄦ埛鍜屾湇鍔＄储寮?
        history: [n_users, n_services, tau]
        user_neighbors: [n_users, top_k]
        service_neighbors: [n_services, top_k]
        global_mean, global_std: 褰掍竴鍖栧弬鏁?
        seq_len: 搴忓垪闀垮害
    Returns:
        user_signal: [seq_len] 褰掍竴鍖栫殑鐢ㄦ埛閭诲眳鍗忓悓淇″彿
        service_signal: [seq_len] 褰掍竴鍖栫殑鏈嶅姟閭诲眳鍗忓悓淇″彿
    """
    tau = history.shape[2]
    actual_len = min(seq_len, tau)
    
    # 鐢ㄦ埛鍗忓悓淇″彿: 鐩镐技鐢ㄦ埛鍦╯ervice s涓婄殑QoS
    u_nbrs = user_neighbors[u]  # [top_k]
    nbr_vals = history[u_nbrs, s, :actual_len]  # [top_k, actual_len]
    nbr_nonzero = (nbr_vals > 0).astype(np.float32)
    nbr_sum = nbr_vals.sum(axis=0)       # [actual_len]
    nbr_count = nbr_nonzero.sum(axis=0)  # [actual_len]
    user_signal = np.zeros(seq_len, dtype=np.float32)
    valid = nbr_count > 0
    user_signal[:actual_len][valid] = (nbr_sum[valid] / nbr_count[valid] - global_mean) / global_std
    
    # 鏈嶅姟鍗忓悓淇″彿: 鐩镐技鏈嶅姟鍦╱ser u涓婄殑QoS
    s_nbrs = service_neighbors[s]  # [top_k]
    nbr_vals = history[u, s_nbrs, :actual_len]  # [top_k, actual_len]
    nbr_nonzero = (nbr_vals > 0).astype(np.float32)
    nbr_sum = nbr_vals.sum(axis=0)
    nbr_count = nbr_nonzero.sum(axis=0)
    service_signal = np.zeros(seq_len, dtype=np.float32)
    valid = nbr_count > 0
    service_signal[:actual_len][valid] = (nbr_sum[valid] / nbr_count[valid] - global_mean) / global_std
    
    return user_signal, service_signal


def compute_nacc_features(u: int, s: int, history: np.ndarray,
                           user_neighbors: np.ndarray, service_neighbors: np.ndarray,
                           global_mean: float, global_std: float) -> np.ndarray:
    """
    V60 NACC: 璁＄畻閭诲眳鑱氬悎鍗忓悓涓婁笅鏂囩壒寰侊紙6缁达級
    
    璺ㄦ墍鏈夋椂闂存?鑱氬悎锛屼骇鐢熺?瀵嗙殑pair-specific鐗瑰緛锛?
    1. user_neighbor_mean: K涓?浉浼肩敤鎴峰湪service s涓婄殑闈為浂QoS鍧囧€?
    2. user_neighbor_std: 鍚屼笂鐨勬爣鍑嗗樊
    3. user_neighbor_density: K涓?敤鎴蜂腑鏈夐潪闆惰?褰曠殑姣斾緥
    4. service_neighbor_mean: K涓?浉浼兼湇鍔″湪user u涓婄殑闈為浂QoS鍧囧€?
    5. service_neighbor_std: 鍚屼笂鐨勬爣鍑嗗樊
    6. service_neighbor_density: K涓?湇鍔′腑鏈夐潪闆惰?褰曠殑姣斾緥
    
    Args:
        u, s: 鐢ㄦ埛鍜屾湇鍔＄储寮?
        history: [n_users, n_services, tau]
        user_neighbors: [n_users, top_k]
        service_neighbors: [n_services, top_k]
        global_mean, global_std: 褰掍竴鍖栧弬鏁?
    Returns:
        features: [6] NACC鐗瑰緛鍚戦噺
    """
    features = np.zeros(6, dtype=np.float32)
    
    # 鐢ㄦ埛閭诲眳鍦╯ervice s涓婄殑QoS锛堣法鎵€鏈夋椂闂存?锛?
    u_nbrs = user_neighbors[u]  # [top_k]
    u_nbr_vals = history[u_nbrs, s, :]  # [top_k, tau]
    u_nbr_flat = u_nbr_vals.flatten()
    u_nbr_nonzero = u_nbr_flat[u_nbr_flat > 0]
    
    if len(u_nbr_nonzero) > 0:
        features[0] = (u_nbr_nonzero.mean() - global_mean) / global_std
        features[1] = u_nbr_nonzero.std() / global_std if len(u_nbr_nonzero) > 1 else 0.0
    # 瀵嗗害锛氭瘡涓?偦灞呮槸鍚︽湁鑷冲皯涓€涓?潪闆惰?褰?
    u_nbr_has_record = (u_nbr_vals > 0).any(axis=1)  # [top_k]
    features[2] = u_nbr_has_record.mean()
    
    # 鏈嶅姟閭诲眳鍦╱ser u涓婄殑QoS锛堣法鎵€鏈夋椂闂存?锛?
    s_nbrs = service_neighbors[s]  # [top_k]
    s_nbr_vals = history[u, s_nbrs, :]  # [top_k, tau]
    s_nbr_flat = s_nbr_vals.flatten()
    s_nbr_nonzero = s_nbr_flat[s_nbr_flat > 0]
    
    if len(s_nbr_nonzero) > 0:
        features[3] = (s_nbr_nonzero.mean() - global_mean) / global_std
        features[4] = s_nbr_nonzero.std() / global_std if len(s_nbr_nonzero) > 1 else 0.0
    # 瀵嗗害
    s_nbr_has_record = (s_nbr_vals > 0).any(axis=1)  # [top_k]
    features[5] = s_nbr_has_record.mean()
    
    # Clip鏋佺?鍊?
    features = np.clip(features, -10, 10)
    
    return features




class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logging(log_dir: str, qos_type: str, gamma: float, density: float = None, tag: str = '') -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    d_label = f"d{int(density*100)}" if density else ""
    tag_label = f"_{tag}" if tag else ""
    log_file = os.path.join(log_dir, f'{qos_type}_{d_label}_gamma{gamma}{tag_label}_{timestamp}.log')
    
    file_handler = FlushFileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    logger = logging.getLogger(__name__)
    logger.log_file_path = log_file
    return logger


def split_fingerprint(indices: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def rolling_source_hashes():
    root = PROJECT_ROOT
    sources = {
        'runner': os.path.abspath(__file__),
        'protocol': os.path.join(root, 'shared', 'protocol.py'),
        'model': os.path.join(root, 'qaph', 'model.py'),
        'heads': os.path.join(root, 'qaph', 'heads.py'),
        'losses': os.path.join(root, 'qaph', 'losses.py'),
    }
    hashes = {}
    for label, path in sources.items():
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        hashes[label] = digest.hexdigest()
    return hashes


def reset_cuda_peak_memory(device):
    """Reset CUDA peak-memory stats without changing model behavior."""
    if device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def get_cuda_peak_memory_mb(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def get_cuda_current_memory_mb(device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_allocated(device) / (1024 ** 2))


def file_size_mb(path):
    if not path or not os.path.exists(path):
        return None
    return float(os.path.getsize(path) / (1024 ** 2))


def count_top_level_module_params(model):
    """Return a compact module-level parameter breakdown for efficiency analysis."""
    breakdown = {}
    assigned = 0
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters(recurse=True))
        if count > 0:
            breakdown[name] = int(count)
            assigned += count
    total = sum(p.numel() for p in model.parameters())
    direct = int(total - assigned)
    if direct > 0:
        breakdown['_direct_parameters'] = direct
    breakdown['total'] = int(total)
    breakdown['trainable'] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return breakdown


def _get_lambda0_result(test_results):
    if not test_results:
        return None
    if 0 in test_results:
        return test_results[0]
    if '0' in test_results:
        return test_results['0']
    return None


def append_history_window_raw_result(result: dict, output_csv: str = None):
    """Append lambda=0 result for the diagnostic history-window sensitivity experiment."""
    config_name = str(result.get('ablation', ''))
    if config_name.startswith('debug_') or not config_name.startswith('history_'):
        return
    lambda0 = _get_lambda0_result(result.get('test_results') or {})
    if lambda0 is None:
        return
    output_csv = output_csv or os.path.join(
        PROJECT_ROOT, 'results', 'history_window_sensitivity', 'raw_results.csv'
    )
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fields = [
        'qos_type', 'density', 'history_length', 'seed', 'mae', 'rmse',
        'best_epoch', 'best_val_mae', 'config_name', 'timestamp'
    ]
    row = {
        'qos_type': result.get('qos_type'),
        'density': result.get('density'),
        'history_length': result.get('history_length'),
        'seed': result.get('seed'),
        'mae': lambda0.get('mae'),
        'rmse': lambda0.get('rmse'),
        'best_epoch': result.get('best_epoch', ''),
        'best_val_mae': result.get('best_val_mae_lambda0', ''),
        'config_name': config_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_efficiency_raw_result(result: dict, output_csv: str):
    """Append a compact deployment-efficiency row when explicitly requested."""
    if not output_csv:
        return
    lambda0 = _get_lambda0_result(result.get('test_results') or {})
    efficiency = result.get('efficiency') or {}
    flops_profile = result.get('flops_profile') or {}
    module_params = result.get('module_param_count') or {}
    fields = [
        'qos_type', 'density', 'history_length', 'seed', 'config_name',
        'mae', 'rmse', 'best_epoch', 'best_val_mae',
        'param_count', 'trainable_param_count',
        'forward_flops_per_sample', 'forward_flops_g_per_sample',
        'flops_profile_batch_size',
        'peak_train_memory_mb', 'peak_inference_memory_mb',
        'current_memory_after_model_mb',
        'train_wall_time_s', 'total_run_time_s',
        'avg_train_epoch_time_s', 'avg_val_eval_time_s', 'avg_total_epoch_time_s',
        'final_test_inference_time_s', 'test_samples',
        'latency_ms_per_sample', 'throughput_samples_per_s',
        'checkpoint_size_mb',
        'timing_profile_csv', 'dataset_prepare_profile_csv',
        'module_param_count_json', 'timestamp'
    ]
    row = {
        'qos_type': result.get('qos_type'),
        'density': result.get('density'),
        'history_length': result.get('history_length'),
        'seed': result.get('seed'),
        'config_name': result.get('ablation'),
        'mae': lambda0.get('mae') if lambda0 else '',
        'rmse': lambda0.get('rmse') if lambda0 else '',
        'best_epoch': result.get('best_epoch', ''),
        'best_val_mae': result.get('best_val_mae_lambda0', ''),
        'param_count': result.get('param_count', ''),
        'trainable_param_count': module_params.get('trainable', ''),
        'forward_flops_per_sample': efficiency.get(
            'forward_flops_per_sample',
            flops_profile.get('forward_flops_per_sample', ''),
        ),
        'forward_flops_g_per_sample': efficiency.get(
            'forward_flops_g_per_sample',
            flops_profile.get('forward_flops_g_per_sample', ''),
        ),
        'flops_profile_batch_size': efficiency.get(
            'flops_profile_batch_size',
            flops_profile.get('batch_size', ''),
        ),
        'peak_train_memory_mb': efficiency.get('peak_train_memory_mb', ''),
        'peak_inference_memory_mb': efficiency.get('peak_inference_memory_mb', ''),
        'current_memory_after_model_mb': efficiency.get('current_memory_after_model_mb', ''),
        'train_wall_time_s': efficiency.get('train_wall_time_s', ''),
        'total_run_time_s': efficiency.get('total_run_time_s', ''),
        'avg_train_epoch_time_s': efficiency.get('avg_train_epoch_time_s', ''),
        'avg_val_eval_time_s': efficiency.get('avg_val_eval_time_s', ''),
        'avg_total_epoch_time_s': efficiency.get('avg_total_epoch_time_s', ''),
        'final_test_inference_time_s': efficiency.get('final_test_inference_time_s', ''),
        'test_samples': efficiency.get('test_samples', ''),
        'latency_ms_per_sample': efficiency.get('latency_ms_per_sample', ''),
        'throughput_samples_per_s': efficiency.get('throughput_samples_per_s', ''),
        'checkpoint_size_mb': efficiency.get('checkpoint_size_mb', ''),
        'timing_profile_csv': result.get('timing_profile_csv', ''),
        'dataset_prepare_profile_csv': result.get('dataset_prepare_profile_csv', ''),
        'module_param_count_json': json.dumps(module_params, sort_keys=True),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    exists = os.path.exists(output_csv)
    if exists:
        with open(output_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            if existing_fields != fields:
                old_rows = list(reader)
        if existing_fields != fields:
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({field: old_row.get(field, '') for field in fields})
    with open(output_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_epoch(model, dataloader, optimizer, criterion, device, logger, epoch=0,
                use_gpam=False, use_mfvp=False, use_hap=False, use_rapp=False,
                rapp_balance_weight=0.01, use_nacc=False, use_qaph=False,
                use_aacaph=False):
    model.train()
    total_loss, num_batches = 0, 0
    
    for batch in dataloader:
        # 瑙ｆ瀽batch tensors
        # 甯冨眬: [user_ids, service_ids, qos_history, targets, (extra_features)?, (anchor)?, (anchor_info)?, (nacc)?]
        idx = 4
        gpam_features, mfvp_features, anchor, anchor_info, nacc_features = None, None, None, None, None
        
        user_ids, service_ids, qos_history, targets = batch[0], batch[1], batch[2], batch[3]
        
        if (use_mfvp or use_gpam) and len(batch) > idx:
            extra_features = batch[idx].to(device)
            gpam_features = extra_features if use_gpam else None
            mfvp_features = extra_features if use_mfvp else None
            idx += 1
        
        if (use_hap or use_qaph) and len(batch) > idx:
            anchor = batch[idx].to(device)
            idx += 1
        
        if use_aacaph and len(batch) > idx:
            anchor_info = batch[idx].to(device)
            idx += 1
        
        if use_nacc and len(batch) > idx:
            nacc_features = batch[idx].to(device)
            idx += 1
        
        user_ids, service_ids = user_ids.to(device), service_ids.to(device)
        qos_history, targets = qos_history.to(device), targets.to(device)
        
        attention_mask = (qos_history > 0).bool()
        
        optimizer.zero_grad()
        outputs = model(
            user_ids=user_ids,
            service_ids=service_ids,
            qos_history=qos_history,
            attention_mask=attention_mask,
            gpam_features=gpam_features,
            mfvp_features=mfvp_features,
            anchor=anchor,
            anchor_info=anchor_info,
            nacc_features=nacc_features
        )
        bad_buffers, bad_buffer_count = _sanitize_nonfinite_buffers(model)
        if bad_buffers and (epoch == 0 or num_batches < 3):
            logger.warning(
                f"  [Finite Guard] repaired {bad_buffer_count} non-finite buffer entries in {bad_buffers}"
            )
        
        pred = outputs['prediction'].squeeze()
        
        # V69: 棰勬祴NaN妫€娴嬶紙鍦╨oss璁＄畻鍓嶆嫤鎴?紝闃叉?NaN浼犳挱鍒發oss鍜屾?搴︼級
        if not torch.isfinite(pred).all():
            nan_count = (~torch.isfinite(pred)).sum().item()
            if epoch == 0 or num_batches < 3:
                logger.warning(f"  [Finite Guard] {nan_count}/{pred.numel()} non-finite predictions detected")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=10000.0, neginf=0.0)
        
        loss = criterion(pred, targets)
        
        # V56: RAPP璐熻浇鍧囪　鎹熷け
        if use_rapp and 'router_weights' in outputs:
            rapp_head = None
            if hasattr(model, 'cqpm') and hasattr(model.cqpm, 'rapp_head'):
                rapp_head = model.cqpm.rapp_head
            if rapp_head is not None:
                balance_loss = rapp_head.load_balance_loss(outputs['router_weights'])
                loss = loss + rapp_balance_weight * balance_loss
        
        if not torch.isfinite(loss):
            if epoch == 0 or num_batches < 3:
                logger.warning("  [Finite Guard] non-finite loss detected, skipping batch")
            continue
        
        loss.backward()
        all_params = [p for group in optimizer.param_groups for p in group['params']]
        bad_grads, bad_grad_count = _sanitize_nonfinite_grads(model)
        if bad_grads and (epoch == 0 or num_batches < 3):
            logger.warning(
                f"  [Finite Guard] sanitized {bad_grad_count} non-finite gradient entries in {bad_grads}"
            )
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        param_snapshot = _snapshot_trainable_params(model) if use_qaph else None
        optimizer.step()
        bad_params = _find_nonfinite_param_names(model)
        if bad_params:
            logger.warning(f"  [Finite Guard] non-finite parameters after optimizer step in {bad_params}; rolling back batch")
            if param_snapshot is not None:
                _restore_trainable_params(model, param_snapshot)
            _clear_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            continue
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


def train_epoch_optimized(model, dataloader, optimizer, criterion, device, logger, epoch=0,
                          use_gpam=False, use_mfvp=False, use_hap=False, use_rapp=False,
                          rapp_balance_weight=0.01, use_nacc=False, use_qaph=False,
                          use_aacaph=False, use_amp=False, scaler=None):
    """Training epoch with the same batch semantics plus optional CUDA AMP."""
    model.train()
    total_loss, num_batches = 0, 0
    amp_enabled = bool(use_amp and device.type == 'cuda')
    if amp_enabled and scaler is None:
        raise ValueError("use_amp=True requires a GradScaler")

    for batch in dataloader:
        idx = 4
        gpam_features, mfvp_features, anchor, anchor_info, nacc_features = None, None, None, None, None

        user_ids, service_ids, qos_history, targets = batch[0], batch[1], batch[2], batch[3]

        if (use_mfvp or use_gpam) and len(batch) > idx:
            extra_features = batch[idx].to(device)
            gpam_features = extra_features if use_gpam else None
            mfvp_features = extra_features if use_mfvp else None
            idx += 1

        if (use_hap or use_qaph) and len(batch) > idx:
            anchor = batch[idx].to(device)
            idx += 1

        if use_aacaph and len(batch) > idx:
            anchor_info = batch[idx].to(device)
            idx += 1

        if use_nacc and len(batch) > idx:
            nacc_features = batch[idx].to(device)
            idx += 1

        user_ids, service_ids = user_ids.to(device), service_ids.to(device)
        qos_history, targets = qos_history.to(device), targets.to(device)
        attention_mask = (qos_history > 0).bool()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                user_ids=user_ids,
                service_ids=service_ids,
                qos_history=qos_history,
                attention_mask=attention_mask,
                gpam_features=gpam_features,
                mfvp_features=mfvp_features,
                anchor=anchor,
                anchor_info=anchor_info,
                nacc_features=nacc_features
            )
            bad_buffers, bad_buffer_count = _sanitize_nonfinite_buffers(model)
            if bad_buffers and (epoch == 0 or num_batches < 3):
                logger.warning(
                    f"  [Finite Guard] repaired {bad_buffer_count} non-finite buffer entries in {bad_buffers}"
                )

            pred = outputs['prediction'].squeeze()
            if not torch.isfinite(pred).all():
                nan_count = (~torch.isfinite(pred)).sum().item()
                if epoch == 0 or num_batches < 3:
                    logger.warning(f"  [Finite Guard] {nan_count}/{pred.numel()} non-finite predictions detected")
                pred = torch.nan_to_num(pred, nan=0.0, posinf=10000.0, neginf=0.0)

            loss = criterion(pred, targets)

            if use_rapp and 'router_weights' in outputs:
                rapp_head = None
                if hasattr(model, 'cqpm') and hasattr(model.cqpm, 'rapp_head'):
                    rapp_head = model.cqpm.rapp_head
                if rapp_head is not None:
                    balance_loss = rapp_head.load_balance_loss(outputs['router_weights'])
                    loss = loss + rapp_balance_weight * balance_loss

        if not torch.isfinite(loss):
            if epoch == 0 or num_batches < 3:
                logger.warning("  [Finite Guard] non-finite loss detected, skipping batch")
            continue

        all_params = [p for group in optimizer.param_groups for p in group['params']]
        param_snapshot = _snapshot_trainable_params(model) if use_qaph else None
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            bad_grads, bad_grad_count = _sanitize_nonfinite_grads(model)
            if bad_grads and (epoch == 0 or num_batches < 3):
                logger.warning(
                    f"  [Finite Guard] sanitized {bad_grad_count} non-finite gradient entries in {bad_grads}"
                )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            bad_grads, bad_grad_count = _sanitize_nonfinite_grads(model)
            if bad_grads and (epoch == 0 or num_batches < 3):
                logger.warning(
                    f"  [Finite Guard] sanitized {bad_grad_count} non-finite gradient entries in {bad_grads}"
                )
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

        bad_params = _find_nonfinite_param_names(model)
        if bad_params:
            logger.warning(f"  [Finite Guard] non-finite parameters after optimizer step in {bad_params}; rolling back batch")
            if param_snapshot is not None:
                _restore_trainable_params(model, param_snapshot)
            _clear_optimizer_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            continue

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def _compute_lambda_inlier_mask(
    targets: np.ndarray,
    lambda_value: int,
    outlier_protocol: str = 'legacy_test_fit_predict',
    reference_targets: np.ndarray = None,
):
    """
    Compute the inlier mask for 位>0 evaluation.

    Protocols:
    - legacy_test_fit_predict: keep the historical behavior for full backward compatibility.
      Fit IsolationForest directly on evaluation targets and use fit_predict().
    - scheme_b_exact: fit IsolationForest on reference targets (typically train targets),
      score evaluation targets, and remove exactly 位% most-anomalous samples.
    """
    if lambda_value <= 0:
        return np.ones(len(targets), dtype=bool)

    if outlier_protocol == 'legacy_test_fit_predict':
        iso = IsolationForest(contamination=lambda_value / 100.0, random_state=42, n_estimators=100)
        labels = iso.fit_predict(targets.reshape(-1, 1))
        return labels == 1

    if outlier_protocol == 'scheme_b_exact':
        if reference_targets is None or len(reference_targets) == 0:
            raise ValueError("scheme_b_exact requires non-empty reference_targets")

        iso = IsolationForest(contamination=lambda_value / 100.0, random_state=42, n_estimators=100)
        iso.fit(reference_targets.reshape(-1, 1))

        # Lower score means more anomalous in scikit-learn.
        scores = iso.score_samples(targets.reshape(-1, 1))
        n_remove = int(len(targets) * lambda_value / 100.0)
        if n_remove <= 0:
            return np.ones(len(targets), dtype=bool)

        remove_idx = np.argsort(scores)[:n_remove]
        mask = np.ones(len(targets), dtype=bool)
        mask[remove_idx] = False
        return mask

    raise ValueError(f"Unknown outlier_protocol: {outlier_protocol}")


def _refresh_eval_caches(model):
    """
    Refresh detached graph-enhanced caches before evaluation/export.

    Minimal hotfix:
    - Do not change training objectives or model structure.
    - Only recompute SEGA / Independent-GCN cached embeddings from the
      current model parameters before metric evaluation or gate CSV export.

    The caches are plain attributes rather than state_dict buffers, so
    `load_state_dict(best_state_dict)` does not restore them automatically.
    """
    cache_owner = model.module if hasattr(model, 'module') and hasattr(model.module, 'state_dict') else model

    if hasattr(cache_owner, 'precompute_sega_embeddings'):
        cache_owner.precompute_sega_embeddings()
    if hasattr(cache_owner, 'precompute_indep_gcn_embeddings'):
        cache_owner.precompute_indep_gcn_embeddings()


def _load_checkpoint_state_dict(model, checkpoint_path, device, logger=None):
    """
    Load an existing checkpoint for post-hoc evaluation / gate export.

    This keeps the fix minimal: restore model weights, then rely on the
    refreshed evaluation/export caches to keep graph-enhanced features
    consistent with the loaded checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    candidate_state_dicts = []

    if isinstance(checkpoint, dict):
        for nested_key in ('state_dict', 'model_state_dict', 'model'):
            nested_state = checkpoint.get(nested_key)
            if isinstance(nested_state, dict):
                candidate_state_dicts.append((nested_key, nested_state))
        candidate_state_dicts.append(('raw', checkpoint))
    else:
        candidate_state_dicts.append(('raw', checkpoint))

    qaph_head = getattr(getattr(model, 'cqpm', None), 'qaph_head', None)
    allowed_missing_prefixes = tuple()
    if qaph_head is not None and getattr(qaph_head, 'anchor_usage', None) == 'hap':
        # Backward compatibility:
        # older HAP-mode QAPH checkpoints may predate anchor-as-feature support,
        # so they legitimately lack anchor_feature_mlp weights. That branch is
        # inactive when anchor_usage='hap', so these missing keys are safe to ignore.
        allowed_missing_prefixes = ("cqpm.qaph_head.anchor_feature_mlp.",)

    load_errors = []
    for candidate_name, state_dict in candidate_state_dicts:
        try:
            model.load_state_dict(state_dict, strict=True)
            if logger is not None:
                logger.info(f"Loaded checkpoint state_dict from {checkpoint_path} ({candidate_name})")
            return
        except Exception as exc:
            load_errors.append(f"{candidate_name}: {exc}")

        if isinstance(state_dict, dict):
            stripped_state_dict = {}
            for key, value in state_dict.items():
                if key == 'n_averaged':
                    continue
                new_key = key[len('module.'):] if key.startswith('module.') else key
                stripped_state_dict[new_key] = value
            try:
                model.load_state_dict(stripped_state_dict, strict=True)
                if logger is not None:
                    logger.info(
                        f"Loaded checkpoint state_dict from {checkpoint_path} "
                        f"({candidate_name}, stripped module./n_averaged)"
                    )
                return
            except Exception as exc:
                load_errors.append(f"{candidate_name} [stripped]: {exc}")

        for variant_name, variant_state_dict in ((candidate_name, state_dict), (f"{candidate_name} [stripped]", stripped_state_dict if isinstance(state_dict, dict) else None)):
            if variant_state_dict is None:
                continue
            try:
                incompatible = model.load_state_dict(variant_state_dict, strict=False)
            except Exception:
                continue

            missing_keys = list(getattr(incompatible, 'missing_keys', []))
            unexpected_keys = list(getattr(incompatible, 'unexpected_keys', []))
            disallowed_missing = [
                key for key in missing_keys
                if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
            ]
            if not disallowed_missing and not unexpected_keys:
                if logger is not None:
                    if missing_keys:
                        logger.warning(
                            "Loaded checkpoint %s (%s) with backward-compatible missing keys: %s",
                            checkpoint_path,
                            variant_name,
                            ", ".join(missing_keys),
                        )
                    else:
                        logger.info(f"Loaded checkpoint state_dict from {checkpoint_path} ({variant_name}, strict=False)")
                return

    raise RuntimeError(
        f"Failed to load checkpoint {checkpoint_path}. "
        f"Tried {len(load_errors)} state_dict variants. Errors: {' | '.join(load_errors)}"
    )


def parse_lambda_values(value):
    """Parse comma-separated lambda values into an ordered unique int list."""
    if value is None:
        return [0]
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value).split(',')
    lambdas = []
    for item in raw_items:
        item = str(item).strip()
        if not item:
            continue
        lambdas.append(int(float(item)))
    if not lambdas:
        lambdas = [0]
    return list(dict.fromkeys(lambdas))


def make_qaph_dataloader(dataset, batch_size, shuffle, num_workers, device, prefetch_factor=2):
    """Create a DataLoader with CUDA-friendly defaults while preserving tensor order."""
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
    }
    if num_workers and num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def _find_nonfinite_param_names(model, limit=8):
    """Return names of parameters containing NaN/Inf values."""
    bad = []
    for name, param in model.named_parameters():
        if param is not None and param.numel() and not torch.isfinite(param.detach()).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _find_nonfinite_state_names(model, limit=12):
    """Return state_dict entries containing NaN/Inf values, including BN buffers."""
    bad = []
    for name, value in model.state_dict().items():
        if torch.is_tensor(value) and torch.is_floating_point(value) and value.numel():
            if not torch.isfinite(value.detach()).all():
                bad.append(name)
                if len(bad) >= limit:
                    break
    return bad


def _sanitize_nonfinite_buffers(model, limit=8):
    """Repair NaN/Inf buffers such as BatchNorm running stats."""
    bad = []
    total_bad = 0
    with torch.no_grad():
        for name, buf in model.named_buffers():
            if not torch.is_tensor(buf) or not torch.is_floating_point(buf) or not buf.numel():
                continue
            finite_mask = torch.isfinite(buf)
            if finite_mask.all():
                continue
            bad_count = int((~finite_mask).sum().item())
            total_bad += bad_count
            if len(bad) < limit:
                bad.append(f"{name}({bad_count})")
            if name.endswith("running_var"):
                repaired = torch.nan_to_num(buf, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-6)
            else:
                repaired = torch.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0)
            buf.copy_(repaired)
    return bad, total_bad


def _find_nonfinite_grad_names(model, limit=8):
    """Return names of gradients containing NaN/Inf values."""
    bad = []
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.numel() and not torch.isfinite(param.grad.detach()).all():
            bad.append(name)
            if len(bad) >= limit:
                break
    return bad


def _sanitize_nonfinite_grads(model, limit=8):
    """Replace NaN/Inf gradient entries with zero and return a compact report."""
    bad = []
    total_bad = 0
    for name, param in model.named_parameters():
        if param.grad is None or not param.grad.numel():
            continue
        finite_mask = torch.isfinite(param.grad)
        if finite_mask.all():
            continue
        bad_count = int((~finite_mask).sum().item())
        total_bad += bad_count
        if len(bad) < limit:
            bad.append(f"{name}({bad_count})")
        param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)
    return bad, total_bad


def _snapshot_trainable_params(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _restore_trainable_params(model, snapshot):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in snapshot:
                param.copy_(snapshot[name])


def _clear_optimizer_state(optimizer):
    # AdamW moments can also become non-finite after a bad step; clearing is safer than
    # keeping poisoned state for the next batch.
    optimizer.state.clear()


def _dataset_cache_key(args, qos_type, split_name, split_id, ds_kwargs):
    """Build a cache key for TensorDataset outputs without storing target-slice-derived features."""
    key_payload = {
        'version': 2,
        'qos_type': qos_type,
        'split_name': split_name,
        'density': float(args.density),
        'seed': int(args.seed),
        'split_id': int(split_id),
        'tau': int(args.tau),
        'history_length': int(args.history_length),
        'rolling_history_sha256': str(args.rolling_history_sha256),
        'rolling_target_sha256': str(args.rolling_target_sha256),
        'data_signature': _dataset_file_signature(args.data_dir, qos_type),
        'seq_len': int(args.history_length),
        'val_ratio': float(args.val_ratio),
        'use_gpam_input': bool(args.use_gpam_input),
        'gpam_fuse_mfvp': bool(args.gpam_fuse_mfvp),
        'gpam_fill_missing': bool(args.gpam_fill_missing),
        'gpam_f1': int(args.gpam_f1),
        'use_mfvp': bool(args.use_mfvp),
        'mfvp_window': int(args.mfvp_window),
        'use_ptcs': bool(args.use_ptcs),
        'ptcs_top_k': int(args.ptcs_top_k),
        'use_nacc': bool(args.use_nacc),
        'nacc_top_k': int(args.nacc_top_k),
        'use_hap': bool(args.use_hap),
        'use_qaph': bool(args.use_qaph),
        'use_aacaph': bool(args.use_aacaph),
        'anchor_mode': str(args.anchor_mode),
        'anchor_ema_alpha': float(args.anchor_ema_alpha),
        'anchor_trim_ratio': float(args.anchor_trim_ratio),
        'anchor_neighbor_weight': float(args.anchor_neighbor_weight),
        'ptcs_enabled_in_dataset': ds_kwargs.get('ptcs_data') is not None,
        'nacc_enabled_in_dataset': ds_kwargs.get('nacc_data') is not None,
        'gpam_enabled_in_dataset': ds_kwargs.get('gpam_user_factors') is not None,
        'mfvp_enabled_in_dataset': ds_kwargs.get('mfvp_stats') is not None,
    }
    encoded = json.dumps(key_payload, sort_keys=True, default=str).encode('utf-8')
    digest = hashlib.sha1(encoded).hexdigest()[:16]
    d_label = f"d{int(round(float(args.density) * 100))}"
    filename = f"qaph_dataset_{qos_type}_{d_label}_seed{args.seed}_split{split_id}_{split_name}_{digest}.pt"
    return key_payload, filename


def _append_dataset_prepare_profile(profile_csv, row):
    if not profile_csv:
        return
    fields = [
        'split_name', 'num_samples', 'history_gather_time', 'anchor_time',
        'mfvp_time', 'ptcs_signal_time', 'nacc_time', 'tensor_stack_time',
        'total_prepare_time', 'cache_hit'
    ]
    exists = os.path.exists(profile_csv)
    os.makedirs(os.path.dirname(profile_csv), exist_ok=True)
    with open(profile_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in fields})


def _empty_prepare_profile(split_name, num_samples, cache_hit):
    return {
        'split_name': split_name,
        'num_samples': int(num_samples),
        'history_gather_time': 0.0,
        'anchor_time': 0.0,
        'mfvp_time': 0.0,
        'ptcs_signal_time': 0.0,
        'nacc_time': 0.0,
        'tensor_stack_time': 0.0,
        'total_prepare_time': 0.0,
        'cache_hit': cache_hit,
    }


def prepare_dataset_v42_cached(split_name, indices, history, target_matrix, seq_len, args,
                               qos_type, split_id, ds_kwargs, logger,
                               profile_csv=None):
    """Optionally cache prepare_dataset_v42 TensorDataset tensors on disk."""
    use_cache = bool(args.reuse_dataset_cache or args.rebuild_dataset_cache)
    if not use_cache:
        return prepare_dataset_v42(
            indices, history, target_matrix, seq_len,
            split_name=split_name, profile_csv=profile_csv,
            cache_hit='no_cache', logger=logger,
            **ds_kwargs
        )

    key_payload, filename = _dataset_cache_key(args, qos_type, split_name, split_id, ds_kwargs)
    cache_dir = os.path.abspath(os.path.expanduser(args.dataset_cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)

    if args.reuse_dataset_cache and not args.rebuild_dataset_cache and os.path.exists(cache_path):
        try:
            payload = torch.load(cache_path, map_location='cpu')
            if isinstance(payload, dict) and payload.get('key') == key_payload and 'tensors' in payload:
                tensors = tuple(payload['tensors'])
                logger.info(f"Dataset cache hit [{split_name}]: {cache_path}")
                _append_dataset_prepare_profile(
                    profile_csv,
                    _empty_prepare_profile(split_name, len(indices), 'hit_split_tensor_cache')
                )
                return TensorDataset(*tensors)
            logger.warning(f"Dataset cache key mismatch [{split_name}], rebuilding: {cache_path}")
        except Exception as exc:
            logger.warning(f"Dataset cache load failed [{split_name}], rebuilding: {exc}")

    dataset = prepare_dataset_v42(
        indices, history, target_matrix, seq_len,
        split_name=split_name, profile_csv=profile_csv,
        cache_hit='miss_split_tensor_cache', logger=logger,
        **ds_kwargs
    )
    try:
        torch.save({'key': key_payload, 'tensors': tuple(dataset.tensors)}, cache_path)
        logger.info(f"Dataset cache saved [{split_name}]: {cache_path}")
    except Exception as exc:
        logger.warning(f"Dataset cache save failed [{split_name}]: {exc}")
    return dataset


def compute_multi_lambda_metrics(all_preds, all_targets, lambda_values=[0, 2, 4, 6, 8, 10],
                                 outlier_protocol='legacy_test_fit_predict',
                                 reference_targets=None):
    all_preds, all_targets = np.array(all_preds), np.array(all_targets)
    results = {}

    for lv in lambda_values:
        if lv == 0:
            mae = np.mean(np.abs(all_preds - all_targets))
            rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        else:
            mask = _compute_lambda_inlier_mask(
                all_targets,
                lv,
                outlier_protocol=outlier_protocol,
                reference_targets=reference_targets,
            )
            mae = np.mean(np.abs(all_preds[mask] - all_targets[mask]))
            rmse = np.sqrt(np.mean((all_preds[mask] - all_targets[mask]) ** 2))
        results[lv] = {'mae': float(mae), 'rmse': float(rmse)}

    return results


def evaluate_multi_lambda(model, dataloader, device, lambda_values=[0, 2, 4, 6, 8, 10],
                          use_gpam=False, use_mfvp=False, use_hap=False, use_nacc=False,
                          use_qaph=False, use_aacaph=False,
                          outlier_protocol='legacy_test_fit_predict',
                          reference_targets=None,
                          refresh_caches=True,
                          use_amp=False):
    model.eval()
    if refresh_caches:
        _refresh_eval_caches(model)
    all_preds, all_targets = [], []
    amp_enabled = bool(use_amp and device.type == 'cuda')
    
    with torch.no_grad():
        for batch in dataloader:
            idx = 4
            gpam_features, mfvp_features, anchor, anchor_info, nacc_features = None, None, None, None, None
            
            user_ids, service_ids, qos_history, targets = batch[0], batch[1], batch[2], batch[3]
            
            if (use_mfvp or use_gpam) and len(batch) > idx:
                extra_features = batch[idx].to(device)
                gpam_features = extra_features if use_gpam else None
                mfvp_features = extra_features if use_mfvp else None
                idx += 1
            
            if (use_hap or use_qaph) and len(batch) > idx:
                anchor = batch[idx].to(device)
                idx += 1
            
            if use_aacaph and len(batch) > idx:
                anchor_info = batch[idx].to(device)
                idx += 1
            
            if use_nacc and len(batch) > idx:
                nacc_features = batch[idx].to(device)
                idx += 1
            
            user_ids, service_ids = user_ids.to(device), service_ids.to(device)
            qos_history = qos_history.to(device)
            
            attention_mask = (qos_history > 0).bool()
            
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(
                    user_ids=user_ids,
                    service_ids=service_ids,
                    qos_history=qos_history,
                    attention_mask=attention_mask,
                    gpam_features=gpam_features,
                    mfvp_features=mfvp_features,
                    anchor=anchor,
                    anchor_info=anchor_info,
                    nacc_features=nacc_features
                )
            
            all_preds.extend(outputs['prediction'].squeeze().cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
    
    return compute_multi_lambda_metrics(
        all_preds,
        all_targets,
        lambda_values=lambda_values,
        outlier_protocol=outlier_protocol,
        reference_targets=reference_targets,
    )


def profile_qaph_forward_flops(model, dataloader, device,
                               use_gpam=False, use_mfvp=False, use_hap=False,
                               use_nacc=False, use_qaph=False, use_aacaph=False,
                               use_amp=False, warmup_runs=2, logger=None):
    """Profile online batched forward FLOPs after cached features/graphs are ready."""
    if not hasattr(torch, 'profiler'):
        return {'supported': False, 'reason': 'torch.profiler is unavailable'}
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return {'supported': False, 'reason': 'empty dataloader'}

    def _forward_once(batch):
        idx = 4
        gpam_features, mfvp_features, anchor, anchor_info, nacc_features = None, None, None, None, None
        user_ids, service_ids, qos_history = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        if (use_mfvp or use_gpam) and len(batch) > idx:
            extra_features = batch[idx].to(device)
            gpam_features = extra_features if use_gpam else None
            mfvp_features = extra_features if use_mfvp else None
            idx += 1
        if (use_hap or use_qaph) and len(batch) > idx:
            anchor = batch[idx].to(device)
            idx += 1
        if use_aacaph and len(batch) > idx:
            anchor_info = batch[idx].to(device)
            idx += 1
        if use_nacc and len(batch) > idx:
            nacc_features = batch[idx].to(device)
            idx += 1

        attention_mask = (qos_history > 0).bool()
        amp_enabled = bool(use_amp and device.type == 'cuda')
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(
                user_ids=user_ids,
                service_ids=service_ids,
                qos_history=qos_history,
                attention_mask=attention_mask,
                gpam_features=gpam_features,
                mfvp_features=mfvp_features,
                anchor=anchor,
                anchor_info=anchor_info,
                nacc_features=nacc_features,
            )
        return outputs, int(user_ids.shape[0])

    model.eval()
    _refresh_eval_caches(model)
    try:
        with torch.no_grad():
            for _ in range(max(0, int(warmup_runs))):
                _forward_once(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda' and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=True,
            ) as prof:
                _, batch_size = _forward_once(batch)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

        total_flops = 0
        for event in prof.key_averages():
            flops = getattr(event, 'flops', None)
            if flops:
                total_flops += int(flops)
        flops_per_sample = float(total_flops / max(batch_size, 1))
        profile = {
            'supported': True,
            'batch_size': int(batch_size),
            'forward_flops_total': int(total_flops),
            'forward_flops_per_sample': flops_per_sample,
            'forward_flops_g_per_sample': float(flops_per_sample / 1e9),
            'note': (
                'Profiler-supported dense forward FLOPs only; offline feature '
                'construction and graph building are excluded.'
            ),
        }
        if logger is not None:
            logger.info(
                "Forward FLOPs profile: "
                f"batch={batch_size}, total={total_flops}, "
                f"per_sample={flops_per_sample:.2f} ({flops_per_sample / 1e9:.6f} G)"
            )
        return profile
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Forward FLOPs profiling failed: {exc}")
        return {'supported': False, 'reason': str(exc)}


def _pair_ids_from_indices(indices, num_services):
    arr = np.asarray(indices, dtype=np.int64)
    return arr[:, 0] * int(num_services) + arr[:, 1]


def compute_ptcs_signals_batch(users, services, history, user_neighbors, service_neighbors,
                               global_mean, global_std, seq_len=63):
    users = np.asarray(users, dtype=np.int64)
    services = np.asarray(services, dtype=np.int64)
    actual_len = min(seq_len, history.shape[2])
    bsz = len(users)
    user_signal = np.zeros((bsz, seq_len), dtype=np.float32)
    service_signal = np.zeros((bsz, seq_len), dtype=np.float32)
    if bsz == 0:
        return user_signal, service_signal

    user_nb = user_neighbors[users]
    user_vals = history[user_nb, services[:, None], :actual_len]
    user_mask = user_vals > 0
    user_count = user_mask.sum(axis=1)
    user_sum = user_vals.sum(axis=1)
    valid = user_count > 0
    user_mean = np.divide(user_sum, user_count, out=np.zeros_like(user_sum, dtype=np.float32), where=valid)
    user_signal[:, :actual_len] = np.where(valid, (user_mean - global_mean) / global_std, 0.0)

    service_nb = service_neighbors[services]
    service_vals = history[users[:, None], service_nb, :actual_len]
    service_mask = service_vals > 0
    service_count = service_mask.sum(axis=1)
    service_sum = service_vals.sum(axis=1)
    valid = service_count > 0
    service_mean = np.divide(service_sum, service_count, out=np.zeros_like(service_sum, dtype=np.float32), where=valid)
    service_signal[:, :actual_len] = np.where(valid, (service_mean - global_mean) / global_std, 0.0)
    return user_signal, service_signal


def compute_ema_anchor_batch(histories, alpha=0.1):
    anchors = np.zeros(histories.shape[0], dtype=np.float32)
    for i, h in enumerate(histories):
        anchors[i] = compute_ema_anchor(h, alpha=alpha)
    return anchors


def compute_trimmed_anchor_batch(histories, trim_ratio=0.1):
    anchors = np.zeros(histories.shape[0], dtype=np.float32)
    for i, h in enumerate(histories):
        anchors[i] = compute_trimmed_anchor(h, trim_ratio=trim_ratio)
    return anchors


def compute_mfvp_for_batch(histories, global_mean, global_std, window=5,
                           ptcs_user_signal=None, ptcs_service_signal=None):
    histories = histories.astype(np.float32, copy=False)
    bsz, seq_len = histories.shape
    use_ptcs = ptcs_user_signal is not None and ptcs_service_signal is not None
    feat_dim = 9 if use_ptcs else 7
    features = np.zeros((bsz, seq_len, feat_dim), dtype=np.float32)
    features[:, :, 0] = (histories - global_mean) / global_std

    log_vals = np.log1p(np.clip(histories, 0, None))
    log_mean = np.log1p(global_mean)
    log_std = np.log1p(global_std)
    features[:, :, 1] = (log_vals - log_mean) / (log_std + 1e-8)

    half_w = window // 2
    nonzero = histories > 0
    for t in range(seq_len):
        start = max(0, t - half_w)
        end = min(seq_len, t + half_w + 1)
        local = histories[:, start:end]
        local_mask = nonzero[:, start:end]
        count = local_mask.sum(axis=1)
        local_sum = np.where(local_mask, local, 0.0).sum(axis=1)
        mean = np.divide(local_sum, count, out=np.zeros(bsz, dtype=np.float32), where=count > 0)
        sq = np.where(local_mask, (local - mean[:, None]) ** 2, 0.0).sum(axis=1)
        std = np.sqrt(np.divide(sq, count, out=np.zeros(bsz, dtype=np.float32), where=count > 0))
        valid = count > 0
        features[valid, t, 2] = (mean[valid] - global_mean) / global_std
        features[valid, t, 3] = std[valid] / global_std
        features[valid, t, 6] = std[valid] / (mean[valid] + 1e-8)

    diff = np.zeros_like(histories, dtype=np.float32)
    diff[:, 1:] = histories[:, 1:] - histories[:, :-1]
    features[:, :, 4] = diff / global_std
    features[:, :, 5] = nonzero.astype(np.float32)
    if use_ptcs:
        features[:, :, 7] = ptcs_user_signal[:, :seq_len]
        features[:, :, 8] = ptcs_service_signal[:, :seq_len]
    return np.clip(features, -10, 10).astype(np.float32)


def compute_nacc_features_batch(users, services, history, user_neighbors, service_neighbors,
                                global_mean, global_std):
    users = np.asarray(users, dtype=np.int64)
    services = np.asarray(services, dtype=np.int64)
    bsz = len(users)
    features = np.zeros((bsz, 6), dtype=np.float32)
    if bsz == 0:
        return features

    user_nb = user_neighbors[users]
    u_vals = history[user_nb, services[:, None], :]
    u_mask = u_vals > 0
    u_count = u_mask.sum(axis=(1, 2))
    u_sum = u_vals.sum(axis=(1, 2))
    u_mean = np.divide(u_sum, u_count, out=np.zeros(bsz, dtype=np.float32), where=u_count > 0)
    u_sq = np.where(u_mask, (u_vals - u_mean[:, None, None]) ** 2, 0.0).sum(axis=(1, 2))
    u_std = np.sqrt(np.divide(u_sq, u_count, out=np.zeros(bsz, dtype=np.float32), where=u_count > 0))
    valid = u_count > 0
    features[valid, 0] = (u_mean[valid] - global_mean) / global_std
    features[valid, 1] = u_std[valid] / global_std
    features[:, 2] = u_mask.any(axis=2).mean(axis=1)

    service_nb = service_neighbors[services]
    s_vals = history[users[:, None], service_nb, :]
    s_mask = s_vals > 0
    s_count = s_mask.sum(axis=(1, 2))
    s_sum = s_vals.sum(axis=(1, 2))
    s_mean = np.divide(s_sum, s_count, out=np.zeros(bsz, dtype=np.float32), where=s_count > 0)
    s_sq = np.where(s_mask, (s_vals - s_mean[:, None, None]) ** 2, 0.0).sum(axis=(1, 2))
    s_std = np.sqrt(np.divide(s_sq, s_count, out=np.zeros(bsz, dtype=np.float32), where=s_count > 0))
    valid = s_count > 0
    features[valid, 3] = (s_mean[valid] - global_mean) / global_std
    features[valid, 4] = s_std[valid] / global_std
    features[:, 5] = s_mask.any(axis=2).mean(axis=1)
    return np.clip(features, -10, 10).astype(np.float32)


def export_qaph_gate_analysis_csv(model, dataloader, dataset_tensors, sample_indices,
                                  full_history,
                                  qos_type, density, split_id, output_csv, device,
                                  use_gpam=False, use_mfvp=False, use_hap=False,
                                  use_nacc=False, use_qaph=False, use_aacaph=False,
                                  logger=None):
    """
    Export sample-level QAPH gate values and related test metadata.

    The CSV schema is aligned with analyze_qaph_gate_interpretability.py and
    includes a few extra prediction columns for optional follow-up analysis.
    """
    if not use_qaph:
        if logger is not None:
            logger.info("Gate CSV export skipped because QAPH is disabled.")
        return None

    user_ids_tensor = dataset_tensors[0]
    service_ids_tensor = dataset_tensors[1]
    history_tensor = dataset_tensors[2]
    targets_tensor = dataset_tensors[3]

    anchor_tensor = None
    anchor_pos = 4
    if use_mfvp or use_gpam:
        anchor_pos += 1
    if (use_hap or use_qaph) and len(dataset_tensors) > anchor_pos:
        anchor_tensor = dataset_tensors[anchor_pos]

    rows = []
    row_offset = 0
    model.eval()
    _refresh_eval_caches(model)

    with torch.no_grad():
        for batch in dataloader:
            idx = 4
            gpam_features, mfvp_features, anchor, anchor_info, nacc_features = None, None, None, None, None

            user_ids = batch[0].to(device)
            service_ids = batch[1].to(device)
            qos_history = batch[2].to(device)
            targets = batch[3]
            batch_size = targets.shape[0]

            if (use_mfvp or use_gpam) and len(batch) > idx:
                extra_features = batch[idx].to(device)
                gpam_features = extra_features if use_gpam else None
                mfvp_features = extra_features if use_mfvp else None
                idx += 1

            if (use_hap or use_qaph) and len(batch) > idx:
                anchor = batch[idx].to(device)
                idx += 1

            if use_aacaph and len(batch) > idx:
                anchor_info = batch[idx].to(device)
                idx += 1

            if use_nacc and len(batch) > idx:
                nacc_features = batch[idx].to(device)
                idx += 1

            attention_mask = (qos_history > 0).bool()
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(
                    user_ids=user_ids,
                    service_ids=service_ids,
                    qos_history=qos_history,
                    attention_mask=attention_mask,
                    gpam_features=gpam_features,
                    mfvp_features=mfvp_features,
                    anchor=anchor,
                    anchor_info=anchor_info,
                    nacc_features=nacc_features
                )

            gate_np = outputs.get('gate', torch.zeros(batch_size, 1, device=device)).squeeze(-1).cpu().numpy()
            pred_np = outputs['prediction'].squeeze(-1).cpu().numpy()
            direct_pred_np = outputs.get('direct_pred', outputs['prediction']).squeeze(-1).cpu().numpy()
            anchor_pred_np = outputs.get('anchor_pred', outputs['prediction']).squeeze(-1).cpu().numpy()

            batch_user_ids = user_ids_tensor[row_offset:row_offset + batch_size].numpy()
            batch_service_ids = service_ids_tensor[row_offset:row_offset + batch_size].numpy()
            batch_targets = targets_tensor[row_offset:row_offset + batch_size].numpy()
            batch_anchors = (
                anchor_tensor[row_offset:row_offset + batch_size].numpy()
                if anchor_tensor is not None else np.zeros(batch_size, dtype=np.float32)
            )

            for i in range(batch_size):
                u, s = sample_indices[row_offset + i]
                hist_nonzero_count = int(np.count_nonzero(full_history[u, s, :] > 0))
                anchor_value = float(batch_anchors[i])
                y_true_value = float(batch_targets[i])
                g_value = float(gate_np[i])
                y_pred_value = float(pred_np[i])
                direct_pred_value = float(direct_pred_np[i])
                anchor_pred_value = float(anchor_pred_np[i])
                anchor_is_zero = 1 if anchor_value <= 0 else 0
                anchor_ratio_raw = (y_true_value / anchor_value) if anchor_value > 0 else np.nan
                rows.append({
                    'dataset': qos_type.upper(),
                    'density': int(round(density * 100)),
                    'split_id': int(split_id),
                    'user_id': int(batch_user_ids[i]),
                    'service_id': int(batch_service_ids[i]),
                    'g': g_value,
                    'hap_weight': g_value,
                    'direct_weight': 1.0 - g_value,
                    'y_true': y_true_value,
                    'y_pred': y_pred_value,
                    'direct_pred': direct_pred_value,
                    'anchor_pred': anchor_pred_value,
                    'abs_err_final': abs(y_pred_value - y_true_value),
                    'abs_err_direct': abs(direct_pred_value - y_true_value),
                    'abs_err_anchor': abs(anchor_pred_value - y_true_value),
                    'anchor': anchor_value,
                    'anchor_is_zero': anchor_is_zero,
                    'anchor_ratio_raw': anchor_ratio_raw,
                    'hist_nonzero_count': hist_nonzero_count,
                })

            row_offset += batch_size

    if row_offset != len(sample_indices):
        raise RuntimeError(
            f"Gate CSV export mismatch: got {row_offset} rows, expected {len(sample_indices)}"
        )

    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    fieldnames = [
        'dataset', 'density', 'split_id', 'user_id', 'service_id',
        'g', 'hap_weight', 'direct_weight',
        'y_true', 'y_pred', 'direct_pred', 'anchor_pred',
        'abs_err_final', 'abs_err_direct', 'abs_err_anchor',
        'anchor', 'anchor_is_zero', 'anchor_ratio_raw', 'hist_nonzero_count'
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if logger is not None:
        logger.info(f"QAPH gate CSV exported to {output_csv} ({len(rows)} rows)")
    return output_csv


def prepare_dataset_v42(indices, history, target_matrix, seq_len=63,
                        gpam_user_factors=None, gpam_service_factors=None,
                        mfvp_stats=None, mfvp_window=5, ptcs_data=None,
                        use_hap=False, nacc_data=None,
                        gpam_fuse_mfvp=False, use_aacaph=False,
                        anchor_mode='mean', anchor_ema_alpha=0.1,
                        anchor_trim_ratio=0.1, anchor_neighbor_weight=0.3,
                        split_name='dataset', profile_csv=None,
                        cache_hit='miss', logger=None):
    """鍑嗗?鏁版嵁闆?
    
    V50: 鏀?寔GPAM鍥犲瓙杈撳叆
    V52: 鏀?寔MFVP澶氱淮鐗瑰緛杈撳叆
    V55: 鏀?寔PTCS鍗忓悓淇″彿锛堟墿灞昅FVP浠?缁村埌9缁达級
    V56: 鏀?寔HAP anchor棰勮?绠?
    V60: 鏀?寔NACC閭诲眳鑱氬悎鍗忓悓涓婁笅鏂囷紙6缁磒air-specific鐗瑰緛锛?
    V62: 鏀?寔GPAM+MFVP铻嶅悎锛圙PAM鍥犲瓙杩藉姞鍒癕FVP鐗瑰緛鍚庨潰锛?
    V66: 鏀?寔AACAPH anchor_info棰勮?绠楋紙3缁? log1p_anchor, nonzero_ratio, cv锛?
    V70: 鏀?寔澶氱?anchor璁＄畻妯″紡锛坢ean/ema/trimmed/neighbor锛?
    - gpam_user_factors: [n_users, tau, f1] 姣忔椂闂存?鐨勭敤鎴種MF鍥犲瓙
    - gpam_service_factors: [n_services, tau, f1] 姣忔椂闂存?鐨勬湇鍔?MF鍥犲瓙
    - mfvp_stats: dict with 'global_mean', 'global_std' (from compute_mfvp_features)
    - mfvp_window: MFVP婊戝姩绐楀彛澶у皬
    - ptcs_data: dict from compute_ptcs_neighbors() (V55)
    - use_hap: bool, 鏄?惁棰勮?绠梐nchor锛堝巻鍙查潪闆跺潎鍊硷級
    - nacc_data: dict from compute_ptcs_neighbors() for NACC (V60)
    - gpam_fuse_mfvp: V62 - 鏄?惁灏咷PAM鍥犲瓙杩藉姞鍒癕FVP鐗瑰緛
    - use_aacaph: V66 - 鏄?惁棰勮?绠梐nchor_info锛?缁磋緟鍔╃壒寰侊級
    - anchor_mode: V70 - anchor璁＄畻妯″紡 (mean/ema/trimmed/neighbor)
    - anchor_ema_alpha: V70 - EMA琛板噺鐜?
    - anchor_trim_ratio: V70 - 鎴?柇鍧囧€兼瘮渚?
    - anchor_neighbor_weight: V70 - 閭诲眳anchor铻嶅悎鏉冮噸
    """
    user_ids_list, service_ids_list = [], []
    history_list, target_list = [], []
    anchor_list = [] if use_hap else None
    anchor_info_list = [] if use_aacaph else None
    # V62: GPAM+MFVP铻嶅悎妯″紡涓嬶紝gpam_list涓嶅啀鐙?珛浣跨敤
    gpam_list = [] if (gpam_user_factors is not None and not gpam_fuse_mfvp) else None
    mfvp_list = [] if mfvp_stats is not None else None
    nacc_list = [] if nacc_data is not None else None
    profile = _empty_prepare_profile(split_name, len(indices), cache_hit)
    total_start = time.time()
    profile_enabled = profile_csv is not None
    
    for u, s in tqdm(indices, desc="Preparing dataset"):
        user_ids_list.append(u)
        service_ids_list.append(s)

        timer = time.time()
        h = history[u, s, :]
        if len(h) < seq_len:
            h = np.pad(h, (0, seq_len - len(h)), mode='edge')
        elif len(h) > seq_len:
            h = h[-seq_len:]

        history_list.append(h)
        target_list.append(target_matrix[u, s])
        if profile_enabled:
            profile['history_gather_time'] += time.time() - timer
        
        # V56/V70: 璁＄畻anchor锛堟敮鎸佸?绉嶆ā寮忥級
        if use_hap:
            timer = time.time()
            if anchor_mode == 'ema':
                anchor_list.append(compute_ema_anchor(h, alpha=anchor_ema_alpha))
            elif anchor_mode == 'trimmed':
                anchor_list.append(compute_trimmed_anchor(h, trim_ratio=anchor_trim_ratio))
            elif anchor_mode == 'neighbor' and ptcs_data is not None:
                anchor_list.append(compute_neighbor_weighted_anchor(
                    u, s, h, ptcs_data['history'],
                    ptcs_data['user_neighbors'], ptcs_data['service_neighbors'],
                    neighbor_weight=anchor_neighbor_weight,
                    ema_alpha=anchor_ema_alpha
                ))
            else:
                # 榛樿?: 绠€鍗曞潎鍊?(V56鍘熷?閫昏緫)
                nonzero_vals = h[h > 0]
                if len(nonzero_vals) > 0:
                    anchor_list.append(float(nonzero_vals.mean()))
                else:
                    anchor_list.append(0.0)
            if profile_enabled:
                profile['anchor_time'] += time.time() - timer
        
        # V66: 璁＄畻anchor_info锛?缁磋緟鍔╃壒寰侊級
        if use_aacaph:
            timer = time.time()
            nonzero_vals = h[h > 0]
            if len(nonzero_vals) > 0:
                anchor_mean = float(nonzero_vals.mean())
                anchor_std = float(nonzero_vals.std()) if len(nonzero_vals) > 1 else 0.0
                log1p_anchor = float(np.log1p(anchor_mean))
                nonzero_ratio = float(len(nonzero_vals)) / max(len(h), 1)
                cv = anchor_std / (anchor_mean + 1e-8)
            else:
                log1p_anchor = 0.0
                nonzero_ratio = 0.0
                cv = 0.0
            anchor_info_list.append([log1p_anchor, nonzero_ratio, cv])
            if profile_enabled:
                profile['anchor_time'] += time.time() - timer
        
        # V52/V55/V62: 璁＄畻MFVP鐗瑰緛锛堝彲閫塒TCS鎵╁睍 + 鍙?€塆PAM铻嶅悎锛?
        if mfvp_stats is not None:
            ptcs_user_signal, ptcs_service_signal = None, None
            if ptcs_data is not None:
                timer = time.time()
                ptcs_user_signal, ptcs_service_signal = compute_ptcs_signals(
                    u, s, ptcs_data['history'],
                    ptcs_data['user_neighbors'], ptcs_data['service_neighbors'],
                    ptcs_data['global_mean'], ptcs_data['global_std'],
                    seq_len=seq_len
                )
                if profile_enabled:
                    profile['ptcs_signal_time'] += time.time() - timer
            timer = time.time()
            mfvp_feat = compute_mfvp_for_sample(
                h, mfvp_stats['global_mean'], mfvp_stats['global_std'], mfvp_window,
                ptcs_user_signal=ptcs_user_signal,
                ptcs_service_signal=ptcs_service_signal
            )  # [seq_len, 7 or 9]
            
            # V62: 灏咷PAM NMF鍥犲瓙杩藉姞鍒癕FVP鐗瑰緛鍚庨潰
            if gpam_fuse_mfvp and gpam_user_factors is not None:
                uf = gpam_user_factors[u]  # [tau, f1]
                sf = gpam_service_factors[s]  # [tau, f1]
                if uf.shape[0] < seq_len:
                    uf = np.pad(uf, ((0, seq_len - uf.shape[0]), (0, 0)), mode='edge')
                elif uf.shape[0] > seq_len:
                    uf = uf[-seq_len:]
                if sf.shape[0] < seq_len:
                    sf = np.pad(sf, ((0, seq_len - sf.shape[0]), (0, 0)), mode='edge')
                elif sf.shape[0] > seq_len:
                    sf = sf[-seq_len:]
                gpam_feat = np.concatenate([uf, sf], axis=-1)  # [seq_len, 2*f1]
                mfvp_feat = np.concatenate([mfvp_feat, gpam_feat], axis=-1)  # [seq_len, 7/9 + 2*f1]
            
            mfvp_list.append(mfvp_feat)
            if profile_enabled:
                profile['mfvp_time'] += time.time() - timer
        
        # V60: 璁＄畻NACC鐗瑰緛锛?缁达級
        if nacc_data is not None:
            timer = time.time()
            nacc_feat = compute_nacc_features(
                u, s, nacc_data['history'],
                nacc_data['user_neighbors'], nacc_data['service_neighbors'],
                nacc_data['global_mean'], nacc_data['global_std']
            )  # [6]
            nacc_list.append(nacc_feat)
            if profile_enabled:
                profile['nacc_time'] += time.time() - timer
        
        # V50: 鎷兼帴鐢ㄦ埛鍜屾湇鍔＄殑NMF鍥犲瓙 鈫?[seq_len, 2*f1]锛堜粎闈炶瀺鍚堟ā寮忥級
        if gpam_user_factors is not None and not gpam_fuse_mfvp:
            timer = time.time()
            uf = gpam_user_factors[u]  # [tau, f1]
            sf = gpam_service_factors[s]  # [tau, f1]
            # 纭?繚闀垮害鍖归厤
            if uf.shape[0] < seq_len:
                uf = np.pad(uf, ((0, seq_len - uf.shape[0]), (0, 0)), mode='edge')
            elif uf.shape[0] > seq_len:
                uf = uf[-seq_len:]
            if sf.shape[0] < seq_len:
                sf = np.pad(sf, ((0, seq_len - sf.shape[0]), (0, 0)), mode='edge')
            elif sf.shape[0] > seq_len:
                sf = sf[-seq_len:]
            gpam_feat = np.concatenate([uf, sf], axis=-1)  # [seq_len, 2*f1]
            gpam_list.append(gpam_feat)
            if profile_enabled:
                profile['mfvp_time'] += time.time() - timer

    stack_start = time.time()
    tensors = [
        torch.LongTensor(user_ids_list),
        torch.LongTensor(service_ids_list),
        torch.FloatTensor(np.array(history_list)),
        torch.FloatTensor(target_list)
    ]
    
    # 绗?涓猼ensor: MFVP鎴朑PAM锛堜簰鏂ワ級
    if mfvp_list is not None:
        tensors.append(torch.FloatTensor(np.array(mfvp_list)))
    elif gpam_list is not None:
        tensors.append(torch.FloatTensor(np.array(gpam_list)))
    
    # V56: anchor tensor
    if anchor_list is not None:
        tensors.append(torch.FloatTensor(anchor_list))
    
    # V66: anchor_info tensor [N, 3]
    if anchor_info_list is not None:
        tensors.append(torch.FloatTensor(np.array(anchor_info_list)))
    
    # V60: NACC tensor [N, 6]
    if nacc_list is not None:
        tensors.append(torch.FloatTensor(np.array(nacc_list)))

    profile['tensor_stack_time'] = time.time() - stack_start
    profile['total_prepare_time'] = time.time() - total_start
    _append_dataset_prepare_profile(profile_csv, profile)
    if logger is not None:
        logger.info(
            f"Dataset prepared [{split_name}]: n={len(indices):,}, "
            f"cache={cache_hit}, total={profile['total_prepare_time']:.2f}s"
        )

    return TensorDataset(*tensors)


def _dataset_file_signature(data_dir, qos_type):
    fname = 'rtdata.txt' if qos_type == 'rt' else 'tpdata.txt'
    path = os.path.join(data_dir, fname)
    if os.path.exists(path):
        st = os.stat(path)
        return {'path': os.path.abspath(path), 'size': int(st.st_size), 'mtime': float(st.st_mtime)}
    return {'path': os.path.abspath(path), 'size': -1, 'mtime': -1.0}


def _pair_feature_cache_key(args, qos_type, ds_kwargs, history):
    key_payload = {
        'version': 2,
        'qos_type': qos_type,
        'tau': int(args.tau),
        'history_length': int(args.history_length),
        'rolling_history_sha256': str(args.rolling_history_sha256),
        'rolling_target_sha256': str(args.rolling_target_sha256),
        'seq_len': int(args.history_length),
        'history_shape': tuple(int(x) for x in history.shape),
        'data_signature': _dataset_file_signature(args.data_dir, qos_type),
        'use_gpam_input': bool(args.use_gpam_input),
        'gpam_fuse_mfvp': bool(args.gpam_fuse_mfvp),
        'gpam_fill_missing': bool(args.gpam_fill_missing),
        'gpam_f1': int(args.gpam_f1),
        'use_mfvp': bool(args.use_mfvp),
        'mfvp_window': int(args.mfvp_window),
        'use_ptcs': bool(args.use_ptcs),
        'ptcs_top_k': int(args.ptcs_top_k),
        'use_nacc': bool(args.use_nacc),
        'nacc_top_k': int(args.nacc_top_k),
        'use_hap': bool(args.use_hap),
        'use_qaph': bool(args.use_qaph),
        'use_aacaph': bool(args.use_aacaph),
        'anchor_mode': str(args.anchor_mode),
        'anchor_ema_alpha': float(args.anchor_ema_alpha),
        'anchor_trim_ratio': float(args.anchor_trim_ratio),
        'anchor_neighbor_weight': float(args.anchor_neighbor_weight),
        'ptcs_enabled_in_dataset': ds_kwargs.get('ptcs_data') is not None,
        'nacc_enabled_in_dataset': ds_kwargs.get('nacc_data') is not None,
        'gpam_enabled_in_dataset': ds_kwargs.get('gpam_user_factors') is not None,
        'mfvp_enabled_in_dataset': ds_kwargs.get('mfvp_stats') is not None,
    }
    digest = hashlib.sha1(json.dumps(key_payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]
    filename = f"qaph_pair_feature_bank_{qos_type}_tau{args.tau}_L{args.history_length}_{digest}.pt"
    return key_payload, filename


def _slice_or_pad_time(arr, seq_len):
    if arr.shape[1] == seq_len:
        return arr
    if arr.shape[1] > seq_len:
        return arr[:, -seq_len:]
    pad_width = [(0, 0), (0, seq_len - arr.shape[1])]
    if arr.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(arr, tuple(pad_width), mode='edge')


def _build_pair_feature_bank_tensors(valid_indices, history, target_matrix, seq_len, ds_kwargs,
                                     chunk_size, logger=None, profile_csv=None):
    num_services = history.shape[1]
    chunks = []
    total_profile = _empty_prepare_profile('pair_feature_bank', len(valid_indices), 'miss_pair_feature_bank')
    total_start = time.time()
    mfvp_stats = ds_kwargs.get('mfvp_stats')
    ptcs_data = ds_kwargs.get('ptcs_data')
    nacc_data = ds_kwargs.get('nacc_data')
    gpam_user_factors = ds_kwargs.get('gpam_user_factors')
    gpam_service_factors = ds_kwargs.get('gpam_service_factors')
    gpam_fuse_mfvp = ds_kwargs.get('gpam_fuse_mfvp', False)
    use_hap = ds_kwargs.get('use_hap', False)
    use_aacaph = ds_kwargs.get('use_aacaph', False)
    anchor_mode = ds_kwargs.get('anchor_mode', 'mean')

    for start in tqdm(range(0, len(valid_indices), chunk_size), desc="Building pair feature bank"):
        chunk = np.asarray(valid_indices[start:start + chunk_size], dtype=np.int64)
        users = chunk[:, 0]
        services = chunk[:, 1]
        chunk_tensors = [
            torch.LongTensor(users),
            torch.LongTensor(services),
        ]

        timer = time.time()
        h_batch = history[users, services, :]
        h_batch = _slice_or_pad_time(h_batch, seq_len).astype(np.float32, copy=False)
        targets = target_matrix[users, services].astype(np.float32, copy=False)
        total_profile['history_gather_time'] += time.time() - timer
        chunk_tensors.extend([torch.FloatTensor(h_batch), torch.FloatTensor(targets)])

        extra_features = None
        if mfvp_stats is not None:
            ptcs_user_signal, ptcs_service_signal = None, None
            if ptcs_data is not None:
                timer = time.time()
                ptcs_user_signal, ptcs_service_signal = compute_ptcs_signals_batch(
                    users, services, ptcs_data['history'],
                    ptcs_data['user_neighbors'], ptcs_data['service_neighbors'],
                    ptcs_data['global_mean'], ptcs_data['global_std'],
                    seq_len=seq_len
                )
                total_profile['ptcs_signal_time'] += time.time() - timer
            timer = time.time()
            extra_features = compute_mfvp_for_batch(
                h_batch, mfvp_stats['global_mean'], mfvp_stats['global_std'],
                window=ds_kwargs.get('mfvp_window', 5),
                ptcs_user_signal=ptcs_user_signal,
                ptcs_service_signal=ptcs_service_signal
            )
            total_profile['mfvp_time'] += time.time() - timer
            if gpam_fuse_mfvp and gpam_user_factors is not None:
                timer = time.time()
                uf = _slice_or_pad_time(gpam_user_factors[users], seq_len)
                sf = _slice_or_pad_time(gpam_service_factors[services], seq_len)
                extra_features = np.concatenate([extra_features, uf, sf], axis=-1).astype(np.float32)
                total_profile['mfvp_time'] += time.time() - timer
            chunk_tensors.append(torch.FloatTensor(extra_features))
        elif gpam_user_factors is not None and not gpam_fuse_mfvp:
            timer = time.time()
            uf = _slice_or_pad_time(gpam_user_factors[users], seq_len)
            sf = _slice_or_pad_time(gpam_service_factors[services], seq_len)
            extra_features = np.concatenate([uf, sf], axis=-1).astype(np.float32)
            total_profile['mfvp_time'] += time.time() - timer
            chunk_tensors.append(torch.FloatTensor(extra_features))

        if use_hap:
            timer = time.time()
            if anchor_mode == 'ema':
                anchors = compute_ema_anchor_batch(h_batch, alpha=ds_kwargs.get('anchor_ema_alpha', 0.1))
            elif anchor_mode == 'trimmed':
                anchors = compute_trimmed_anchor_batch(h_batch, trim_ratio=ds_kwargs.get('anchor_trim_ratio', 0.1))
            elif anchor_mode == 'neighbor' and ptcs_data is not None:
                anchors = np.array([
                    compute_neighbor_weighted_anchor(
                        int(u), int(s), h_batch[i], ptcs_data['history'],
                        ptcs_data['user_neighbors'], ptcs_data['service_neighbors'],
                        neighbor_weight=ds_kwargs.get('anchor_neighbor_weight', 0.3),
                        ema_alpha=ds_kwargs.get('anchor_ema_alpha', 0.1)
                    )
                    for i, (u, s) in enumerate(zip(users, services))
                ], dtype=np.float32)
            else:
                mask = h_batch > 0
                counts = mask.sum(axis=1)
                sums = h_batch.sum(axis=1)
                anchors = np.divide(sums, counts, out=np.zeros(len(users), dtype=np.float32), where=counts > 0)
            total_profile['anchor_time'] += time.time() - timer
            chunk_tensors.append(torch.FloatTensor(anchors))

        if use_aacaph:
            timer = time.time()
            mask = h_batch > 0
            counts = mask.sum(axis=1).astype(np.float32)
            sums = h_batch.sum(axis=1)
            means = np.divide(sums, counts, out=np.zeros(len(users), dtype=np.float32), where=counts > 0)
            sq = np.where(mask, (h_batch - means[:, None]) ** 2, 0.0).sum(axis=1)
            stds = np.sqrt(np.divide(sq, counts, out=np.zeros(len(users), dtype=np.float32), where=counts > 0))
            anchor_info = np.stack([
                np.log1p(means),
                counts / max(seq_len, 1),
                np.divide(stds, means + 1e-8, out=np.zeros_like(stds), where=means > 0)
            ], axis=1).astype(np.float32)
            total_profile['anchor_time'] += time.time() - timer
            chunk_tensors.append(torch.FloatTensor(anchor_info))

        if nacc_data is not None:
            timer = time.time()
            nacc_features = compute_nacc_features_batch(
                users, services, nacc_data['history'],
                nacc_data['user_neighbors'], nacc_data['service_neighbors'],
                nacc_data['global_mean'], nacc_data['global_std']
            )
            total_profile['nacc_time'] += time.time() - timer
            chunk_tensors.append(torch.FloatTensor(nacc_features))

        pair_ids = torch.LongTensor(users * int(num_services) + services)
        chunks.append((pair_ids, chunk_tensors))

    stack_start = time.time()
    pair_ids_all = torch.cat([c[0] for c in chunks], dim=0)
    tensors = []
    for tensor_idx in range(len(chunks[0][1])):
        tensors.append(torch.cat([c[1][tensor_idx] for c in chunks], dim=0))
    total_profile['tensor_stack_time'] += time.time() - stack_start
    total_profile['total_prepare_time'] = time.time() - total_start
    _append_dataset_prepare_profile(profile_csv, total_profile)
    if logger is not None:
        logger.info(
            f"Pair feature bank built: n={len(valid_indices):,}, "
            f"total={total_profile['total_prepare_time']:.2f}s"
        )
    return {'pair_ids': pair_ids_all, 'tensors': tuple(tensors)}


def load_or_build_pair_feature_bank(valid_indices, history, target_matrix, seq_len,
                                    args, qos_type, ds_kwargs, logger,
                                    profile_csv=None):
    use_bank = bool(args.reuse_pair_feature_cache or args.rebuild_pair_feature_cache)
    if not use_bank:
        return None
    key_payload, filename = _pair_feature_cache_key(args, qos_type, ds_kwargs, history)
    cache_dir = os.path.abspath(os.path.expanduser(args.pair_feature_cache_dir))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)
    if args.reuse_pair_feature_cache and not args.rebuild_pair_feature_cache and os.path.exists(cache_path):
        try:
            payload = torch.load(cache_path, map_location='cpu')
            if isinstance(payload, dict) and payload.get('key') == key_payload:
                logger.info(f"Pair feature cache hit: {cache_path}")
                _append_dataset_prepare_profile(
                    profile_csv,
                    _empty_prepare_profile('pair_feature_bank', len(valid_indices), 'hit_pair_feature_bank')
                )
                return payload['bank']
            logger.warning(f"Pair feature cache key mismatch, rebuilding: {cache_path}")
        except Exception as exc:
            logger.warning(f"Pair feature cache load failed, rebuilding: {exc}")

    bank = _build_pair_feature_bank_tensors(
        valid_indices, history, target_matrix, seq_len, ds_kwargs,
        chunk_size=args.pair_feature_cache_chunk_size,
        logger=logger,
        profile_csv=profile_csv,
    )
    try:
        torch.save({'key': key_payload, 'bank': bank}, cache_path)
        logger.info(f"Pair feature cache saved: {cache_path}")
    except Exception as exc:
        logger.warning(f"Pair feature cache save failed: {exc}")
    return bank


def prepare_dataset_v42_from_feature_bank(split_name, indices, bank, num_services,
                                          profile_csv=None, logger=None):
    total_start = time.time()
    pair_ids = _pair_ids_from_indices(indices, num_services)
    bank_pair_ids = bank['pair_ids'].numpy()
    max_bank_id = int(bank_pair_ids.max()) if len(bank_pair_ids) else 0
    max_pair_id = int(pair_ids.max()) if len(pair_ids) else 0
    max_id = max(max_bank_id, max_pair_id)
    pos_map = np.full(max_id + 1, -1, dtype=np.int64)
    pos_map[bank_pair_ids] = np.arange(len(bank_pair_ids), dtype=np.int64)
    positions = pos_map[pair_ids]
    if (positions < 0).any():
        missing = pair_ids[positions < 0][:5].tolist()
        raise KeyError(f"Pair feature bank missing pair ids for {split_name}: {missing}")
    positions_t = torch.LongTensor(positions)
    stack_start = time.time()
    tensors = tuple(t.index_select(0, positions_t) for t in bank['tensors'])
    profile = _empty_prepare_profile(split_name, len(indices), 'hit_pair_feature_bank')
    profile['tensor_stack_time'] = time.time() - stack_start
    profile['total_prepare_time'] = time.time() - total_start
    _append_dataset_prepare_profile(profile_csv, profile)
    if logger is not None:
        logger.info(
            f"Dataset gathered from pair bank [{split_name}]: n={len(indices):,}, "
            f"total={profile['total_prepare_time']:.2f}s"
        )
    return TensorDataset(*tensors)


def validate_pair_feature_bank_equivalence(sample_indices, bank, history, target_matrix,
                                           seq_len, ds_kwargs, num_services, logger):
    if len(sample_indices) == 0:
        return True
    old_ds = prepare_dataset_v42(
        sample_indices, history, target_matrix, seq_len,
        split_name='pair_bank_validation_old',
        profile_csv=None,
        cache_hit='validation_old',
        logger=None,
        **ds_kwargs
    )
    new_ds = prepare_dataset_v42_from_feature_bank(
        'pair_bank_validation_new', sample_indices, bank, num_services,
        profile_csv=None, logger=None
    )
    if len(old_ds.tensors) != len(new_ds.tensors):
        logger.warning(f"Pair bank validation failed: tensor count {len(old_ds.tensors)} != {len(new_ds.tensors)}")
        return False
    for i, (old_t, new_t) in enumerate(zip(old_ds.tensors, new_ds.tensors)):
        if old_t.dtype in (torch.long, torch.int64, torch.int32):
            ok = torch.equal(old_t, new_t)
            max_diff = 0
        else:
            ok = torch.allclose(old_t, new_t, rtol=1e-5, atol=1e-6)
            max_diff = float((old_t - new_t).abs().max().item()) if old_t.numel() else 0.0
        if not ok:
            logger.warning(f"Pair bank validation failed at tensor {i}, max_diff={max_diff}")
            return False
    logger.info(f"Pair bank validation passed on {len(sample_indices)} sampled pairs.")
    return True


def compute_nmd_factors(history: np.ndarray, embed_dim: int, logger):
    """
    瀵瑰巻鍙睶oS鐭╅樀鍋歂MF鍒嗚В锛屾彁鍙栫敤鎴峰拰鏈嶅姟鍥犲瓙
    
    浣跨敤鍘嗗彶鍧囧€肩煩闃碉紙蹇界暐0鍊硷級锛孨MF瑕佹眰闈炶礋杈撳叆銆?
    
    Args:
        history: [n_users, n_services, tau] 鍘嗗彶QoS寮犻噺
        embed_dim: 宓屽叆缁村害
    Returns:
        user_factors: [n_users, embed_dim]
        service_factors: [n_services, embed_dim]
    """
    # 浣跨敤鍘嗗彶鍧囧€肩煩闃碉紙蹇界暐0鍊硷級
    qos_sum = history.sum(axis=2)
    qos_count = (history > 0).sum(axis=2).clip(min=1)
    qos_mean = qos_sum / qos_count
    qos_mean = np.nan_to_num(qos_mean, nan=0.0)
    
    # NMF瑕佹眰闈炶礋杈撳叆
    qos_mean = qos_mean.clip(min=0)
    
    logger.info(f"NMD: Computing NMF decomposition, matrix shape={qos_mean.shape}, rank={embed_dim}")
    logger.info(f"NMD: Matrix stats - min={qos_mean.min():.4f}, max={qos_mean.max():.4f}, "
                f"mean={qos_mean[qos_mean>0].mean():.4f}, sparsity={1-np.count_nonzero(qos_mean)/qos_mean.size:.2%}")
    
    nmf = NMF(n_components=embed_dim, init='nndsvda', max_iter=500, random_state=42)
    W = nmf.fit_transform(qos_mean)  # [n_users, embed_dim]
    H = nmf.components_               # [embed_dim, n_services]
    
    reconstruction_error = nmf.reconstruction_err_
    logger.info(f"NMD: Reconstruction error={reconstruction_error:.4f}, iterations={nmf.n_iter_}")
    logger.info(f"NMD: W shape={W.shape}, H shape={H.T.shape}")
    
    return W, H.T  # [n_users, embed_dim], [n_services, embed_dim]


def build_knn_graph(feature_matrix: np.ndarray, k: int, logger) -> torch.Tensor:
    """
    鏋勫缓KNN鐩镐技搴﹀浘鐨別dge_index (COO鏍煎紡)銆?
    
    Args:
        feature_matrix: [num_nodes, num_features] 鑺傜偣鐗瑰緛鐭╅樀
        k: 姣忎釜鑺傜偣鐨勯偦灞呮暟
        logger: 鏃ュ織鍣?
    Returns:
        edge_index: [2, num_edges] COO鏍煎紡杈圭储寮?(瀵圭О鍥?
    """
    num_nodes = feature_matrix.shape[0]
    
    # 澶勭悊鍏ㄩ浂琛岋細鐢ㄦ瀬灏忓€兼浛鎹㈤伩鍏峜osine_similarity杩斿洖NaN
    row_norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    zero_mask = (row_norms.squeeze() < 1e-10)
    if zero_mask.any():
        feature_matrix = feature_matrix.copy()
        feature_matrix[zero_mask] = 1e-10
        logger.info(f"  {zero_mask.sum()} zero-norm nodes detected, filled with epsilon")
    
    # 璁＄畻浣欏鸡鐩镐技搴?
    sim_matrix = sk_cosine_similarity(feature_matrix)
    np.fill_diagonal(sim_matrix, -1)  # 鎺掗櫎鑷?幆
    
    # 瀵规瘡涓?妭鐐瑰彇Top-K閭诲眳
    src_list, dst_list = [], []
    for i in range(num_nodes):
        topk_idx = np.argsort(sim_matrix[i])[-k:]
        for j in topk_idx:
            if sim_matrix[i, j] > 0:  # 鍙?繚鐣欐?鐩镐技搴︾殑杈?
                src_list.append(i)
                dst_list.append(j)
                # 瀵圭О杈?
                src_list.append(j)
                dst_list.append(i)
    
    # 鍘婚噸
    edge_set = set()
    unique_src, unique_dst = [], []
    for s, d in zip(src_list, dst_list):
        if (s, d) not in edge_set:
            edge_set.add((s, d))
            unique_src.append(s)
            unique_dst.append(d)
    
    edge_index = torch.tensor([unique_src, unique_dst], dtype=torch.long)
    logger.info(f"  KNN graph: {num_nodes} nodes, {edge_index.shape[1]} edges, K={k}")
    return edge_index


def run_experiment(gamma: float, qos_type: str, args, device, logger, ablation_label: str):
    run_start_time = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"V42 [{ablation_label}]: {qos_type.upper()}, gamma={gamma}")
    logger.info(f"MS-TGEM: {args.use_ms_tgem}, BAL: {args.use_bal}, CAPH: {args.use_caph}, AGH: {args.use_agh}")
    logger.info(f"{'='*60}")
    seq_len = int(args.history_length)
    
    # 鍔犺浇鏁版嵁
    qos_data = load_wsdream2_tensor(data_dir=args.data_dir, qos_type=qos_type)
    num_users, num_services, num_times = qos_data.shape
    logger.info(f"Data: {num_users} users, {num_services} services, {num_times} times")
    if args.target_slice > num_times:
        raise ValueError(f"target_slice={args.target_slice} exceeds dataset length {num_times}")

    # Split before feature construction. Every historical input is restricted
    # to [target_slice-history_length, target_slice-1] in one-based indexing.
    split_data = prepare_rolling_split(
        qos_data, target_slice=args.target_slice, density=args.density,
        val_ratio=args.val_ratio, seed=args.split_id, history_length=seq_len,
    )
    train_idx, val_idx, test_idx = split_data['train_indices'], split_data['val_indices'], split_data['test_indices']
    history, target_matrix = split_data['history'], split_data['target_matrix']
    assert target_matrix.shape == qos_data[:, :, args.tau].shape
    args.rolling_history_sha256 = history_sha256(history)
    args.rolling_target_sha256 = hashlib.sha256(target_matrix.tobytes(order='C')).hexdigest()
    logger.info(
        f"Rolling target slice {args.target_slice}: historical slices "
        f"[{split_data['history_start_slice']}, {split_data['history_end_slice']}], "
        f"history_sha256={args.rolling_history_sha256}"
    )
    split_mask_path = os.path.join(
        args.log_dir,
        f"split_masks_{qos_type}_t{args.target_slice}_L{seq_len}_"
        f"d{int(round(args.density * 100))}_seed{args.seed}_split{args.split_id}.npz",
    )
    split_metadata = save_split_indices(
        split_mask_path, split_data, qos_type, args.density, args.seed, args.split_id
    )
    logger.info(f"Saved rolling split masks: {split_mask_path}")

    # Legacy statistical_features_{qos}.pkl has no target boundary metadata and
    # may contain observations after an early rolling target. Never load it here.
    user_stat, service_stat = (None, None)
    stat_metadata = None
    if args.use_stat_features:
        user_stat, service_stat, stat_metadata = load_or_compute_rolling_statistics(
            history, qos_type, args.target_slice, args.stat_cache_dir
        )
        logger.info(
            f"Rolling statistical features: {stat_metadata['cache_path']}, "
            f"cache_hit={stat_metadata['cache_hit']}"
        )
    train_targets_for_outlier = np.array([target_matrix[u, s] for u, s in train_idx], dtype=np.float32)
    logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    logger.info(
        "Split fingerprint: "
        f"train={split_fingerprint(train_idx)}, "
        f"val={split_fingerprint(val_idx)}, "
        f"test={split_fingerprint(test_idx)}"
    )
    history_observed_pairs = int(np.count_nonzero((history > 0).any(axis=2)))
    anchor_nonzero_ratio = float(((history > 0).any(axis=2)).mean())
    logger.info(f"Historical FIG observed pairs for L={seq_len}: {history_observed_pairs}")
    logger.info(f"Anchor nonzero ratio from cropped history L={seq_len}: {anchor_nonzero_ratio:.6f}")
    logger.info(f"Outlier evaluation protocol: {args.outlier_protocol}")
    
    # V52: 璁＄畻MFVP鐗瑰緛缁熻?閲?
    mfvp_stats = None
    mfvp_input_dim = 0
    ptcs_data = None
    use_mfvp = args.use_mfvp
    if use_mfvp:
        mfvp_stats = compute_mfvp_features(history, seq_len=seq_len, window=args.mfvp_window)
        mfvp_input_dim = mfvp_stats['mfvp_dim']  # 7
        logger.info(f"MFVP: enabled, dim={mfvp_input_dim}, window={args.mfvp_window}, "
                    f"global_mean={mfvp_stats['global_mean']:.4f}, global_std={mfvp_stats['global_std']:.4f}")
        
        # V55: PTCS鍗忓悓淇″彿锛堟墿灞昅FVP浠?缁村埌9缁达級
        if args.use_ptcs:
            ptcs_cache_path = ""
            if args.ptcs_cache_dir:
                ptcs_cache_path = os.path.join(
                    args.ptcs_cache_dir,
                    f"ptcs_{qos_type}_tau{args.tau}_L{seq_len}_"
                    f"h{args.rolling_history_sha256[:16]}_topk{args.ptcs_top_k}.pkl"
                )
            ptcs_data = load_or_compute_ptcs_neighbors(
                history, top_k=args.ptcs_top_k, logger=logger,
                cache_path=ptcs_cache_path, cache_label='PTCS'
            )
            mfvp_input_dim = 9  # 7 base + 2 PTCS
            logger.info(f"PTCS: enabled, top_k={args.ptcs_top_k}, mfvp_input_dim upgraded to {mfvp_input_dim}")
    
    # V50/V62: 鍔犺浇GPAM鍥犲瓙
    gpam_user_factors, gpam_service_factors = None, None
    gpam_input_dim = 0
    gpam_fuse_mfvp = getattr(args, 'gpam_fuse_mfvp', False) and use_mfvp  # V62: 铻嶅悎妯″紡闇€瑕丮FVP寮€鍚?
    # V62: 铻嶅悎妯″紡涓嬪厑璁窯PAM+MFVP鍏卞瓨锛涢潪铻嶅悎妯″紡淇濇寔浜掓枼
    use_gpam = args.use_gpam_input and args.gpam_cache_dir and (gpam_fuse_mfvp or not use_mfvp)
    if use_gpam:
        # V62: 鏀?寔fill_missing鍜宯ormalize鍚庣紑
        gpam_fill = getattr(args, 'gpam_fill_missing', False)
        gpam_suffix = f'_filled_norm' if gpam_fill else ''
        gpam_cache = os.path.join(args.gpam_cache_dir, f'gpam_factors_{qos_type}_f{args.gpam_f1}{gpam_suffix}.pkl')
        if os.path.exists(gpam_cache):
            with open(gpam_cache, 'rb') as f:
                gpam_data = pickle.load(f)
            # precompute_gpam_factors.py 淇濆瓨鏍煎紡: [n_users, f1, tau] 鈫?杞?疆涓?[n_users, tau, f1]
            gpam_user_factors = gpam_data['user_factors'].transpose(0, 2, 1)    # [n_users, tau, f1]
            gpam_service_factors = gpam_data['service_factors'].transpose(0, 2, 1)  # [n_services, tau, f1]
            f1 = gpam_user_factors.shape[2]
            gpam_input_dim = 2 * f1  # user_f1 + service_f1
            logger.info(f"GPAM: loaded {gpam_cache}, user={gpam_user_factors.shape}, "
                       f"service={gpam_service_factors.shape}, gpam_input_dim={gpam_input_dim}")
            if gpam_fuse_mfvp:
                # V62: 铻嶅悎妯″紡 - GPAM鍥犲瓙杩藉姞鍒癕FVP鐗瑰緛锛屾洿鏂癿fvp_input_dim
                mfvp_input_dim += gpam_input_dim
                logger.info(f"V62 GPAM+MFVP fusion: mfvp_input_dim upgraded to {mfvp_input_dim} "
                           f"(base {mfvp_input_dim - gpam_input_dim} + gpam {gpam_input_dim})")
        else:
            logger.warning(f"GPAM cache not found: {gpam_cache}, disabling GPAM input")
            use_gpam = False
            gpam_fuse_mfvp = False
    
    # V56: HAP/RAPP寮€鍏? V65: QAPH寮€鍏? V66: AACAPH寮€鍏?
    use_hap = args.use_hap
    use_rapp = args.use_rapp
    use_qaph = args.use_qaph
    use_aacaph = args.use_aacaph
    # QAPH/HAP闇€瑕乤nchor鏍囬噺; AACAPH闇€瑕乤nchor_info (3缁?
    use_anchor = use_hap or use_qaph
    # AACAPH涔熼渶瑕乤nchor_info棰勮?绠楋紙鐙?珛浜巙se_anchor锛?
    
    # V60: NACC閭诲眳鑱氬悎鍗忓悓涓婁笅鏂?
    nacc_data = None
    use_nacc = args.use_nacc
    nacc_dim = 0
    if use_nacc:
        # 澶嶇敤PTCS鐨凬MF閭诲眳璁＄畻锛堝?鏋滃凡鏈夛級锛屽惁鍒欑嫭绔嬭?绠?
        if ptcs_data is not None:
            nacc_data = ptcs_data
            logger.info(f"NACC: reusing PTCS neighbors, top_k={args.ptcs_top_k}")
        else:
            nacc_cache_path = ""
            if args.ptcs_cache_dir:
                nacc_cache_path = os.path.join(
                    args.ptcs_cache_dir,
                    f"nacc_{qos_type}_tau{args.tau}_L{seq_len}_"
                    f"h{args.rolling_history_sha256[:16]}_topk{args.nacc_top_k}.pkl"
                )
            nacc_data = load_or_compute_ptcs_neighbors(
                history, top_k=args.nacc_top_k, logger=logger,
                cache_path=nacc_cache_path, cache_label='NACC'
            )
            logger.info(f"NACC: computed neighbors, top_k={args.nacc_top_k}")
        nacc_dim = 6
        logger.info(f"NACC: enabled, dim={nacc_dim}")
    
    # V70: Anchor妯″紡
    anchor_mode = getattr(args, 'anchor_mode', 'mean')
    anchor_ema_alpha = getattr(args, 'anchor_ema_alpha', 0.1)
    anchor_trim_ratio = getattr(args, 'anchor_trim_ratio', 0.1)
    anchor_neighbor_weight = getattr(args, 'anchor_neighbor_weight', 0.3)
    if use_anchor and anchor_mode != 'mean':
        logger.info(f"V70 Anchor mode: {anchor_mode}" +
                    (f", ema_alpha={anchor_ema_alpha}" if anchor_mode == 'ema' else "") +
                    (f", trim_ratio={anchor_trim_ratio}" if anchor_mode == 'trimmed' else "") +
                    (f", neighbor_weight={anchor_neighbor_weight}" if anchor_mode == 'neighbor' else ""))
        if anchor_mode == 'neighbor' and ptcs_data is None:
            logger.warning("V70: anchor_mode='neighbor' requires PTCS data, falling back to 'mean'")
            anchor_mode = 'mean'
    
    # 鍑嗗?鏁版嵁闆嗭紙use_anchor: HAP鎴朡APH閮介渶瑕乤nchor棰勮?绠? use_aacaph: 闇€瑕乤nchor_info 3缁达級
    ds_kwargs = dict(
        gpam_user_factors=gpam_user_factors, gpam_service_factors=gpam_service_factors,
        mfvp_stats=mfvp_stats, mfvp_window=args.mfvp_window,
        ptcs_data=ptcs_data, use_hap=use_anchor,
        nacc_data=nacc_data, gpam_fuse_mfvp=gpam_fuse_mfvp,
        use_aacaph=use_aacaph,
        anchor_mode=anchor_mode, anchor_ema_alpha=anchor_ema_alpha,
        anchor_trim_ratio=anchor_trim_ratio, anchor_neighbor_weight=anchor_neighbor_weight,
    )
    d_label = f"d{int(args.density*100)}"
    tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
    dataset_prepare_profile_csv = os.path.join(
        args.log_dir, f'dataset_prepare_profile_{qos_type}_{d_label}_gamma{gamma}_v42{tag}.csv'
    )
    if os.path.exists(dataset_prepare_profile_csv):
        os.remove(dataset_prepare_profile_csv)
    logger.info(f"Dataset prepare profile CSV: {dataset_prepare_profile_csv}")

    test_eval_policy_early = getattr(args, 'test_eval_policy', 'every_epoch')
    defer_test_dataset = (
        bool(args.defer_test_dataset) and
        args.defer_test_dataset_policy == 'auto' and
        test_eval_policy_early == 'final_only' and
        not args.export_gate_csv_only
    )

    valid_indices_all = np.argwhere(target_matrix > 0)
    pair_feature_bank = load_or_build_pair_feature_bank(
        valid_indices_all, history, target_matrix, seq_len,
        args, qos_type, ds_kwargs, logger,
        profile_csv=dataset_prepare_profile_csv
    )
    if pair_feature_bank is not None and args.validate_pair_feature_cache:
        rng = np.random.default_rng(args.seed)
        sample_n = min(512, len(valid_indices_all))
        sample_pos = rng.choice(len(valid_indices_all), size=sample_n, replace=False)
        sample_indices = valid_indices_all[sample_pos]
        if not validate_pair_feature_bank_equivalence(
            sample_indices, pair_feature_bank, history, target_matrix,
            seq_len, ds_kwargs, num_services, logger
        ):
            logger.warning("Pair feature bank validation failed; falling back to split-level dataset preparation.")
            pair_feature_bank = None

    def build_split_dataset(split_name, split_indices):
        if pair_feature_bank is not None:
            return prepare_dataset_v42_from_feature_bank(
                split_name, split_indices, pair_feature_bank, num_services,
                profile_csv=dataset_prepare_profile_csv,
                logger=logger
            )
        return prepare_dataset_v42_cached(
            split_name, split_indices, history, target_matrix, seq_len,
            args, qos_type, args.split_id, ds_kwargs, logger,
            profile_csv=dataset_prepare_profile_csv
        )

    train_ds = build_split_dataset('train', train_idx)
    val_ds = build_split_dataset('val', val_idx)
    test_ds = None
    if defer_test_dataset:
        logger.info("Test dataset construction deferred until final evaluation.")
    else:
        test_ds = build_split_dataset('test', test_idx)

    train_loader = make_qaph_dataloader(
        train_ds, args.batch_size, True, args.num_workers, device,
        prefetch_factor=args.prefetch_factor
    )
    val_loader = make_qaph_dataloader(
        val_ds, args.eval_batch_size, False, args.num_workers, device,
        prefetch_factor=args.prefetch_factor
    )
    test_loader = None
    if test_ds is not None:
        test_loader = make_qaph_dataloader(
            test_ds, args.eval_batch_size, False, args.num_workers, device,
            prefetch_factor=args.prefetch_factor
        )
    
    # 鍒涘缓妯″瀷锛圴50: GPAM, V52: MFVP, V62: GPAM+MFVP铻嶅悎锛?
    # V62铻嶅悎妯″紡: GPAM鍥犲瓙宸茶拷鍔犲埌MFVP锛実pam_input_dim瀵规ā鍨嬮€忔槑
    model_gpam_dim = gpam_input_dim if (use_gpam and not gpam_fuse_mfvp) else 0
    model_mfvp_dim = mfvp_input_dim if use_mfvp else 0
    model = UnifiedQoSModelV42(
        num_users=num_users,
        num_services=num_services,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.tgem_e_layers,  # V73-B2: TGEM E-block涓撶敤灞傛暟锛堥粯璁?锛孊2瀹為獙鐢?锛?
        dropout=args.dropout,
        qos_type=qos_type,
        use_gmm=args.use_gmm,
        use_tgem=args.use_tgem,
        num_neighbors=10,
        gmm_num_heads=4,
        gmm_dropout=0.3,
        user_threshold=1.5,
        service_threshold=1.5,
        bilinear_rank=args.bilinear_rank,
        use_ms_tgem=args.use_ms_tgem,
        use_caph=args.use_caph,
        use_lgcfe=args.use_lgcfe,
        lgcfe_init_alpha=args.lgcfe_init_alpha,
        e_block_ratio=args.e_block_ratio,
        tgem_num_branches=args.tgem_num_branches,
        use_missing_token=args.use_missing_token,
        use_fuse_first=args.use_fuse_first,
        gpam_input_dim=model_gpam_dim,
        mfvp_input_dim=model_mfvp_dim,
        use_hap=use_hap,
        use_rapp=use_rapp,
        rapp_num_experts=args.rapp_num_experts,
        hap_log_scale_min=args.hap_log_scale_min,
        hap_log_scale_max=args.hap_log_scale_max,
        use_cf_feature=args.use_cf_feature,
        cf_factor_dim=args.cf_factor_dim,
        nacc_dim=nacc_dim,
        use_serial_gmm=args.use_serial_gmm,
        use_lightgcn=args.use_lightgcn,
        lightgcn_layers=args.lightgcn_layers,
        lightgcn_dropout=args.lightgcn_dropout,
        use_sega=args.use_sega,
        sega_gcn_dropout=args.sega_gcn_dropout,
        sega_init_gate=args.sega_init_gate,
        pgcn_num_layers=args.pgcn_num_layers,
        use_multi_graph=args.use_multi_graph,
        multi_graph_use_suig=args.multi_graph_use_suig,
        multi_graph_use_ssig=args.multi_graph_use_ssig,
        multi_graph_second_gate_init=args.multi_graph_second_gate_init,
        use_indep_gcn=args.use_indep_gcn,
        indep_gcn_dropout=args.indep_gcn_dropout,
        indep_gcn_init_gate=args.indep_gcn_init_gate,
        use_qaph=use_qaph,
        use_aacaph=use_aacaph,
        qaph_gate_init_bias=args.qaph_gate_init_bias,
        qaph_anchor_usage=args.qaph_anchor_usage,
        qaph_gate_mode=args.qaph_gate_mode,
        use_film=args.use_film,
        use_tpa=args.use_tpa,
        tpa_num_heads=args.tpa_num_heads,
        tpa_max_neighbors=args.tpa_max_neighbors,
        tpa_init_gate=args.tpa_init_gate,
        tpa_dropout=args.tpa_dropout,
        disable_tblock=args.disable_tblock,
        disable_fblock=args.disable_fblock,
    ).to(device)
    
    if use_aacaph:
        logger.info(f"AACAPH: enabled (anchor-augmented unified prediction head)")
    if use_qaph:
        logger.info(f"QAPH: enabled (unified prediction head), anchor precomputed")
        logger.info(f"QAPH anchor usage mode: {args.qaph_anchor_usage}")
        logger.info(f"QAPH gate mode: {args.qaph_gate_mode}")
    if use_hap:
        logger.info(f"HAP: enabled, log_scale_range=[{args.hap_log_scale_min}, {args.hap_log_scale_max}]")
    if use_rapp:
        logger.info(f"RAPP: enabled, num_experts={args.rapp_num_experts}, balance_weight={args.rapp_balance_weight}")
    if use_nacc:
        logger.info(f"NACC: enabled, dim={nacc_dim}, top_k={args.nacc_top_k}")
    if args.use_serial_gmm:
        logger.info(f"Serial GMM鈫扵GEM: enabled")
    
    # 璁剧疆GMM鏁版嵁锛圴55 Fix: 闈為浂鍧囧€硷紝鎺掗櫎缂哄け鍊?锛?
    qos_sum = history.sum(axis=2)
    qos_count = (history > 0).sum(axis=2).astype(np.float32)
    qos_count[qos_count == 0] = 1.0  # 閬垮厤闄ら浂
    qos_matrix = qos_sum / qos_count
    user_qos_matrix = torch.tensor(qos_matrix, dtype=torch.float32, device=device)
    service_qos_matrix = torch.tensor(qos_matrix.T, dtype=torch.float32, device=device)
    user_stat_tensor = torch.tensor(user_stat, dtype=torch.float32, device=device) if user_stat is not None else torch.zeros(num_users, 14, device=device)
    service_stat_tensor = torch.tensor(service_stat, dtype=torch.float32, device=device) if service_stat is not None else torch.zeros(num_services, 14, device=device)
    
    user_mean = user_qos_matrix[user_qos_matrix > 0].mean().item()
    user_std = user_qos_matrix[user_qos_matrix > 0].std().item()
    service_mean = service_qos_matrix[service_qos_matrix > 0].mean().item()
    service_std = service_qos_matrix[service_qos_matrix > 0].std().item()
    
    model.set_qos_data(
        user_qos_matrix, service_qos_matrix,
        user_stat_tensor, service_stat_tensor,
        user_mean, user_std, service_mean, service_std
    )
    
    # V63: 鏋勫缓SEGA ParametricGCN浜岄儴鍥撅紙蹇呴』鍦╭os_matrix璁＄畻涔嬪悗锛?
    graph_build_time_s = 0.0
    if args.use_sega:
        logger.info(f"V63 SEGA: Building bipartite graph for ParametricGCN...")
        fig_weight_mode = getattr(args, 'fig_weight_mode', 'binary')
        fig_min_weight = getattr(args, 'fig_min_weight', 0.1)
        if fig_weight_mode != 'binary':
            logger.info(f"V73 FIG weighting: mode={fig_weight_mode}, min_weight={fig_min_weight}")
        graph_start_time = time.time()
        sega_adj = build_lightgcn_adj(
            qos_matrix,  # [num_users, num_services] 闈為浂鍧囧€肩煩闃?
            num_users, num_services,
            device=device,
            weight_mode=fig_weight_mode,
            min_weight=fig_min_weight,
        )
        graph_build_time_s += time.time() - graph_start_time
        model.set_sega_graph(sega_adj)
        logger.info(f"V63 SEGA: gcn_dropout={args.sega_gcn_dropout}, init_gate={args.sega_init_gate}")
        
        # V76 TPA: 鎻愬彇鐢ㄦ埛閭诲眳绱㈠紩骞惰?缃?埌妯″瀷
        if args.use_tpa:
            from qaph.heads import extract_user_neighbors
            nb_indices, nb_counts = extract_user_neighbors(
                qos_matrix, max_neighbors=args.tpa_max_neighbors
            )
            model.set_user_neighbors(nb_indices, nb_counts)
            avg_nb = nb_counts.float().mean().item()
            max_nb = nb_counts.max().item()
            logger.info(f"V76 TPA: neighbors extracted, avg={avg_nb:.1f}, max={max_nb}, "
                        f"max_neighbors={args.tpa_max_neighbors}, heads={args.tpa_num_heads}")
        
        # V71: 鏋勫缓浜岄樁鍥撅紙SUIG + SSIG锛?
        if args.use_multi_graph:
            logger.info(f"V71 MultiGraph: Building second-order graphs (SUIG={args.multi_graph_use_suig}, SSIG={args.multi_graph_use_ssig})...")
            # 浠巕os_matrix鏋勫缓浜や簰鎺╃爜: [num_users, num_services]
            interaction_mask = torch.from_numpy((qos_matrix > 0).astype(np.float32))
            graph_start_time = time.time()
            suig_adj, ssig_adj = build_second_order_graphs(
                interaction_mask, num_users, num_services,
                device=torch.device('cpu'),  # CPU涓婃瀯寤猴紝鍚庣画GCN forward鏃惰嚜鍔ㄨ縼绉?
                suig_top_k=args.suig_top_k if args.suig_top_k > 0 else None,
                ssig_top_k=args.ssig_top_k,
                build_suig_flag=args.multi_graph_use_suig,
                build_ssig_flag=args.multi_graph_use_ssig,
            )
            graph_build_time_s += time.time() - graph_start_time
            model.set_sega_graph(sega_adj, suig_adj=suig_adj, ssig_adj=ssig_adj)
            logger.info(f"V71 MultiGraph: Second-order graphs built and set")
    
    # V64: 鏋勫缓Independent-GCN浜岄儴鍥撅紙鐙?珛宓屽叆鍚勮嚜GCN澧炲己锛?
    if args.use_indep_gcn:
        logger.info(f"V64 Independent-GCN: Building bipartite graph for 3 independent ParametricGCNs...")
        graph_start_time = time.time()
        indep_gcn_adj = build_lightgcn_adj(
            qos_matrix,  # [num_users, num_services] 闈為浂鍧囧€肩煩闃?
            num_users, num_services,
            device=device
        )
        graph_build_time_s += time.time() - graph_start_time
        model.set_indep_gcn_graph(indep_gcn_adj)
        logger.info(f"V64 Independent-GCN: dropout={args.indep_gcn_dropout}, init_gate={args.indep_gcn_init_gate}")
    
    # V61: 鏋勫缓LightGCN浜岄儴鍥撅紙蹇呴』鍦╭os_matrix璁＄畻涔嬪悗锛?
    if args.use_lightgcn:
        logger.info(f"V61 LightGCN: Building bipartite graph from training data...")
        graph_start_time = time.time()
        lightgcn_adj = build_lightgcn_adj(
            qos_matrix,  # [num_users, num_services] 闈為浂鍧囧€肩煩闃?
            num_users, num_services,
            device=device
        )
        graph_build_time_s += time.time() - graph_start_time
        model.set_lightgcn_graph(lightgcn_adj)
        logger.info(f"V61 LightGCN: layers={args.lightgcn_layers}, dropout={args.lightgcn_dropout}")
    
    # LGCFE: 鏋勫缓KNN鍥?
    if args.use_lgcfe:
        logger.info(f"Building LGCFE KNN graphs (K={args.lgcfe_k})...")
        graph_start_time = time.time()
        user_feature = qos_matrix.copy()  # [num_users, num_services]
        user_edge_index = build_knn_graph(user_feature, args.lgcfe_k, logger).to(device)
        service_feature = qos_matrix.T.copy()  # [num_services, num_users]
        service_edge_index = build_knn_graph(service_feature, args.lgcfe_k, logger).to(device)
        graph_build_time_s += time.time() - graph_start_time
        model.set_lgcfe_graph(user_edge_index, service_edge_index)
        logger.info(f"LGCFE graphs set: user_edges={user_edge_index.shape[1]}, service_edges={service_edge_index.shape[1]}")
    
    # V49: NMD鍒濆?宓屽叆
    if args.use_nmd_init:
        user_factors, service_factors = compute_nmd_factors(history, args.embed_dim, logger)
        model.init_embeddings_from_nmd(user_factors, service_factors)
        logger.info(f"NMD init applied: user_factors={user_factors.shape}, service_factors={service_factors.shape}")
    
    # V54: CF宓屽叆鍒濆?鍖朑MM
    if args.use_cf_embed:
        d_label = f"d{int(args.density*100)}"
        cf_path = os.path.join(args.cf_embed_dir, f'cf_embeddings_{qos_type}_{d_label}.pt')
        if os.path.exists(cf_path):
            cf_data = torch.load(cf_path, map_location='cpu')
            user_cf_raw = cf_data['user_cf_embed']
            service_cf_raw = cf_data['service_cf_embed']
            # 鍏煎?numpy鍜宼ensor涓ょ?鏍煎紡
            user_cf = user_cf_raw.numpy() if hasattr(user_cf_raw, 'numpy') else np.array(user_cf_raw)
            service_cf = service_cf_raw.numpy() if hasattr(service_cf_raw, 'numpy') else np.array(service_cf_raw)
            model.init_gmm_from_cf(user_cf, service_cf)
            logger.info(f"CF embed init applied: user={user_cf.shape}, service={service_cf.shape}, from {cf_path}")
        else:
            logger.warning(f"CF embed cache not found: {cf_path}, skipping CF init")
    
    # V58: CF-Enhanced Embedding锛堟寔涔呭寲NMF鍥犲瓙 + 闂ㄦ帶铻嶅悎锛?
    if args.use_cf_feature:
        cf_user_factors, cf_service_factors = compute_nmd_factors(history, args.cf_factor_dim, logger)
        model.set_cf_factors(cf_user_factors, cf_service_factors)
        logger.info(f"V58 CF feature: factors set, dim={args.cf_factor_dim}, "
                    f"user={cf_user_factors.shape}, service={cf_service_factors.shape}")
    
    module_param_count = count_top_level_module_params(model)
    param_count = module_param_count['total']
    logger.info(f"Model params: {param_count:,}")
    logger.info(f"Module params: {json.dumps(module_param_count, sort_keys=True)}")
    current_memory_after_model_mb = get_cuda_current_memory_mb(device)
    if current_memory_after_model_mb is not None:
        logger.info(f"CUDA memory after model/graph setup: {current_memory_after_model_mb:.2f} MB")

    train_eval_lambdas = list(getattr(args, 'train_eval_lambda_values', [0]))
    final_eval_lambdas = list(getattr(args, 'final_eval_lambda_values', [0]))
    test_eval_policy = getattr(args, 'test_eval_policy', 'every_epoch')
    d_label = f"d{int(args.density*100)}"
    tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
    timing_profile_csv = os.path.join(
        args.log_dir, f'timing_profile_{qos_type}_{d_label}_gamma{gamma}_v42{tag}.csv'
    )
    timing_fields = [
        'epoch', 'sega_precompute_time', 'gcn_update_time', 'train_epoch_time',
        'val_eval_time', 'test_eval_time', 'total_epoch_time', 'train_loss',
        'val_mae_lambda0', 'test_mae_lambda0', 'test_eval_policy'
    ]
    os.makedirs(args.log_dir, exist_ok=True)
    with open(timing_profile_csv, 'w', newline='', encoding='utf-8') as f:
        csv.DictWriter(f, fieldnames=timing_fields).writeheader()
    logger.info(f"Timing profile CSV: {timing_profile_csv}")

    if args.export_gate_csv_only:
        if not args.load_checkpoint_path:
            raise ValueError("--export_gate_csv_only requires --load_checkpoint_path")

        checkpoint_path = os.path.abspath(os.path.expanduser(args.load_checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Export-only mode: loading checkpoint {checkpoint_path}")
        _load_checkpoint_state_dict(model, checkpoint_path, device, logger=logger)

        export_test_results = evaluate_multi_lambda(
            model, test_loader, device, final_eval_lambdas,
            use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
            use_nacc=use_nacc, use_qaph=use_qaph, use_aacaph=use_aacaph,
            outlier_protocol=args.outlier_protocol,
            reference_targets=train_targets_for_outlier,
            refresh_caches=True,
            use_amp=args.use_amp,
        )
        flops_profile = None
        if getattr(args, 'profile_flops', False):
            flops_profile = profile_qaph_forward_flops(
                model, test_loader, device,
                use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
                use_nacc=use_nacc, use_qaph=use_qaph, use_aacaph=use_aacaph,
                use_amp=args.use_amp,
                warmup_runs=args.flops_warmup_runs,
                logger=logger,
            )

        gate_analysis_csv = None
        if args.export_qaph_gate_csv and use_qaph:
            d_label = f"d{int(args.density*100)}"
            tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
            gate_csv_path = args.gate_csv_path or os.path.join(
                args.log_dir, f'gate_analysis_{qos_type}_{d_label}_gamma{gamma}_v42{tag}.csv'
            )
            gate_analysis_csv = export_qaph_gate_analysis_csv(
                model=model,
                dataloader=test_loader,
                dataset_tensors=test_ds.tensors,
                sample_indices=test_idx,
                full_history=history,
                qos_type=qos_type,
                density=args.density,
                split_id=args.split_id,
                output_csv=gate_csv_path,
                device=device,
                use_gpam=use_gpam,
                use_mfvp=use_mfvp,
                use_hap=use_hap,
                use_nacc=use_nacc,
                use_qaph=use_qaph,
                use_aacaph=use_aacaph,
                logger=logger,
            )

        result = {
            'qos_type': qos_type,
            'gamma': gamma,
            'density': args.density,
            'history_length': args.history_length,
            'version': 'V56',
            'ablation': ablation_label,
            'outlier_protocol': args.outlier_protocol,
            'seed': args.seed,
            'split_id': args.split_id,
            'checkpoint_path': checkpoint_path,
            'log_path': getattr(logger, 'log_file_path', None),
            'config': {
                'ms_tgem': args.use_ms_tgem,
                'bal': args.use_bal,
                'caph': args.use_caph,
                'agh': args.use_agh,
                'qaph_anchor_usage': args.qaph_anchor_usage,
                'qaph_gate_mode': args.qaph_gate_mode
            },
            'args': vars(args),
            'param_count': param_count,
            'module_param_count': module_param_count,
            'flops_profile': flops_profile,
            'best_val_mae_lambda0': None,
            'test_results': export_test_results
        }
        if gate_analysis_csv is not None:
            result['gate_analysis_csv'] = gate_analysis_csv
        return result
    
    # 鍒涘缓鎹熷け鍑芥暟锛堟牴鎹?秷铻嶅紑鍏筹級
    bal_alpha, bal_scale, agh_gamma = 0.0, gamma, 0.0
    
    # 纭?畾AGH 纬鍊煎拰鏄?惁鍙??涔?
    agh_init_gamma = args.fixed_agh_gamma if args.fixed_agh_gamma > 0 else 0.5
    agh_learnable = (args.fixed_agh_gamma <= 0)  # fixed_agh_gamma>0鏃跺浐瀹氾紝鍚﹀垯鍙??涔?
    
    if args.use_lscl:
        # LSCL: 瀵规暟绌洪棿Cauchy Loss
        base_loss = LogSpaceCauchyLoss(gamma=args.lscl_gamma).to(device)
        logger.info(f"LSCL: 纬={args.lscl_gamma}")
        
        if args.use_agh:
            criterion = AdaptiveGradientHarmonization(
                base_loss=base_loss, init_gamma=agh_init_gamma,
                gamma_lo=0.1, gamma_hi=3.0, learnable=agh_learnable
            ).to(device)
            logger.info(f"AGH: 纬={agh_init_gamma}, learnable={agh_learnable}")
        else:
            criterion = base_loss
    elif args.use_scl:
        # SCL: 鍒嗘?Cauchy Loss
        base_loss = SegmentedCauchyLoss(
            gamma_small=args.scl_gamma_small,
            gamma_large=args.scl_gamma_large,
            threshold=args.scl_threshold
        ).to(device)
        logger.info(f"SCL: 纬_small={args.scl_gamma_small}, 纬_large={args.scl_gamma_large}, threshold={args.scl_threshold}")
        
        if args.use_agh:
            criterion = AdaptiveGradientHarmonization(
                base_loss=base_loss, init_gamma=agh_init_gamma,
                gamma_lo=0.1, gamma_hi=3.0, learnable=agh_learnable
            ).to(device)
            logger.info(f"AGH: 纬={agh_init_gamma}, learnable={agh_learnable}")
        else:
            criterion = base_loss
    elif args.use_bal:
        # BAL: Barron鑷?€傚簲鎹熷け
        bal = BarronAdaptiveLoss(
            init_alpha=0.0, init_scale=1.0,
            alpha_lo=-3.0, alpha_hi=2.0
        ).to(device)
        
        if args.use_agh:
            criterion = AdaptiveGradientHarmonization(
                base_loss=bal, init_gamma=agh_init_gamma,
                gamma_lo=0.1, gamma_hi=3.0, learnable=agh_learnable
            ).to(device)
        else:
            criterion = bal
    else:
        # 鍥哄畾Cauchy Loss锛堟棤BAL锛?
        cauchy = CauchyLoss(gamma=gamma).to(device)
        
        if args.use_agh:
            criterion = AdaptiveGradientHarmonization(
                base_loss=cauchy, init_gamma=agh_init_gamma,
                gamma_lo=0.1, gamma_hi=3.0, learnable=agh_learnable
            ).to(device)
            logger.info(f"Cauchy(纬={gamma}) + AGH(纬={agh_init_gamma}, learnable={agh_learnable})")
        else:
            criterion = cauchy
            logger.info(f"Cauchy(纬={gamma})")
    
    # 浼樺寲鍣?
    model_params = list(model.parameters())
    loss_params = [p for p in criterion.parameters()] if hasattr(criterion, 'parameters') else []
    
    if loss_params:
        optimizer = optim.AdamW([
            {'params': model_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': loss_params, 'lr': args.lr * 0.01, 'weight_decay': 0.0}
        ])
    else:
        optimizer = optim.AdamW(model_params, lr=args.lr, weight_decay=args.weight_decay)
    logger.info(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    
    # 璋冨害鍣?細鏀?寔StepLR銆丆osineAnnealing銆丷educeLROnPlateau銆乄armup+Cosine
    use_plateau = (args.scheduler == 'plateau')
    if args.scheduler == 'step':
        # V45/HCTN椋庢牸锛歋tepLR閫愭?琛板噺
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
        logger.info(f"Scheduler: StepLR (step_size={args.step_size}, gamma={args.step_gamma})")
    elif args.scheduler == 'plateau':
        # V46: ReduceLROnPlateau - val MAE鍋滄粸鏃惰嚜鍔ㄩ檷LR
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.plateau_factor,
            patience=args.plateau_patience, min_lr=1e-6, verbose=False
        )
        logger.info(f"Scheduler: ReduceLROnPlateau (factor={args.plateau_factor}, patience={args.plateau_patience})")
    elif args.no_warmup:
        # V36椋庢牸锛氱函CosineAnnealingLR锛屾棤warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        logger.info("Scheduler: CosineAnnealingLR (no warmup)")
    else:
        # V42椋庢牸锛歐armup + CosineAnnealing
        warmup_epochs = 5
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        logger.info("Scheduler: Warmup(5) + CosineAnnealing")
    
    # V46: SWA (Stochastic Weight Averaging)
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(args.epochs * 0.8)  # 鏈€鍚?0%鐨別pochs鍚?敤SWA
    if args.use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
        logger.info(f"SWA: enabled, start_epoch={swa_start_epoch}, swa_lr={args.swa_lr}")
    
    # V63 SEGA: GCN鍙傛暟鍗曠嫭浼樺寲鍣?
    optimizer_gcn = None
    if args.use_sega and hasattr(model, 'pgcn') and model.pgcn is not None:
        gcn_params = list(model.pgcn.parameters())
        if gcn_params:
            optimizer_gcn = optim.AdamW(gcn_params, lr=args.lr * 0.1, weight_decay=args.weight_decay)
            logger.info(f"V63 SEGA: GCN optimizer created, {sum(p.numel() for p in gcn_params)} params, "
                       f"lr={args.lr * 0.1:.6f}, update_interval={args.sega_gcn_update_interval}")
    
    # V64 Independent-GCN: 3涓狦CN鐨勫弬鏁板崟鐙?紭鍖栧櫒
    optimizer_indep_gcn = None
    if args.use_indep_gcn:
        indep_gcn_params = []
        for gcn_name in ['base_gcn', 'gmm_gcn', 'tgem_gcn']:
            gcn_module = getattr(model, gcn_name, None)
            if gcn_module is not None:
                indep_gcn_params.extend(list(gcn_module.parameters()))
        if indep_gcn_params:
            optimizer_indep_gcn = optim.AdamW(indep_gcn_params, lr=args.lr * 0.1, weight_decay=args.weight_decay)
            logger.info(f"V64 Independent-GCN: optimizer created, {sum(p.numel() for p in indep_gcn_params)} params, "
                       f"lr={args.lr * 0.1:.6f}, update_interval={args.indep_gcn_update_interval}")
    
    # 璁?粌寰?幆
    best_val_mae_l0, best_test_results = float('inf'), None
    best_state_dict = None
    best_checkpoint_path = None
    best_epoch = None
    patience_counter = 0
    amp_enabled = bool(args.use_amp and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    logger.info(
        f"Fast eval config: eval_batch_size={args.eval_batch_size}, "
        f"test_eval_policy={test_eval_policy}, train_eval_lambdas={train_eval_lambdas}, "
        f"final_eval_lambdas={final_eval_lambdas}, amp={amp_enabled}, "
        f"dataset_cache={'on' if (args.reuse_dataset_cache or args.rebuild_dataset_cache) else 'off'}"
    )

    epoch_efficiency_rows = []
    reset_cuda_peak_memory(device)
    train_loop_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        sega_precompute_time = 0.0
        gcn_update_time = 0.0
        train_epoch_time = 0.0
        val_eval_time = 0.0
        test_eval_time = 0.0
        test_results = None
        # V63 SEGA: 姣廵poch寮€濮嬫椂棰勮?绠桮CN澧炲己宓屽叆锛坉etach缂撳瓨锛?
        precompute_start_time = time.time()
        if args.use_sega and hasattr(model, 'precompute_sega_embeddings'):
            model.precompute_sega_embeddings()
        
        # V64 Independent-GCN: 姣廵poch寮€濮嬫椂棰勮?绠?涓狦CN鐨勫?寮哄祵鍏ワ紙detach缂撳瓨锛?
        if args.use_indep_gcn and hasattr(model, 'precompute_indep_gcn_embeddings'):
            model.precompute_indep_gcn_embeddings()
        sega_precompute_time += time.time() - precompute_start_time
        
        # V64 Independent-GCN: 鍛ㄦ湡鎬ф洿鏂?涓狦CN鍙傛暟
        if (args.use_indep_gcn and optimizer_indep_gcn is not None and
            epoch % args.indep_gcn_update_interval == 0):
            gcn_update_start_time = time.time()
            model.train()
            gcn_batch = next(iter(train_loader))
            idx = 4
            gpam_features_gcn, mfvp_features_gcn, anchor_gcn, anchor_info_gcn, nacc_features_gcn = None, None, None, None, None
            user_ids_gcn, service_ids_gcn = gcn_batch[0].to(device), gcn_batch[1].to(device)
            qos_history_gcn, targets_gcn = gcn_batch[2].to(device), gcn_batch[3].to(device)
            
            if (use_mfvp or use_gpam) and len(gcn_batch) > idx:
                extra_gcn = gcn_batch[idx].to(device)
                mfvp_features_gcn = extra_gcn if use_mfvp else None
                gpam_features_gcn = extra_gcn if (use_gpam and not use_mfvp) else None
                idx += 1
            if (use_hap or use_qaph) and len(gcn_batch) > idx:
                anchor_gcn = gcn_batch[idx].to(device)
                idx += 1
            if use_aacaph and len(gcn_batch) > idx:
                anchor_info_gcn = gcn_batch[idx].to(device)
                idx += 1
            if use_nacc and len(gcn_batch) > idx:
                nacc_features_gcn = gcn_batch[idx].to(device)
                idx += 1
            
            # 甯︽?搴︾殑GCN forward锛堣繑鍥炲甫姊?害澧炲己宓屽叆锛?
            grad_results = model.indep_gcn_forward_with_grad()
            
            # 涓存椂鏇挎崲delta缂撳瓨涓哄甫姊?害鐗堟湰锛堜娇forward鑳戒紶鎾??搴﹀埌GCN鍙傛暟锛?
            # forward涓?娇鐢?embed(ids) + delta[ids]锛宒elta闇€瑕佹湁姊?害鎵嶈兘浼犳挱鍒癎CN
            saved_cache = dict(model._indep_gcn_cache)
            # 璁＄畻甯︽?搴︾殑delta = enhanced - raw_embed.detach()
            if 'base_user' in grad_results:
                model._indep_gcn_cache['base_user_delta'] = (
                    grad_results['base_user'] - model.user_embedding.weight.detach()
                )
                model._indep_gcn_cache['base_service_delta'] = (
                    grad_results['base_service'] - model.service_embedding.weight.detach()
                )
            if 'gmm_user' in grad_results:
                model._indep_gcn_cache['gmm_user_delta'] = (
                    grad_results['gmm_user'] - model.gmm_user_embed.weight.detach()
                )
                model._indep_gcn_cache['gmm_service_delta'] = (
                    grad_results['gmm_service'] - model.gmm_service_embed.weight.detach()
                )
            if 'tgem_user' in grad_results:
                model._indep_gcn_cache['tgem_user_delta'] = (
                    grad_results['tgem_user'] - model.tgem_user_embed.weight.detach()
                )
                model._indep_gcn_cache['tgem_service_delta'] = (
                    grad_results['tgem_service'] - model.tgem_service_embed.weight.detach()
                )
            
            attention_mask_gcn = (qos_history_gcn > 0).bool()
            outputs_gcn = model(
                user_ids=user_ids_gcn, service_ids=service_ids_gcn,
                qos_history=qos_history_gcn, attention_mask=attention_mask_gcn,
                gpam_features=gpam_features_gcn, mfvp_features=mfvp_features_gcn,
                anchor=anchor_gcn, anchor_info=anchor_info_gcn,
                nacc_features=nacc_features_gcn
            )
            gcn_loss = criterion(outputs_gcn['prediction'].squeeze(), targets_gcn)
            
            if torch.isfinite(gcn_loss):
                optimizer_indep_gcn.zero_grad()
                gcn_loss.backward()
                bad_grads, bad_grad_count = _sanitize_nonfinite_grads(model)
                if bad_grads and epoch == 0:
                    logger.warning(
                        f"  [Finite Guard] sanitized {bad_grad_count} non-finite Indep-GCN gradient entries in {bad_grads}"
                    )
                indep_gcn_all_params = []
                for gcn_name in ['base_gcn', 'gmm_gcn', 'tgem_gcn']:
                    gcn_module = getattr(model, gcn_name, None)
                    if gcn_module is not None:
                        indep_gcn_all_params.extend(list(gcn_module.parameters()))
                torch.nn.utils.clip_grad_norm_(indep_gcn_all_params, max_norm=1.0)
                optimizer_indep_gcn.step()
            
            # 鎭㈠?detach缂撳瓨锛圙CN鍙傛暟宸叉洿鏂帮紝閲嶆柊棰勮?绠楋級
            model._indep_gcn_cache = saved_cache
            model.precompute_indep_gcn_embeddings()
            
            if epoch % 50 == 0:
                gate_vals = []
                for gcn_name in ['base_gcn', 'gmm_gcn', 'tgem_gcn']:
                    gcn_module = getattr(model, gcn_name, None)
                    if gcn_module is not None:
                        gate_vals.append(f"{gcn_name}={torch.sigmoid(gcn_module.gate_alpha).item():.4f}")
                logger.info(f"  Indep-GCN update: loss={gcn_loss.item():.4f}, gates=[{', '.join(gate_vals)}]")
            gcn_update_time += time.time() - gcn_update_start_time
        
        # V63 SEGA: 鍛ㄦ湡鎬ф洿鏂癎CN鍙傛暟锛堢敤璁?粌闆嗙?涓€涓猙atch鐨勪富鎹熷け锛?
        if (args.use_sega and optimizer_gcn is not None and
            epoch % args.sega_gcn_update_interval == 0):
            gcn_update_start_time = time.time()
            model.train()
            # 鍙栬?缁冮泦绗?竴涓猙atch
            gcn_batch = next(iter(train_loader))
            idx = 4
            gpam_features_gcn, mfvp_features_gcn, anchor_gcn, anchor_info_gcn, nacc_features_gcn = None, None, None, None, None
            user_ids_gcn, service_ids_gcn = gcn_batch[0].to(device), gcn_batch[1].to(device)
            qos_history_gcn, targets_gcn = gcn_batch[2].to(device), gcn_batch[3].to(device)
            
            if (use_mfvp or use_gpam) and len(gcn_batch) > idx:
                extra_gcn = gcn_batch[idx].to(device)
                mfvp_features_gcn = extra_gcn if use_mfvp else None
                gpam_features_gcn = extra_gcn if (use_gpam and not use_mfvp) else None
                idx += 1
            if (use_hap or use_qaph) and len(gcn_batch) > idx:
                anchor_gcn = gcn_batch[idx].to(device)
                idx += 1
            if use_aacaph and len(gcn_batch) > idx:
                anchor_info_gcn = gcn_batch[idx].to(device)
                idx += 1
            if use_nacc and len(gcn_batch) > idx:
                nacc_features_gcn = gcn_batch[idx].to(device)
                idx += 1
            
            # 甯︽?搴︾殑GCN forward锛堟洿鏂扮紦瀛?+ 淇濈暀GCN姊?害鍥撅級
            enhanced_user, enhanced_service = model.sega_gcn_forward_with_grad()
            
            if enhanced_user is not None:
                # 涓存椂鏇挎崲缂撳瓨涓哄甫姊?害鐗堟湰锛堜娇forward鑳戒紶鎾??搴﹀埌GCN鍙傛暟锛?
                model._sega_user_cache = enhanced_user
                model._sega_service_cache = enhanced_service
                
                attention_mask_gcn = (qos_history_gcn > 0).bool()
                outputs_gcn = model(
                    user_ids=user_ids_gcn, service_ids=service_ids_gcn,
                    qos_history=qos_history_gcn, attention_mask=attention_mask_gcn,
                    gpam_features=gpam_features_gcn, mfvp_features=mfvp_features_gcn,
                    anchor=anchor_gcn, anchor_info=anchor_info_gcn,
                    nacc_features=nacc_features_gcn
                )
                gcn_loss = criterion(outputs_gcn['prediction'].squeeze(), targets_gcn)
                bad_buffers, bad_buffer_count = _sanitize_nonfinite_buffers(model)
                if bad_buffers and epoch == 0:
                    logger.warning(
                        f"  [Finite Guard] repaired {bad_buffer_count} non-finite buffers after SEGA forward in {bad_buffers}"
                    )
                
                if torch.isfinite(gcn_loss):
                    optimizer_gcn.zero_grad()
                    gcn_loss.backward()
                    bad_grads, bad_grad_count = _sanitize_nonfinite_grads(model)
                    if bad_grads and epoch == 0:
                        logger.warning(
                            f"  [Finite Guard] sanitized {bad_grad_count} non-finite SEGA gradient entries in {bad_grads}"
                        )
                    torch.nn.utils.clip_grad_norm_(model.pgcn.parameters(), max_norm=1.0)
                    optimizer_gcn.step()
                
                # GCN鍙傛暟宸叉洿鏂帮紝閲嶆柊棰勮?绠梔etach缂撳瓨渚涘悗缁??缁僢atch浣跨敤
                model.precompute_sega_embeddings()
                
                if epoch % 50 == 0:
                    gate_val = torch.sigmoid(model.pgcn.gate_alpha).item()
                    gate_info = f"  SEGA GCN update: loss={gcn_loss.item():.4f}, gate={gate_val:.4f}"
                    if hasattr(model.pgcn, 'second_order_gate'):
                        so_gate = torch.sigmoid(model.pgcn.second_order_gate).item()
                        gate_info += f", second_order_gate={so_gate:.4f}"
                    logger.info(gate_info)
            gcn_update_time += time.time() - gcn_update_start_time
        
        train_start_time = time.time()
        train_loss = train_epoch_optimized(
            model, train_loader, optimizer, criterion, device, logger, epoch,
            use_gpam=use_gpam, use_mfvp=use_mfvp,
            use_hap=use_hap, use_rapp=use_rapp,
            rapp_balance_weight=args.rapp_balance_weight,
            use_nacc=use_nacc, use_qaph=use_qaph,
            use_aacaph=use_aacaph,
            use_amp=args.use_amp,
            scaler=scaler,
        )
        train_epoch_time = time.time() - train_start_time

        val_start_time = time.time()
        val_results = evaluate_multi_lambda(
            model, val_loader, device, [0],
            use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
            use_nacc=use_nacc, use_qaph=use_qaph,
            use_aacaph=use_aacaph,
            outlier_protocol=args.outlier_protocol,
            reference_targets=train_targets_for_outlier,
            refresh_caches=True,
            use_amp=args.use_amp,
        )
        val_eval_time = time.time() - val_start_time
        val_mae_l0 = val_results[0]['mae']

        is_new_best = val_mae_l0 < best_val_mae_l0
        bad_buffers_for_checkpoint, bad_buffer_count = _sanitize_nonfinite_buffers(model)
        if bad_buffers_for_checkpoint:
            logger.warning(
                f"Repaired {bad_buffer_count} non-finite buffer entries before checkpoint: "
                f"{bad_buffers_for_checkpoint}"
            )
        bad_state_for_checkpoint = _find_nonfinite_state_names(model)
        if bad_state_for_checkpoint:
            logger.warning(
                "Skipping best-checkpoint update because model state contains non-finite "
                f"entries: {bad_state_for_checkpoint}"
            )
            is_new_best = False
        should_eval_test = (
            test_eval_policy == 'every_epoch' or
            (test_eval_policy == 'on_improve' and is_new_best)
        )
        if should_eval_test:
            test_start_time = time.time()
            test_results = evaluate_multi_lambda(
                model, test_loader, device, train_eval_lambdas,
                use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
                use_nacc=use_nacc, use_qaph=use_qaph,
                use_aacaph=use_aacaph,
                outlier_protocol=args.outlier_protocol,
                reference_targets=train_targets_for_outlier,
                refresh_caches=False,
                use_amp=args.use_amp,
            )
            test_eval_time = time.time() - test_start_time
        
        # Scheduler step (ReduceLROnPlateau闇€瑕佷紶鍏?al metric)
        if use_plateau:
            scheduler.step(val_results[0]['mae'])
        elif args.use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        val_mae_l0 = val_results[0]['mae']
        
        # 鑾峰彇鍙??涔犳崯澶卞弬鏁?
        loss_info = ""
        if hasattr(criterion, 'base_loss') and isinstance(criterion.base_loss, BarronAdaptiveLoss):
            bal_alpha = criterion.base_loss.alpha.item()
            bal_scale = criterion.base_loss.scale.item()
            agh_gamma = criterion.gamma.item() if hasattr(criterion.gamma, 'item') else criterion.gamma
            loss_info = f" BAL[伪={bal_alpha:.2f},c={bal_scale:.2f}] AGH[纬={agh_gamma:.2f}]"
        elif isinstance(criterion, BarronAdaptiveLoss):
            bal_alpha = criterion.alpha.item()
            bal_scale = criterion.scale.item()
            loss_info = f" BAL[伪={bal_alpha:.2f},c={bal_scale:.2f}]"
        elif hasattr(criterion, 'gamma'):
            agh_gamma = criterion.gamma.item() if hasattr(criterion.gamma, 'item') else float(criterion.gamma)
            loss_info = f" AGH[纬={agh_gamma:.2f}]"
        
        # V65: QAPH gate鏃ュ織锛堟瘡50涓猠poch璁板綍涓€娆★級
        qaph_info = ""
        if use_qaph and epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                gate_vals = []
                for batch in val_loader:
                    idx_b = 4
                    u_ids = batch[0].to(device)
                    s_ids = batch[1].to(device)
                    qos_h = batch[2].to(device)
                    a_mask = (qos_h > 0).bool()
                    mfvp_b, anchor_b = None, None
                    if (use_mfvp or use_gpam) and len(batch) > idx_b:
                        mfvp_b = batch[idx_b].to(device) if use_mfvp else None
                        idx_b += 1
                    if len(batch) > idx_b:
                        anchor_b = batch[idx_b].to(device)
                        idx_b += 1
                    out_b = model(user_ids=u_ids, service_ids=s_ids, qos_history=qos_h,
                                  attention_mask=a_mask, mfvp_features=mfvp_b, anchor=anchor_b)
                    if 'gate' in out_b:
                        gate_vals.append(out_b['gate'].mean().item())
                    break  # 鍙?彇绗?竴涓猙atch
                if gate_vals:
                    qaph_info = f" QAPH_gate={gate_vals[0]:.4f}"
            model.train()
        
        if test_results is not None and 0 in test_results:
            test_log_msg = f"Test MAE(lambda=0): {test_results[0]['mae']:.4f}"
            if 10 in test_results:
                test_log_msg += f" - Test MAE(lambda=10): {test_results[10]['mae']:.4f}"
        else:
            test_log_msg = "Test MAE(lambda=0): skipped"
        total_epoch_time = time.time() - epoch_start_time
        test_mae_l0_for_timing = (
            test_results[0]['mae'] if test_results is not None and 0 in test_results else ''
        )
        with open(timing_profile_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=timing_fields)
            writer.writerow({
                'epoch': epoch + 1,
                'sega_precompute_time': f"{sega_precompute_time:.6f}",
                'gcn_update_time': f"{gcn_update_time:.6f}",
                'train_epoch_time': f"{train_epoch_time:.6f}",
                'val_eval_time': f"{val_eval_time:.6f}",
                'test_eval_time': f"{test_eval_time:.6f}",
                'total_epoch_time': f"{total_epoch_time:.6f}",
                'train_loss': f"{train_loss:.8f}",
                'val_mae_lambda0': f"{val_mae_l0:.8f}",
                'test_mae_lambda0': test_mae_l0_for_timing,
                'test_eval_policy': test_eval_policy,
            })
        epoch_efficiency_rows.append({
            'train_epoch_time': float(train_epoch_time),
            'val_eval_time': float(val_eval_time),
            'test_eval_time': float(test_eval_time),
            'total_epoch_time': float(total_epoch_time),
        })

        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - "
                   f"Val MAE(lambda=0): {val_mae_l0:.4f} - "
                   f"{test_log_msg} - "
                   f"Time[s] precompute={sega_precompute_time:.2f}, gcn={gcn_update_time:.2f}, "
                   f"train={train_epoch_time:.2f}, val={val_eval_time:.2f}, test={test_eval_time:.2f}"
                   f"{loss_info}{qaph_info}")
        
        if is_new_best:
            best_val_mae_l0 = val_mae_l0
            best_epoch = epoch + 1
            if test_results is not None:
                best_test_results = test_results
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            d_label = f"d{int(args.density*100)}"
            tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
            best_checkpoint_path = os.path.join(args.log_dir, f'best_model_{qos_type}_{d_label}_gamma{gamma}_v42{tag}.pth')
            torch.save(model.state_dict(), best_checkpoint_path)
            logger.info(f"  -> NEW BEST MAE: {val_mae_l0:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    train_wall_time_s = time.time() - train_loop_start_time
    peak_train_memory_mb = get_cuda_peak_memory_mb(device)
    if peak_train_memory_mb is not None:
        logger.info(f"Peak CUDA training memory: {peak_train_memory_mb:.2f} MB")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if test_loader is None:
            logger.info("Constructing deferred test dataset for final evaluation.")
            test_ds = build_split_dataset('test', test_idx)
            test_loader = make_qaph_dataloader(
                test_ds, args.eval_batch_size, False, args.num_workers, device,
                prefetch_factor=args.prefetch_factor
            )
        reset_cuda_peak_memory(device)
        final_eval_start_time = time.time()
        best_test_results = evaluate_multi_lambda(
            model, test_loader, device, final_eval_lambdas,
            use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
            use_nacc=use_nacc, use_qaph=use_qaph,
            use_aacaph=use_aacaph,
            outlier_protocol=args.outlier_protocol,
            reference_targets=train_targets_for_outlier,
            refresh_caches=True,
            use_amp=args.use_amp,
        )
        logger.info(
            f"Final best-checkpoint test evaluation complete in "
            f"{time.time() - final_eval_start_time:.2f}s, lambdas={final_eval_lambdas}"
        )
        final_test_inference_time_s = time.time() - final_eval_start_time
        peak_inference_memory_mb = get_cuda_peak_memory_mb(device)
        if peak_inference_memory_mb is not None:
            logger.info(f"Peak CUDA inference memory: {peak_inference_memory_mb:.2f} MB")
    else:
        logger.warning("No best checkpoint state captured; final test evaluation skipped.")
        final_test_inference_time_s = None
        peak_inference_memory_mb = None

    # V46: SWA - 璁?粌缁撴潫鍚庢洿鏂癇N缁熻?閲忓苟璇勴及
    if args.use_swa and swa_model is not None:
        logger.info("SWA: Updating BatchNorm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        
        # 鐢⊿WA妯″瀷閲嶆柊璇勪及
        swa_test_results = evaluate_multi_lambda(swa_model, test_loader, device, final_eval_lambdas,
                                                use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
                                                use_nacc=use_nacc, use_qaph=use_qaph,
                                                use_aacaph=use_aacaph,
                                                outlier_protocol=args.outlier_protocol,
                                                reference_targets=train_targets_for_outlier,
                                                refresh_caches=True,
                                                use_amp=args.use_amp)
        swa_val_results = evaluate_multi_lambda(swa_model, val_loader, device, [0],
                                               use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
                                               use_nacc=use_nacc, use_qaph=use_qaph,
                                               use_aacaph=use_aacaph,
                                               outlier_protocol=args.outlier_protocol,
                                               reference_targets=train_targets_for_outlier,
                                               refresh_caches=True,
                                               use_amp=args.use_amp)
        
        swa_test_log = f"Test MAE(lambda=0): {swa_test_results[0]['mae']:.4f}"
        if 10 in swa_test_results:
            swa_test_log += f", Test MAE(lambda=10): {swa_test_results[10]['mae']:.4f}"
        logger.info(f"SWA Results - Val MAE(lambda=0): {swa_val_results[0]['mae']:.4f}, " + swa_test_log)
        
        # 濡傛灉SWA鏇村ソ锛屼娇鐢⊿WA缁撴灉
        if best_test_results is None or swa_test_results[0]['mae'] < best_test_results[0]['mae']:
            prev_best_mae = best_test_results[0]['mae'] if best_test_results is not None else float('inf')
            logger.info(f"SWA improved Test MAE(lambda=0): {prev_best_mae:.4f} -> {swa_test_results[0]['mae']:.4f}")
            best_test_results = swa_test_results
            best_val_mae_l0 = swa_val_results[0]['mae']
            best_epoch = 'swa'
            best_state_dict = {k: v.detach().cpu().clone() for k, v in swa_model.module.state_dict().items()}
            # 淇濆瓨SWA妯″瀷
            d_label = f"d{int(args.density*100)}"
            tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
            best_checkpoint_path = os.path.join(args.log_dir, f'best_model_{qos_type}_{d_label}_gamma{gamma}_v42{tag}_swa.pth')
            torch.save(swa_model.state_dict(), best_checkpoint_path)
        else:
            logger.info(f"SWA did NOT improve: SWA={swa_test_results[0]['mae']:.4f} vs Best={best_test_results[0]['mae']:.4f}")
    
    gate_analysis_csv = None
    if args.export_qaph_gate_csv and use_qaph:
        if best_state_dict is None:
            logger.warning("Gate CSV export skipped: no best checkpoint state was captured.")
        else:
            model.load_state_dict(best_state_dict)
            d_label = f"d{int(args.density*100)}"
            tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
            gate_csv_path = args.gate_csv_path or os.path.join(
                args.log_dir, f'gate_analysis_{qos_type}_{d_label}_gamma{gamma}_v42{tag}.csv'
            )
            gate_analysis_csv = export_qaph_gate_analysis_csv(
                model=model,
                dataloader=test_loader,
                dataset_tensors=test_ds.tensors,
                sample_indices=test_idx,
                full_history=history,
                qos_type=qos_type,
                density=args.density,
                split_id=args.split_id,
                output_csv=gate_csv_path,
                device=device,
                use_gpam=use_gpam,
                use_mfvp=use_mfvp,
                use_hap=use_hap,
                use_nacc=use_nacc,
                use_qaph=use_qaph,
                use_aacaph=use_aacaph,
                logger=logger,
            )

    flops_profile = None
    if getattr(args, 'profile_flops', False) and test_loader is not None:
        flops_profile = profile_qaph_forward_flops(
            model, test_loader, device,
            use_gpam=use_gpam, use_mfvp=use_mfvp, use_hap=use_hap,
            use_nacc=use_nacc, use_qaph=use_qaph, use_aacaph=use_aacaph,
            use_amp=args.use_amp,
            warmup_runs=args.flops_warmup_runs,
            logger=logger,
        )

    steady_rows = epoch_efficiency_rows[1:] if len(epoch_efficiency_rows) >= 5 else epoch_efficiency_rows

    def _mean_efficiency_value(key):
        vals = [float(row[key]) for row in steady_rows if key in row]
        return float(np.mean(vals)) if vals else None

    test_samples = int(len(test_ds)) if 'test_ds' in locals() and test_ds is not None else int(len(test_idx))
    latency_ms_per_sample = None
    throughput_samples_per_s = None
    if final_test_inference_time_s and test_samples > 0:
        latency_ms_per_sample = float(final_test_inference_time_s * 1000.0 / test_samples)
        throughput_samples_per_s = float(test_samples / final_test_inference_time_s)
    total_run_time_s = time.time() - run_start_time
    efficiency_profile = {
        'graph_build_time_s': float(graph_build_time_s),
        'current_memory_after_model_mb': current_memory_after_model_mb,
        'peak_train_memory_mb': peak_train_memory_mb,
        'peak_inference_memory_mb': peak_inference_memory_mb,
        'train_wall_time_s': float(train_wall_time_s),
        'total_run_time_s': float(total_run_time_s),
        'avg_train_epoch_time_s': _mean_efficiency_value('train_epoch_time'),
        'avg_val_eval_time_s': _mean_efficiency_value('val_eval_time'),
        'avg_test_eval_time_s': _mean_efficiency_value('test_eval_time'),
        'avg_total_epoch_time_s': _mean_efficiency_value('total_epoch_time'),
        'final_test_inference_time_s': final_test_inference_time_s,
        'test_samples': test_samples,
        'latency_ms_per_sample': latency_ms_per_sample,
        'throughput_samples_per_s': throughput_samples_per_s,
        'forward_flops_per_sample': (flops_profile or {}).get('forward_flops_per_sample'),
        'forward_flops_g_per_sample': (flops_profile or {}).get('forward_flops_g_per_sample'),
        'flops_profile_batch_size': (flops_profile or {}).get('batch_size'),
        'checkpoint_size_mb': file_size_mb(best_checkpoint_path),
        'num_profiled_epochs': len(epoch_efficiency_rows),
        'steady_state_excludes_first_epoch': bool(len(epoch_efficiency_rows) >= 5),
    }

    if rolling_source_hashes() != args.rolling_source_sha256:
        raise RuntimeError('Rolling source files changed during this run')

    result = {
        'qos_type': qos_type,
        'gamma': gamma,
        'density': args.density,
        'history_length': args.history_length,
        'history_mode': args.rolling_history_mode,
        'target_slice': args.target_slice,
        'history_start_slice': split_data['history_start_slice'],
        'history_end_slice': split_data['history_end_slice'],
        'history_sha256': args.rolling_history_sha256,
        'target_sha256': args.rolling_target_sha256,
        'split_mask_path': split_mask_path,
        'split_metadata': split_metadata,
        'statistical_features': stat_metadata,
        'rolling_protocol': args.rolling_protocol,
        'source_sha256': args.rolling_source_sha256,
        'version': 'V42-rolling-v0',
        'ablation': ablation_label,
        'outlier_protocol': args.outlier_protocol,
        'seed': args.seed,
        'split_id': args.split_id,
        'checkpoint_path': best_checkpoint_path,
        'log_path': getattr(logger, 'log_file_path', None),
        'timing_profile_csv': timing_profile_csv,
        'dataset_prepare_profile_csv': dataset_prepare_profile_csv,
        'test_eval_policy': test_eval_policy,
        'train_eval_lambdas': train_eval_lambdas,
        'final_eval_lambdas': final_eval_lambdas,
        'eval_batch_size': args.eval_batch_size,
        'use_amp': bool(args.use_amp),
        'config': {
            'ms_tgem': args.use_ms_tgem,
            'bal': args.use_bal,
            'caph': args.use_caph,
            'agh': args.use_agh,
            'qaph_anchor_usage': args.qaph_anchor_usage,
            'qaph_gate_mode': args.qaph_gate_mode
        },
        'args': vars(args),
        'param_count': param_count,
        'module_param_count': module_param_count,
        'efficiency': efficiency_profile,
        'flops_profile': flops_profile,
        'best_epoch': best_epoch,
        'best_val_mae_lambda0': float(best_val_mae_l0),
        'test_results': best_test_results
    }
    if gate_analysis_csv is not None:
        result['gate_analysis_csv'] = gate_analysis_csv
    return result


def main():
    parser = argparse.ArgumentParser(description='V42 Experiment - MS-TGEM + BAL + CAPH + AGH')
    parser.add_argument('--qos_type', type=str, default='tp', choices=['rt', 'tp'])
    parser.add_argument('--gamma', type=float, required=True)
    parser.add_argument('--density', type=float, default=0.10)
    parser.add_argument('--outlier_protocol', type=str, default='legacy_test_fit_predict',
                        choices=['legacy_test_fit_predict', 'scheme_b_exact'],
                        help='Outlier evaluation protocol: legacy keeps fit_predict(test); '
                             'scheme_b_exact fits on train targets and removes exactly 位%% on test scores')
    parser.add_argument('--use_stat_features', action='store_true', default=True)
    parser.add_argument('--no_stat_features', action='store_true')
    parser.add_argument('--use_gmm', action='store_true', default=True)
    parser.add_argument('--no_gmm', action='store_true')
    parser.add_argument('--use_tgem', action='store_true', default=True)
    parser.add_argument('--no_tgem', action='store_true')
    # V42娑堣瀺寮€鍏?
    parser.add_argument('--no_ms_tgem', action='store_true')
    parser.add_argument('--no_caph', action='store_true')
    parser.add_argument('--no_bal', action='store_true')
    parser.add_argument('--no_agh', action='store_true')
    parser.add_argument('--ablation_tag', type=str, default='')
    # V49: NMD鍒濆?宓屽叆
    parser.add_argument('--use_nmd_init', action='store_true')
    # V54: CF宓屽叆鍒濆?鍖朑MM
    parser.add_argument('--use_cf_embed', action='store_true')
    parser.add_argument('--cf_embed_dir', type=str, default='cache/cf_embeddings')
    # V50: GPAM-Aligned Input
    parser.add_argument('--use_gpam_input', action='store_true')
    parser.add_argument('--gpam_cache_dir', type=str, default='cache/gpam_factors')
    parser.add_argument('--gpam_f1', type=int, default=16)
    # V62: GPAM+MFVP铻嶅悎
    parser.add_argument('--gpam_fuse_mfvp', action='store_true')
    parser.add_argument('--gpam_fill_missing', action='store_true')
    # V52: MFVP (Multi-Feature Value Projection)
    parser.add_argument('--use_mfvp', action='store_true')
    parser.add_argument('--mfvp_window', type=int, default=5)
    # V55: PTCS (Pairwise Temporal Collaborative Signal)
    parser.add_argument('--use_ptcs', action='store_true')
    parser.add_argument('--ptcs_top_k', type=int, default=20)
    # LGCFE鍙傛暟
    parser.add_argument('--use_lgcfe', action='store_true')
    parser.add_argument('--lgcfe_k', type=int, default=10)
    parser.add_argument('--lgcfe_init_alpha', type=float, default=0.1)
    # V47鍙傛暟
    parser.add_argument('--e_block_ratio', type=int, default=4)
    parser.add_argument('--tgem_e_layers', type=int, default=2,
                        help='V73-B2: TGEM E-block Transformer灞傛暟 (榛樿?2, B2瀹為獙鐢?, 閰嶅悎dropout+0.05)')
    parser.add_argument('--tgem_num_branches', type=int, default=4, choices=[4, 5])
    parser.add_argument('--use_missing_token', action='store_true')
    parser.add_argument('--use_fuse_first', action='store_true')
    # V43浼樺寲鍙傛暟
    parser.add_argument('--fixed_agh_gamma', type=float, default=0.0,
                        help='鍥哄畾AGH 纬鍊?>0鏃跺浐瀹氾紝=0鏃跺彲瀛︿範)')
    parser.add_argument('--no_warmup', action='store_true')
    parser.add_argument('--patience', type=int, default=200)
    # V45: StepLR璋冨害鍣ㄥ弬鏁?
    parser.add_argument('--scheduler', type=str, default='step', choices=['step', 'cosine', 'plateau'],
                        help='LR scheduler type: step, cosine, or plateau')
    parser.add_argument('--step_size', type=int, default=500)
    parser.add_argument('--step_gamma', type=float, default=0.5)
    parser.add_argument('--plateau_factor', type=float, default=0.5)
    parser.add_argument('--plateau_patience', type=int, default=30)
    # V46: SWA鍙傛暟
    parser.add_argument('--use_swa', action='store_true')
    parser.add_argument('--swa_lr', type=float, default=0.0005)
    # V58: CF-Enhanced Embedding
    parser.add_argument('--use_cf_feature', action='store_true')
    parser.add_argument('--cf_factor_dim', type=int, default=64)
    # V56: HAP/RAPP鍙傛暟
    parser.add_argument('--use_hap', action='store_true')
    parser.add_argument('--hap_log_scale_min', type=float, default=-2.0)
    parser.add_argument('--hap_log_scale_max', type=float, default=3.0)
    parser.add_argument('--use_rapp', action='store_true')
    parser.add_argument('--rapp_num_experts', type=int, default=3)
    parser.add_argument('--rapp_balance_weight', type=float, default=0.01)
    # V60: NACC + Serial GMM鍙傛暟
    parser.add_argument('--use_nacc', action='store_true')
    parser.add_argument('--nacc_top_k', type=int, default=20)
    parser.add_argument('--use_serial_gmm', action='store_true')
    # V61: LightGCN宓屽叆澧炲己
    parser.add_argument('--use_lightgcn', action='store_true')
    parser.add_argument('--lightgcn_layers', type=int, default=2)
    parser.add_argument('--lightgcn_dropout', type=float, default=0.0)
    # V63: SEGA (Shared-Embedding Graph-Enhanced Architecture)
    parser.add_argument('--use_sega', action='store_true')
    parser.add_argument('--sega_gcn_dropout', type=float, default=0.1)
    parser.add_argument('--sega_init_gate', type=float, default=-2.0)
    parser.add_argument('--sega_gcn_update_interval', type=int, default=5)
    parser.add_argument('--pgcn_num_layers', type=int, default=2)
    parser.add_argument('--pgcn_layers', type=int, default=None)
    # V71: MultiGraph (FIG + SUIG + SSIG)
    parser.add_argument('--use_multi_graph', action='store_true')
    parser.add_argument('--multi_graph_use_suig', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--multi_graph_use_ssig', type=lambda x: x.lower() != 'false', default=True)
    parser.add_argument('--multi_graph_second_gate_init', type=float, default=-2.0)
    parser.add_argument('--suig_top_k', type=int, default=0)
    parser.add_argument('--ssig_top_k', type=int, default=50)
    # V64: Independent-GCN (鐙?珛宓屽叆 + 鍚勮嚜GCN澧炲己)
    parser.add_argument('--use_indep_gcn', action='store_true')
    parser.add_argument('--indep_gcn_dropout', type=float, default=0.1)
    parser.add_argument('--indep_gcn_init_gate', type=float, default=0.0)
    parser.add_argument('--indep_gcn_update_interval', type=int, default=5)
    # V65: QAPH (QoS-Adaptive Prediction Head)
    parser.add_argument('--use_qaph', action='store_true')
    parser.add_argument('--qaph_gate_init_bias', type=float, default=-1.0)
    parser.add_argument(
        '--qaph_anchor_usage',
        type=str,
        default='hap',
        choices=['hap', 'feature'],
        help='QAPH anchor usage mode: hap=Full QAPH anchor-relative HAP; feature=Anchor-as-Feature ablation where normalized anchor is concatenated into an MLP.'
    )
    parser.add_argument(
        '--qaph_gate_mode',
        type=str,
        default='learnable',
        choices=['learnable', 'fixed_0_5'],
        help='QAPH gate mode: learnable=sample-wise adaptive gate; fixed_0_5=use constant g=0.5 for fixed direct-anchor fusion.'
    )
    # V66: AACAPH (Anchor-Augmented Cross-Attention Prediction Head)
    parser.add_argument('--use_aacaph', action='store_true')
    # V70: Anchor璁＄畻澧炲己
    parser.add_argument('--anchor_mode', type=str, default='mean', choices=['mean', 'ema', 'trimmed', 'neighbor'],
                        help='V70: Anchor璁＄畻妯″紡 (mean=绠€鍗曞潎鍊? ema=鎸囨暟鍔犳潈, trimmed=鎴?柇鍧囧€? neighbor=閭诲眳鍔犳潈)')
    parser.add_argument('--anchor_ema_alpha', type=float, default=0.1)
    parser.add_argument('--anchor_trim_ratio', type=float, default=0.1)
    parser.add_argument('--anchor_neighbor_weight', type=float, default=0.3)
    # V73 FIG graph weighting
    parser.add_argument('--fig_weight_mode', type=str, default='binary', choices=['binary', 'reliability', 'qos_mean'],
                        help='FIG graph edge weight mode (binary=all 1s, reliability=QoS reliability, qos_mean=QoS value weighting)')
    parser.add_argument('--fig_min_weight', type=float, default=0.1)
    # V75 FiLM
    parser.add_argument('--use_film', action='store_true')
    # V76 TPA
    parser.add_argument('--use_tpa', action='store_true')
    parser.add_argument('--tpa_num_heads', type=int, default=4)
    parser.add_argument('--tpa_max_neighbors', type=int, default=256)
    parser.add_argument('--tpa_init_gate', type=float, default=-2.0)
    parser.add_argument('--tpa_dropout', type=float, default=0.1)
    # TGEM sub-module ablation
    parser.add_argument('--disable_tblock', action='store_true')
    parser.add_argument('--disable_fblock', action='store_true')
    # SCL鍙傛暟
    # LSCL鍙傛暟
    parser.add_argument('--use_lscl', action='store_true')
    parser.add_argument('--lscl_gamma', type=float, default=1.0)
    parser.add_argument('--use_scl', action='store_true')
    parser.add_argument('--scl_gamma_small', type=float, default=2.0)
    parser.add_argument('--scl_gamma_large', type=float, default=10.0)
    parser.add_argument('--scl_threshold', type=float, default=10.0)
    # 璁?粌鍙傛暟
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--bilinear_rank', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split_id', type=int, default=None,
                        help='Optional explicit split identifier for exported gate CSV; defaults to seed')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size for validation/test evaluation; defaults to --batch_size')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='DataLoader prefetch_factor when num_workers > 0')
    parser.add_argument('--test_eval_policy', type=str, default='final_only',
                        choices=['every_epoch', 'on_improve', 'final_only'],
                        help='When to evaluate the test split during training')
    parser.add_argument('--train_eval_lambdas', type=str, default='0',
                        help='Comma-separated lambda values for optional epoch-time test evaluation')
    parser.add_argument('--final_eval_lambdas', type=str, default='0',
                        help='Comma-separated lambda values for the final best-checkpoint test evaluation')
    parser.add_argument('--skip_epoch_test', action='store_true',
                        help='Shortcut for --test_eval_policy final_only')
    parser.add_argument('--dataset_cache_dir', type=str, default='cache/rolling_dataset_tensors',
                        help='Directory for cached TensorDataset tensors')
    parser.add_argument('--reuse_dataset_cache', action='store_true',
                        help='Reuse cached TensorDataset tensors when the cache key matches')
    parser.add_argument('--rebuild_dataset_cache', action='store_true',
                        help='Rebuild and overwrite TensorDataset tensor cache files')
    parser.add_argument('--defer_test_dataset', action='store_true',
                        help='Defer test TensorDataset construction when epoch-time test evaluation is disabled')
    parser.add_argument('--defer_test_dataset_policy', type=str, default='auto',
                        choices=['auto', 'never'])
    parser.add_argument('--pair_feature_cache_dir', type=str, default='cache/rolling_pair_feature_bank')
    parser.add_argument('--reuse_pair_feature_cache', action='store_true')
    parser.add_argument('--rebuild_pair_feature_cache', action='store_true')
    parser.add_argument('--pair_feature_cache_chunk_size', type=int, default=8192)
    parser.add_argument('--validate_pair_feature_cache', action='store_true')
    parser.add_argument('--efficiency_result_csv', type=str, default='',
                        help='Optional CSV path for appending deployment-efficiency metrics. '
                             'Empty keeps the legacy behavior and writes only the normal result JSON.')
    parser.add_argument('--profile_flops', action='store_true',
                        help='Profile one real evaluation batch with torch.profiler and append online '
                             'forward FLOPs/sample to efficiency outputs. Disabled by default.')
    parser.add_argument('--flops_warmup_runs', type=int, default=2,
                        help='Warmup forward passes before FLOPs profiling.')
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable CUDA automatic mixed precision for training and evaluation')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_dir', type=str, default='log/v71_phase2a_experiments')
    parser.add_argument('--data_dir', type=str, default='datasets/wsdream_dataset2/dataset2')
    parser.add_argument('--ptcs_cache_dir', type=str, default='cache/rolling_ptcs_neighbors',
                        help='Directory for cached PTCS/NACC neighbor precomputations; empty string disables caching')
    parser.add_argument('--export_qaph_gate_csv', action='store_true',
                        help='Export sample-level QAPH gate CSV on the test set for interpretability analysis')
    parser.add_argument('--gate_csv_path', type=str, default='',
                        help='Optional explicit output path for the exported QAPH gate CSV')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Existing checkpoint path for post-hoc evaluation / gate export without retraining')
    parser.add_argument('--export_gate_csv_only', action='store_true',
                        help='Load an existing checkpoint, refresh graph caches, evaluate, and optionally export gate CSV without retraining')
    parser.add_argument('--stat_cache_dir', type=str, default='cache/rolling_statistical_features')
    parser.add_argument('--target_slice', type=int, required=True,
                        help='One-based target time slice, from 2 to 64 for WS-DREAM2.')
    parser.add_argument('--history_length', type=int, default=None,
                        help='Use the latest L slices before the target; default uses all preceding slices.')
    parser.add_argument('--val_ratio', type=float, default=0.20)
    args = parser.parse_args()

    if args.export_gate_csv_only and not args.load_checkpoint_path:
        parser.error("--export_gate_csv_only requires --load_checkpoint_path")
    if not 2 <= args.target_slice <= 64:
        parser.error('--target_slice must be in [2, 64] for WS-DREAM2')
    args.tau = args.target_slice - 1
    args.rolling_history_mode = (
        'expanding' if args.history_length is None
        else f'fixed_L{args.history_length}'
    )
    if args.history_length is None:
        args.history_length = args.tau
    if args.history_length < 1 or args.history_length > args.tau:
        parser.error(
            '--history_length must be in [1, target_slice - 1].'
        )
    if args.use_gpam_input or args.use_cf_embed:
        parser.error('GPAM/CF embedding caches lack rolling target provenance; disable these options')
    if args.use_swa:
        parser.error('SWA selection uses test performance in the legacy loop; disable it for rolling runs')
    
    # V70: pgcn_layers鍒?悕澶勭悊
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.skip_epoch_test:
        args.test_eval_policy = 'final_only'
    if args.test_eval_policy != 'final_only':
        parser.error('rolling evaluation requires --test_eval_policy final_only')
    args.defer_test_dataset = True
    args.train_eval_lambda_values = parse_lambda_values(args.train_eval_lambdas)
    args.final_eval_lambda_values = parse_lambda_values(args.final_eval_lambdas)
    if args.train_eval_lambda_values != [0] or args.final_eval_lambda_values != [0]:
        parser.error('rolling evaluation currently reports only unfiltered lambda=0')
    if 0 not in args.train_eval_lambda_values:
        args.train_eval_lambda_values.insert(0, 0)
    if 0 not in args.final_eval_lambda_values:
        args.final_eval_lambda_values.insert(0, 0)

    if args.pgcn_layers is not None:
        args.pgcn_num_layers = args.pgcn_layers

    if args.split_id is None:
        args.split_id = args.seed
    args.rolling_protocol = 'history_only_v1'
    args.rolling_source_sha256 = rolling_source_hashes()
    rolling_tag = f"rolling_t{args.target_slice}_L{args.history_length}_s{args.seed}_sp{args.split_id}"
    args.ablation_tag = f"{args.ablation_tag}_{rolling_tag}" if args.ablation_tag else rolling_tag

    if args.no_stat_features:
        args.use_stat_features = False
    if args.no_gmm:
        args.use_gmm = False
    if args.no_tgem:
        args.use_tgem = False
    
    # 娑堣瀺寮€鍏?
    args.use_ms_tgem = not args.no_ms_tgem
    args.use_caph = not args.no_caph
    args.use_bal = not args.no_bal
    args.use_agh = not args.no_agh
    # V47寮€鍏筹紙榛樿?绂佺敤锛岄渶瑕佹樉寮忓惎鐢?級
    args.use_missing_token = False  # V48鍥為€€锛氶粯璁ょ?鐢?
    args.use_fuse_first = False     # V48鍥為€€锛氶粯璁ょ?鐢?
    
    # 鏋勫缓娑堣瀺鏍囩?
    ablation_parts = []
    if not args.use_ms_tgem: ablation_parts.append('noMS')
    if not args.use_bal: ablation_parts.append('noBAL')
    if not args.use_caph: ablation_parts.append('noCAPH')
    if not args.use_agh: ablation_parts.append('noAGH')
    ablation_label = '-'.join(ablation_parts) if ablation_parts else 'Full'
    if args.ablation_tag:
        ablation_label = args.ablation_tag
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # V63: 纭?畾鎬ц?缁冩ā寮?
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # 鏌愪簺PyTorch鐗堟湰涓嶆敮鎸?
    
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('requested CUDA runtime is unavailable')
    device = torch.device(args.device)
    if device.type == 'cuda' and (device.index or 0) >= torch.cuda.device_count():
        parser.error(f'requested CUDA device is unavailable: {args.device}')
    os.makedirs(args.log_dir, exist_ok=True)
    logger = setup_logging(args.log_dir, args.qos_type, args.gamma, args.density, ablation_label)
    
    logger.info(f"V56 [{ablation_label}]")
    logger.info(f"MS-TGEM: {args.use_ms_tgem}, BAL: {args.use_bal}, CAPH: {args.use_caph}, AGH: {args.use_agh}")
    logger.info(f"NMD init: {args.use_nmd_init}, GPAM input: {args.use_gpam_input}, MFVP: {args.use_mfvp}, PTCS: {args.use_ptcs}, CF embed: {args.use_cf_embed}, CF feature: {args.use_cf_feature}")
    logger.info(f"HAP: {args.use_hap}, RAPP: {args.use_rapp}, QAPH: {args.use_qaph}, AACAPH: {args.use_aacaph}" +
                (f", rapp_experts={args.rapp_num_experts}" if args.use_rapp else ""))
    if args.use_qaph:
        logger.info(f"QAPH anchor usage mode: {args.qaph_anchor_usage}")
        logger.info(f"QAPH gate mode: {args.qaph_gate_mode}")
    logger.info(f"NACC: {args.use_nacc}" + (f", nacc_top_k={args.nacc_top_k}" if args.use_nacc else "") +
                f", Serial GMM: {args.use_serial_gmm}" +
                f", LightGCN: {args.use_lightgcn}" + (f" (layers={args.lightgcn_layers})" if args.use_lightgcn else "") +
                f", SEGA: {args.use_sega}" + (f" (gcn_dropout={args.sega_gcn_dropout}, init_gate={args.sega_init_gate})" if args.use_sega else "") +
                f", Indep-GCN: {args.use_indep_gcn}" + (f" (dropout={args.indep_gcn_dropout}, init_gate={args.indep_gcn_init_gate})" if args.use_indep_gcn else "") +
                f", GPAM+MFVP fuse: {args.gpam_fuse_mfvp}" + (f" (fill={args.gpam_fill_missing}, f1={args.gpam_f1})" if args.gpam_fuse_mfvp else ""))
    logger.info(f"e_block_ratio={args.e_block_ratio}, tgem_branches={args.tgem_num_branches}, missing_token={args.use_missing_token}, fuse_first={args.use_fuse_first}")
    logger.info(f"{args.qos_type.upper()}, density={args.density}, gamma={args.gamma}")
    logger.info(f"Rolling protocol: target_slice={args.target_slice}, "
                f"history_length={args.history_length}, tau={args.tau}, split_id={args.split_id}")
    logger.info(f"epochs={args.epochs}, patience={args.patience}, scheduler={args.scheduler}, dropout={args.dropout}, weight_decay={args.weight_decay}")
    logger.info(f"SWA: {args.use_swa}" + (f", swa_lr={args.swa_lr}" if args.use_swa else ""))
    if args.export_gate_csv_only:
        logger.info(f"Export-only checkpoint: {args.load_checkpoint_path}")
    logger.info(f"Device: {device}")
    
    result = run_experiment(args.gamma, args.qos_type, args, device, logger, ablation_label)
    
    # 鎵撳嵃缁撴灉
    logger.info("\n" + "="*80)
    logger.info(f"V42 [{ablation_label}] RESULTS: {args.qos_type.upper()}, gamma={args.gamma}")
    logger.info("="*80)
    for lv in sorted(result['test_results'].keys(), key=lambda x: int(x)):
        logger.info(f"lambda={lv}: MAE={result['test_results'][lv]['mae']:.4f}, RMSE={result['test_results'][lv]['rmse']:.4f}")
    
    # Fixed-slice-64 reference numbers are not valid comparators for earlier
    # rolling targets; retain only the measured lambda=0 result above.
    
    # 淇濆瓨缁撴灉
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    d_label = f"d{int(args.density*100)}"
    tag = f"_{ablation_label}" if ablation_label != 'Full' else ''
    result_file = os.path.join(args.log_dir, f'result_{args.qos_type}_{d_label}_gamma{args.gamma}_v42{tag}_{timestamp}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    logger.info(f"\nResults saved to {result_file}")
    append_history_window_raw_result(result)
    if str(result.get('ablation', '')).startswith('history_'):
        logger.info("History-window raw result appended to results/history_window_sensitivity/raw_results.csv")
    if args.efficiency_result_csv:
        append_efficiency_raw_result(result, args.efficiency_result_csv)
        logger.info(f"Efficiency raw result appended to {args.efficiency_result_csv}")


if __name__ == '__main__':
    main()

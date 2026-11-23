"""History-only GPAM, hypergraph, and GMM inputs for an HCTN reproduction.

All arrays derive exclusively from historical time slices. Missing/nonpositive
entries are absent edges; NMF uses an explicitly documented zero-fill choice.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


FEATURE_KEYS = ("gpam", "incidence", "statistics", "greysheep")


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _profiles(matrix):
    """Fourteen paper-listed statistics, using positive observations only."""
    valid = matrix > 0
    count = valid.sum(axis=1).astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    values = np.where(valid, matrix, 0.0).astype(np.float64)
    mean = values.sum(axis=1) / safe_count
    centered = np.where(valid, values - mean[:, None], 0.0)
    variance = np.square(centered).sum(axis=1) / safe_count
    std = np.sqrt(variance)
    skew = np.zeros(len(matrix), dtype=np.float64)
    kurt = np.zeros(len(matrix), dtype=np.float64)
    enough = (count > 2) & (std > 1e-10)
    skew[enough] = (np.power(centered[enough], 3).sum(axis=1) /
                    safe_count[enough] / np.power(std[enough], 3))
    enough = (count > 3) & (std > 1e-10)
    kurt[enough] = (np.power(centered[enough], 4).sum(axis=1) /
                    safe_count[enough] / np.power(std[enough], 4) - 3.0)
    minimum = np.where(valid, matrix, np.inf).min(axis=1)
    maximum = np.where(valid, matrix, -np.inf).max(axis=1)
    minimum[count == 0] = 0.0
    maximum[count == 0] = 0.0
    nan_values = np.where(valid, matrix, np.nan)
    # Sorting one profile at a time avoids huge temporary arrays from a
    # whole-matrix nanpercentile on the 4500 service profiles.
    quartiles = np.zeros((len(matrix), 3), dtype=np.float64)
    median_abs = np.zeros(len(matrix), dtype=np.float64)
    entropy = np.zeros(len(matrix), dtype=np.float64)
    for i in range(len(matrix)):
        observed = nan_values[i, valid[i]]
        if not len(observed):
            continue
        quartiles[i] = np.percentile(observed, (25, 50, 75))
        median_abs[i] = np.median(np.abs(observed - quartiles[i, 1]))
        if len(observed) > 1 and maximum[i] > minimum[i]:
            hist = np.histogram(observed, bins=10)[0].astype(np.float64)
            probability = hist[hist > 0] / len(observed)
            entropy[i] = -(probability * np.log(probability)).sum()
    mean_abs = np.abs(centered).sum(axis=1) / safe_count
    energy = np.square(values).sum(axis=1)
    rms = np.sqrt(energy / safe_count)
    result = np.stack((minimum, maximum, mean, quartiles[:, 1], std,
                       skew, kurt, quartiles[:, 2] - quartiles[:, 0],
                       mean_abs, median_abs, rms, energy, entropy,
                       maximum - minimum), axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=0.0,
                         neginf=0.0).astype(np.float32)


def _greysheep(matrix, c_user, c_service):
    """Appendix A GDI (Eqs. 6-11) with safe low-degree conventions."""
    mask = matrix > 0
    q = np.where(mask, matrix, 0.0).astype(np.float64)
    n_user = mask.sum(axis=1)
    n_service = mask.sum(axis=0)
    user_mean = q.sum(axis=1) / np.maximum(n_user, 1)
    service_mean = q.sum(axis=0) / np.maximum(n_service, 1)
    user_min = np.where(mask, q, np.inf).min(axis=1)
    user_max = np.where(mask, q, -np.inf).max(axis=1)
    service_min = np.where(mask, q, np.inf).min(axis=0)
    service_max = np.where(mask, q, -np.inf).max(axis=0)
    user_min[n_user == 0] = user_max[n_user == 0] = 0.0
    service_min[n_service == 0] = service_max[n_service == 0] = 0.0
    user_trim = np.where(n_user > 2,
                         (q.sum(axis=1) - user_min - user_max) /
                         np.maximum(n_user - 2, 1), user_mean)
    service_trim = np.where(n_service > 2,
                            (q.sum(axis=0) - service_min - service_max) /
                            np.maximum(n_service - 2, 1), service_mean)
    user_var = ((np.square(q - user_mean[:, None]) * mask).sum(axis=1) /
                np.maximum(n_user, 1))
    service_var = ((np.square(q - service_mean[None, :]) * mask).sum(axis=0) /
                   np.maximum(n_service, 1))
    user_std = np.sqrt(user_var)
    service_std = np.sqrt(service_var)

    def inverse_minmax(values):
        spread = values.max() - values.min()
        if spread < 1e-10:
            return np.ones_like(values)
        return 1.0 - (values - values.min()) / spread

    user_weight = inverse_minmax(user_std)
    service_weight = inverse_minmax(service_std)
    user_gdi = ((np.abs(q - user_mean[:, None] - service_trim[None, :]) *
                 mask * service_weight[None, :]).sum(axis=1) /
                np.maximum(n_user, 1))
    service_gdi = ((np.abs(q - service_mean[None, :] - user_trim[:, None]) *
                    mask * user_weight[:, None]).sum(axis=0) /
                   np.maximum(n_service, 1))
    user_flag = (user_gdi > user_gdi.mean() + c_user * user_gdi.std())
    service_flag = (service_gdi > service_gdi.mean() +
                    c_service * service_gdi.std())
    return np.concatenate((user_flag, service_flag)).astype(np.float32)


def _nmf_window(matrices, rank, iterations, seed, device):
    """Deterministic multiplicative NMF, warm-started between adjacent slices."""
    rng = np.random.RandomState(seed)
    n_users, n_services = matrices.shape[1:]
    user_init = rng.uniform(0.05, 0.15, (n_users, rank)).astype(np.float32)
    service_init = rng.uniform(0.05, 0.15, (n_services, rank)).astype(np.float32)
    user = torch.from_numpy(user_init).to(device)
    service = torch.from_numpy(service_init).to(device)
    frames = []
    with torch.no_grad():
        for t, matrix in enumerate(matrices):
            target = torch.from_numpy(np.ascontiguousarray(matrix)).to(device)
            for _ in range(iterations):
                service *= ((target.T @ user) /
                            (service @ (user.T @ user) + 1e-8))
                service.clamp_(min=1e-8)
                user *= ((target @ service) /
                         (user @ (service.T @ service) + 1e-8))
                user.clamp_(min=1e-8)
            # Fix component scale ambiguity while preserving the product.
            u_norm = torch.sqrt(torch.sum(user * user, dim=0).clamp(min=1e-8))
            s_norm = torch.sqrt(torch.sum(service * service, dim=0).clamp(min=1e-8))
            balance = torch.sqrt(s_norm / u_norm)
            user *= balance
            service /= balance
            frame = torch.cat((user, service), dim=0).cpu().numpy().copy()
            if not np.isfinite(frame).all():
                raise RuntimeError("nonfinite NMF factors at history frame {}".format(t))
            frames.append(frame)
            print("[GPAM] history frame {}/{} NMF complete".format(
                t + 1, len(matrices)), flush=True)
    return np.stack(frames, axis=0).astype(np.float32)


def build_history_features(history, window, rank, nmf_iterations,
                           c_user, c_service, nmf_seed, device):
    if (history.ndim != 3 or window < 2 or window > history.shape[2] or
            rank < 1 or nmf_iterations < 1):
        raise ValueError("invalid HCTN feature configuration")
    selected = np.asarray(history[:, :, -window:], dtype=np.float32)
    if not np.isfinite(selected).all():
        raise ValueError("nonfinite historical tensor")
    positive = selected[selected > 0]
    if not len(positive):
        raise ValueError("no positive historical observations")
    scale = float(np.quantile(positive, 0.95))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid history-only scale")
    matrices = np.stack([np.where(selected[:, :, t] > 0,
                                  selected[:, :, t] / scale, 0.0)
                         for t in range(window)], axis=0).astype(np.float32)
    gpam = _nmf_window(matrices, rank, nmf_iterations, nmf_seed, device)
    stats = []
    flags = []
    incidence = []
    for t, matrix in enumerate(matrices):
        stats.append(np.concatenate((_profiles(matrix), _profiles(matrix.T)),
                                    axis=0))
        flags.append(_greysheep(matrix, c_user, c_service))
        incidence.append(matrix > 0)
        print("[GMM] history frame {}/{} statistics/GDI complete".format(
            t + 1, window), flush=True)
    statistics = np.stack(stats, axis=1)
    statistics = np.sign(statistics) * np.log1p(np.abs(statistics))
    center = statistics.mean(axis=(0, 1), keepdims=True)
    spread = statistics.std(axis=(0, 1), keepdims=True)
    statistics = ((statistics - center) / np.maximum(spread, 1e-5)).astype(
        np.float32)
    result = {
        "gpam": gpam,
        "incidence": np.stack(incidence, axis=0),
        "statistics": statistics,
        "greysheep": np.stack(flags, axis=1)[..., None],
    }
    _validate(result, history, window, rank)
    return result, scale


def _validate(features, history, window, rank):
    n, m = history.shape[:2]
    expected = {
        "gpam": (window, n + m, rank),
        "incidence": (window, n, m),
        "statistics": (n + m, window, 14),
        "greysheep": (n + m, window, 1),
    }
    for key, shape in expected.items():
        array = features[key]
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("invalid {} feature shape/values".format(key))
    if not np.array_equal(features["incidence"],
                          np.moveaxis(history[:, :, -window:] > 0, 2, 0)):
        raise ValueError("history graph does not match positive observations")


def load_or_build(history, history_sha256, qos, window, rank, nmf_iterations,
                  c_user, c_service, nmf_seed, cache_dir, device):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = {
        "protocol": "hctn_paper_history_only_v1",
        "source_sha256": source_sha256(),
        "history_sha256": history_sha256,
        "qos_type": qos,
        "window": window,
        "rank": rank,
        "nmf_iterations": nmf_iterations,
        "c_user": c_user,
        "c_service": c_service,
        "nmf_seed": nmf_seed,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = cache_dir / ("{}_w{}_r{}_{}.npz".format(qos, window, rank,
                                                   digest[:16]))
    if path.exists():
        with np.load(str(path), allow_pickle=False) as saved:
            stored = json.loads(saved["metadata"].item())
            features = {name: saved[name] for name in FEATURE_KEYS}
            scale = float(saved["scale"].item())
        if stored != key or scale <= 0 or not np.isfinite(scale):
            raise ValueError("HCTN feature cache metadata invalid")
        _validate(features, history, window, rank)
        return features, scale, str(path), True
    features, scale = build_history_features(
        history, window, rank, nmf_iterations, c_user, c_service,
        nmf_seed, device)
    temporary = str(path) + ".tmp.npz"
    try:
        np.savez_compressed(temporary, metadata=json.dumps(key, sort_keys=True),
                            scale=np.float32(scale), **features)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return features, scale, str(path), False

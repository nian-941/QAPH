"""Leakage-safe rolling target-slice inputs for WS-DREAM2 QAPH experiments.

Time slices are one-based at this boundary. The target slice is never included
in history-derived features, graphs, neighbors, or anchors.
"""

import hashlib
import json
import os
import pickle
import tempfile

import numpy as np
from scipy import stats


def prepare_rolling_split(qos_tensor, target_slice, density, val_ratio, seed,
                          history_length=None):
    """Return the legacy sparse-label split for a chosen target time slice.

    The seeded permutation matches ``HCTNDataSplitter.split`` without changing
    NumPy's global RNG state. History always ends at target_slice - 1.
    """
    if qos_tensor.ndim != 3:
        raise ValueError("qos_tensor must have shape [users, services, time]")
    n_users, n_services, n_times = qos_tensor.shape
    if not 2 <= target_slice <= n_times:
        raise ValueError("target_slice must be between 2 and the last available slice")
    if not 0 < density <= 1 or not 0 <= val_ratio < 1:
        raise ValueError("density must be in (0, 1] and val_ratio in [0, 1)")

    tau = target_slice - 1
    if history_length is None:
        history_length = tau
    if not 1 <= history_length <= tau:
        raise ValueError("history_length must be in [1, target_slice - 1]")

    history = qos_tensor[:, :, tau - history_length:tau]
    target = qos_tensor[:, :, tau]
    valid_indices = np.argwhere(target > 0)
    n_valid = len(valid_indices)
    n_sampled = int(n_valid * density)
    n_val = int(n_sampled * val_ratio)
    if n_sampled - n_val < 1 or n_val < 1 or n_valid - n_sampled < 1:
        raise ValueError("target slice lacks train, validation, or test entries")

    shuffled = np.random.RandomState(seed).permutation(n_valid)
    sampled = valid_indices[shuffled[:n_sampled]]
    test = valid_indices[shuffled[n_sampled:]]
    val = sampled[:n_val]
    train = sampled[n_val:]
    return {
        "history": history,
        "target_matrix": target,
        "valid_indices": valid_indices,
        "train_indices": train,
        "val_indices": val,
        "test_indices": test,
        "target_slice": target_slice,
        "tau": tau,
        "history_length": history_length,
        "history_start_slice": tau - history_length + 1,
        "history_end_slice": tau,
        "n_users": n_users,
        "n_services": n_services,
        "n_timesteps": n_times,
    }


def history_sha256(history):
    """Hash precisely the QoS values from which rolling features are built."""
    return hashlib.sha256(history.tobytes(order="C")).hexdigest()


def compute_14_statistical_features(values):
    """Match the 14-feature definition in precompute_statistical_features.py."""
    valid_values = values[values > 0]
    if len(valid_values) == 0:
        return np.zeros(14, dtype=np.float32)

    features = np.zeros(14, dtype=np.float32)
    min_val = np.min(valid_values)
    max_val = np.max(valid_values)
    mean_val = np.mean(valid_values)
    median_val = np.median(valid_values)
    std_val = np.std(valid_values) if len(valid_values) > 1 else 0.0
    features[:5] = (min_val, max_val, mean_val, median_val, std_val)
    features[5] = stats.skew(valid_values) if len(valid_values) > 2 and std_val > 1e-8 else 0.0
    features[6] = stats.kurtosis(valid_values) if len(valid_values) > 3 and std_val > 1e-8 else 0.0
    q1, q3 = np.percentile(valid_values, [25, 75])
    features[7] = q3 - q1
    features[8] = np.mean(np.abs(valid_values - mean_val))
    features[9] = np.median(np.abs(valid_values - median_val))
    features[10] = np.sqrt(np.mean(valid_values ** 2))
    features[11] = np.sum(valid_values ** 2)
    hist, _ = np.histogram(valid_values, bins=min(10, len(valid_values)), density=True)
    hist = hist[hist > 0]
    features[12] = -np.sum(hist * np.log(hist + 1e-10)) if len(hist) > 0 else 0.0
    features[13] = max_val - min_val
    return np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def _normalize_features(features):
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return np.clip((features - mean) / std, -10, 10).astype(np.float32)


def load_or_compute_rolling_statistics(history, qos_type, target_slice, cache_dir):
    """Compute/cache per-target historical statistics; reject legacy caches."""
    if history.ndim != 3 or history.shape[2] < 1:
        raise ValueError("history must contain at least one preceding time slice")
    n_users, n_services, history_length = history.shape
    digest = history_sha256(history)
    key = {
        "protocol": "rolling_history_only_v1",
        "qos_type": qos_type,
        "target_slice": int(target_slice),
        "history_length": int(history_length),
        "history_sha256": digest,
        "shape": (int(n_users), int(n_services), int(history_length)),
    }
    filename = "statistical_features_{}_t{}_L{}_{}.pkl".format(
        qos_type, target_slice, history_length, digest[:16]
    )
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    if os.path.exists(path):
        try:
            with open(path, "rb") as stream:
                payload = pickle.load(stream)
            if (isinstance(payload, dict) and payload.get("key") == key and
                    payload["user_features"].shape == (n_users, 14) and
                    payload["service_features"].shape == (n_services, 14) and
                    np.isfinite(payload["user_features"]).all() and
                    np.isfinite(payload["service_features"]).all()):
                return (_normalize_features(payload["user_features"]),
                        _normalize_features(payload["service_features"]),
                        {"cache_path": path, "cache_hit": True, **key})
        except (KeyError, AttributeError, OSError, TypeError, ValueError,
                pickle.UnpicklingError):
            pass

    user_features = np.zeros((n_users, 14), dtype=np.float32)
    service_features = np.zeros((n_services, 14), dtype=np.float32)
    for user_id in range(n_users):
        user_features[user_id] = compute_14_statistical_features(history[user_id].ravel())
    for service_id in range(n_services):
        service_features[service_id] = compute_14_statistical_features(history[:, service_id, :].ravel())
    payload = {
        "key": key,
        "user_features": user_features,
        "service_features": service_features,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=cache_dir, suffix=".tmp",
                                         delete=False) as stream:
            temporary_path = stream.name
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return (_normalize_features(user_features),
            _normalize_features(service_features),
            {"cache_path": path, "cache_hit": False, **key})


def save_split_indices(path, split, qos_type, density, seed, split_id):
    """Persist exactly the three target-slice masks used for one run."""
    n_services = split["n_services"]
    valid = split["valid_indices"]
    groups = [split[name] for name in ("train_indices", "val_indices", "test_indices")]
    encoded = [group[:, 0] * n_services + group[:, 1] for group in groups]
    merged = np.concatenate(encoded)
    valid_encoded = valid[:, 0] * n_services + valid[:, 1]
    if (len(np.unique(merged)) != len(merged) or
            len(merged) != len(valid_encoded) or
            not np.array_equal(np.sort(merged), np.sort(valid_encoded))):
        raise ValueError("rolling split masks overlap or do not cover all valid target entries")
    metadata = {
        "qos_type": qos_type,
        "density": float(density),
        "seed": int(seed),
        "split_id": int(split_id),
        "target_slice": split["target_slice"],
        "history_start_slice": split["history_start_slice"],
        "history_end_slice": split["history_end_slice"],
        "history_length": split["history_length"],
        "history_sha256": history_sha256(split["history"]),
        "target_sha256": hashlib.sha256(
            split["target_matrix"].tobytes(order="C")
        ).hexdigest(),
        "n_train": int(len(groups[0])),
        "n_val": int(len(groups[1])),
        "n_test": int(len(groups[2])),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, train_indices=groups[0], val_indices=groups[1],
                        test_indices=groups[2], metadata=json.dumps(metadata, sort_keys=True))
    return metadata

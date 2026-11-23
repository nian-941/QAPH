"""History-only, deterministic neighborhood sampling for GACL reproduction.

The paper does not specify a neighbor sampler. QAPH's shared protocol retains
all positive historical observations, making exact two-hop expansion on this
dense graph impractical. We sample without replacement and record the cap.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


GRAPH_KEYS = (
    "user_first_ids", "user_first_weight", "user_first_mask",
    "service_first_ids", "service_first_weight", "service_first_mask",
    "user_second_ids", "user_second_weight", "user_second_mask",
    "service_second_ids", "service_second_weight", "service_second_mask",
)


def _sample_axis(matrix, by_user, first_cap, second_cap, rng):
    n_nodes = matrix.shape[0] if by_user else matrix.shape[1]
    ids_1 = np.zeros((n_nodes, first_cap), dtype=np.int64)
    weights_1 = np.zeros((n_nodes, first_cap), dtype=np.float32)
    mask_1 = np.zeros((n_nodes, first_cap), dtype=np.bool_)
    ids_2 = np.zeros((n_nodes, second_cap), dtype=np.int64)
    weights_2 = np.zeros((n_nodes, second_cap), dtype=np.float32)
    mask_2 = np.zeros((n_nodes, second_cap), dtype=np.bool_)
    for node in range(n_nodes):
        row = matrix[node] if by_user else matrix[:, node]
        positive = np.flatnonzero(row > 0)
        if len(positive) > first_cap:
            chosen = rng.choice(positive, size=first_cap, replace=False)
        else:
            chosen = positive
        count_1 = len(chosen)
        count_2 = min(count_1, second_cap)
        ids_1[node, :count_1] = chosen
        weights_1[node, :count_1] = row[chosen]
        mask_1[node, :count_1] = True
        ids_2[node, :count_2] = chosen[:count_2]
        weights_2[node, :count_2] = row[chosen[:count_2]]
        mask_2[node, :count_2] = True
    return (ids_1, weights_1, mask_1, ids_2, weights_2, mask_2)


def build_window(history, window, first_cap, second_cap, seed):
    if (history.ndim != 3 or window < 1 or window > history.shape[2] or
            first_cap < 1 or second_cap < 1 or second_cap > first_cap):
        raise ValueError("invalid graph/window/cap configuration")
    by_key = {key: [] for key in GRAPH_KEYS}
    rng = np.random.RandomState(seed)
    for time_id in range(history.shape[2] - window, history.shape[2]):
        matrix = np.asarray(history[:, :, time_id], dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("nonfinite historical QoS values")
        user = _sample_axis(matrix, True, first_cap, second_cap, rng)
        service = _sample_axis(matrix, False, first_cap, second_cap, rng)
        for prefix, sampled in (("user", user), ("service", service)):
            for suffix, value in zip(("first_ids", "first_weight", "first_mask",
                                      "second_ids", "second_weight", "second_mask"),
                                     sampled):
                by_key[prefix + "_" + suffix].append(value)
    return {key: np.stack(value, axis=0) for key, value in by_key.items()}


def source_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def load_or_build_window(history, history_sha256, qos, window, first_cap,
                         second_cap, seed, cache_dir):
    """Validate a cache by source, history bytes, seed and sampling protocol."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = {
        "protocol": "gacl_paper_history_only_sampled_v1",
        "source_sha256": source_sha256(),
        "history_sha256": history_sha256,
        "qos_type": qos,
        "window": window,
        "first_cap": first_cap,
        "second_cap": second_cap,
        "seed": seed,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    path = cache_dir / ("{}_w{}_s{}_{}.npz".format(qos, window, seed, digest[:16]))
    if path.exists():
        with np.load(str(path), allow_pickle=False) as saved:
            stored = json.loads(saved["metadata"].item())
            graph = {name: saved[name] for name in GRAPH_KEYS}
        if stored != key:
            raise ValueError("graph cache metadata mismatch: {}".format(path))
        _validate_graph(graph, history.shape[0], history.shape[1], window,
                        first_cap, second_cap)
        return graph, str(path), True
    graph = build_window(history, window, first_cap, second_cap, seed)
    _validate_graph(graph, history.shape[0], history.shape[1], window,
                    first_cap, second_cap)
    temporary = str(path) + ".tmp.npz"
    try:
        np.savez_compressed(temporary, metadata=json.dumps(key, sort_keys=True),
                            **graph)
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return graph, str(path), False


def _validate_graph(graph, n_users, n_services, window, first_cap, second_cap):
    for prefix, n_nodes, n_other in (("user", n_users, n_services),
                                     ("service", n_services, n_users)):
        for level, cap in (("first", first_cap), ("second", second_cap)):
            ids = graph[prefix + "_" + level + "_ids"]
            weights = graph[prefix + "_" + level + "_weight"]
            mask = graph[prefix + "_" + level + "_mask"]
            shape = (window, n_nodes, cap)
            if (ids.shape != shape or weights.shape != shape or
                    mask.shape != shape or np.any(ids < 0) or
                    np.any(ids >= n_other) or not np.isfinite(weights).all() or
                    np.any(weights[mask] <= 0) or np.any(weights[~mask] != 0)):
                raise ValueError("invalid sampled graph {} {}".format(prefix, level))

"""Validate and reuse a saved QAPH target-slice split for baseline experiments."""

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def load_shared_split(mask_path, tensor, qos_type, target_slice, density, seed, split_id):
    """Return history, target, and ordered masks after checking their provenance.

    The check intentionally reads the current tensor. A stale tensor cache cannot
    silently turn a shared-mask comparison into a different-data comparison.
    """
    mask_path = Path(mask_path).resolve(strict=True)
    if tensor.ndim != 3 or not 2 <= target_slice <= tensor.shape[2]:
        raise ValueError("invalid tensor shape or target slice")
    history = tensor[:, :, :target_slice - 1]
    target = tensor[:, :, target_slice - 1]
    with np.load(mask_path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        groups = {
            name: np.asarray(saved[name], dtype=np.int64)
            for name in ("train_indices", "val_indices", "test_indices")
        }

    expected = {
        "qos_type": qos_type,
        "target_slice": int(target_slice),
        "history_start_slice": 1,
        "history_end_slice": int(target_slice - 1),
        "history_length": int(target_slice - 1),
        "density": float(density),
        "seed": int(seed),
        "split_id": int(split_id),
        "history_sha256": _sha256(history),
        "target_sha256": _sha256(target),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError("shared split {} mismatch: {}".format(field, mask_path))

    n_users, n_services = target.shape
    flat_groups = []
    for name, indices in groups.items():
        if indices.ndim != 2 or indices.shape[1] != 2 or len(indices) == 0:
            raise ValueError("invalid or empty {} in {}".format(name, mask_path))
        if (np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= n_users) or
                np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= n_services)):
            raise ValueError("out-of-range {} in {}".format(name, mask_path))
        if np.any(target[indices[:, 0], indices[:, 1]] <= 0):
            raise ValueError("nonpositive target in {}: {}".format(name, mask_path))
        if metadata.get("n_" + name.split("_")[0]) != len(indices):
            raise ValueError("count mismatch in {}: {}".format(name, mask_path))
        flat_groups.append(indices[:, 0] * n_services + indices[:, 1])

    all_flat = np.concatenate(flat_groups)
    valid = np.argwhere(target > 0)
    valid_flat = valid[:, 0] * n_services + valid[:, 1]
    if (len(np.unique(all_flat)) != len(all_flat) or
            not np.array_equal(np.sort(all_flat), np.sort(valid_flat))):
        raise ValueError("shared masks overlap or omit valid targets: {}".format(mask_path))

    n_sampled = int(len(valid) * density)
    n_val = int(n_sampled * 0.20)
    ordered = valid[np.random.RandomState(split_id).permutation(len(valid))]
    expected_groups = (ordered[n_val:n_sampled], ordered[:n_val], ordered[n_sampled:])
    if any(not np.array_equal(actual, expected_indices)
           for actual, expected_indices in zip(groups.values(), expected_groups)):
        raise ValueError("saved mask order differs from seeded QAPH split: {}".format(mask_path))

    return {
        "history": history,
        "target": target,
        "valid_indices": valid,
        **groups,
        "mask_path": str(mask_path),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "metadata": metadata,
    }

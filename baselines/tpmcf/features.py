"""History-only spatial inputs for the paper-aligned TPMCF adapter.

All functions in this module operate on one preceding time slice.  The target
slice is never passed to them.  The feature types follow TPMCF Section III.A:
five statistics, matrix-decomposition factors, and cosine similarities (the
latter are encoded by a separate autoencoder in the runner).
"""

import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


def _five_stats(values):
    """Return min/max/median/mean/std of positive entries along rows."""
    values = np.asarray(values, dtype=np.float32)
    positive = np.where(values > 0, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        columns = [
            np.nanmin(positive, axis=1),
            np.nanmax(positive, axis=1),
            np.nanmedian(positive, axis=1),
            np.nanmean(positive, axis=1),
            np.nanstd(positive, axis=1),
        ]
    out = np.stack(columns, axis=1)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    for column in range(out.shape[1]):
        lo = float(np.min(out[:, column]))
        hi = float(np.max(out[:, column]))
        if hi > lo:
            out[:, column] = (out[:, column] - lo) / (hi - lo)
        else:
            out[:, column] = 0.0
    return out.astype(np.float32)


def statistical_features(matrix):
    """Return normalized 5-column user and service features."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or np.any(matrix < 0):
        raise ValueError("one nonnegative QoS matrix is required")
    return _five_stats(matrix), _five_stats(matrix.T)


def qos_svd_features(matrix, rank, seed):
    """Return user/service factors with deterministic signs.

The author code calls ``svds`` and keeps U and V without singular values.
We preserve that choice.  ``rank`` is 100 for the paper configuration and
can be reduced only for a smoke test.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    if not 1 <= rank < min(matrix.shape):
        raise ValueError("SVD rank must be below the smaller matrix dimension")
    rng = np.random.RandomState(seed)
    v0 = rng.standard_normal(min(matrix.shape)).astype(np.float32)
    users, sigma, services_t = svds(sparse.csr_matrix(matrix), k=rank, v0=v0)
    order = np.argsort(sigma)[::-1]
    users = users[:, order]
    services_t = services_t[order, :]
    for component in range(rank):
        pivot = np.argmax(np.abs(users[:, component]))
        if users[pivot, component] < 0:
            users[:, component] *= -1
            services_t[component, :] *= -1
    return users.astype(np.float32), services_t.T.astype(np.float32)


def correlation_matrices(matrix):
    """Cosine similarity inputs for the two released-code autoencoders."""
    matrix = np.asarray(matrix, dtype=np.float32)
    return (
        cosine_similarity(matrix).astype(np.float32),
        cosine_similarity(matrix.T).astype(np.float32),
    )


def normalized_bipartite_graph(matrix):
    """Return COO indices and values of D^-1/2 (A+I) D^-1/2.

    The user/service graph has one edge for each positive historical value.
    Target-slice labels cannot enter this function because it takes only one
    historical matrix.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("one QoS matrix is required")
    n_users, n_services = matrix.shape
    users, services = np.nonzero(matrix > 0)
    services = services + n_users
    n_nodes = n_users + n_services
    degrees = np.ones(n_nodes, dtype=np.float32)
    np.add.at(degrees, users, 1.0)
    np.add.at(degrees, services, 1.0)
    nodes = np.arange(n_nodes, dtype=np.int64)
    rows = np.concatenate((nodes, users, services)).astype(np.int64)
    cols = np.concatenate((nodes, services, users)).astype(np.int64)
    vals = np.reciprocal(np.sqrt(degrees[rows] * degrees[cols])).astype(np.float32)
    return np.stack((rows, cols), axis=0), vals, n_nodes

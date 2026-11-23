"""Paper equations implemented with blocked, FP64 Torch operations.

CTF follows the user-supplied CTF.cpp, including block refresh and epsilon.
NNTF literal follows Eq.15--17 with a dense (zero-filled) denominator.
The separately named nntf_observed sensitivity changes that denominator only.
"""
import numpy as np
import torch


EPS = 1e-10


def reconstruct(factors):
    user, service, time = factors
    return (user[:, None, :] * service[None, :, :]) @ time.T


def contraction(values, factors, axis):
    user, service, time = factors
    if axis == 0:
        return torch.einsum("ijk,jr,kr->ir", values, service, time)
    if axis == 1:
        return torch.einsum("ijk,ir,kr->jr", values, user, time)
    return torch.einsum("ijk,ir,jr->kr", values, user, service)


def objective(values, factors, config):
    pred = reconstruct(factors)
    observed = values > 0
    residual = pred - values
    method = config["method"]
    if method == "nntf":
        return .5 * residual.square().sum()
    if method == "ctf":
        data = .5 * (torch.log1p((residual / config["gamma"]).square()) * observed).sum()
    else:
        data = .5 * (residual.square() * observed).sum()
    penalty = .5 * config.get("regularization", 0) * sum(f.square().sum() for f in factors)
    if method == "wspred":
        pair_mean = values.sum(2) / observed.sum(2).clamp(min=1)
        data = data + .5 * config["alpha"] * ((pred - pair_mean[:, :, None]).square() * observed).sum()
    return data + penalty


def initialize(shape, config, seed, device):
    rng = np.random.RandomState(seed)
    rank = config["rank"]
    # Preserve the WSPred Cython factor-major RNG layout.
    if config["method"] == "wspred":
        arrays = [rng.rand(rank, size).T.copy() for size in shape]
    else:
        arrays = [rng.rand(size, rank) for size in shape]
    return [torch.as_tensor(a, dtype=torch.float64, device=device) for a in arrays]


def factor_step(values, factors, config):
    """One complete U,S,T sweep, with refreshed prediction after each block."""
    method = config["method"]
    if method == "wspred":
        observed = values > 0
        pred = reconstruct(factors)
        pair_mean = values.sum(2) / observed.sum(2).clamp(min=1)
        residual = ((pred - values) + config["alpha"] * (pred - pair_mean[:, :, None])) * observed
        gradients = [contraction(residual, factors, axis) + config["regularization"] * f
                     for axis, f in enumerate(factors)]
        previous = float(objective(values, factors, config))
        step = config["learning_rate"]
        # A rejected final proposal stops this run instead of applying a stale loss.
        for attempt in range(20):
            proposed = [f - step * g for f, g in zip(factors, gradients)]
            loss = float(objective(values, proposed, config))
            if np.isfinite(loss) and loss < previous:
                return proposed, {"step_size": step, "backtracks": attempt}
            step *= .5
        raise RuntimeError("WSPred line search exhausted 20 proposals; retained previous state")

    for axis in range(3):
        if method == "nntf":
            others = [f for index, f in enumerate(factors) if index != axis]
            numerator = contraction(values, factors, axis)
            denominator = factors[axis] @ ((others[0].T @ others[0]) * (others[1].T @ others[1]))
        else:
            pred = reconstruct(factors)
            weight = (values > 0).to(values.dtype)
            if method == "ctf":
                weight = weight / (config["gamma"] ** 2 + (values - pred).square())
            numerator = contraction(weight * values, factors, axis)
            denominator = contraction(weight * pred, factors, axis)
            denominator = denominator + config.get("regularization", 0) * factors[axis]
        factors[axis] = factors[axis] * numerator / (denominator + EPS)
    return factors, {}


def target_matrix(factors):
    return (factors[0] * factors[2][-1]) @ factors[1].T


def pair_statistics(matrix, start, stop, need_rbs=True, need_pcc=True):
    """Exact all-neighbor RBS, centered PCC and Jaccard in a bounded block."""
    valid = matrix > 0
    count = valid.sum(1)
    means = matrix.sum(1) / count.clamp(min=1)
    left = matrix[start:stop]
    left_valid = valid[start:stop]
    common = left_valid.to(matrix.dtype) @ valid.to(matrix.dtype).T
    ratio = torch.zeros_like(common)
    if need_rbs:
        low = torch.minimum(left[:, None, :], matrix[None, :, :])
        high = torch.maximum(left[:, None, :], matrix[None, :, :])
        ratio = (low / high.clamp(min=EPS)).sum(2) / common.clamp(min=1)
    corr = torch.zeros_like(common)
    if need_pcc:
        centered = torch.where(valid, matrix - means[:, None], torch.zeros_like(matrix))
        a = centered[start:stop]
        numerator = a @ centered.T
        left_norm = a.square() @ valid.to(matrix.dtype).T
        right_norm = left_valid.to(matrix.dtype) @ centered.square().T
        denominator = torch.sqrt(left_norm * right_norm)
        corr = torch.where(denominator > EPS, numerator / denominator.clamp(min=EPS), torch.zeros_like(numerator))
    union = count[start:stop, None] + count[None, :] - common
    jaccard = common / union.clamp(min=1)
    ids = torch.arange(start, stop, device=matrix.device)
    ratio[torch.arange(stop-start, device=matrix.device), ids] = 0
    corr[torch.arange(stop-start, device=matrix.device), ids] = 0
    return means, valid, ratio, corr, jaccard


def reinforced_average(matrix, config):
    """TRCF Eq.2--15; each missing value filled from original neighbors once."""
    count = (matrix > 0).sum(1)
    original_mean = matrix.sum(1) / count.clamp(min=1)
    average = torch.empty_like(original_mean)
    block = config.get("neighbor_block", 64)
    valid_float = (matrix > 0).to(matrix.dtype)
    for start in range(0, len(matrix), block):
        stop = min(start + block, len(matrix))
        means, valid, rbs, _, _ = pair_statistics(matrix, start, stop, need_pcc=False)
        selected = rbs > config["rbs_threshold"]
        scale = torch.where(means[None, :] >= means[start:stop, None], rbs, 1 / rbs.clamp(min=EPS))
        weight = torch.where(selected, scale, torch.zeros_like(scale))
        number = selected.to(matrix.dtype) @ valid_float
        fill = (weight @ matrix) / number.clamp(min=1)
        supplemented = torch.where(valid[start:stop], matrix[start:stop], fill)
        valid_count = (supplemented > 0).sum(1)
        average[start:stop] = supplemented.sum(1) / valid_count.clamp(min=1)
    return average


def deviation_prediction(matrix, baseline, config):
    """TRCF Eq.16--29. Neighbor target values restricted to target-train."""
    current = matrix[:, -1]
    available = current > 0
    additive = torch.empty_like(baseline)
    multiplicative = torch.empty_like(baseline)
    block = config.get("neighbor_block", 64)
    for start in range(0, len(matrix), block):
        stop = min(start + block, len(matrix))
        _, _, _, pcc, reliability = pair_statistics(matrix, start, stop, need_rbs=False)
        weight = torch.where((pcc > config["pcc_threshold"]) & available[None, :], pcc * reliability, torch.zeros_like(pcc))
        denom = weight.sum(1)
        safe_denom = torch.where(denom.abs() > EPS, denom, torch.ones_like(denom))
        delta = torch.where(denom.abs() > EPS, (weight @ (current - baseline)) / safe_denom, torch.zeros_like(denom))
        additive[start:stop] = baseline[start:stop] + delta
        ratio = torch.where(available & (baseline > 0), current / baseline.clamp(min=EPS), torch.zeros_like(current))
        ratio_weight = weight * (baseline > 0)[None, :]
        ratio_denom = ratio_weight.sum(1)
        safe_ratio = torch.where(ratio_denom.abs() > EPS, ratio_denom, torch.ones_like(ratio_denom))
        multiplier = (ratio_weight @ ratio) / safe_ratio
        multiplier = torch.where(ratio_denom.abs() > EPS, multiplier, torch.ones_like(multiplier))
        multiplicative[start:stop] = baseline[start:stop] * multiplier
    return additive, multiplicative


def trcf(values, config, progress=None):
    users, services, _ = values.shape
    user_average = torch.empty((users, services), dtype=values.dtype, device=values.device)
    service_average = torch.empty_like(user_average)
    for user in range(users):
        user_average[user] = reinforced_average(values[user], config)
        if progress:
            progress("REINFORCE_SERVICE_NEIGHBORS", user + 1, users)
    for service in range(services):
        service_average[:, service] = reinforced_average(values[:, service, :], config)
        if progress and (service + 1) % 100 == 0:
            progress("REINFORCE_USER_NEIGHBORS", service + 1, services)
    # A completely unobserved pair with no usable neighbor receives a declared fallback.
    fallback = values.sum() / (values > 0).sum().clamp(min=1)
    user_average = torch.where(user_average > 0, user_average, fallback)
    service_average = torch.where(service_average > 0, service_average, fallback)
    user_add = torch.empty_like(user_average)
    user_mul = torch.empty_like(user_average)
    service_add = torch.empty_like(user_average)
    service_mul = torch.empty_like(user_average)
    for service in range(services):
        user_add[:, service], user_mul[:, service] = deviation_prediction(values[:, service, :], user_average[:, service], config)
        if progress and (service + 1) % 100 == 0:
            progress("DEVIATE_USER_NEIGHBORS", service + 1, services)
    for user in range(users):
        service_add[user], service_mul[user] = deviation_prediction(values[user], service_average[user], config)
        if progress:
            progress("DEVIATE_SERVICE_NEIGHBORS", user + 1, users)
    alpha = config["alpha"]
    prediction = alpha * user_add + (1-alpha) * service_add
    repaired = alpha * user_mul + (1-alpha) * service_mul
    negative = prediction < 0
    return torch.where(negative, repaired, prediction), {"negative_repair_count": int(negative.sum()), "fallback": float(fallback)}

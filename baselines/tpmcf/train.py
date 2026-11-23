"""Paper-aligned TPMCF adaptation for QAPH's shared target-slice protocol.

This is a PyTorch port of the architecture described in TPMCF Sections III-IV
with the source release as a second reference.  It is *not* an unmodified run
of the authors' TensorFlow program.  All spatial inputs come from the eight
historical slices immediately before the target.  Target labels are used only
for PTE train/validation/final-test roles defined by a verified QAPH mask.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from baselines.tpmcf.model import CosineAutoencoder, GraphConvolutionalMF, PredictiveBlock, TPMCFPTE
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
from shared.splits import load_shared_split  # noqa: E402
from baselines.tpmcf.features import (  # noqa: E402
    correlation_matrices,
    normalized_bipartite_graph,
    qos_svd_features,
    statistical_features,
)

N_USERS = 142
N_SERVICES = 4500
N_TIMESTEPS = 64


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_tensor(qos_type, data_dir, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / (qos_type + "_tensor_float32.npy")
    if cache.exists():
        tensor = np.load(str(cache), mmap_mode="r")
        print("[Data] cache={}".format(cache), flush=True)
    else:
        raw = data_dir / ("rtdata.txt" if qos_type == "rt" else "tpdata.txt")
        if not raw.exists():
            raise FileNotFoundError(str(raw))
        tensor = np.zeros((N_USERS, N_SERVICES, N_TIMESTEPS), dtype=np.float32)
        for frame in pd.read_csv(
                str(raw), sep=r"\s+", header=None,
                names=["user", "service", "time", "value"],
                dtype={"user": np.int32, "service": np.int32,
                       "time": np.int32, "value": np.float32},
                chunksize=1_000_000):
            users, services, times = (frame.user.to_numpy(),
                                      frame.service.to_numpy(), frame.time.to_numpy())
            if (np.any(users < 0) or np.any(users >= N_USERS) or
                    np.any(services < 0) or np.any(services >= N_SERVICES) or
                    np.any(times < 0) or np.any(times >= N_TIMESTEPS)):
                raise ValueError("raw tensor index outside WS-DREAM2 dimensions")
            tensor[users, services, times] = frame.value.to_numpy()
        np.save(str(cache), tensor)
        print("[Data] created cache={}".format(cache), flush=True)
    if tensor.shape != (N_USERS, N_SERVICES, N_TIMESTEPS) or tensor.dtype != np.float32:
        raise ValueError("unexpected WS-DREAM2 tensor shape/dtype")
    return tensor




def fit_cosine_ae(matrix, output_dim, epochs, batch_size, seed, device, service):
    set_seed(seed)
    data = torch.from_numpy(np.asarray(matrix, dtype=np.float32))
    hidden_a, hidden_b = (1024, 256) if service else (120, 80)
    model = CosineAutoencoder(data.shape[1], hidden_a, hidden_b, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(seed)
    model.train()
    for _ in range(epochs):
        for selected in np.array_split(rng.permutation(len(data)),
                                       max(1, int(math.ceil(len(data) / float(batch_size))))):
            batch = data[selected].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch), batch)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite cosine autoencoder loss")
            loss.backward()
            optimizer.step()
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            output.append(model.encoder(data[start:start + batch_size].to(device)).cpu().numpy())
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return np.concatenate(output, axis=0).astype(np.float32), parameters




def fit_graph_mf(matrix, node_features, dim, epochs, patience, gamma, seed, device):
    set_seed(seed)
    graph_indices, graph_values, n_nodes = normalized_bipartite_graph(matrix)
    adjacency = torch.sparse_coo_tensor(
        torch.from_numpy(graph_indices).to(device),
        torch.from_numpy(graph_values).to(device),
        (n_nodes, n_nodes), device=device,
    ).coalesce()
    features = torch.from_numpy(node_features).to(device)
    n_users = matrix.shape[0]
    edge_user, edge_service = np.nonzero(matrix > 0)
    if len(edge_user) == 0:
        raise ValueError("historical graph has no observed edges")
    edge_u = torch.from_numpy(edge_user.astype(np.int64)).to(device)
    edge_s = torch.from_numpy(edge_service.astype(np.int64)).to(device)
    edge_y = torch.from_numpy(matrix[edge_user, edge_service].astype(np.float32)).to(device)
    model = GraphConvolutionalMF(features.shape[1], dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_loss, best_state, wait = math.inf, None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        embedding = model(features, adjacency)
        predicted = (embedding[edge_u] * embedding[n_users + edge_s]).sum(dim=1)
        loss = torch.log1p(((predicted - edge_y) / gamma) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite graph reconstruction loss")
        loss.backward()
        optimizer.step()
        # Select the state using the loss of that same, updated state.
        model.eval()
        with torch.no_grad():
            updated = model(features, adjacency)
            updated_prediction = (
                updated[edge_u] * updated[n_users + edge_s]
            ).sum(dim=1)
            updated_loss = torch.log1p(
                ((updated_prediction - edge_y) / gamma) ** 2
            ).mean()
        if not torch.isfinite(updated_loss):
            raise RuntimeError("nonfinite updated graph reconstruction loss")
        value = float(updated_loss.cpu())
        if value < best_loss - 1e-6:
            best_loss = value
            best_state = {key: val.detach().cpu().clone()
                          for key, val in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embedding = model(features, adjacency).cpu().numpy().astype(np.float32)
    return embedding, {
        "observed_edges": int(len(edge_user)),
        "graph_epochs_ran": epoch + 1,
        "best_graph_loss": best_loss,
        "graph_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def make_spatial_bank(history, args, device):
    n_users, n_services, n_history = history.shape
    if args.feature_steps > n_history:
        raise ValueError("feature_steps exceeds available history")
    selected = list(range(n_history - args.feature_steps, n_history))
    bank, stage_rows = [], []
    for time_id in selected:
        start = time.time()
        matrix = np.asarray(history[:, :, time_id], dtype=np.float32)
        user_stats, service_stats = statistical_features(matrix)
        user_qos, service_qos = qos_svd_features(matrix, args.svd_rank,
                                                 args.seed + time_id * 10 + 1)
        initial_feature_time = time.time() - start
        user_cosine, service_cosine = correlation_matrices(matrix)
        user_collab, user_ae_parameters = fit_cosine_ae(
            user_cosine, args.collab_dim, args.ae_epochs,
            args.ae_batch_size, args.seed + time_id * 10 + 2, device, False)
        del user_cosine
        service_collab, service_ae_parameters = fit_cosine_ae(
            service_cosine, args.collab_dim, args.ae_epochs,
            args.ae_batch_size, args.seed + time_id * 10 + 3, device, True)
        del service_cosine
        ae_time = time.time() - start - initial_feature_time
        user_features = np.concatenate((user_stats, user_qos, user_collab), axis=1)
        service_features = np.concatenate((service_stats, service_qos, service_collab), axis=1)
        node_features = np.concatenate((user_features, service_features), axis=0)
        if not np.isfinite(node_features).all():
            raise RuntimeError("nonfinite spatial features")
        embedding, graph_record = fit_graph_mf(
            matrix, node_features.astype(np.float32), args.gcn_dim,
            args.gcn_epochs, args.gcn_patience, args.gcn_gamma,
            args.seed + time_id * 10 + 4, device,
        )
        if not np.isfinite(embedding).all():
            raise RuntimeError("nonfinite graph embeddings")
        graph_time = time.time() - start - initial_feature_time - ae_time
        bank.append(embedding)
        record = {"history_slice": time_id + 1,
                  "feature_dim": int(node_features.shape[1]),
                  "embedding_dim": int(embedding.shape[1]),
                  "feature_time_sec": initial_feature_time,
                  "autoencoder_time_sec": ae_time,
                  "graph_time_sec": graph_time,
                  "user_ae_parameter_count": user_ae_parameters,
                  "service_ae_parameter_count": service_ae_parameters,
                  "time_sec": time.time() - start, **graph_record}
        stage_rows.append(record)
        print("[CSFE] slice={} edges={} feat={} graph_epochs={} loss={:.6f} time={:.1f}s".format(
            time_id + 1, record["observed_edges"], record["feature_dim"],
            record["graph_epochs_ran"], record["best_graph_loss"], record["time_sec"]),
            flush=True)
    return np.stack(bank, axis=0), stage_rows


def load_or_build_spatial_bank(split, args, device, cache_dir):
    """Cache only history-derived embeddings, keyed by data and source hashes."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = {
        "protocol": "tpmcf_paper_aligned_history_only_v1",
        "history_sha256": split["metadata"]["history_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "model_sha256": sha256_file(ROOT / "baselines" / "tpmcf" / "model.py"),
        "features_sha256": sha256_file(ROOT / "baselines" / "tpmcf" / "features.py"),
        "seed": args.seed, "feature_steps": args.feature_steps,
        "svd_rank": args.svd_rank, "collab_dim": args.collab_dim,
        "ae_epochs": args.ae_epochs, "ae_batch_size": args.ae_batch_size,
        "gcn_dim": args.gcn_dim, "gcn_epochs": args.gcn_epochs,
        "gcn_patience": args.gcn_patience, "gcn_gamma": args.gcn_gamma,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    path = cache_dir / ("{}_t{}_s{}_{}.npz".format(
        args.qos_type, args.target_slice, args.seed, digest[:16]))
    if path.exists():
        with np.load(str(path), allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            bank = np.asarray(saved["bank"], dtype=np.float32)
            rows = json.loads(saved["spatial_rows"].item())
        expected_shape = (args.feature_steps, N_USERS + N_SERVICES,
                          2 * args.gcn_dim)
        if metadata != key or bank.shape != expected_shape or not np.isfinite(bank).all():
            raise ValueError("invalid spatial cache: {}".format(path))
        print("[CSFE] verified cache={}".format(path), flush=True)
        return bank, rows, str(path), True
    bank, rows = make_spatial_bank(split["history"], args, device)
    temp_path = str(path) + ".tmp.npz"
    try:
        np.savez_compressed(temp_path, bank=bank,
                            metadata=json.dumps(key, sort_keys=True),
                            spatial_rows=json.dumps(rows))
        os.replace(temp_path, str(path))
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    print("[CSFE] saved cache={}".format(path), flush=True)
    return bank, rows, str(path), False






def evaluate(model, indices, target, user_bank, service_bank, batch_size, device,
             record_rt_boundary=False):
    model.eval()
    abs_sum, sq_sum, count = 0.0, 0.0, 0
    boundary_abs_sum, boundary_n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            group = indices[start:start + batch_size]
            users = torch.as_tensor(group[:, 0], dtype=torch.long, device=device)
            services = torch.as_tensor(group[:, 1], dtype=torch.long, device=device)
            prediction = model(users, services, user_bank, service_bank).clamp_min(0.0)
            truth = torch.as_tensor(target[group[:, 0], group[:, 1]],
                                    dtype=torch.float32, device=device)
            error = prediction - truth
            abs_sum += float(error.abs().sum().item())
            sq_sum += float(error.square().sum().item())
            count += len(group)
            if record_rt_boundary:
                at_boundary = truth == 20.0
                if at_boundary.any():
                    boundary_abs_sum += float(error[at_boundary].abs().sum().item())
                    boundary_n += int(at_boundary.sum().item())
    if count == 0:
        raise ValueError("empty evaluation split")
    return {"mae": abs_sum / count, "rmse": math.sqrt(sq_sum / count),
            "n": count,
            "rt_20_n": boundary_n if record_rt_boundary else None,
            "rt_20_mae": (boundary_abs_sum / boundary_n
                          if record_rt_boundary and boundary_n else None)}


def train_pte(split, user_bank, service_bank, args, device):
    target = split["target"]
    train = split["train_indices"]
    val = split["val_indices"]
    test = split["test_indices"]
    set_seed(args.seed)
    model = TPMCFPTE(args.feature_steps, args.gcn_dim, args.pte_blocks,
                     args.pte_heads, args.conv_filters, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.pte_lr)
    rng = np.random.RandomState(args.seed)
    best_val, best_state, wait = math.inf, None, 0
    rows = []
    start_train = time.time()
    for epoch in range(1, args.pte_epochs + 1):
        model.train()
        loss_total, n_batches = 0.0, 0
        shuffled = train[rng.permutation(len(train))]
        for offset in range(0, len(train), args.train_batch_size):
            group = shuffled[offset:offset + args.train_batch_size]
            users = torch.as_tensor(group[:, 0], dtype=torch.long, device=device)
            services = torch.as_tensor(group[:, 1], dtype=torch.long, device=device)
            truth = torch.as_tensor(target[group[:, 0], group[:, 1]],
                                    dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(users, services, user_bank, service_bank)
            loss = torch.log1p(((prediction - truth) / args.pte_gamma) ** 2).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite PTE loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu())
            n_batches += 1
        val_result = evaluate(model, val, target, user_bank, service_bank,
                              args.eval_batch_size, device,
                              record_rt_boundary=args.qos_type == "rt")
        row = {"epoch": epoch, "train_loss": loss_total / n_batches,
               "val_mae": val_result["mae"], "val_rmse": val_result["rmse"]}
        rows.append(row)
        print("[PTE] epoch={} loss={:.6f} val_mae={:.6f}".format(
            epoch, row["train_loss"], row["val_mae"]), flush=True)
        if row["val_mae"] < best_val:
            best_val = row["val_mae"]
            best_state = {key: val.detach().cpu().clone()
                          for key, val in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.pte_patience:
                break
    train_sec = time.time() - start_train
    model.load_state_dict(best_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_train_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        peak_train_mb = None
    if args.test_eval_policy == "final_only":
        start_test = time.time()
        test_result = evaluate(model, test, target, user_bank, service_bank,
                               args.eval_batch_size, device,
                               record_rt_boundary=args.qos_type == "rt")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_test_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        else:
            peak_test_mb = None
        test_sec = time.time() - start_test
    else:
        test_result = {"mae": None, "rmse": None,
                       "rt_20_n": None, "rt_20_mae": None}
        peak_test_mb, test_sec = None, 0.0
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return rows, {"best_val_mae": best_val, "test_mae": test_result["mae"],
                  "test_rmse": test_result["rmse"],
                  "test_rt_20_n": test_result["rt_20_n"],
                  "test_rt_20_mae": test_result["rt_20_mae"],
                  "pte_epochs_ran": len(rows), "pte_parameter_count": parameters,
                  "pte_train_time_sec": train_sec, "final_test_time_sec": test_sec,
                  "peak_train_cuda_mb": peak_train_mb,
                  "peak_test_cuda_mb": peak_test_mb}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qos_type", choices=("rt", "tp"), required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--target_slice", type=int, default=64)
    parser.add_argument("--data_dir", default="datasets/wsdream_dataset2/dataset2")
    parser.add_argument("--cache_dir", default="cache/tpmcf_qaph_protocol")
    parser.add_argument("--spatial_cache_dir", default="cache/tpmcf_paper_spatial")
    parser.add_argument("--output_dir", default="log/fair_baseline_fixed64/tpmcf_paper_aligned")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default="pilot")
    parser.add_argument("--feature_steps", type=int, default=8)
    parser.add_argument("--svd_rank", type=int, default=100)
    parser.add_argument("--collab_dim", type=int, default=50)
    parser.add_argument("--ae_epochs", type=int, default=30)
    parser.add_argument("--ae_batch_size", type=int, default=256)
    parser.add_argument("--gcn_dim", type=int, default=64)
    parser.add_argument("--gcn_epochs", type=int, default=100)
    parser.add_argument("--gcn_patience", type=int, default=20)
    parser.add_argument("--gcn_gamma", type=float, default=1.0)
    parser.add_argument("--pte_blocks", type=int, default=4)
    parser.add_argument("--pte_heads", type=int, default=4)
    parser.add_argument("--conv_filters", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--pte_epochs", type=int, default=200)
    parser.add_argument("--pte_patience", type=int, default=10)
    parser.add_argument("--pte_lr", type=float, default=1e-4)
    parser.add_argument("--pte_gamma", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--test_eval_policy", choices=("final_only", "skip"),
                        default="final_only")
    args = parser.parse_args()
    if not (0 < args.density < 1 and 1 <= args.feature_steps <= 63 and
            1 <= args.svd_rank < N_USERS and args.collab_dim > 0 and
            args.ae_epochs > 0 and args.gcn_epochs > 0 and args.pte_epochs > 0 and
            args.gcn_gamma > 0 and args.pte_gamma > 0 and
            args.pte_heads > 0 and args.train_batch_size > 0 and
            args.eval_batch_size > 0):
        parser.error("invalid dimensions, budgets, or loss scales")
    return args


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for this TPMCF adaptation")
    device = torch.device(args.device)
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is unavailable")
    set_seed(args.seed)
    output_dir = (ROOT / args.output_dir).resolve()
    data_dir = (ROOT / args.data_dir).resolve()
    cache_dir = (ROOT / args.cache_dir).resolve()
    spatial_cache_dir = (ROOT / args.spatial_cache_dir).resolve()
    mask_path = (ROOT / args.mask).resolve(strict=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("result_{}_s{}_{}.json".format(args.qos_type, args.seed, args.tag))
    if result_path.exists():
        raise FileExistsError(str(result_path))
    start = time.time()
    tensor = load_tensor(args.qos_type, data_dir, cache_dir)
    split = load_shared_split(mask_path, tensor, args.qos_type, args.target_slice,
                              args.density, args.seed, args.seed)
    print("[Split] verified={} train={} val={} test={}".format(
        split["mask_sha256"], len(split["train_indices"]),
        len(split["val_indices"]), len(split["test_indices"])), flush=True)
    start_features = time.time()
    spatial_bank, spatial_rows, spatial_cache_path, spatial_cache_hit = \
        load_or_build_spatial_bank(split, args, device, spatial_cache_dir)
    user_bank = torch.from_numpy(spatial_bank[:, :N_USERS, :]).to(device)
    service_bank = torch.from_numpy(spatial_bank[:, N_USERS:, :]).to(device)
    spatial_time = time.time() - start_features
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        spatial_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        spatial_peak_mb = None
    rows, pte = train_pte(split, user_bank, service_bank, args, device)
    spatial_parameter_count = sum(
        row["user_ae_parameter_count"] + row["service_ae_parameter_count"] +
        row["graph_parameter_count"] for row in spatial_rows
    )
    result = {
        "method": "TPMCF-paper-aligned-QAPH-adaptation",
        "status": "smoke_only" if "smoke" in args.tag else "candidate_not_yet_certified",
        "qos_type": args.qos_type, "target_slice": args.target_slice,
        "history_slices": [args.target_slice - args.feature_steps, args.target_slice - 1],
        "density": args.density, "seed": args.seed, "split_id": args.seed,
        "n_train": len(split["train_indices"]), "n_val": len(split["val_indices"]),
        "n_test": len(split["test_indices"]),
        "shared_split_sha256": split["mask_sha256"],
        "shared_split_metadata": split["metadata"],
        "source_sha256": {
            "runner": sha256_file(Path(__file__)),
            "model": sha256_file(ROOT / "baselines" / "tpmcf" / "model.py"),
            "features": sha256_file(ROOT / "baselines" / "tpmcf" / "features.py"),
            "split": sha256_file(ROOT / "shared" / "splits.py"),
        },
        "config": {key: value for key, value in vars(args).items()
                   if key not in ("mask", "data_dir", "cache_dir", "spatial_cache_dir",
                                  "output_dir")},
        "adaptation": {
            "spatial_input": "last 8 strictly historical slices; target labels never enter graph or features",
            "target_labels": "PTE train/validation/final-test roles from verified common mask",
            "outlier_policy": "no IsolationForest filtering; all positive targets retained, including RT=20",
            "feature_types": "five statistics + truncated SVD factors + cosine autoencoder",
            "graph": "two normalized bipartite graph-convolution layers, concatenated, Cauchy reconstruction",
            "pte": "shared multihead attention, Conv1D residual blocks, temporal max pool, two FC layers",
            "loss": "raw-unit Cauchy log1p(((prediction-target)/gamma)^2)",
            "selection": "minimum validation MAE; test evaluated once after training",
            "prediction_clip": "zero lower bound at evaluation",
        },
        "spatial_stage": spatial_rows,
        "spatial_parameter_count": spatial_parameter_count,
        "total_parameter_count": spatial_parameter_count + pte["pte_parameter_count"],
        "spatial_time_sec": spatial_time,
        "spatial_cache_path": spatial_cache_path,
        "spatial_cache_hit": spatial_cache_hit,
        "spatial_peak_cuda_mb": spatial_peak_mb,
        "total_time_sec": time.time() - start,
        **pte,
    }
    history_path = output_dir / ("history_{}_s{}_{}.csv".format(args.qos_type, args.seed, args.tag))
    pd.DataFrame(rows).to_csv(str(history_path), index=False)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.test_eval_policy == "final_only":
        print("[Result] MAE={:.6f} RMSE={:.6f} file={}".format(
            result["test_mae"], result["test_rmse"], result_path), flush=True)
    else:
        print("[Result] test=skipped best_val_mae={:.6f} file={}".format(
            result["best_val_mae"], result_path), flush=True)


if __name__ == "__main__":
    main()

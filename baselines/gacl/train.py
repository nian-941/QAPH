"""Paper-reproduced GACL on QAPH's frozen history-only target-slice split.

This is an independent implementation of the published equations, with an
explicit neighbor-sampling approximation for the dense shared historical graph.
It is not author code and does not reproduce the paper's historical sampling.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
from shared.splits import load_shared_split  # noqa: E402
from baselines.gacl.graph import GRAPH_KEYS, load_or_build_window  # noqa: E402
from baselines.gacl.model import GACLPaperReproduction  # noqa: E402

N_USERS, N_SERVICES, N_TIMESTEPS = 142, 4500, 64


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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
        if not raw.is_file():
            raise FileNotFoundError(str(raw))
        tensor = np.zeros((N_USERS, N_SERVICES, N_TIMESTEPS), dtype=np.float32)
        for frame in pd.read_csv(
                str(raw), sep=r"\s+", header=None,
                names=["user", "service", "time", "value"],
                dtype={"user": np.int32, "service": np.int32,
                       "time": np.int32, "value": np.float32},
                chunksize=1_000_000):
            user = frame.user.to_numpy()
            service = frame.service.to_numpy()
            moment = frame.time.to_numpy()
            if (np.any(user < 0) or np.any(user >= N_USERS) or
                    np.any(service < 0) or np.any(service >= N_SERVICES) or
                    np.any(moment < 0) or np.any(moment >= N_TIMESTEPS)):
                raise ValueError("raw tensor index outside WS-DREAM2 dimensions")
            tensor[user, service, moment] = frame.value.to_numpy()
        np.save(str(cache), tensor)
        print("[Data] created cache={}".format(cache), flush=True)
    if tensor.shape != (N_USERS, N_SERVICES, N_TIMESTEPS) or tensor.dtype != np.float32:
        raise ValueError("unexpected tensor shape or dtype")
    return tensor


def to_device_graph(sampled, device):
    result = {}
    for name in GRAPH_KEYS:
        values = np.ascontiguousarray(sampled[name])
        result[name] = torch.from_numpy(values).to(device)
    return result


def evaluate(model, indices, target, graph, batch_size, device, qos):
    model.eval()
    absolute = squared = 0.0
    at_20_absolute = at_20_count = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            user = torch.from_numpy(batch[:, 0].astype(np.int64)).to(device)
            service = torch.from_numpy(batch[:, 1].astype(np.int64)).to(device)
            actual = np.asarray(target[batch[:, 0], batch[:, 1]], dtype=np.float64)
            predicted = model(user, service, graph).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(predicted).all():
                raise RuntimeError("nonfinite model predictions")
            error = predicted - actual
            absolute += float(np.abs(error).sum())
            squared += float(np.square(error).sum())
            if qos == "rt":
                boundary = actual == 20.0
                at_20_count += int(boundary.sum())
                at_20_absolute += float(np.abs(error[boundary]).sum())
    return {
        "mae": absolute / len(indices),
        "rmse": math.sqrt(squared / len(indices)),
        "rt_20_n": at_20_count if qos == "rt" else None,
        "rt_20_mae": at_20_absolute / at_20_count
                     if qos == "rt" and at_20_count else None,
    }


def train(split, model, graph, args, device):
    target = split["target"]
    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    if args.train_limit:
        train_indices = train_indices[:args.train_limit]
    if args.val_limit:
        val_indices = val_indices[:args.val_limit]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = math.inf
    best_epoch = 0
    best_state = None
    waits = 0
    records = []
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        order = np.random.RandomState(args.seed + epoch + 1701).permutation(len(train_indices))
        total_loss = 0.0
        total_seen = 0
        for start in range(0, len(order), args.batch_size):
            batch = train_indices[order[start:start + args.batch_size]]
            user = torch.from_numpy(batch[:, 0].astype(np.int64)).to(device)
            service = torch.from_numpy(batch[:, 1].astype(np.int64)).to(device)
            actual = torch.from_numpy(np.asarray(
                target[batch[:, 0], batch[:, 1]], dtype=np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(user, service, graph)
            loss = torch.mean((predicted - actual) ** 2)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            total_seen += len(batch)
        scheduler.step()
        validation = evaluate(model, val_indices, target, graph,
                              args.eval_batch_size, device, args.qos_type)
        value = validation["mae"]
        if value < best_val - 1e-6:
            best_val = value
            best_epoch = epoch + 1
            best_state = {key: tensor.detach().cpu().clone()
                          for key, tensor in model.state_dict().items()}
            waits = 0
        else:
            waits += 1
        record = {
            "epoch": epoch + 1,
            "train_mse": total_loss / total_seen,
            "val_mae": value,
            "val_rmse": validation["rmse"],
            "time_sec": time.time() - epoch_start,
        }
        records.append(record)
        print("[Epoch] {}/{} train_mse={:.6f} val_mae={:.6f} val_rmse={:.6f} time={:.2f}s".format(
            epoch + 1, args.epochs, record["train_mse"], value,
            validation["rmse"], record["time_sec"]), flush=True)
        if waits >= args.patience:
            print("[EarlyStop] patience={} best_epoch={}".format(
                args.patience, best_epoch), flush=True)
            break
    if best_state is None:
        raise RuntimeError("no validation-selected model")
    model.load_state_dict(best_state)
    model.to(device)
    return records, best_val, best_epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qos_type", choices=("rt", "tp"), required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target_slice", type=int, default=64)
    parser.add_argument("--density", type=float, default=0.1)
    parser.add_argument("--data_dir", default="datasets/wsdream_dataset2/dataset2")
    parser.add_argument("--cache_dir", default="cache/tpmcf_qaph_protocol")
    parser.add_argument("--graph_cache_dir", default="cache/gacl_paper_graphs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--first_cap", type=int, default=32)
    parser.add_argument("--second_cap", type=int, default=8)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--graph_layers", type=int, default=2)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_eval_policy", choices=("skip", "final_only"),
                        default="final_only")
    parser.add_argument("--reuse_result", default="",
                        help="reuse a validated tuning checkpoint for final seed-42 evaluation")
    args = parser.parse_args()
    if (args.seed < 0 or args.window < 1 or args.first_cap < 1 or
            args.second_cap < 1 or args.second_cap > args.first_cap or
            args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or
            args.eval_batch_size < 1 or args.lr <= 0 or args.weight_decay < 0 or
            args.grad_clip < 0 or args.density <= 0 or args.density >= 1 or
            args.train_limit < 0 or args.val_limit < 0):
        parser.error("invalid numeric experiment configuration")
    if (args.train_limit or args.val_limit) and args.test_eval_policy != "skip":
        parser.error("limited smoke runs must skip test evaluation")
    if args.reuse_result and args.test_eval_policy != "final_only":
        parser.error("a reused tuning checkpoint is only for final evaluation")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    data_dir = (ROOT / args.data_dir).resolve()
    cache_dir = (ROOT / args.cache_dir).resolve()
    graph_cache_dir = (ROOT / args.graph_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("result_{}_s{}_{}.json".format(
        args.qos_type, args.seed, args.tag))
    if result_path.exists():
        raise FileExistsError(str(result_path))
    start = time.time()
    tensor = load_tensor(args.qos_type, data_dir, cache_dir)
    split = load_shared_split(args.mask, tensor, args.qos_type,
                              args.target_slice, args.density,
                              args.seed, args.seed)
    history_window = split["history"][:, :, -args.window:]
    positive = np.asarray(history_window)[np.asarray(history_window) > 0]
    if not len(positive):
        raise ValueError("no positive historical QoS values")
    edge_scale = float(np.quantile(positive, 0.95))
    if edge_scale <= 0 or not math.isfinite(edge_scale):
        raise ValueError("invalid history-only edge scale")
    sampled, graph_path, graph_hit = load_or_build_window(
        split["history"], split["metadata"]["history_sha256"],
        args.qos_type, args.window, args.first_cap, args.second_cap,
        args.seed, graph_cache_dir)
    graph = to_device_graph(sampled, device)
    print("[Graph] window={} slices={}..{} first_cap={} second_cap={} edge_scale_p95={:.6f} cache_hit={}".format(
        args.window, args.target_slice - args.window, args.target_slice - 1,
        args.first_cap, args.second_cap, edge_scale, graph_hit), flush=True)
    model = GACLPaperReproduction(
        N_USERS, N_SERVICES, args.dim, args.window, args.graph_layers,
        args.transformer_layers, args.heads, edge_scale).to(device)
    sampled_train = split["train_indices"]
    if args.train_limit:
        sampled_train = sampled_train[:args.train_limit]
    initial_values = split["target"][sampled_train[:, 0], sampled_train[:, 1]]
    model.output.bias.data.fill_(float(np.median(initial_values)))
    n_parameters = sum(parameter.numel() for parameter in model.parameters())
    print("[Model] params={} train={} val={} test={} device={}".format(
        n_parameters, len(split["train_indices"]), len(split["val_indices"]),
        len(split["test_indices"]), device), flush=True)
    source = {
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "baselines" / "gacl" / "model.py"),
        "graph": sha256_file(ROOT / "baselines" / "gacl" / "graph.py"),
        "split": sha256_file(ROOT / "shared" / "splits.py"),
    }
    preparation_time = time.time() - start
    reused_result = None
    if args.reuse_result:
        prior_path = Path(args.reuse_result).resolve(strict=True)
        prior = json.loads(prior_path.read_text(encoding="utf-8-sig"))
        prior_config = prior["config"]
        frozen_keys = (
            "qos_type", "seed", "target_slice", "density", "device", "window",
            "first_cap", "second_cap", "dim", "graph_layers",
            "transformer_layers", "heads", "epochs", "patience", "lr",
            "weight_decay", "batch_size", "eval_batch_size", "grad_clip",
            "train_limit", "val_limit",
        )
        if (prior["source_sha256"] != source or
                prior["shared_split_sha256"] != split["mask_sha256"] or
                prior["status"] != "candidate_not_yet_certified" or
                prior_config["test_eval_policy"] != "skip" or
                prior["test_mae"] is not None or
                any(prior_config[key] != getattr(args, key) for key in frozen_keys) or
                not math.isclose(prior["edge_scale_history_p95"], edge_scale,
                                 rel_tol=0, abs_tol=1e-8)):
            raise ValueError("tuning result is not reusable under this final protocol")
        old_checkpoint = prior_path.parent / ("best_{}_s{}_{}.pth".format(
            args.qos_type, args.seed, prior_config["tag"]))
        if sha256_file(old_checkpoint) != prior["checkpoint_sha256"]:
            raise ValueError("tuning checkpoint SHA mismatch")
        model.load_state_dict(torch.load(str(old_checkpoint), map_location=device))
        records = prior["history"]
        best_val = float(prior["best_val_mae"])
        best_epoch = int(prior["best_epoch"])
        train_time = 0.0
        reused_result = str(prior_path)
        print("[Reuse] validated tuning checkpoint={}".format(old_checkpoint),
              flush=True)
    else:
        tick = time.time()
        records, best_val, best_epoch = train(split, model, graph, args, device)
        train_time = time.time() - tick
    test = None
    test_time = 0.0
    if args.test_eval_policy == "final_only":
        tick = time.time()
        test = evaluate(model, split["test_indices"], split["target"], graph,
                        args.eval_batch_size, device, args.qos_type)
        test_time = time.time() - tick
    checkpoint_path = output_dir / ("best_{}_s{}_{}.pth".format(
        args.qos_type, args.seed, args.tag))
    torch.save(model.state_dict(), str(checkpoint_path))
    config = {name: value for name, value in vars(args).items()
              if name not in ("mask", "data_dir", "cache_dir", "graph_cache_dir",
                              "output_dir")}
    result = {
        "method": "GACL-paper-reproduction-history-only-neighbor-sampled",
        "status": "smoke_only" if args.train_limit or args.val_limit
                  else "candidate_not_yet_certified",
        "qos_type": args.qos_type,
        "target_slice": args.target_slice,
        "history_slices": [args.target_slice - args.window,
                           args.target_slice - 1],
        "density": args.density,
        "seed": args.seed,
        "split_id": args.seed,
        "n_train": len(split["train_indices"]),
        "n_val": len(split["val_indices"]),
        "n_test": len(split["test_indices"]),
        "shared_split_sha256": split["mask_sha256"],
        "shared_split_metadata": split["metadata"],
        "source_sha256": source,
        "config": config,
        "adaptation": {
            "target_slice_graph_used": False,
            "target_labels_as_supervision_only": True,
            "history_sampling": "full_positive_history_before_bounded_neighbor_sampling",
            "neighbor_sampling": "fixed_without_replacement_per_seed_and_slice",
            "norm_operator": "L2_normalize_eps_1e-6",
            "edge_weight_scale": "positive_history_window_p95",
            "training_loss": "raw_QoS_MSE",
            "selection_metric": "validation_MAE",
            "output_activation": "ReLU",
            "first_layer_weight_sharing": "same_weights_user_and_service_directions",
            "gradient_clipping": args.grad_clip,
        },
        "graph_cache_path": graph_path,
        "graph_cache_hit": graph_hit,
        "edge_scale_history_p95": edge_scale,
        "parameter_count": n_parameters,
        "preparation_time_sec": preparation_time,
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "total_time_sec": time.time() - start,
        "best_epoch": best_epoch,
        "epochs_ran": len(records),
        "best_val_mae": best_val,
        "test_mae": test["mae"] if test else None,
        "test_rmse": test["rmse"] if test else None,
        "test_rt_20_n": test["rt_20_n"] if test else None,
        "test_rt_20_mae": test["rt_20_mae"] if test else None,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reused_training_result": reused_result,
        "peak_cuda_memory_mb": (torch.cuda.max_memory_allocated(device) / 1048576.0
                                if device.type == "cuda" else None),
        "history": records,
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    if test:
        print("[Result] val_mae={:.6f} test_mae={:.6f} test_rmse={:.6f} file={}".format(
            best_val, test["mae"], test["rmse"], result_path), flush=True)
    else:
        print("[Result] val_mae={:.6f} test=skipped file={}".format(
            best_val, result_path), flush=True)


if __name__ == "__main__":
    main()

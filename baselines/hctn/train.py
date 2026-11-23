"""Independent HCTN paper reproduction on the frozen QAPH target-slice split."""

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
from baselines.hctn.features import load_or_build  # noqa: E402
from baselines.hctn.model import HCTNPaperReproduction, graph_operators  # noqa: E402

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


def as_inputs(features, device):
    gpam = torch.from_numpy(features["gpam"]).to(device)
    stats = torch.from_numpy(features["statistics"]).to(device)
    greysheep = torch.from_numpy(features["greysheep"]).to(device)
    graphs = [graph_operators(torch.from_numpy(frame).to(device))
              for frame in features["incidence"]]
    return gpam, graphs, stats, greysheep


def evaluate_predictions(predicted, indices, target, qos):
    row, col = indices[:, 0], indices[:, 1]
    actual = np.asarray(target[row, col], dtype=np.float64)
    prediction = predicted[row, col].astype(np.float64)
    if not np.isfinite(prediction).all():
        raise RuntimeError("nonfinite HCTN predictions")
    errors = prediction - actual
    at_20 = actual == 20.0 if qos == "rt" else None
    return {
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.square(errors).mean())),
        "rt_20_n": int(at_20.sum()) if qos == "rt" else None,
        "rt_20_mae": float(np.abs(errors[at_20]).mean())
                     if qos == "rt" and at_20.any() else None,
    }


def predict(model, inputs, scale):
    model.eval()
    with torch.no_grad():
        result = model(*inputs) * scale
    return result.detach().cpu().numpy()


def train(model, inputs, split, scale, args, device):
    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    if args.train_limit:
        train_indices = train_indices[:args.train_limit]
    if args.val_limit:
        val_indices = val_indices[:args.val_limit]
    train_rows = torch.from_numpy(train_indices[:, 0].copy()).long().to(device)
    train_cols = torch.from_numpy(train_indices[:, 1].copy()).long().to(device)
    truth = torch.from_numpy(np.asarray(
        split["target"][train_indices[:, 0], train_indices[:, 1]],
        dtype=np.float32)).to(device) / scale
    gamma = args.gamma / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_state, best_val, best_epoch, waits = None, math.inf, 0, 0
    records = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        matrix = model(*inputs)
        residual = (matrix[train_rows, train_cols] - truth) / gamma
        loss = torch.log1p(residual.square()).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("nonfinite Cauchy training loss")
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        val = evaluate_predictions(predict(model, inputs, scale), val_indices,
                                   split["target"], args.qos_type)
        if val["mae"] < best_val - 1e-6:
            best_val = val["mae"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
            waits = 0
        else:
            waits += 1
        record = {"epoch": epoch, "train_cauchy": float(loss.detach().cpu()),
                  "val_mae": val["mae"], "val_rmse": val["rmse"],
                  "time_sec": time.time() - start}
        records.append(record)
        print("[Epoch] {}/{} loss={:.6f} val_mae={:.6f} time={:.2f}s".format(
            epoch, args.epochs, record["train_cauchy"], val["mae"],
            record["time_sec"]), flush=True)
        if waits >= args.patience:
            print("[EarlyStop] best_epoch={}".format(best_epoch), flush=True)
            break
    if best_state is None:
        raise RuntimeError("no validation-selected checkpoint")
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
    parser.add_argument("--feature_cache_dir", default="cache/hctn_paper_features")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--nmf_iterations", type=int, default=100)
    parser.add_argument("--nmf_seed", type=int, default=42)
    parser.add_argument("--c_user", type=float, default=1.0)
    parser.add_argument("--c_service", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--output_width", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=0)
    parser.add_argument("--test_eval_policy", choices=("skip", "final_only"),
                        default="final_only")
    parser.add_argument("--reuse_result", default="")
    args = parser.parse_args()
    if (args.seed < 0 or args.target_slice != 64 or args.window not in (2, 4, 6, 8, 10, 12, 14) or
            args.rank < 1 or args.nmf_iterations < 1 or args.nmf_seed < 0 or
            args.width < 4 or args.width % 4 or args.layers < 1 or
            args.heads < 1 or (args.width // 4) % args.heads or
            args.output_width < 1 or not 0 <= args.dropout < 1 or
            args.gamma <= 0 or args.epochs < 1 or args.patience < 1 or
            args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0 or
            args.train_limit < 0 or args.val_limit < 0 or
            not 0 < args.density < 1):
        parser.error("invalid numeric HCTN configuration")
    if (args.train_limit or args.val_limit) and args.test_eval_policy != "skip":
        parser.error("limited smoke runs must skip the test set")
    if args.reuse_result and args.test_eval_policy != "final_only":
        parser.error("reused tuning checkpoints are for final evaluation only")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / ("result_{}_s{}_{}.json".format(
        args.qos_type, args.seed, args.tag))
    if result_path.exists():
        raise FileExistsError(str(result_path))
    start = time.time()
    tensor = load_tensor(args.qos_type, (ROOT / args.data_dir).resolve(),
                         (ROOT / args.cache_dir).resolve())
    split = load_shared_split(args.mask, tensor, args.qos_type,
                              args.target_slice, args.density,
                              args.seed, args.seed)
    features, scale, feature_path, cache_hit = load_or_build(
        split["history"], split["metadata"]["history_sha256"],
        args.qos_type, args.window, args.rank, args.nmf_iterations,
        args.c_user, args.c_service, args.nmf_seed,
        (ROOT / args.feature_cache_dir).resolve(), device)
    inputs = as_inputs(features, device)
    print("[Feature] window={} slices={}..{} rank={} cache_hit={} scale={:.6f}".format(
        args.window, args.target_slice - args.window,
        args.target_slice - 1, args.rank, cache_hit, scale), flush=True)
    model = HCTNPaperReproduction(
        N_USERS, N_SERVICES, args.rank, args.width, args.window, args.layers,
        args.heads, args.output_width, args.dropout).to(device)
    parameter_count = sum(item.numel() for item in model.parameters())
    source = {
        "runner": sha256_file(Path(__file__)),
        "model": sha256_file(ROOT / "baselines" / "hctn" / "model.py"),
        "features": sha256_file(ROOT / "baselines" / "hctn" / "features.py"),
        "split": sha256_file(ROOT / "shared" / "splits.py"),
    }
    print("[Model] params={} train={} val={} test={} device={}".format(
        parameter_count, len(split["train_indices"]),
        len(split["val_indices"]), len(split["test_indices"]), device),
        flush=True)
    preparation_time = time.time() - start
    reused_result = None
    if args.reuse_result:
        prior_path = Path(args.reuse_result).resolve(strict=True)
        prior = json.loads(prior_path.read_text(encoding="utf-8-sig"))
        frozen = ("qos_type", "seed", "target_slice", "density", "device",
                  "window", "rank", "nmf_iterations", "nmf_seed", "c_user",
                  "c_service", "width", "layers", "heads", "output_width",
                  "dropout", "gamma", "epochs", "patience", "lr",
                  "weight_decay", "grad_clip", "train_limit", "val_limit")
        if (prior["source_sha256"] != source or
                prior["shared_split_sha256"] != split["mask_sha256"] or
                prior["status"] != "candidate_not_yet_certified" or
                prior["config"]["test_eval_policy"] != "skip" or
                prior["test_mae"] is not None or
                any(prior["config"][key] != getattr(args, key) for key in frozen) or
                not math.isclose(prior["history_scale_p95"], scale,
                                 rel_tol=0, abs_tol=1e-6)):
            raise ValueError("tuning result cannot be reused under final protocol")
        old_checkpoint = prior_path.parent / ("best_{}_s{}_{}.pth".format(
            args.qos_type, args.seed, prior["config"]["tag"]))
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
        records, best_val, best_epoch = train(model, inputs, split, scale,
                                              args, device)
        train_time = time.time() - tick
    test = None
    test_time = 0.0
    if args.test_eval_policy == "final_only":
        tick = time.time()
        test = evaluate_predictions(predict(model, inputs, scale),
                                    split["test_indices"], split["target"],
                                    args.qos_type)
        test_time = time.time() - tick
    checkpoint_path = output_dir / ("best_{}_s{}_{}.pth".format(
        args.qos_type, args.seed, args.tag))
    torch.save(model.state_dict(), str(checkpoint_path))
    config = {name: value for name, value in vars(args).items()
              if name not in ("mask", "data_dir", "cache_dir",
                              "feature_cache_dir", "output_dir")}
    result = {
        "method": "HCTN-paper-reproduction-history-only-factored-hypergraph",
        "status": "smoke_only" if args.train_limit or args.val_limit
                  else "candidate_not_yet_certified",
        "qos_type": args.qos_type,
        "target_slice": args.target_slice,
        "history_slices": [args.target_slice - args.window,
                           args.target_slice - 1],
        "density": args.density, "seed": args.seed, "split_id": args.seed,
        "n_train": len(split["train_indices"]),
        "n_val": len(split["val_indices"]),
        "n_test": len(split["test_indices"]),
        "shared_split_sha256": split["mask_sha256"],
        "shared_split_metadata": split["metadata"],
        "source_sha256": source, "config": config,
        "adaptation": {
            "target_slice_graph_used": False,
            "target_labels_as_supervision_only": True,
            "nmf_missing_value": "zero_fill_positive_history_only",
            "nmf_training": "deterministic_multiplicative_warm_start_history_only",
            "hypergraph": "exact_incidence_factored_no_neighbor_sampling",
            "grey_sheep": "historical_GDI_per_slice",
            "history_scaling": "positive_window_p95",
            "training_loss": "Cauchy_log1p_squared_residual_gamma_raw_units",
            "selection_metric": "validation_MAE_raw_units",
            "output_activation": "unbounded_inner_product",
            "outlier_filter": "disabled_for_shared_positive_mask",
        },
        "feature_cache_path": feature_path,
        "feature_cache_hit": cache_hit,
        "history_scale_p95": scale,
        "parameter_count": parameter_count,
        "preparation_time_sec": preparation_time,
        "train_time_sec": train_time,
        "test_time_sec": test_time,
        "total_time_sec": time.time() - start,
        "best_epoch": best_epoch, "epochs_ran": len(records),
        "best_val_mae": best_val,
        "test_mae": test["mae"] if test else None,
        "test_rmse": test["rmse"] if test else None,
        "test_rt_20_n": test["rt_20_n"] if test else None,
        "test_rt_20_mae": test["rt_20_mae"] if test else None,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "reused_training_result": reused_result,
        "peak_cuda_memory_mb":
            (torch.cuda.max_memory_allocated(device) / 1048576.0
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

"""Validation-selected training using the archived baseline algorithm kernels."""
import copy
from pathlib import Path
import numpy as np
import torch
from baselines._shared import tensor_factorization as algorithms
from shared.runtime import array_sha, atomic_json

def train(args, tensor, split):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ix = split['train_indices']
    val = split['val_indices']
    test = split['test_indices']
    history = np.asarray(split['history'])
    target = split['target']
    y = target[ix[:, 0], ix[:, 1]]
    yv = target[val[:, 0], val[:, 1]]
    config = dict(method=args.method, rank=args.rank, regularization=args.regularization, learning_rate=args.lr, max_epochs=args.epochs, patience=args.patience, alpha=args.alpha, gamma=args.gamma)
    curve = []
    best = float('inf')
    best_epoch = 0
    if args.method in ('wspred', 'nntf'):
        values = np.zeros(tensor.shape, dtype=np.float64)
        values[:, :, :history.shape[2]] = history
        values[ix[:, 0], ix[:, 1], -1] = y
        data = torch.as_tensor(values, device=args.device, dtype=torch.float64)
        with torch.no_grad():
            factors = algorithms.initialize(data.shape, config, args.seed, args.device)
            chosen = None
            for epoch in range(1, args.epochs + 1):
                factors, _ = algorithms.factor_step(data, factors, config)
                if not all((torch.isfinite(f).all().item() for f in factors)):
                    raise RuntimeError('Nonfinite factors')
                pred = algorithms.target_matrix(factors).cpu().numpy()
                loss = float(np.abs(pred[val[:, 0], val[:, 1]] - yv).mean())
                curve.append(dict(epoch=epoch, val_mae=loss))
                if loss < best:
                    best = loss
                    best_epoch = epoch
                    chosen = [f.clone() for f in factors]
                if epoch - best_epoch >= args.patience:
                    break
            if chosen is None:
                raise RuntimeError('No valid validation checkpoint')
            pred = algorithms.target_matrix(chosen).cpu().numpy()
            np.savez_compressed(str(out / 'best_factors.npz'), **{k: v.cpu().numpy() for k, v in zip(('user', 'service', 'time'), chosen)})
    elif args.method in ('bnlft', 'ctf'):
        from baselines._shared import history as history_stage, target as target_stage
        hconfig = dict(method=args.method, rank=args.rank, regularization=args.regularization, epochs=args.history_epochs, history_length=history.shape[2], qos=args.qos)
        tconfig = dict(method=args.method, rank=args.rank, regularization=args.regularization, epochs=args.epochs)
        if args.method == 'bnlft':
            hconfig['bias_regularization'] = tconfig['bias_regularization'] = args.bias_regularization
        else:
            hconfig['gamma'] = tconfig['gamma'] = args.gamma
        state, hcurve = history_stage.fit_history(history, hconfig, args.seed, args.device)
        chosen = [None]
        validation = []

        def observe(epoch, weights, row):
            score = float(np.abs(target_stage.predict(weights, val).cpu().numpy() - yv).mean())
            validation.append(dict(epoch=epoch, val_mae=score))
            if len(validation) == 1 or score < min((v['val_mae'] for v in validation[:-1])):
                chosen[0] = weights
        target_stage.fit_target(state, ix, y, tconfig, args.device, observe)
        best_row = min(validation, key=lambda r: r['val_mae'])
        best = best_row['val_mae']
        best_epoch = best_row['epoch']
        curve = validation
        pred = target_stage.predict(chosen[0]).cpu().numpy()
        np.savez_compressed(str(out / 'best_factors.npz'), **{k: v.cpu().numpy() for k, v in chosen[0].items()})
        atomic_json(out / 'history_curve.json', hcurve)
    elif args.method == 'trcf':
        from baselines.trcf import model as history_adapter
        components = history_adapter.build_history_components(history, rbs_threshold=args.rbs_threshold, pcc_threshold=args.pcc_threshold, target_slice=tensor.shape[2], neighbor_block=args.neighbor_block, device=args.device)
        mixture, candidates = history_adapter.fit_mixture(components, ix, y)
        pred = history_adapter.predict(components, mixture).cpu().numpy()
        best = float(np.abs(pred[val[:, 0], val[:, 1]] - yv).mean())
        atomic_json(out / 'mixture.json', dict(alpha=mixture.alpha, training_candidates=candidates))
    elif args.method == 'gmcl':
        from baselines.gmcl import model as gmcl
        values = np.zeros(tensor.shape, dtype=np.float32)
        values[:, :, :history.shape[2]] = history
        values[ix[:, 0], ix[:, 1], -1] = y
        config.update(dropout=args.dropout, layers=args.layers, node_dropout=args.node_dropout, temperature=args.temperature, beta=args.beta, evd_rank=args.evd_rank, batch_size=args.batch_size)
        atomic_json(out / 'config.json', dict(bundle_id='public-source-release'))
        data = torch.as_tensor(values, device=args.device)

        def progress(stage, index, total, **kw):
            print(stage, index, total, flush=True)
        pred, curve, best, best_epoch, _ = gmcl.fit(data, ix, val, yv.astype(np.float64), config, args.seed, out, out, array_sha(values), progress)
    else:
        raise ValueError(args.method)
    truth = target[test[:, 0], test[:, 1]].astype(np.float64)
    predictions = pred[test[:, 0], test[:, 1]].astype(np.float64)
    if not np.isfinite(predictions).all():
        raise RuntimeError('Nonfinite final predictions')
    error = predictions - truth
    np.savez_compressed(str(out / 'predictions.npz'), indices=test, y_true=truth, y_pred=predictions)
    atomic_json(out / 'epochs.json', curve)
    result = dict(method=args.method, qos=args.qos, mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error ** 2).mean())), best_val_mae=best, best_epoch=best_epoch, n_test=len(test), mask_sha256=split['mask_sha256'], test_passes=1)
    atomic_json(out / 'result.json', result)
    print(result)
    return result

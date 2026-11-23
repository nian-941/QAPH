"""Run one baseline using a verified tensor and shared train/val/test mask."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from shared.splits import load_shared_split
METHODS = ('wspred', 'nntf', 'bnlft', 'ctf', 'trcf', 'gmcl', 'hctn', 'tpmcf', 'gacl')
ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--tensor', type=Path, required=True)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--epochs', type=int, required=True, help='User-selected training budget; TRCF is noniterative')
    p.add_argument('--history-epochs', type=int, default=None)
    p.add_argument('--rank', type=int, default=16)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--regularization', type=float, default=0.001)
    p.add_argument('--bias-regularization', type=float, default=0.001)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--rbs-threshold', type=float, default=0.5)
    p.add_argument('--pcc-threshold', type=float, default=0.5)
    p.add_argument('--neighbor-block', type=int, default=64)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--node-dropout', type=float, default=0.1)
    p.add_argument('--temperature', type=float, default=0.2)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--evd-rank', type=int, default=16)
    args, extra = p.parse_known_args()
    if args.epochs < 1 or args.rank < 1:
        raise ValueError('Positive budget and rank required')
    args.history_epochs = args.history_epochs or args.epochs
    tensor = np.load(str(args.tensor), mmap_mode='r', allow_pickle=False)
    with np.load(str(args.mask), allow_pickle=False) as z:
        meta = json.loads(z['metadata'].item())
    args.qos = meta['qos_type']
    args.seed = int(meta['seed'])
    split = load_shared_split(args.mask, tensor, args.qos, int(meta['target_slice']), meta['density'], args.seed, int(meta['split_id']))
    if int(meta['target_slice']) != tensor.shape[2] or meta['history_start_slice'] != 1:
        raise ValueError('Baseline entry point expects full prefix history and the last tensor slot as target')
    out = Path(args.output_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    if args.method not in ('hctn', 'tpmcf', 'gacl'):
        if extra:
            raise ValueError('Unrecognized options: ' + str(extra))
        from baselines._shared.training import train
        train(args, tensor, split)
        return
    import importlib
    module = importlib.import_module('baselines.' + args.method + '.train')
    module.N_USERS, module.N_SERVICES, module.N_TIMESTEPS = tensor.shape
    module.load_tensor = lambda *a, **kw: tensor

    def save_predictions(indices, pred):
        truth = split['target'][indices[:, 0], indices[:, 1]]
        np.savez_compressed(str(out / 'predictions.npz'), indices=indices, y_true=truth, y_pred=pred)
    if args.method == 'hctn':
        evaluate = module.evaluate_predictions

        def capture(pred, indices, target, *a, **kw):
            result = evaluate(pred, indices, target, *a, **kw)
            if np.array_equal(indices, split['test_indices']):
                save_predictions(indices, pred[indices[:, 0], indices[:, 1]])
            return result
        module.evaluate_predictions = capture
    else:
        evaluate = module.evaluate

        def capture(model, indices, *a, **kw):
            if not np.array_equal(indices, split['test_indices']):
                return evaluate(model, indices, *a, **kw)
            parts = []

            def collect(m, i, o):
                pred = o.detach().cpu().numpy().reshape(-1)
                if args.method == 'tpmcf':
                    pred = np.maximum(pred, 0)
                parts.append(pred)
            hook = model.register_forward_hook(collect)
            try:
                result = evaluate(model, indices, *a, **kw)
            finally:
                hook.remove()
            save_predictions(indices, np.concatenate(parts))
            return result
        module.evaluate = capture
    argv = ['--qos_type', args.qos, '--mask', str(args.mask.resolve()), '--seed', str(args.seed), '--density', str(meta['density']), '--target_slice', str(meta['target_slice']), '--output_dir', str(out), '--device', args.device, '--tag', 'public', '--test_eval_policy', 'final_only', '--cache_dir', str(out / 'cache/raw')]
    if args.method == 'tpmcf':
        argv += ['--pte_epochs', str(args.epochs), '--pte_patience', str(args.patience), '--pte_lr', str(args.lr), '--pte_gamma', str(args.gamma), '--dropout', str(args.dropout), '--spatial_cache_dir', str(out / 'cache/spatial')]
    else:
        argv += ['--epochs', str(args.epochs), '--patience', str(args.patience), '--lr', str(args.lr)]
    if args.method == 'hctn':
        argv += ['--gamma', str(args.gamma), '--rank', str(args.rank), '--layers', str(args.layers), '--dropout', str(args.dropout), '--feature_cache_dir', str(out / 'cache/features')]
    if args.method == 'gacl':
        argv += ['--graph_cache_dir', str(out / 'cache/graphs')]
    sys.argv = [str(module.__file__)] + argv + extra
    module.main()
if __name__ == '__main__':
    main()

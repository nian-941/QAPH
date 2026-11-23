"""Portable entry point for the frozen v0 Full QAPH model."""
import sys
from pathlib import Path
import numpy as np

def main():
    from qaph.training import engine as native
    user = sys.argv[1:]
    if '--help' in user or '-h' in user:
        native.main()
        return
    architecture = ['--use_qaph', '--use_sega', '--use_mfvp', '--use_ptcs', '--no_bal']
    policy = ['--test_eval_policy', 'final_only', '--train_eval_lambdas', '0', '--final_eval_lambdas', '0']
    for flag in ('--test_eval_policy', '--train_eval_lambdas', '--final_eval_lambdas'):
        if flag in user:
            raise ValueError('This entry point fixes final-only lambda-0 evaluation; use the native runner for other policies')
    output = Path(user[user.index('--log_dir') + 1]) if '--log_dir' in user else Path('runs/qaph')
    datasets = {}
    for name in ('prepare_dataset_v42', 'prepare_dataset_v42_from_feature_bank'):
        fn = getattr(native, name)

        def dataset(*a, _fn=fn, _name=name, **kw):
            value = _fn(*a, **kw)
            label = a[0] if _name.endswith('from_feature_bank') and a else kw.get('split_name')
            if label in ('train', 'val', 'test'):
                datasets[label] = value
            return value
        setattr(native, name, dataset)
    evaluate = native.evaluate_multi_lambda

    def capture(model, loader, *a, **kw):
        if loader.dataset is not datasets.get('test'):
            return evaluate(model, loader, *a, **kw)
        parts = []
        hook = model.register_forward_hook(lambda m, i, o: parts.append(o['prediction'].detach().cpu().numpy().reshape(-1)))
        try:
            result = evaluate(model, loader, *a, **kw)
        finally:
            hook.remove()
        t = loader.dataset.tensors
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output / 'predictions.npz'), indices=np.column_stack((t[0].numpy(), t[1].numpy())), y_true=t[3].numpy(), y_pred=np.concatenate(parts))
        return result
    native.evaluate_multi_lambda = capture
    if '--log_dir' not in user:
        user += ['--log_dir', str(output)]
    sys.argv = [str(Path(native.__file__))] + [flag for flag in architecture if flag not in user] + policy + user
    native.main()
if __name__ == '__main__':
    main()

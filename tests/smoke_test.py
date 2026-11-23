"""Small synthetic integration checks; never reproduce benchmark results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def run(work):
    from shared.prepare_data import prepare
    rng = np.random.RandomState(71)
    tensor = rng.uniform(0.2, 12, (24, 32, 64)).astype(np.float32)
    tensor[:, :, :63][rng.rand(24, 32, 63) < 0.25] = 0
    tensor[0, 0, 63] = 20
    raw = work / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='-1', PYTORCH_JIT='0', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    checks = []

    def execute(name, argv):
        result = subprocess.run([sys.executable, '-X', 'utf8', '-B'] + argv, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, encoding='utf-8', timeout=300)
        (work / (name + '.out')).write_text(result.stdout, encoding='utf-8')
        (work / (name + '.err')).write_text(result.stderr, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(name + ' failed:\n' + result.stderr[-6000:] + '\n' + result.stdout[-1500:])

    def verify(name, folder, expected=None):
        with np.load(str(folder / 'predictions.npz'), allow_pickle=False) as z:
            ids = z['indices']
            truth = tensor[ids[:, 0], ids[:, 1], 63]
            assert np.array_equal(z['y_true'].reshape(-1), truth)
            error = z['y_pred'].reshape(-1).astype(np.float64) - truth.astype(np.float64)
            metrics = [float(np.abs(error).mean()), float(np.sqrt((error ** 2).mean()))]
            assert np.isfinite(metrics).all()
        if expected is not None:
            np.testing.assert_allclose(metrics, expected, atol=1e-05, rtol=1e-05)
        checks.append(dict(method=name, predictions_finite=True, truth_matches=True, n_test=len(ids), metric_replay_passed=True))
        print('PASS ' + name, flush=True)
    for q in ('rt', 'tp'):
        prepared = work / ('prepared_' + q)
        prepare(tensor, q, 0.3, 17, 64, prepared)
        with (raw / (q + 'data.txt')).open('w', encoding='utf-8') as f:
            for u, s, t in np.ndindex(tensor.shape):
                f.write('{} {} {} {:.9g}\n'.format(u, s, t, float(tensor[u, s, t])))
        folder = work / ('qaph_' + q)
        argv = ['-m', 'qaph.train', '--qos_type', q, '--data_dir', str(raw), '--target_slice', '64', '--history_length', '63', '--density', '.3', '--seed', '17', '--split_id', '17', '--gamma', '1', '--epochs', '2', '--batch_size', '64', '--device', 'cpu', '--log_dir', str(folder), '--stat_cache_dir', str(work / 'qaph_cache' / q / 'statistics'), '--ptcs_cache_dir', str(work / 'qaph_cache' / q / 'ptcs'), '--pair_feature_cache_dir', str(work / 'qaph_cache' / q / 'pair'), '--dataset_cache_dir', str(work / 'qaph_cache' / q / 'dataset')]
        if q == 'tp':
            argv += ['--use_scl']
        execute('qaph_' + q, argv)
        files = list(folder.glob('result*.json'))
        assert len(files) == 1
        r = json.loads(files[0].read_text(encoding='utf-8-sig'))['test_results']['0']
        verify('qaph_' + q, folder, [r['mae'], r['rmse']])
    prepared = work / 'prepared_rt'
    methods = ('wspred', 'nntf', 'bnlft', 'ctf', 'trcf', 'gmcl', 'hctn', 'tpmcf', 'gacl')
    for method in methods:
        folder = work / method
        argv = ['-m', 'baselines.train', '--method', method, '--tensor', str(prepared / 'rt.npy'), '--mask', str(prepared / 'split.npz'), '--output-dir', str(folder), '--epochs', '2', '--history-epochs', '2', '--rank', '4', '--evd-rank', '4', '--batch-size', '64', '--device', 'cpu', '--lr', '.00001']
        if method == 'hctn':
            argv += ['--window', '4', '--nmf_iterations', '2', '--width', '16', '--heads', '1', '--output_width', '8']
        if method == 'gacl':
            argv += ['--window', '4', '--dim', '16', '--heads', '2', '--first_cap', '4', '--second_cap', '2']
        if method == 'tpmcf':
            argv += ['--feature_steps', '4', '--svd_rank', '4', '--collab_dim', '4', '--ae_epochs', '1', '--gcn_dim', '4', '--gcn_epochs', '2', '--pte_blocks', '1', '--pte_heads', '2', '--conv_filters', '2']
        execute(method, argv)
        files = list(folder.glob('result*.json'))
        assert len(files) == 1
        r = json.loads(files[0].read_text(encoding='utf-8-sig'))
        metrics = [r['mae'], r['rmse']] if 'mae' in r else [r['test_mae'], r['test_rmse']]
        verify(method, folder, metrics)
    try:
        from importlib.metadata import version
    except ImportError:
        import pkg_resources
        version = lambda n: pkg_resources.get_distribution(n).version
    report = dict(passed=True, synthetic_only=True, real_dataset_training=False, checks=checks, environment=dict(python=sys.version, packages={n: version(n) for n in ('torch', 'torch-geometric', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'tqdm', 'psutil')}), tested_code_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.rglob('*.py') if '__pycache__' not in p.parts})
    (work / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('All eleven synthetic execution checks passed.', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--work-dir', type=Path)
    a = p.parse_args()
    if a.work_dir:
        a.work_dir.mkdir(parents=True, exist_ok=True)
        run(a.work_dir.resolve())
    else:
        with tempfile.TemporaryDirectory(prefix='qaph_smoke_') as d:
            run(Path(d).resolve())

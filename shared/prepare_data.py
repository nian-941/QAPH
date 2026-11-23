"""Convert zero-based WS-DREAM Dataset 2 records and generate reusable masks."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from shared.protocol import prepare_rolling_split, save_split_indices

def load_records(path, shape):
    tensor = np.zeros(shape, dtype=np.float32)
    for chunk in pd.read_csv(path, sep='\\s+', header=None, names=['user', 'service', 'time', 'value'], chunksize=500000):
        ids = chunk[['user', 'service', 'time']].to_numpy()
        values = chunk['value'].to_numpy(dtype=np.float32)
        if not np.isfinite(ids).all() or not np.equal(ids, np.floor(ids)).all():
            raise ValueError('Noninteger IDs')
        ids = ids.astype(np.int64)
        if (ids < 0).any() or any(((ids[:, i] >= shape[i]).any() for i in range(3))):
            raise ValueError('IDs out of bounds; expect zero-based IDs')
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Nonfinite/negative observations')
        tensor[ids[:, 0], ids[:, 1], ids[:, 2]] = values
    return tensor

def prepare(tensor, qos, density, seed, target, output):
    if tensor.ndim != 3 or not np.isfinite(tensor).all() or (tensor < 0).any():
        raise ValueError('Expected finite nonnegative U x S x T tensor')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    split = prepare_rolling_split(tensor, target, density, 0.2, seed, target - 1)
    meta = save_split_indices(str(output / 'split.npz'), split, qos, density, seed, seed)
    np.save(str(output / (qos + '.npy')), tensor.astype(np.float32), allow_pickle=False)
    (output / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='rtdata.txt or tpdata.txt; zero-based four-column records')
    p.add_argument('--qos', choices=('rt', 'tp'), required=True)
    p.add_argument('--density', type=float, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shape', type=int, nargs=3, default=(142, 4500, 64))
    p.add_argument('--target', type=int, default=64)
    a = p.parse_args()
    prepare(load_records(a.input, tuple(a.shape)), a.qos, a.density, a.seed, a.target, a.output)
    print('Prepared tensor and shared split in ' + str(a.output))
if __name__ == '__main__':
    main()

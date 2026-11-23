"""Portable utilities; no batch controller or private experiment manifests."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def atomic_json(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(v, indent=2), encoding='utf-8')
    os.replace(str(temp), str(p))

@contextlib.contextmanager
def file_lock(path, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        f.seek(0, 2)
        if not f.tell():
            f.write(b'0')
            f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

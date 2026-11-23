"""Proposed TRCF-RTA history-only adaptation, not the native paper algorithm.

All evidence is built on H[1:t-1]. Native TRCF deviations at t-1 are persisted
one step. Target train labels can fit only a global mixture scalar. No official
experiments are run by importing this module; native frozen code stays intact.
"""
import hashlib
import importlib.util
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

NATIVE_PATH=Path(__file__).resolve().parent.parent/'_shared'/'tensor_factorization.py'
NATIVE_SHA256='1c7a73ac24dac3572f2ce96b5c63d622887ceb051107c839f4bc51e88cf02f50'


def _native():
    assert hashlib.sha256(NATIVE_PATH.read_bytes()).hexdigest()==NATIVE_SHA256,'Native source identity changed'
    spec=importlib.util.spec_from_file_location('trcf_original_frozen_algorithms',str(NATIVE_PATH))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class HistoryComponents:
    user_add: torch.Tensor
    service_add: torch.Tensor
    user_mul: torch.Tensor
    service_mul: torch.Tensor
    history_sha256: str
    target_slice: int
    rbs_threshold: float
    pcc_threshold: float


@dataclass(frozen=True)
class FittedMixture:
    alpha: float
    fitted_on: str='target_train_labels_only'


def build_history_components(history, rbs_threshold=.68, pcc_threshold=.5,
                             target_slice=64, neighbor_block=64, device='cpu',
                             expected_history_sha256=None, progress=None):
    """No target values, masks or labels are accepted by this feature builder.

    Caller must supply the original history float32 array and its authenticated
    MAIN split metadata hash when using this prototype on real data.
    """
    history=np.asarray(history)
    if history.ndim!=3 or history.shape[2]!=target_slice-1:
        raise ValueError('Expected slices 1..target-1 only; no target slot')
    if not np.isfinite(history).all() or (history<0).any():
        raise ValueError('History must be finite and nonnegative')
    history_hash=hashlib.sha256(np.ascontiguousarray(history).tobytes()).hexdigest()
    if expected_history_sha256 is not None and history_hash!=expected_history_sha256:
        raise ValueError('History identity differs from authenticated split metadata')
    native=_native()
    config=dict(rbs_threshold=float(rbs_threshold),pcc_threshold=float(pcc_threshold),neighbor_block=int(neighbor_block))
    values=torch.as_tensor(history.astype(np.float64),device=device,dtype=torch.float64)
    users,services,_=values.shape
    with torch.no_grad():
        user_average=torch.empty((users,services),device=device,dtype=torch.float64)
        service_average=torch.empty_like(user_average)
        for user in range(users):
            user_average[user]=native.reinforced_average(values[user],config)
            if progress:progress('HISTORY_SERVICE_AVERAGES',user+1,users)
        for service in range(services):
            service_average[:,service]=native.reinforced_average(values[:,service,:],config)
            if progress and (service+1)%100==0:progress('HISTORY_USER_AVERAGES',service+1,services)
        fallback=values.sum()/(values>0).sum().clamp(min=1)
        user_average=torch.where(user_average>0,user_average,fallback)
        service_average=torch.where(service_average>0,service_average,fallback)
        ua,um,sa,sm=[torch.empty_like(user_average) for _ in range(4)]
        for service in range(services):
            ua[:,service],um[:,service]=native.deviation_prediction(values[:,service,:],user_average[:,service],config)
            if progress and (service+1)%100==0:progress('HISTORY_USER_DEVIATION',service+1,services)
        for user in range(users):
            sa[user],sm[user]=native.deviation_prediction(values[user],service_average[user],config)
            if progress:progress('HISTORY_SERVICE_DEVIATION',user+1,users)
    return HistoryComponents(ua,sa,um,sm,history_hash,int(target_slice),float(rbs_threshold),float(pcc_threshold))


def _query(components,indices):
    raw=np.asarray(indices)
    if raw.ndim!=2 or raw.shape[1]!=2 or raw.dtype.kind not in 'iu':
        raise ValueError('Expected integer user/service index pairs')
    shape=components.user_add.shape
    if (raw<0).any() or (raw[:,0]>=shape[0]).any() or (raw[:,1]>=shape[1]).any():
        raise ValueError('Query outside history dimensions')
    return torch.as_tensor(raw,dtype=torch.long,device=components.user_add.device)


def predict(components,mixture,indices=None):
    """Accept only history-derived evidence, fitted scalar and query identities."""
    alpha=float(mixture.alpha)
    if not 0<=alpha<=1:raise ValueError('Mixture alpha outside [0,1]')
    a,b,c,d=components.user_add,components.service_add,components.user_mul,components.service_mul
    if indices is not None:
        ix=_query(components,indices)
        a,b,c,d=[x[ix[:,0],ix[:,1]] for x in [a,b,c,d]]
    additive=alpha*a+(1-alpha)*b
    repaired=alpha*c+(1-alpha)*d
    output=torch.where(additive<0,repaired,additive)
    if not torch.isfinite(output).all():raise ValueError('Nonfinite predictions')
    return output


def fit_mixture(components,train_indices,y_train,alpha_grid=None):
    """Train alpha with train MAE; validation/test labels are not API inputs.

    The original paper's 0,.1,...,1 alpha domain is retained. Preserve the exact
    combined-negative repair while evaluating every candidate. Ties prefer the
    original RTA default .5, then the smaller alpha. This is a proposed adapter
    training rule, not a claim about the native paper's parameter selection.
    """
    ix=_query(components,train_indices)
    raw=np.asarray(train_indices)
    truth=np.asarray(y_train,dtype=np.float64).reshape(-1)
    if len(ix)==0 or len(truth)!=len(ix) or not np.isfinite(truth).all() or (truth<=0).any():
        raise ValueError('Need finite positive training labels matching train indices')
    if len(np.unique(raw,axis=0))!=len(raw):raise ValueError('Duplicate training indices')
    grid=[i/10. for i in range(11)] if alpha_grid is None else list(alpha_grid)
    if not grid or len(set(float(a) for a in grid))!=len(grid):raise ValueError('Empty or duplicate alpha grid')
    y=torch.tensor(truth,device=components.user_add.device,dtype=torch.float64)
    candidates=[]
    with torch.no_grad():
        for a in grid:
            fit=FittedMixture(float(a))
            value=float(torch.abs(predict(components,fit,raw)-y).mean())
            candidates.append(dict(alpha=float(a),train_mae=value))
    winner=min(candidates,key=lambda r:(r['train_mae'],abs(r['alpha']-.5),r['alpha']))
    return FittedMixture(winner['alpha']),candidates

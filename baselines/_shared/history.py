"""Independent H-only pretraining and immutable, content-addressed history states."""
import time
import shutil
from pathlib import Path
import numpy as np
import torch
from baselines.bnlft import model as native_bnlft
from baselines._shared import tensor_factorization as native_ctf
from shared.runtime import ROOT,array_sha,atomic_json,digest,file_lock,read_json,require,sha256

SOURCE_IDS={'../bnlft/model.py': '689bf9259e7971607631ab2c31f610ec267c5ecc3d56185753f170b7f8ab57a5', 'tensor_factorization.py': '1c7a73ac24dac3572f2ce96b5c63d622887ceb051107c839f4bc51e88cf02f50'}
NAMES=('user','service','time','user_bias','service_bias','time_bias')


def validate(history,config):
    allowed={'method','rank','regularization','bias_regularization','gamma','epochs','history_length','qos'}
    if set(config)-allowed:raise ValueError('Historical config cannot contain target data, split or density')
    h=np.asarray(history)
    if config['history_length']!=63 or h.ndim!=3 or h.shape[2]!=63:raise ValueError('History must contain slots1..63 only')
    if not np.isfinite(h).all() or (h<0).any():raise ValueError('Invalid history')
    if config['method'] not in ('bnlft','ctf'):raise ValueError('Unknown history method')
    if not isinstance(config['epochs'],int) or config['epochs']<=0 or config['rank']<=0:raise ValueError('Invalid history budget/rank')
    if config['regularization']<0 or (config['method']=='ctf' and config['gamma']<=0) or (config['method']=='bnlft' and config['bias_regularization']<0):raise ValueError('Invalid history loss parameters')
    for name,expected in SOURCE_IDS.items():require(sha256(Path(__file__).parent/name)==expected,'Frozen history source changed: '+name)
    return h


def fit_history(history,history_config,history_seed,device='cpu',on_epoch=None):
    h=validate(history,history_config);module=native_bnlft if history_config['method']=='bnlft' else native_ctf
    data=torch.tensor(np.array(h,copy=True),device=device,dtype=torch.float64)
    f=module.initialize(data.shape,history_config,history_seed,device)
    curve=[];best=float('inf');best_epoch=0;best_arrays=None
    with torch.no_grad():
        for epoch in range(1,history_config['epochs']+1):
            tick=time.time()
            f,_=(module.step(data,f,history_config) if history_config['method']=='bnlft' else module.factor_step(data,f,history_config))
            require(all(torch.isfinite(v).all().item() and (v>=0).all().item() for v in f),'Historical factors invalid')
            loss=float(module.objective(data,f,history_config));require(np.isfinite(loss),'Historical objective nonfinite')
            row=dict(epoch=epoch,history_objective=loss,seconds=time.time()-tick);curve.append(row)
            if loss<best:
                best=loss;best_epoch=epoch;best_arrays={name:v.detach().cpu().numpy().copy() for name,v in zip(NAMES,f)}
            if on_epoch is not None:on_epoch(epoch,dict(row))
    for a in best_arrays.values():a.setflags(write=False)
    return dict(method=history_config['method'],arrays=best_arrays,history_sha256=array_sha(h),config=dict(history_config),
                history_seed=history_seed,best_epoch=best_epoch,best_history_objective=best),curve


def contract(history,history_config,history_seed):
    h=validate(history,history_config)
    return dict(history_sha256=array_sha(h),history_start=1,history_end=h.shape[2],history_config=dict(history_config),
                history_seed=int(history_seed),dtype='float64',native_source_sha256=SOURCE_IDS,history_source_sha256=sha256(__file__),
                selection='minimum historical objective among fixed-budget history epochs; never target train/val/test')


def get_history(history,history_config,history_seed,expected_sha,session,device,progress):
    info=contract(history,history_config,history_seed);require(info['history_sha256']==expected_sha,'History input differs from shared-mask provenance')
    key=digest(info);folder=session/'cache/history'/key;folder.mkdir(parents=True,exist_ok=True)
    progress('HISTORY_CACHE_WAIT',0,1)
    with file_lock(folder/'history.lock',blocking=True):
        hit=(folder/'record.json').exists()
        if hit:
            record=read_json(folder/'record.json')
            require(record['contract']==info and sha256(folder/'history_state.npz')==record['state_sha256'] and
                    sha256(folder/'history_curve.json')==record['curve_sha256'],'History cache contract/artifact changed')
        else:
            def observe(epoch,row):
                progress('HISTORY_PRETRAIN',epoch,history_config['epochs'],history_objective=row['history_objective'],epoch_seconds=row['seconds'])
            fit_started=time.time();state,curve=fit_history(history,history_config,history_seed,device,observe)
            fit_wall=time.time()-fit_started
            with (folder/'history.tmp').open('wb') as stream:np.savez_compressed(stream,**state['arrays'])
            import os
            os.replace(str(folder/'history.tmp'),str(folder/'history_state.npz'))
            atomic_json(folder/'history_curve.json',curve)
            record=dict(contract=info,history_id=key,state_sha256=sha256(folder/'history_state.npz'),curve_sha256=sha256(folder/'history_curve.json'),
                        epochs_completed=len(curve),best_epoch=state['best_epoch'],best_history_objective=state['best_history_objective'],
                        cold_history_seconds=fit_wall,history_optimization_seconds=sum(r['seconds'] for r in curve))
            atomic_json(folder/'record.json',record)
        evidence=session/'history_models'/key;evidence.mkdir(parents=True,exist_ok=True)
        for name in ('history_state.npz','history_curve.json'):
            if not (evidence/name).exists():shutil.copyfile(folder/name,evidence/name)
            expected=record['state_sha256'] if name.endswith('.npz') else record['curve_sha256']
            require(sha256(evidence/name)==expected,'Immutable exported history state changed')
        atomic_json(evidence/'record.json',record)
        with np.load(folder/'history_state.npz',allow_pickle=False) as a:arrays={name:a[name].copy() for name in a.files}
        for v in arrays.values():v.setflags(write=False)
        state=dict(method=history_config['method'],arrays=arrays,history_sha256=expected_sha,config=dict(history_config),history_seed=history_seed,
                   best_epoch=record['best_epoch'],best_history_objective=record['best_history_objective'])
    return state,dict(record,cache_hit=hit,evidence_relative='history_models/'+key)

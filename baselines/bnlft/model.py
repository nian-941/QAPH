"""BNLFT Eq.13,14,18,19: six sequential MU blocks, per-observation L2."""
import numpy as np
import torch
from baselines._shared.tensor_factorization import contraction,reconstruct

EPS=1e-10


def initialize(shape,config,seed,device):
    rng=np.random.RandomState(seed)
    return [torch.tensor(rng.uniform(0,.05,(n,config['rank'])),dtype=torch.float64,device=device) for n in shape]+[
        torch.tensor(rng.uniform(0,.05,n),dtype=torch.float64,device=device) for n in shape]


def prediction(f):
    return reconstruct(f[:3])+f[3][:,None,None]+f[4][None,:,None]+f[5][None,None,:]


def target(f):
    return (f[0]*f[2][-1])@f[1].T+f[3][:,None]+f[4][None,:]+f[5][-1]


def counts(data,axis):
    return (data>0).sum(tuple(a for a in range(3) if a!=axis)).to(data.dtype)


def objective(data,f,config):
    observed=data>0
    value=.5*((prediction(f)-data).square()*observed).sum()
    for axis in range(3):
        degree=counts(data,axis)
        value+=.5*config['regularization']*(f[axis].square()*degree[:,None]).sum()
        value+=.5*config['bias_regularization']*(f[axis+3].square()*degree).sum()
    return value


def step(data,f,config):
    observed=(data>0).to(data.dtype)
    for axis in range(3):
        pred=prediction(f)
        numerator=contraction(data,f[:3],axis)
        denominator=contraction(observed*pred,f[:3],axis)+config['regularization']*counts(data,axis)[:,None]*f[axis]
        f[axis]=f[axis]*numerator/(denominator+EPS)
    for axis in range(3):
        other=tuple(a for a in range(3) if a!=axis)
        numerator=data.sum(other)
        denominator=(observed*prediction(f)).sum(other)+config['bias_regularization']*counts(data,axis)*f[axis+3]
        f[axis+3]=f[axis+3]*numerator/(denominator+EPS)
    return f,{}

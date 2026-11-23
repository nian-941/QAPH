"""Target-only BNLFT/CTF MU. No historical QoS observation tensor is accepted.

Dense 2D arrays are internal deterministic matrix-product workspaces. The data
objective and residual weights have exactly the supplied target-train support.
"""
import hashlib
import numpy as np
import torch

EPS=1e-10
PARAMS=('user','service','vector','user_bias','service_bias','bias')


def fingerprint(arrays):
    return {k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in arrays.items()}


def initialize_target(history_state,device='cpu'):
    a=history_state['arrays']
    expected={'user','service','time'}|({'user_bias','service_bias','time_bias'} if history_state['method']=='bnlft' else set())
    if set(a)!=expected:raise ValueError('History state must contain factors/biases only')
    state={k:torch.tensor(np.array(a[k],copy=True),device=device,dtype=torch.float64) for k in ('user','service')}
    state['vector']=torch.tensor(np.array(a['time'][-1],copy=True),device=device,dtype=torch.float64)
    if history_state['method']=='bnlft':
        for name in ('user_bias','service_bias'):state[name]=torch.tensor(np.array(a[name],copy=True),device=device,dtype=torch.float64)
        state['bias']=torch.tensor(float(a['time_bias'][-1]),device=device,dtype=torch.float64)
    return state


def prepare_train(state,train_indices,train_labels):
    raw=np.asarray(train_indices);y=np.asarray(train_labels,dtype=np.float64).reshape(-1)
    u,s=len(state['user']),len(state['service'])
    if raw.ndim!=2 or raw.shape[1]!=2 or raw.dtype.kind not in 'iu' or not len(raw):raise ValueError('Expected nonempty target train indices')
    if len(y)!=len(raw) or not np.isfinite(y).all() or (y<=0).any():raise ValueError('Invalid positive target supervision')
    if (raw<0).any() or (raw[:,0]>=u).any() or (raw[:,1]>=s).any() or len(np.unique(raw,axis=0))!=len(raw):raise ValueError('Invalid target train support')
    device=state['user'].device
    ix=torch.tensor(raw,device=device,dtype=torch.long)
    values=torch.zeros((u,s),device=device,dtype=torch.float64)
    mask=torch.zeros_like(values)
    values[ix[:,0],ix[:,1]]=torch.tensor(y,device=device,dtype=torch.float64)
    mask[ix[:,0],ix[:,1]]=1.
    return dict(indices=ix,labels=torch.tensor(y,device=device,dtype=torch.float64),values=values,mask=mask,
                user_degree=mask.sum(1),service_degree=mask.sum(0),n=len(raw))


def predict(state,indices=None):
    p=(state['user']*state['vector'])@state['service'].T
    if 'bias' in state:p=p+state['user_bias'][:,None]+state['service_bias'][None,:]+state['bias']
    if indices is None:return p
    raw=np.asarray(indices)
    if raw.ndim!=2 or raw.shape[1]!=2 or raw.dtype.kind not in 'iu':raise ValueError('Expected integer query pairs')
    if (raw<0).any() or (raw[:,0]>=p.shape[0]).any() or (raw[:,1]>=p.shape[1]).any():raise ValueError('Query outside target model')
    ix=torch.tensor(raw,device=p.device,dtype=torch.long)
    return p[ix[:,0],ix[:,1]]


def objective(state,train,config):
    p=predict(state);ix=train['indices'];residual=p[ix[:,0],ix[:,1]]-train['labels']
    if config['method']=='bnlft':
        result=.5*residual.square().sum()
        regularizer=(state['user'].square()*train['user_degree'][:,None]).sum()
        regularizer+=(state['service'].square()*train['service_degree'][:,None]).sum()+train['n']*state['vector'].square().sum()
        biases=(state['user_bias'].square()*train['user_degree']).sum()+(state['service_bias'].square()*train['service_degree']).sum()+train['n']*state['bias'].square()
        return result+.5*config['regularization']*regularizer+.5*config['bias_regularization']*biases
    if config['method']!='ctf':raise ValueError('Unknown target method')
    result=.5*torch.log1p((residual/config['gamma']).square()).sum()
    return result+.5*config['regularization']*sum(state[k].square().sum() for k in ('user','service','vector'))


def step(state,train,config):
    f=dict(state);method=config['method'];lam=config['regularization']
    def weights():
        p=predict(f)
        if method=='bnlft':return p,train['mask']
        # No residual is evaluated against a synthetic zero at an unobserved pair.
        ix=train['indices'];r=train['labels']-p[ix[:,0],ix[:,1]]
        w=torch.zeros_like(train['mask']);w[ix[:,0],ix[:,1]]=1/(config['gamma']**2+r.square())
        return p,w
    p,w=weights();other=f['service']*f['vector']
    up=(w*train['values'])@other;down=(w*p)@other
    multiplier=train['user_degree'][:,None] if method=='bnlft' else 1.
    proposed=f['user']*up/(down+lam*multiplier*f['user']+EPS)
    active=(train['user_degree']>0) if method=='bnlft' or lam==0 else torch.ones_like(train['user_degree'],dtype=torch.bool)
    f['user']=torch.where(active[:,None],proposed,f['user'])
    p,w=weights();other=f['user']*f['vector']
    up=(w*train['values']).T@other;down=(w*p).T@other
    multiplier=train['service_degree'][:,None] if method=='bnlft' else 1.
    proposed=f['service']*up/(down+lam*multiplier*f['service']+EPS)
    active=(train['service_degree']>0) if method=='bnlft' or lam==0 else torch.ones_like(train['service_degree'],dtype=torch.bool)
    f['service']=torch.where(active[:,None],proposed,f['service'])
    p,w=weights()
    up=(f['user']*((w*train['values'])@f['service'])).sum(0)
    down=(f['user']*((w*p)@f['service'])).sum(0)
    f['vector']=f['vector']*up/(down+lam*(train['n'] if method=='bnlft' else 1.)*f['vector']+EPS)
    if method=='bnlft':
        lb=config['bias_regularization']
        p=predict(f);up=(train['mask']*train['values']).sum(1);down=(train['mask']*p).sum(1)+lb*train['user_degree']*f['user_bias']
        f['user_bias']=torch.where(train['user_degree']>0,f['user_bias']*up/(down+EPS),f['user_bias'])
        p=predict(f);up=(train['mask']*train['values']).sum(0);down=(train['mask']*p).sum(0)+lb*train['service_degree']*f['service_bias']
        f['service_bias']=torch.where(train['service_degree']>0,f['service_bias']*up/(down+EPS),f['service_bias'])
        p=predict(f);f['bias']=f['bias']*(train['mask']*train['values']).sum()/((train['mask']*p).sum()+lb*train['n']*f['bias']+EPS)
    return f


def fit_target(history_state,train_indices,train_labels,target_config,device='cpu',on_epoch=None):
    allowed={'method','rank','regularization','bias_regularization','gamma','epochs'}
    if set(target_config)-allowed:raise ValueError('Target trainer accepts no historical observations or heldout supervision')
    if target_config['method']!=history_state['method']:raise ValueError('Historical/target method mismatch')
    if not isinstance(target_config['epochs'],int) or target_config['epochs']<=0:raise ValueError('Invalid target budget')
    if target_config['regularization']<0 or (target_config['method']=='ctf' and target_config['gamma']<=0) or (target_config['method']=='bnlft' and target_config['bias_regularization']<0):raise ValueError('Invalid target loss parameters')
    before=fingerprint(history_state['arrays']);state=initialize_target(history_state,device)
    if target_config['rank']!=state['user'].shape[1]:raise ValueError('Historical/target rank mismatch')
    train=prepare_train(state,train_indices,train_labels);curve=[]
    import time
    with torch.no_grad():
        for epoch in range(1,target_config['epochs']+1):
            tick=time.time();state=step(state,train,target_config)
            if not all(torch.isfinite(v).all().item() and (v>=0).all().item() for v in state.values()):raise RuntimeError('Nonfinite/negative target parameters')
            loss=float(objective(state,train,target_config))
            if not np.isfinite(loss):raise RuntimeError('Nonfinite target objective')
            row=dict(epoch=epoch,target_train_objective=loss,seconds=time.time()-tick)
            curve.append(row)
            if on_epoch is not None:
                # Observers get independent copies, so validation cannot mutate learning.
                on_epoch(epoch,{k:v.detach().clone() for k,v in state.items()},dict(row))
    if fingerprint(history_state['arrays'])!=before:raise RuntimeError('Target fitting polluted historical state')
    return state,curve,dict(history_unchanged=True,target_train_count=train['n'],
                           cold_users=int((train['user_degree']==0).sum()),cold_services=int((train['service_degree']==0).sum()),
                           train_indices_sha256=hashlib.sha256(np.ascontiguousarray(train_indices).tobytes()).hexdigest(),
                           loss_scope='only supplied target-train support; no history reconstruction',
                           data_interface='historical factors; target train indices/labels; target configuration')

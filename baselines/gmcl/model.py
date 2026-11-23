"""Paper-derived GMCL MAIN adaptation; exact structured graph products.

The graph is never truncated to top-k neighbors. EVD rank is separate from edges.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Graph:
    def __init__(self,values):
        self.u,self.s,self.t=values.shape
        self.g=self.u+self.s+self.t
        self.n=self.g+self.t*(self.u+self.s)
        self.edge=(values>0).permute(2,0,1).contiguous().to(torch.float32)
        local_u=self.edge.sum(2)+2
        local_s=self.edge.sum(1)+2
        temporal=torch.full((self.t,),float(self.u+self.s),device=values.device)
        if self.t>1:
            temporal[0]+=1;temporal[-1]+=1
        if self.t>2:temporal[1:-1]+=2
        self.degree=torch.cat([torch.full((self.u+self.s,),float(self.t),device=values.device),temporal,local_u.flatten(),local_s.flatten()])
        self.inv=self.degree.to(torch.float64).rsqrt()
        self._edges={torch.float32:self.edge}

    def apply(self,x,dropout=0.):
        if x.ndim==1:return self.apply(x[:,None],dropout)[:,0]
        inv=self.inv.to(x.dtype)[:,None]
        z=x*inv
        gu=z[:self.u];gs=z[self.u:self.u+self.s];gt=z[self.u+self.s:self.g]
        lu=z[self.g:self.g+self.t*self.u].reshape(self.t,self.u,-1)
        ls=z[self.g+self.t*self.u:].reshape(self.t,self.s,-1)
        def drop(a):return F.dropout(a,p=dropout,training=dropout>0)
        if x.dtype not in self._edges:self._edges[x.dtype]=self.edge.to(x.dtype)
        e=self._edges[x.dtype]
        # Edge dropout is applied to normalized adjacency entries; inverse degrees
        # remain those of the original graph, as in paper Eq.1.
        out_gu=(drop(torch.ones((self.t,self.u,1),device=x.device,dtype=x.dtype))*lu).sum(0)
        out_gs=(drop(torch.ones((self.t,self.s,1),device=x.device,dtype=x.dtype))*ls).sum(0)
        out_gt=(drop(torch.ones((self.t,self.u,1),device=x.device,dtype=x.dtype))*lu).sum(1)
        out_gt=out_gt+(drop(torch.ones((self.t,self.s,1),device=x.device,dtype=x.dtype))*ls).sum(1)
        if self.t>1:
            previous=F.pad(drop(torch.ones((self.t-1,1),device=x.device,dtype=x.dtype))*gt[:-1],(0,0,1,0))
            following=F.pad(drop(torch.ones((self.t-1,1),device=x.device,dtype=x.dtype))*gt[1:],(0,0,0,1))
            out_gt=out_gt+previous+following
        out_lu=torch.bmm(drop(e),ls)+drop(torch.ones((self.t,self.u,1),device=x.device,dtype=x.dtype))*(gu[None,:,:])
        out_lu=out_lu+drop(torch.ones((self.t,self.u,1),device=x.device,dtype=x.dtype))*gt[:,None,:]
        out_ls=torch.bmm(drop(e.transpose(1,2)),lu)+drop(torch.ones((self.t,self.s,1),device=x.device,dtype=x.dtype))*gs[None,:,:]
        out_ls=out_ls+drop(torch.ones((self.t,self.s,1),device=x.device,dtype=x.dtype))*gt[:,None,:]
        return torch.cat([out_gu,out_gs,out_gt,out_lu.flatten(0,1),out_ls.flatten(0,1)])*inv


def eigensystem(graph,rank,seed,progress):
    from scipy.sparse.linalg import LinearOperator,eigsh
    rank=min(rank,graph.n-2)
    count=[0];started=time.time()
    def apply(a):
        with torch.no_grad():
            x=torch.as_tensor(np.array(a,copy=True),dtype=torch.float64,device=graph.edge.device)
            out=graph.apply(x).cpu().numpy()
        count[0]+=1
        if count[0]%50==0:progress('EVD_MATVEC',count[0],None,seconds=time.time()-started)
        return out
    operator=LinearOperator((graph.n,graph.n),matvec=apply,dtype=np.float64)
    values,vectors=eigsh(operator,k=rank,which='LA',tol=1e-6,maxiter=3000,
                         v0=np.random.RandomState(seed).normal(size=graph.n))
    order=np.argsort(values)[::-1];values=values[order];vectors=vectors[:,order]
    residual=np.max(np.linalg.norm(apply(vectors)-vectors*values[None,:],axis=0))
    orthogonal=np.max(np.abs(vectors.T@vectors-np.eye(rank)))
    if not np.isfinite(residual) or residual>1e-5 or orthogonal>1e-5:
        raise RuntimeError('EVD certificate failed: residual={} orthogonal={}'.format(residual,orthogonal))
    return values,vectors,dict(rank=rank,residual=float(residual),orthogonality=float(orthogonal),
                               matvec_calls=count[0],seconds=time.time()-started,which='largest_algebraic',tolerance=1e-6)


def get_eigen(graph,config,session,directory,input_sha,progress):
    from shared.runtime import file_lock,atomic_json,read_json,sha256,require,digest
    contract=dict(schema='GMCL_structured_v1',bundle_id=read_json(session/'config.json')['bundle_id'],input_sha256=input_sha,rank=config['evd_rank'],seed=20261119,
                  normalization='symmetric_no_self_loops',tolerance=1e-6,which='LA')
    key=digest(contract);folder=session/'cache/evd'/key;folder.mkdir(parents=True,exist_ok=True)
    progress('EVD_CACHE_WAIT',0,1)
    with file_lock(folder/'cache.lock',blocking=True):
        cached=(folder/'record.json').exists()
        if cached:
            record=read_json(folder/'record.json')
            require(record['contract']==contract and sha256(folder/'eigen.npz')==record['sha256'],'EVD cache identity differs')
        else:
            values,vectors,certificate=eigensystem(graph,config['evd_rank'],20261119,progress)
            np.savez_compressed(str(folder/'eigen.npz'),values=values,vectors=vectors)
            record=dict(contract=contract,certificate=certificate,sha256=sha256(folder/'eigen.npz'))
            atomic_json(folder/'record.json',record)
        import shutil
        shutil.copyfile(str(folder/'eigen.npz'),str(directory/'eigen.npz'))
        atomic_json(directory/'eigen_record.json',dict(record,cache_hit=cached))
    with np.load(directory/'eigen.npz',allow_pickle=False) as a:
        values=torch.tensor(a['values'],device=graph.edge.device,dtype=torch.float32)
        vectors=torch.tensor(a['vectors'],device=graph.edge.device,dtype=torch.float32)
    # Free FP64 graph weights used only by the eigensolver before training.
    graph._edges.pop(torch.float64,None)
    return values,vectors,dict(record,cache_hit=cached)


class Model(nn.Module):
    def __init__(self,graph,config,eigenvalues,eigenvectors):
        super().__init__();self.graph=graph;self.config=config
        d=config['rank'];self.embedding=nn.Parameter(torch.empty(graph.n,d))
        nn.init.normal_(self.embedding,std=.1)
        self.register_buffer('eigenvalues',eigenvalues,persistent=False)
        self.register_buffer('eigenvectors',eigenvectors,persistent=False)
        self.mlp=nn.Sequential(nn.Linear(3*d,2*d),nn.BatchNorm1d(2*d),nn.ReLU(),nn.Dropout(config['dropout']),
                               nn.Linear(2*d,d),nn.BatchNorm1d(d),nn.ReLU(),nn.Dropout(config['dropout']))
        self.output=nn.Linear(2*d,1);nn.init.constant_(self.output.bias,.1)

    def views(self,contrast=True):
        v=self.embedding;main=v
        g=v;augmented=g
        for layer in range(self.config['layers']):
            v=self.graph.apply(v,self.config['dropout'] if self.training else 0.)
            main=main+v
            if contrast:
                g=self.eigenvectors@(self.eigenvalues[:,None]*(self.eigenvectors.T@g))
                augmented=augmented+g
        return main,augmented

    def predict(self,emb,indices):
        u=emb[indices[:,0]];s=emb[self.graph.u+indices[:,1]]
        t=emb[self.graph.u+self.graph.s+self.graph.t-1].expand_as(u)
        high=self.mlp(torch.cat([u,s,t],1));low=u*s*t
        return F.relu(self.output(torch.cat([high,low],1))).flatten()

    def contrast(self,main,augmented,indices):
        ids=torch.unique(torch.cat([indices[:,0],self.graph.u+indices[:,1],
                    torch.full((1,),self.graph.u+self.graph.s+self.graph.t-1,device=indices.device,dtype=torch.long)]))
        keep=torch.rand(len(ids),device=ids.device)>=self.config['node_dropout']
        ids=ids[keep]
        if len(ids)<2:return main.sum()*0
        logits=main[ids]@augmented[ids].T/self.config['temperature']
        return F.cross_entropy(logits,torch.arange(len(ids),device=ids.device),reduction='sum')


def fit(data,train,val,truth_val,config,seed,session,directory,input_sha,progress,smoke_epochs=None):
    graph=Graph(data)
    ev,vec,eigen_record=get_eigen(graph,config,session,directory,input_sha,progress)
    epochs=[];best=float('inf');best_epoch=0
    with torch.enable_grad():
        model=Model(graph,config,ev,vec).to(data.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['regularization'])
        train=torch.tensor(train,device=data.device,dtype=torch.long)
        val=torch.tensor(val,device=data.device,dtype=torch.long)
        truth=torch.tensor(truth_val,device=data.device,dtype=torch.float32)
        limit=smoke_epochs or config['max_epochs']
        for epoch in range(1,limit+1):
            started=time.time();model.train();order=torch.randperm(len(train),device=data.device)
            batches=list(order.split(config['batch_size']))
            if len(batches)>1 and len(batches[-1])==1:
                batches[-2]=torch.cat([batches[-2],batches[-1]]);batches.pop()
            objective=0.;seen=0
            for bi,rows in enumerate(batches):
                ix=train[rows];optimizer.zero_grad(set_to_none=True)
                main,aug=model.views();pred=model.predict(main,ix)
                y=data[ix[:,0],ix[:,1],-1]
                loss=F.l1_loss(pred,y)+config['beta']*model.contrast(main,aug,ix)
                if not torch.isfinite(loss):raise RuntimeError('GMCL nonfinite loss')
                loss.backward();optimizer.step()
                objective+=float(loss.detach())*len(ix);seen+=len(ix)
                if bi%10==0:progress('GMCL_TRAIN',bi+1,len(batches),epoch=epoch)
            model.eval()
            with torch.no_grad():
                main,_=model.views(False)
                vp=torch.cat([model.predict(main,b) for b in val.split(8192)])
                mae=float((vp-truth).abs().mean())
                if not np.isfinite(mae):raise RuntimeError('GMCL validation MAE is nonfinite')
            if mae<best:
                best=mae;best_epoch=epoch
                torch.save(model.state_dict(),str(directory/'best_model.pt'))
            epochs.append(dict(epoch=epoch,val_mae=mae,objective=objective/seen,train_seen=seen,seconds=time.time()-started))
            from shared.runtime import atomic_json
            atomic_json(directory/'epochs.json',epochs)
            progress('GMCL_VALIDATION',epoch,limit,val_mae=mae,best_epoch=best_epoch,epoch_seconds=epochs[-1]['seconds'])
            if epoch-best_epoch>=config['patience']:break
        model.load_state_dict(torch.load(str(directory/'best_model.pt'),map_location=data.device))
        model.eval()
        with torch.no_grad():
            main,_=model.views(False)
            indices=torch.cartesian_prod(torch.arange(graph.u,device=data.device),torch.arange(graph.s,device=data.device))
            prediction=torch.cat([model.predict(main,b) for b in indices.split(8192)]).reshape(graph.u,graph.s).cpu().numpy()
            # Store exact FP64 CPU scoring of the restored validation predictions.
            val_cpu=val.cpu().numpy()
            best=float(np.abs(prediction[val_cpu[:,0],val_cpu[:,1]].astype(np.float64)-truth_val).mean())
    return prediction,epochs,best,best_epoch,dict(parameters=sum(p.numel() for p in model.parameters()),graph_nodes=graph.n,
                graph_directed_edges=int(graph.degree.to(torch.int64).sum()),training_scope='target_train_only_MAIN_adaptation',
                information_boundary='history+target_train graph; validation/test labels excluded',evd_rank=len(ev),
                eigen_cache_hit=eigen_record['cache_hit'],eigen_cold_seconds=eigen_record['certificate']['seconds'])

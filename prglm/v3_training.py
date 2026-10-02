"""Continuous fixed-data-budget graph scaling; no old training entrypoint called."""
import copy
import hashlib
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .byte_data import prepare_bytes,load_byte_data
from .learning_curve import atomic_json,atomic_checkpoint,capture_rng,restore_rng
from .models import backend
from .v1_experiment import seed_all,batches,validation_positions
from .v3_model import V3Config,PRGLMv3
from .v3_metrics import empty,add,finish,memory,pack_model


def construct(spec,family,scale):
    return PRGLMv3(V3Config(**(spec['model']|{'family':family,'nodes':spec['scales'][scale]})))


def gradient_diagnostic(model):
    m=copy.deepcopy(model).eval();acts=[]
    def retain(module,args,value):value.retain_grad();acts.append(value)
    hook=m.emb.register_forward_hook(retain)
    x=torch.tensor([[65,66,67,68]],device=next(m.parameters()).device)[:,:min(m.c.context,4)]
    m.zero_grad(set_to_none=True);y,_=m(x,routing_mode='argmax',fixed_repeat=1)
    y[0,-1,69].backward();hook.remove()
    return {'prefix_embedding_gradient_L1':sum(0. if a.grad is None else float(a.grad.abs().sum()) for a in acts[:-1]),
            'current_embedding_gradient_L1':0. if acts[-1].grad is None else float(acts[-1].grad.abs().sum()),
            'definition':'Last next-byte scalar69; argmax single-pass, activation gradients not shared embedding parameter gradients; actual graph readout path.'}


def clip(model):
    grads=[];total=0.
    for p in model.parameters():
        if p.grad is None:continue
        if p.grad.is_sparse:
            p.grad=p.grad.coalesce();g=p.grad._values()
        else:g=p.grad
        grads.append(g);total+=g.detach().square().sum()
    factor=(1/(torch.sqrt(total)+1e-6)).clamp_max(1.)
    for g in grads:g.mul_(factor)


@torch.no_grad()
def evaluate(model,ids,spec,seed,device,fixed=None,budgets=None):
    model.eval();result={}
    for budget in budgets or spec['validation_bytes']:
        positions=validation_positions(ids,model.c.context,budget)
        if len(positions)*model.c.context!=budget:raise ValueError('insufficient validation')
        loss,total=0.,0;stats=empty(model.c)
        for index,(x,y) in enumerate(batches(ids,positions,model.c.context,spec['val_batch'])):
            pred,_=model(x.to(device),routing_mode='sampled',seed=seed+index,fixed_repeat=fixed,collect=True)
            loss+=float(F.cross_entropy(pred.flatten(0,1),y.to(device).flatten(),reduction='sum'));total+=y.numel()
            add(stats,model.last_stats)
        result[str(budget)]={'loss':loss/total,'ppl':math.exp(min(loss/total,50)),
            'validation_tokens':total,'repetition':finish(stats,model.c)}
    return result


def runtime(model,seed,device):
    x=torch.tensor([[65,66,67,68]],device=device)[:,-model.c.context:];generated=16
    # Prototype single-sequence inference, persistent state context recomputed like old baseline.
    with torch.no_grad():
        model(x,routing_mode='sampled',seed=seed)
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        for step in range(generated):
            y,_=model(x[:,-model.c.context:],routing_mode='sampled',seed=seed+step)
            nxt=y[:,-1].argmax(-1,keepdim=True);x=torch.cat((x,nxt),dim=1)
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
    return {'inference_tokens_per_second':generated/elapsed,'latency_seconds_per_token':elapsed/generated,
            'generated_tokens':generated,'note':'Short prototype sample, context recomputed, batched candidate gates even for inactive edges. No physical efficiency claim.'}


def initialize(spec,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    files=['prglm/v3_model.py','prglm/v3_metrics.py','prglm/v3_training.py']
    protocol={'spec':spec,'config_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        'dataset_sha256':hashlib.sha256(Path(spec['corpus']).read_bytes()).hexdigest(),
        'implementation_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    p=root/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('immutable experiment protocol changed')
    if not p.exists():atomic_json(p,protocol)
    if not (root/'train.bin').exists():prepare_bytes(spec['corpus'],root)
    return load_byte_data(root)


def run_one(spec,root,family,scale,seed,train,val,stop_after=None,report_callback=None):
    directory=Path(root)/scale/family/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    result_path=directory/'results.json';rows=json.loads(result_path.read_text())['results'] if result_path.exists() else []
    target=stop_after or max(spec['milestones'])
    if any(row['training_tokens']==target for row in rows):return rows
    seed_all(seed);device=backend();model=construct(spec,family,scale).to(device)
    edge_optimizer=torch.optim.SparseAdam([model.edges.weight],lr=spec['lr'])
    neural_optimizer=torch.optim.AdamW([p for name,p in model.named_parameters() if name!='edges.weight' and p.requires_grad],lr=spec['lr'])
    optimizers=[edge_optimizer,neural_optimizer]
    unit=spec['batch']*model.c.context
    if any(t%unit for t in spec['milestones']):raise ValueError('milestone alignment')
    starts=np.random.default_rng(seed).integers(0,len(train)-model.c.context-1,size=max(spec['milestones'])//unit*spec['batch'])
    stream_hash=hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
    steps,history,elapsed=0,[],0.;latest=directory/'latest.pt'
    if latest.exists():
        old=torch.load(latest,map_location='cpu',weights_only=False)
        if old['spec']!=spec or old['stream_sha256']!=stream_hash:raise ValueError('resume mismatch')
        model.load_state_dict(old['state'])
        for optimizer,state in zip(optimizers,old['optimizers']):optimizer.load_state_dict(state)
        steps,history,elapsed=old['step'],old['history'],old['training_seconds'];restore_rng(old['rng'])
    def checkpoint(step):
        return {'spec':spec,'config':model.c.dict(),'family':family,'scale':scale,'seed':seed,
            'step':step,'training_tokens':step*unit,'state':model.state_dict(),
            'optimizers':[o.state_dict() for o in optimizers],'rng':capture_rng(),
            'stream_sha256':stream_hash,'history':history,'training_seconds':elapsed}
    def assess(step):
        tokens=step*unit
        if any(r['training_tokens']==tokens for r in rows):return
        rng=capture_rng();path=directory/f'tokens_{tokens}.pt'
        if not path.exists():atomic_checkpoint(path,checkpoint(step))
        validation=evaluate(model,val,spec,seed,device)
        gradient=gradient_diagnostic(model)
        inference_path=directory/f'model_tokens_{tokens}.pt'
        if not inference_path.exists():atomic_checkpoint(inference_path,pack_model(model))
        mem=memory(model)
        optimizer_bytes=sum(v.numel()*v.element_size() for o in optimizers for st in o.state.values() for v in st.values() if isinstance(v,torch.Tensor))
        row={'family':family,'scale':scale,'seed':seed,'edge_slots':model.c.nodes*model.c.slots,
            'training_tokens':tokens,'train_loss':history[-1]['loss'],'validation':validation,
            'gradient_diagnostics':gradient,'memory':mem,'backend':str(device),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'actual_pytorch_resume_bytes':path.stat().st_size,'actual_inference_export_bytes':inference_path.stat().st_size,
            'optimizer_training_only_tensor_bytes':optimizer_bytes,
            'checkpoint_identifier':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'stream_sha256':stream_hash,'training_seconds':elapsed,'training_tokens_per_second':tokens/max(elapsed,1e-12),
            'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            'peak_reserved_cuda_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None}
        if tokens==max(spec['milestones']):
            row['runtime']=runtime(model,seed,device)
            if family=='repeat' and scale in spec['diagnostic_scales']:
                row['fixed_repeat_diagnostics']={str(k):evaluate(model,val,spec,seed,device,fixed=k,
                    budgets=[spec['diagnostic_validation_bytes']]) for k in spec['fixed_repeats']}
        restore_rng(rng);rows.append(row);atomic_json(result_path,{'spec':spec,'results':rows})
        if report_callback:report_callback()
    if steps*unit in spec['milestones']:assess(steps)
    if steps*unit>=target:return rows
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for step,(x,y) in enumerate(batches(train,starts[steps*spec['batch']:],model.c.context,spec['batch']),steps+1):
        model.train()
        for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
        prediction,_=model(x.to(device));loss=F.cross_entropy(prediction.flatten(0,1),y.to(device).flatten())
        loss.backward();clip(model)
        for optimizer in optimizers:optimizer.step()
        history.append({'step':step,'loss':float(loss.detach())})
        if step%spec['checkpoint_interval']==0 or step*unit in spec['milestones']:
            elapsed+=time.perf_counter()-start;atomic_checkpoint(latest,checkpoint(step))
            atomic_json(directory/'progress.json',{'training_tokens':step*unit,'target_tokens':max(spec['milestones'])})
            if step*unit in spec['milestones']:assess(step)
            start=time.perf_counter()
        if step*unit>=target:break
    return rows

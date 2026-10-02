"""Independent resumable trainer. Frozen trainers/models/results are never edited."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .byte_data import ByteTokenizer,prepare_bytes,load_byte_data
from .learning_curve import atomic_checkpoint,atomic_json,capture_rng,restore_rng
from .models import backend
from .v1_experiment import seed_all,batches,validation_positions
from .v21_config import V21Config
from .v21_model import PRGLMv21
from .v21_diagnostics import count_vector,routing_metrics,prefix_gradient
from .v21_memory import memory


@torch.no_grad()
def evaluate(model,ids,spec,seed,device):
    model.eval();results={}
    modes=['sampled','argmax']+([] if model.c.minimum_gateway_hops else ['expected'])
    for budget in spec['validation_bytes']:
        results[str(budget)]={}
        positions=validation_positions(ids,model.c.context,budget)
        if len(positions)*model.c.context!=budget:raise ValueError('insufficient validation')
        for mode in modes:
            loss,total=0.,0;counts=np.zeros(model.c.gateways+11)
            for index,(x,y) in enumerate(batches(ids,positions,model.c.context,spec['val_batch'])):
                x,y=x.to(device),y.to(device)
                logits,_=model(x,routing_mode=mode,seed=seed+index)
                loss+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'))
                total+=y.numel()
                if mode!='expected':counts+=count_vector(model)
            results[str(budget)][mode]={'loss':loss/total,'ppl':math.exp(min(loss/total,50)),
                'validation_tokens':total,'routing':routing_metrics(counts,model.c.gateways) if mode!='expected' else None}
    return results


def construct(spec,name,seed):
    return PRGLMv21(V21Config(**(spec['v1']|spec['variants'][name]|{'seed':seed})))


def run_one(spec,root,name,seed,train_ids,val_ids,stop_after=None):
    root=Path(root);directory=root/f'seed{seed}'/name;directory.mkdir(parents=True,exist_ok=True)
    results_path=directory/'results.json'
    rows=json.loads(results_path.read_text())['results'] if results_path.exists() else []
    target=stop_after or max(spec['milestones'])
    if any(row['training_tokens']==target for row in rows):return rows
    seed_all(seed);device=backend();model=construct(spec,name,seed).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'])
    unit=spec['batch']*model.c.context
    if any(t%unit for t in spec['milestones']):raise ValueError('fractional update milestone')
    starts=np.random.default_rng(seed).integers(0,len(train_ids)-model.c.context-1,
        size=max(spec['milestones'])//unit*spec['batch'])
    fingerprint=hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
    done,history,elapsed=0,[],0.
    ema=np.zeros(model.c.gateways+11);totals=ema.copy()
    latest=directory/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        if saved['spec']!=spec or saved['token_stream_sha256']!=fingerprint:raise ValueError('resume protocol changed')
        model.load_state_dict(saved['state']);optimizer.load_state_dict(saved['optimizer'])
        # torch maps optimizer states to parameter device during load_state_dict.
        done,history,elapsed=saved['step'],saved['history'],saved['training_seconds']
        ema=np.asarray(saved['routing_ema']);totals=np.asarray(saved['routing_totals'])
        restore_rng(saved['rng'])
    def payload(step):
        return {'name':name,'config':model.c.dict(),'spec':spec,'seed':seed,'step':step,
            'training_tokens':step*unit,'state':model.state_dict(),'optimizer':optimizer.state_dict(),
            'rng':capture_rng(),'token_stream_sha256':fingerprint,'history':history,
            'training_seconds':elapsed,'routing_ema':ema.tolist(),'routing_totals':totals.tolist()}
    def assess(step):
        tokens=step*unit
        if any(r['training_tokens']==tokens for r in rows):return
        rng=capture_rng();path=directory/f'tokens_{tokens}.pt'
        if not path.exists():atomic_checkpoint(path,payload(step))
        val=evaluate(model,val_ids,spec,seed,device)
        gradient=prefix_gradient(model)
        export_path=directory/f'model_tokens_{tokens}.pt'
        if not export_path.exists():atomic_checkpoint(export_path,{'name':name,'config':model.c.dict(),'state':model.state_dict()})
        mem=memory(model);mem['training_only_routing_statistics_bytes']=int(ema.nbytes+totals.nbytes)
        mem['training_only_optimizer_tensor_bytes']=sum(v.numel()*v.element_size() for st in optimizer.state.values() for v in st.values() if isinstance(v,torch.Tensor))
        rows.append({'seed':seed,'model':name,'milestone':tokens,'training_tokens':tokens,
            'config':model.c.dict(),'train_loss':history[-1]['train_loss'],'validation':val,
            'routing_training_ema':routing_metrics(ema,model.c.gateways),
            'routing_training_cumulative':routing_metrics(totals,model.c.gateways),
            'temporal_credit':gradient,'context_gate':float(torch.sigmoid(model.context_gate)),
            'memory':mem,'checkpoint_identifier':str(path),'serialized_model_bytes':export_path.stat().st_size,
            'serialized_resume_bytes':path.stat().st_size,'token_stream_sha256':fingerprint,
            'training_seconds':elapsed,'training_tokens_per_second':tokens/max(elapsed,1e-12),'backend':str(device)})
        restore_rng(rng);atomic_json(results_path,{'spec':spec,'results':rows})
    if done*unit in spec['milestones']:assess(done)
    if done*unit>=target:return rows
    started=time.perf_counter()
    for step,(x,y) in enumerate(batches(train_ids,starts[done*spec['batch']:],model.c.context,spec['batch']),done+1):
        model.train();logits,_=model(x.to(device));loss=F.cross_entropy(logits.flatten(0,1),y.to(device).flatten())
        counts=count_vector(model);ema=spec['ema_decay']*ema+(1-spec['ema_decay'])*counts;totals+=counts
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        history.append({'step':step,'train_loss':float(loss.detach())})
        if step%spec['checkpoint_interval']==0 or step*unit in spec['milestones']:
            elapsed+=time.perf_counter()-started
            atomic_checkpoint(latest,payload(step))
            atomic_json(directory/'progress.json',{'step':step,'training_tokens':step*unit,'target_tokens':max(spec['milestones'])})
            if step*unit in spec['milestones']:assess(step)
            started=time.perf_counter()
        if step*unit>=target:break
    return rows


def initialize(spec,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    protocol={'spec':spec,'corpus_sha256':hashlib.sha256(Path(spec['corpus']).read_bytes()).hexdigest(),
        'implementation_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ('prglm/v21_model.py','prglm/v21_config.py','prglm/v21_training.py','prglm/v21_diagnostics.py','prglm/v21_memory.py')},
        'validation_overlap':'4096 is a prefix subset of16384, not independent samples'}
    p=root/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('changed study protocol/implementation')
    if not p.exists():atomic_json(p,protocol)
    if not (root/'train.bin').exists():prepare_bytes(spec['corpus'],root)
    return load_byte_data(root)

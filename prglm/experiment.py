import argparse, json, math, random, time, io
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .config import Config
from .models import make_model,backend
from .data import prepare,load_tokenizer


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def batches(ids,c,batch,steps,seed):
    rng=np.random.default_rng(seed)
    for _ in range(steps):
        starts=rng.integers(0,len(ids)-c.context-1,size=batch)
        x=np.stack([ids[i:i+c.context] for i in starts]).astype(np.int64)
        y=np.stack([ids[i+1:i+c.context+1] for i in starts]).astype(np.int64)
        yield torch.from_numpy(x),torch.from_numpy(y)


@torch.no_grad()
def evaluate(model,ids,c,device,steps=4,batch=4):
    model.eval(); losses=[]
    for x,y in batches(ids,c,batch,steps,1234):
        x,y=x.to(device),y.to(device)
        pred,_=model(x)
        losses.append(F.cross_entropy(pred.flatten(0,1),y.flatten()).item())
    return sum(losses)/len(losses)


def packed_bytes(model,name,c):
    count=sum(p.numel() for p in model.parameters())
    if name!='prg': return count*4
    # Ternary stencil stores 2 bits/edge; all other tensors remain float32.
    edge=model.local_edge.numel()
    return (count-edge)*4+math.ceil(edge*(2 if c.edge_type=='ternary' else 32)/8)


def train_one(name,c,train_ids,val_ids,out,steps=8,batch=4,lr=3e-4):
    seed_all(c.seed); device=backend(); model=make_model(name,c).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    history=[]; start=time.perf_counter()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for step,(x,y) in enumerate(batches(train_ids,c,batch,steps,c.seed),1):
        model.train(); x,y=x.to(device),y.to(device)
        pred,_=model(x)
        loss=F.cross_entropy(pred.flatten(0,1),y.flatten())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        history.append({"step":step,"train_loss":loss.item()})
    elapsed=time.perf_counter()-start
    val=evaluate(model,val_ids,c,device)
    path=out/f"{name}.pt"
    torch.save({"name":name,"config":c.dict(),"state":model.state_dict()},path)
    params=sum(p.numel() for p in model.parameters())
    metrics={"model":name,"train_loss":history[-1]["train_loss"],"val_loss":val,
             "val_ppl":math.exp(min(val,50)),"parameters":params,
             "serialized_bytes":path.stat().st_size,"packed_bytes":packed_bytes(model,name,c),
             "training_tokens_per_sec":steps*batch*c.context/elapsed,
             "peak_memory_bytes":torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
             "device":str(device),"steps":steps,"seed":c.seed,"history":history}
    # Timed inference uses the same one-token autoregressive loop for every model.
    from .data import load_tokenizer
    from .inference import generate
    model.eval()
    sample=generate(model,load_tokenizer(out/'tokenizer.json'),'Once upon a time',max_tokens=4,seed=c.seed)
    timed=generate(model,load_tokenizer(out/'tokenizer.json'),'Once upon a time',max_tokens=8,seed=c.seed,record_trace=False)
    metrics['inference_tokens_per_sec']=timed['tokens_per_second']
    metrics['inference_latency_seconds_per_token']=timed['latency_seconds']/8
    if name=='prg':
        metrics['trajectory_example']=sample['timeline'][0]['steps']
        steps_trace=[s for token in sample['timeline'] for s in token['steps']]
        metrics['average_active_regions_per_cycle']=sum(len(s['active']) for s in steps_trace)/len(steps_trace)
        metrics['average_recurrent_cycles_per_token']=sum(bool(s['active']) for s in steps_trace)/4
        metrics['region_visits']=[sum(i in s['active'] for s in steps_trace) for i in range(c.regions)]
        metrics['return_frequency']=sum(s['chosen_route'][i]==2 for s in steps_trace for i in s['active'])/max(1,sum(len(s['active']) for s in steps_trace))
        metrics['recurrence_frequency']=sum(s['chosen_route'][i]==0 for s in steps_trace for i in s['active'])/max(1,sum(len(s['active']) for s in steps_trace))
        metrics['fatigue_mean']=sum(sum(s['fatigue']) for s in steps_trace)/(len(steps_trace)*c.regions)
        metrics['trajectory_lengths']=[sum(bool(s['active']) for s in token['steps']) for token in sample['timeline']]
        metrics['active_region_ratio']=metrics['average_active_regions_per_cycle']/c.regions
        import math as _math
        metrics['routing_entropy']=sum(-sum(p*_math.log(max(p,1e-12)) for p in s['route_probability'][i]) for s in steps_trace for i in s['active'])/max(1,sum(len(s['active']) for s in steps_trace))
        matrix=[[0 for _ in range(c.regions)] for _ in range(c.regions)]
        for s in steps_trace:
            for i in s['active']:
                destination=s['destination'][i]
                if destination>=0: matrix[i][destination]+=1
        metrics['transition_matrix']=matrix
    return metrics


def run(args):
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tokpath=out/'tokenizer.json'
    if not tokpath.exists(): vocab=prepare(args.corpus,out,args.vocab)
    else: vocab=len(load_tokenizer(tokpath).get_vocab())
    c=Config(vocab_size=vocab,context=args.context,regions=args.regions,region_size=args.region_size,
             max_cycles=args.cycles,d_model=args.width,layers=args.layers,
             initial_regions=1 if args.single_region else min(2,args.regions),seed=args.seed,
             edge_type=args.edge_type,fatigue=not args.fatigue_off,
             stochastic_recurrence=not args.recurrence_off,local_router=not args.local_router_off,
             fixed_recurrence_probability=args.fixed_recurrence,
             fatigue_strength=args.fatigue_strength,recovery=args.recovery)
    train_ids=np.fromfile(out/'train.bin',dtype=np.int32)
    val_ids=np.fromfile(out/'val.bin',dtype=np.int32)
    results=[]
    for name in args.models:
        result=train_one(name,c,train_ids,val_ids,out,args.steps,args.batch)
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)
    (out/'results.json').write_text(json.dumps({"config":c.dict(),"corpus":args.corpus,"results":results},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',default='data/raw/tinyshakespeare.txt')
    p.add_argument('--out',default='runs/smoke')
    p.add_argument('--steps',type=int,default=8); p.add_argument('--batch',type=int,default=4)
    p.add_argument('--context',type=int,default=16); p.add_argument('--vocab',type=int,default=2048)
    p.add_argument('--regions',type=int,default=4); p.add_argument('--region-size',type=int,default=32)
    p.add_argument('--cycles',type=int,default=3); p.add_argument('--width',type=int,default=64)
    p.add_argument('--layers',type=int,default=2); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--models',nargs='+',default=['transformer','looped','prg'])
    p.add_argument('--single-region',action='store_true')
    p.add_argument('--fatigue-off',action='store_true')
    p.add_argument('--recurrence-off',action='store_true')
    p.add_argument('--local-router-off',action='store_true')
    p.add_argument('--edge-type',choices=['ternary','float'],default='ternary')
    p.add_argument('--fixed-recurrence',type=float)
    p.add_argument('--fatigue-strength',type=float,default=.5)
    p.add_argument('--recovery',type=float,default=.8)
    run(p.parse_args())


if __name__=='__main__': main()

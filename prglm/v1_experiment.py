"""Byte-vocabulary comparison, with v0 preserved and v1 evaluated in three modes."""
import argparse,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .byte_data import prepare_bytes,load_byte_data,ByteTokenizer
from .config import Config
from .models import TransformerLM,PRGLM,backend
from .v1_config import V1Config
from .v1_model import PRGLMv1
from .v1_memory import memory_breakdown,format_memory_report


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def batches(ids,positions,context,batch_size):
    for chunk in range(0,len(positions),batch_size):
        starts=positions[chunk:chunk+batch_size]
        x=np.stack([ids[i:i+context] for i in starts]).astype(np.int64)
        y=np.stack([ids[i+1:i+context+1] for i in starts]).astype(np.int64)
        yield torch.from_numpy(x),torch.from_numpy(y)


def validation_positions(ids,context,max_tokens):
    count=min(max_tokens//context,(len(ids)-context-1)//context)
    return np.arange(count,dtype=np.int64)*context


@torch.no_grad()
def evaluate(model,name,ids,positions,context,batch_size,device,seed):
    model.eval();modes=['expected','argmax','sampled'] if name=='prg_v1' else ['sampled']
    results={}
    for mode in modes:
        weighted=0.;total=0
        for batch_idx,(x,y) in enumerate(batches(ids,positions,context,batch_size)):
            x=x.to(device);y=y.to(device)
            if name=='prg_v1':pred,_=model(x,routing_mode=mode,seed=seed+batch_idx)
            elif name=='prg':pred,_=model(x,stochastic=True,seed=seed+batch_idx)
            else:pred,_=model(x)
            loss=F.cross_entropy(pred.flatten(0,1),y.flatten(),reduction='sum')
            weighted+=float(loss);total+=y.numel()
        results[mode]={'loss':weighted/total,'ppl':math.exp(min(weighted/total,50)),
                       'validation_tokens':total}
    return results


def v1_diagnostics(model,ids,device,context,seed):
    model.eval()
    x=torch.tensor(ids[:context].astype(np.int64)[None],device=device)
    with torch.no_grad():
        _,timeline=model(x,trace=True,routing_mode='sampled',seed=seed)
    records=model.last_diagnostics
    monte_carlo_counts=[]
    with torch.no_grad():
        for draw in range(8):
            model(x,routing_mode='sampled',seed=seed+1000+draw)
            monte_carlo_counts.append(sum(len(d['visited_regions']) for d in model.last_diagnostics)/x.shape[1])
    c=model.c;tokens=len(records);region_counts=np.zeros(c.regions,dtype=np.int64)
    edge_visits=sum(d['edge_visits'] for d in records)
    nonzero_edge_visits=sum(d.get('nonzero_edge_visits',0) for d in records)
    actual_visits=0;base=[];expected=[];acc=[];contrib=[]
    for d in records:
        region_counts+=np.asarray(d['region_visits'][0],dtype=np.int64)
        actual_visits+=len(d['visited_regions'])
        base.extend(d['base_local_probability']);expected.extend(d['expected_nonoutput_probability'])
        acc.extend(d['accumulator_magnitudes']);contrib.extend(d['contribution_magnitudes'])
    expected_counts=[];actual_counts=[];visit_depth=[];acc_at_visit=[];unique_nodes=[]
    for token in timeline:
        by_walker={};depth={};touched=set()
        for step in token:
            for event in step['events']:
                by_walker.setdefault(event['walker'],[]).append(event)
                touched.update((event['region'],node) for node in event['node_indices'])
                depth[event['region']]=depth.get(event['region'],0)+1
                visit_depth.append(depth[event['region']])
                acc_at_visit.append(event['accumulator_magnitude'])
        unique_nodes.append(len(touched))
        for events in by_walker.values():
            survival=1.;count=0.
            for event in events:
                count+=survival
                survival*=1-event['route_probability'][-1]
            expected_counts.append(count)
            actual_counts.append(len(events))
    def corr(a,b):
        if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
        return float(np.corrcoef(a,b)[0,1])
    return {'diagnostic_tokens':tokens,'total_nodes':c.regions*c.region_size,
        'total_local_edges':c.regions*c.region_size*c.edges_per_node,
        'nonzero_ternary_local_edges':int((model.local_edge.detach().abs()>.33).sum()),
        'total_region_edges':c.regions*c.gateways,
        'active_regions_per_token':actual_visits/tokens,
        'active_nodes_per_token':float(np.mean(unique_nodes)),
        'node_visits_per_token':actual_visits*c.active_nodes/tokens,
        'traversed_local_edges_per_token':edge_visits/tokens,
        'effective_nonzero_edges_per_token':nonzero_edge_visits/tokens,
        'active_node_ratio_per_cycle':actual_visits*c.active_nodes/(tokens*c.max_cycles*c.regions*c.region_size),
        'active_edge_ratio_per_cycle':edge_visits/(tokens*c.max_cycles*c.regions*c.region_size*c.edges_per_node),
        'region_utilization':region_counts.tolist(),
        'actual_traversal_count_per_token':actual_visits/tokens,
        'monte_carlo_expected_traversal_count_per_token':float(np.mean(monte_carlo_counts)),
        'monte_carlo_traversal_std':float(np.std(monte_carlo_counts,ddof=1)),
        'expected_traversal_count_per_walker':float(np.mean(expected_counts)) if expected_counts else 0.,
        'actual_traversal_count_per_walker':float(np.mean(actual_counts)) if actual_counts else 0.,
        'base_local_transition_probability':float(np.mean(base)) if base else 0.,
        'mean_accumulator_magnitude':float(np.mean(acc)) if acc else 0.,
        'mean_output_contribution_magnitude':float(np.mean(contrib)) if contrib else 0.,
        'probability_accumulator_correlation':corr(base,acc),
        'visits_accumulator_correlation':corr(visit_depth,acc_at_visit),
        'expected_actual_traversal_correlation':corr(expected_counts,actual_counts),
        'accumulator_contribution_correlation':corr(acc,contrib),
        'trajectory_example':timeline[0] if timeline else []}


def model_config(name,args,seed,width=None,layers=None):
    if name=='prg_v1':
        return V1Config(context=args.context,regions=args.regions,region_size=args.region_size,
            edges_per_node=args.edges_per_node,gateways=args.gateways,active_nodes=args.active_nodes,
            initial_regions=args.initial_regions,max_cycles=args.cycles,d_model=args.v1_width,
            recurrence=not args.recurrence_off,accumulation=not args.accumulation_off,
            fatigue=not args.fatigue_off,seed=seed)
    return Config(vocab_size=256,context=args.context,regions=args.v0_regions,
        region_size=args.v0_region_size,max_cycles=args.cycles,
        d_model=width or args.transformer_width,layers=layers or args.transformer_layers,
        heads=4,seed=seed)


def construct(name,config):
    if name=='prg_v1':return PRGLMv1(config)
    if name=='prg':return PRGLM(config)
    return TransformerLM(config)


def matched_transformers(args):
    target=memory_breakdown(PRGLMv1(model_config('prg_v1',args,args.seeds[0])),'prg_v1')
    options=[]
    for layers in (1,2,3):
        for width in range(16,129,4):
            config=model_config('transformer',args,args.seeds[0],width,layers)
            memory=memory_breakdown(TransformerLM(config),'transformer')
            options.append((width,layers,memory))
    core=min(options,key=lambda x:abs(x[2]['core_static_packed_bytes']-target['core_static_packed_bytes']))
    total=min(options,key=lambda x:abs(x[2]['peak_theoretical_inference_bytes']-target['peak_theoretical_inference_bytes']))
    return {'transformer_core':core[:2],'transformer_total':total[:2]}


def train_one(name,args,train_ids,val_ids,positions,out,seed,width=None,layers=None):
    seed_all(seed);config=model_config(name,args,seed,width,layers).validate()
    device=backend();model=construct(name,config).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,len(train_ids)-args.context-1,size=args.steps*args.batch)
    history=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for step,(x,y) in enumerate(batches(train_ids,starts,args.context,args.batch),1):
        model.train();x=x.to(device);y=y.to(device)
        logits,_=model(x)
        loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        history.append({'step':step,'train_loss':float(loss)})
    elapsed=time.perf_counter()-start
    evaluation=evaluate(model,name,val_ids,positions,args.context,args.val_batch,device,seed)
    path=out/f'{name}.pt'
    torch.save({'name':name,'config':config.dict(),'state':model.state_dict()},path)
    memory=memory_breakdown(model,name if name in ('prg','prg_v1') else 'transformer')
    row={'model':name,'config':config.dict(),'seed':seed,'steps':args.steps,
        'training_tokens':args.steps*args.batch*args.context,
        'train_loss':history[-1]['train_loss'],'validation':evaluation,
        'train_tokens_per_second':args.steps*args.batch*args.context/elapsed,
        'serialized_bytes':path.stat().st_size,'memory':memory,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
        'history':history}
    from .inference import generate
    sample=generate(model,ByteTokenizer(),'To be',max_tokens=8,seed=seed,record_trace=False)
    row['inference_tokens_per_second']=sample['tokens_per_second']
    row['inference_latency_seconds_per_token']=sample['latency_seconds']/8
    if name=='prg_v1':row['dynamics']=v1_diagnostics(model,val_ids,device,args.context,seed)
    return row


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if not (out/'train.bin').exists():prepare_bytes(args.corpus,out)
    train_ids,val_ids=load_byte_data(out)
    positions=validation_positions(val_ids,args.context,args.validation_tokens)
    if len(positions)*args.context<4096 and not args.smoke:
        raise ValueError('final validation requires at least 4096 held-out tokens')
    matched=matched_transformers(args)
    print('matched configurations',matched,flush=True)
    all_results=[]
    for seed in args.seeds:
        seed_dir=out/f'seed{seed}';seed_dir.mkdir(exist_ok=True)
        ByteTokenizer().save(seed_dir/'tokenizer.json')
        for name in args.models:
            width,layers=matched[name] if name in matched else (None,None)
            row=train_one(name,args,train_ids,val_ids,positions,seed_dir,seed,width,layers)
            all_results.append(row)
            if seed==args.seeds[0]:print(format_memory_report(name,row['memory']),flush=True)
            print(json.dumps({'model':name,'seed':seed,'val':row['validation'],
                'static_bytes':row['memory']['total_static_packed_bytes']}),flush=True)
    payload={'corpus':args.corpus,'tokenizer':'utf8-byte-256',
        'validation_positions':'contiguous nonoverlapping windows from held-out final 10%',
        'validation_tokens':len(positions)*args.context,
        'matched_transformers':matched,'results':all_results}
    (out/'results.json').write_text(json.dumps(payload,indent=2))
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',default='data/raw/tinyshakespeare.txt')
    p.add_argument('--out',default='runs/v1_smoke')
    p.add_argument('--models',nargs='+',default=['transformer','prg','prg_v1'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--steps',type=int,default=4);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--context',type=int,default=16);p.add_argument('--validation-tokens',type=int,default=4096)
    p.add_argument('--val-batch',type=int,default=8);p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--regions',type=int,default=4);p.add_argument('--region-size',type=int,default=64)
    p.add_argument('--edges-per-node',type=int,default=4);p.add_argument('--gateways',type=int,default=2)
    p.add_argument('--active-nodes',type=int,default=8);p.add_argument('--initial-regions',type=int,default=2)
    p.add_argument('--cycles',type=int,default=3);p.add_argument('--v1-width',type=int,default=64)
    p.add_argument('--v0-regions',type=int,default=8);p.add_argument('--v0-region-size',type=int,default=64)
    p.add_argument('--transformer-width',type=int,default=64);p.add_argument('--transformer-layers',type=int,default=2)
    p.add_argument('--recurrence-off',action='store_true');p.add_argument('--accumulation-off',action='store_true')
    p.add_argument('--fatigue-off',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    allowed={'transformer','transformer_core','transformer_total','prg','prg_v1'}
    if not set(args.models)<=allowed:raise ValueError(f'models must be from {sorted(allowed)}')
    run(args)


if __name__=='__main__':main()

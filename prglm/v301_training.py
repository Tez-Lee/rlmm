"""Duration-only continuation of immutable v3, with separate state/results."""
import hashlib
import json
import math
import resource
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .byte_data import load_byte_data
from .learning_curve import atomic_json,atomic_checkpoint,capture_rng,restore_rng
from .models import backend
from .v1_experiment import batches,validation_positions,seed_all
from .v3_training import construct,clip,evaluate,gradient_diagnostic,runtime
from .v3_metrics import memory,pack_model


class ResourceLimitError(RuntimeError):pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(spec,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    source=Path(spec['source_root']);original=json.loads((source/'protocol.json').read_text())
    for p,h in original['implementation_sha256'].items():
        if sha(p)!=h:raise ValueError('frozen v3 implementation changed: '+p)
    protocol={'extension':spec,'source_protocol':original,
        'extension_implementation_sha256':sha(__file__),'dataset_sha256':sha(original['spec']['corpus']),
        'source_train_sha256':sha(source/'train.bin'),'source_val_sha256':sha(source/'val.bin')}
    if protocol['dataset_sha256']!=original['dataset_sha256']:raise ValueError('corpus changed')
    path=root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('immutable continuation protocol changed')
    if not path.exists():atomic_json(path,protocol)
    return original['spec'],*load_byte_data(source)


def coverage_window(ids,context,budget,c):
    visited=np.zeros(c.nodes,dtype=bool)
    for start in validation_positions(ids,context,budget):
        address=0
        for token in ids[start:start+context]:
            address=(address*131+int(token)+1)%c.nodes
            visited[(address+np.arange(c.active_sources)*37)%c.nodes]=True
    return {'validation_bytes':budget,'unique_structural_edges':int(visited.sum())*c.slots,
        'visited_edge_fraction':float(visited.mean()),'definition':'Exact topology-defined distinct edges in fixed held-out windows; all slots of each activated source are used.'}


def run_one(spec,root,family,scale,seed,target,stop_after=None,callback=None):
    base,train,val=initialize(spec,root);directory=Path(root)/scale/family/f'seed{seed}'
    directory.mkdir(parents=True,exist_ok=True);result=directory/'results.json'
    rows=json.loads(result.read_text())['results'] if result.exists() else []
    target=stop_after or target
    if any(r['training_tokens']==target for r in rows):return rows
    seed_all(seed);device=backend();model=construct(base,family,scale).to(device)
    optimizers=[torch.optim.SparseAdam([model.edges.weight],lr=base['lr']),
        torch.optim.AdamW([p for n,p in model.named_parameters() if n!='edges.weight' and p.requires_grad],lr=base['lr'])]
    unit=base['batch']*model.c.context
    if any(t%unit for t in spec['milestones']):raise ValueError('milestone alignment')
    starts=np.random.default_rng(seed).integers(0,len(train)-model.c.context-1,
        size=max(spec['milestones'])//unit*base['batch'])
    stream_hash=hashlib.sha256(starts.astype('<i8').tobytes()).hexdigest()
    old_prefix=starts[:spec['base_training_bytes']//unit*base['batch']]
    prefix_hash=hashlib.sha256(old_prefix.astype('<i8').tobytes()).hexdigest()
    latest=directory/'latest.pt';source=Path(spec['source_root'])/scale/family/f'seed{seed}'/f"tokens_{spec['base_training_bytes']}.pt"
    path=latest if latest.exists() else source
    if not path.exists():raise FileNotFoundError('Source checkpoint missing; no automatic retraining: '+str(path))
    old=torch.load(path,map_location='cpu',weights_only=False)
    if old['family']!=family or old['scale']!=scale or old['seed']!=seed:raise ValueError('checkpoint identity')
    if latest.exists():
        if old['extension_spec']!=spec or old['stream_sha256']!=stream_hash:raise ValueError('extension resume mismatch')
        coverage=old['coverage_counts'].copy();diagnostics=old['diagnostics'];elapsed=old['continuation_seconds'];origin=old['origin']
    else:
        if old['spec']!=base or old['training_tokens']!=spec['base_training_bytes'] or old['stream_sha256']!=prefix_hash:raise ValueError('source/prefix mismatch')
        coverage=np.zeros(model.c.nodes,dtype=np.uint32);diagnostics=[];elapsed=0.
        origin={'checkpoint_identifier':str(source),'checkpoint_sha256':sha(source),'source_stream_sha256':prefix_hash,
            'baseline_training_seconds':old['training_seconds'],'baseline_coverage':'not recorded in v3; continuation coverage starts at819200, not whole-history coverage'}
        atomic_json(directory/'origin.json',origin)
    model.load_state_dict(old['state'])
    for optimizer,state in zip(optimizers,old['optimizers']):optimizer.load_state_dict(state)
    step=old['step'];history=old['history'];restore_rng(old['rng']);del old
    def checkpoint(at):
        if shutil.disk_usage(root).free<spec['resource_limits']['minimum_free_disk_bytes']:
            raise ResourceLimitError('preregistered free-disk bound before checkpoint')
        return {'extension_spec':spec,'spec':base,'config':model.c.dict(),'family':family,'scale':scale,'seed':seed,
            'step':at,'training_tokens':at*unit,'state':model.state_dict(),'optimizers':[o.state_dict() for o in optimizers],
            'rng':capture_rng(),'history':history,'stream_sha256':stream_hash,'prefix_stream_sha256':prefix_hash,
            'coverage_counts':coverage,'diagnostics':diagnostics,'continuation_seconds':elapsed,'origin':origin,
            'config_sha256':hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
            'dataset_sha256':sha(base['corpus']),'previous_milestone':max([spec['base_training_bytes']]+[r['training_tokens'] for r in rows])}
    def assess(at):
        tokens=at*unit
        if any(r['training_tokens']==tokens for r in rows):return
        rng=capture_rng();cp=directory/f'tokens_{tokens}.pt'
        if not cp.exists():atomic_checkpoint(cp,checkpoint(at))
        validation=evaluate(model,val,base,seed,device);mem=memory(model)
        export=directory/f'model_tokens_{tokens}.pt'
        if not export.exists():atomic_checkpoint(export,pack_model(model))
        touched=coverage[coverage>0]
        coverage_stats={'unique_structural_edges_since_resume':int((coverage>0).sum())*model.c.slots,
            'visited_edge_fraction_since_resume':float((coverage>0).mean()),
            'per_source_optimizer_update_count_quantiles':np.quantile(touched,[0,.25,.5,.75,1]).tolist() if len(touched) else [],
            'coverage_counter_training_only_bytes':coverage.nbytes,
            'definition':'Exact distinct source rows present in coalesced sparse gradients; all slots are activated. Counts are optimizer steps touching each row, not nonzero parameter updates. Baseline coverage is not reconstructed.'}
        row={'family':family,'scale':scale,'seed':seed,'training_tokens':tokens,'edge_slots':model.c.nodes*model.c.slots,
            'train_loss':history[-1]['loss'],'validation':validation,'memory':mem,'backend':str(device),
            'checkpoint_identifier':str(cp),'checkpoint_sha256':sha(cp),'origin':origin,
            'stream_sha256':stream_hash,'prefix_stream_sha256':prefix_hash,'gradient_diagnostics':gradient_diagnostic(model),
            'undertraining_diagnostics':{'training_coverage':coverage_stats,'sampled_gradient_updates':diagnostics[-100:],
                'held_out_coverage':coverage_window(val,model.c.context,base['validation_bytes'][0],model.c)},
            'continuation_seconds':elapsed,'training_seconds':elapsed+origin['baseline_training_seconds'],
            'training_tokens_per_second':(tokens-spec['base_training_bytes'])/max(elapsed,1e-12),
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'actual_pytorch_resume_bytes':cp.stat().st_size,'actual_inference_export_bytes':export.stat().st_size,
            'optimizer_training_only_tensor_bytes':sum(v.numel()*v.element_size() for o in optimizers for s in o.state.values() for v in s.values() if isinstance(v,torch.Tensor)),
            'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated() if device.type=='cuda' else None,
            'peak_reserved_cuda_bytes':torch.cuda.max_memory_reserved() if device.type=='cuda' else None,
            'runtime':runtime(model,seed,device)}
        restore_rng(rng);rows.append(row);atomic_json(result,{'results':rows})
        if callback:callback()
    if step*unit in spec['milestones']:assess(step)
    if step*unit>=target:return rows
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for step,(x,y) in enumerate(batches(train,starts[step*base['batch']:],model.c.context,base['batch']),step+1):
        model.train()
        for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
        prediction,_=model(x.to(device));loss=F.cross_entropy(prediction.flatten(0,1),y.to(device).flatten())
        loss.backward();model.edges.weight.grad=model.edges.weight.grad.coalesce()
        touched=model.edges.weight.grad.indices()[0].cpu().numpy();coverage[touched]+=1
        probe=step%spec['diagnostic_interval']==0
        if probe:
            norm=math.sqrt(sum(float((p.grad.coalesce().values() if p.grad.is_sparse else p.grad).detach().square().sum()) for p in model.parameters() if p.grad is not None))
            sample=torch.as_tensor(touched[:64],device=device);before=model.edges.weight.detach()[sample].clone()
        clip(model)
        for optimizer in optimizers:optimizer.step()
        if probe:
            delta=(model.edges.weight.detach()[sample]-before).abs()
            diagnostics.append({'step':step,'pre_clip_gradient_norm':norm,'post_clip_norm':min(norm,1.),
                'sampled_edge_latent_mean_abs_update':float(delta.mean()),'sampled_edge_latent_max_abs_update':float(delta.max()),
                'sample_rows':int(len(sample)),'sampling':'first64 sorted touched source rows every100 steps, not an unbiased graph-wide estimate'})
        history.append({'step':step,'loss':float(loss.detach())})
        tokens=step*unit
        if step%base['checkpoint_interval']==0 or tokens in spec['milestones'] or tokens>=target:
            elapsed+=time.perf_counter()-start;atomic_checkpoint(latest,checkpoint(step))
            atomic_json(directory/'progress.json',{'training_tokens':tokens,'target_tokens':max(spec['milestones'])})
            if tokens in spec['milestones']:assess(step)
            start=time.perf_counter()
        if tokens>=target:break
    return rows

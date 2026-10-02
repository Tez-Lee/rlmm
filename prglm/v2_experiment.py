"""v2 phase gate and diagnostic helpers; neural training uses the common runner."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .v1_experiment import v1_diagnostics
from .learning_curve import run_one,atomic_json
from .curve_report import collect
from .byte_data import prepare_bytes,load_byte_data


def dynamics(model,ids,device,context,seed):
    result=v1_diagnostics(model,ids,device,context,seed)
    x=torch.tensor(ids[:context].astype(np.int64)[None],device=device)
    with torch.no_grad():_,trace=model(x,trace=True,routing_mode='sampled',seed=seed)
    events=[e for token in trace for step in token for e in step['events']]
    result['routing_entropy']=float(np.mean([-sum(p*np.log(max(p,1e-30)) for p in e['route_probability']) for e in events]))
    result['recurrence_count']=sum(e['action']==0 for e in events)
    result['return_count']=sum(e['action']==model.c.gateways+1 for e in events)
    result['recurrence_per_token']=result['recurrence_count']/context
    result['return_per_token']=result['return_count']/context
    result['trajectory']=trace
    return result


def phase_a_gate(root):
    result=collect(root)
    if not result['complete']:raise ValueError('Phase A must finish before v2 main training')
    for row in result['results']:
        directory=root/f"seed{row['seed']}"/row['model']
        if not (directory/f"tokens_{row['training_tokens']}.pt").exists():
            raise ValueError('Phase A milestone checkpoint missing')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/mutable_topology.json')
    p.add_argument('--out',default='runs/mutable_v2');p.add_argument('--phase-a',default='runs/phase_a')
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--models',nargs='+')
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if not args.smoke:phase_a_gate(Path(args.phase_a))
    spec=json.loads(Path(args.config).read_text());root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    manifest=root/'protocol.json'
    import hashlib
    protocol={'spec':spec,'corpus_sha256':hashlib.sha256(Path(spec['corpus']).read_bytes()).hexdigest(),
              'phase_a':str(args.phase_a),'smoke':args.smoke}
    if manifest.exists() and json.loads(manifest.read_text())!=protocol:raise ValueError('protocol mismatch')
    atomic_json(manifest,protocol)
    if not (root/'train.bin').exists():prepare_bytes(spec['corpus'],root)
    train,val=load_byte_data(root)
    import fcntl
    for seed in args.seeds or spec['seeds']:
        for name in args.models or spec['models']:
            with (root/f'{name}_seed{seed}.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                run_one(spec,root,name,seed,train,val)
    print('Requested v2 jobs completed',flush=True)

if __name__=='__main__':main()

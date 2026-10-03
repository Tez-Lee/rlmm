"""Finite autonomous duration-only study: stage barriers, resume, report, push."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
REPO=Path(__file__).resolve().parents[1]


def freeze(out):
    from prglm.learning_curve import atomic_json
    p=out/'frozen_manifest.json'
    if p.exists():return
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only','-z','HEAD']).decode().split('\0')
    files=[Path(f) for f in tracked if f and f!='README.md']
    for directory in Path('runs').glob('*'):
        if directory.is_dir() and directory.name!='v3_0_1' and not directory.name.startswith('v301_smoke'):
            files += [f for f in directory.rglob('*') if f.is_file() and (f.suffix=='.pt' or f.name in ('results.json','protocol.json','train.bin','val.bin'))]
    atomic_json(p,{'base_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))},
        'readme_prefix':Path('README.md').read_text()})


def verify(out):
    frozen=json.loads((out/'frozen_manifest.json').read_text())
    for path,h in frozen['files'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen file changed: '+path)
    if not Path('README.md').read_text().startswith(frozen['readme_prefix']):raise RuntimeError('Original README altered')
    return len(frozen['files'])


def publish(out,message,final=False):
    from prglm.learning_curve import atomic_json
    failure=out/'publish_failure.json'
    if failure.exists():failure.unlink()
    paths=['configs/extended_training.json','prglm/v301_training.py','prglm/v301_report.py',
        'scripts/run_extended_training.py','tests/test_v301.py',str(out)]
    if final:paths.append('README.md')
    subprocess.run(['git','add','--',*paths],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True,capture_output=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:subprocess.run(['git','commit','-m',message],check=True,capture_output=True)
    try:subprocess.run(['git','push','origin','main'],check=True,capture_output=True,timeout=180)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
        atomic_json(failure,{'action':'push origin main','error_type':type(e).__name__,'note':'No credentials printed or stored; training continues.'});return False
    return True


def final_readme(data):
    path=Path('README.md');text=path.read_text();marker='\n## PRG-v3.0.1 Extended-Training Verification\n'
    if marker in text:return
    decision=data['decision'];final=max(data['spec']['milestones'])
    text+=marker+'\nThis duration-only verification tests whether v3 S3/S4 same-edge crossover was undertraining under the uniform819200-byte budget. Architecture, topology, optimizer, learning rate, stream sampler, state, byte vocabulary and validation are unchanged. All33 selected runs resume original model/two-optimizer/RNG/cursor checkpoints. The preregistered final budget is13107200 bytes.\n\n'
    text+=f"Complete={data['complete']}; latest completed bytes={data['latest_completed_training_bytes']}; failures/resource limits={len(data['failures'])}. Final same-edge crossover: {decision['same_edge_crossover']}; final same-memory crossover: {decision['same_memory_crossover']}. {decision['case']}\n\n"
    text+='Every seed, milestone, paired difference, coverage diagnostic, runtime limitation, six static curves and all19 research answers: [research/v3_0_1/REPORT.md](research/v3_0_1/REPORT.md). [Machine results](research/v3_0_1/results.json). Original v3 results and earlier disclosure/prior-art sections remain unchanged.\n\nResume: `OMP_NUM_THREADS=1 PYTHONPATH=.deps:. python scripts/run_extended_training.py --push`. Completed milestones skip. Use `--retry-failed` only for previously recorded failures/resource limits; source checkpoints are never retrained automatically.\n'
    path.write_text(text)


def audit(data,root,out):
    from prglm.learning_curve import atomic_json
    from prglm.v301_training import sha
    import torch
    hashes={seed:set() for seed in data['spec']['seeds']};verified=0
    for row in data['results']:
        if row.get('historical_baseline'):continue
        assert sha(row['checkpoint_identifier'])==row['checkpoint_sha256']
        assert sha(row['origin']['checkpoint_identifier'])==row['origin']['checkpoint_sha256']
        hashes[row['seed']].add(row['stream_sha256']);verified+=1
        for budget in data['base_spec']['validation_bytes']:
            assert row['validation'][str(budget)]['validation_tokens']==budget
            assert row['validation'][str(budget)]['repetition']['identity_failures']==0
    for p in root.glob('S*/*/seed*/latest.pt'):
        saved=torch.load(p,map_location='cpu',weights_only=False)
        assert len(saved['optimizers'])==2 and saved['optimizers'][0]['state'] and saved['rng']['torch'].numel()
        assert saved['training_tokens']==saved['step']*data['base_spec']['batch']*data['base_spec']['model']['context']
        assert all(torch.isfinite(v).all() for v in saved['state'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
    assert all(len(h)<=1 for h in hashes.values())
    atomic_json(out/'integrity.json',{'complete':data['complete'],'verified_extension_milestones':verified,'frozen_file_count':verify(out),'per_seed_stream_hashes':{str(k):list(v) for k,v in hashes.items()}})


def main():
    os.chdir(REPO);p=argparse.ArgumentParser();p.add_argument('--config',default='configs/extended_training.json')
    p.add_argument('--out',default='runs/v3_0_1');p.add_argument('--report',default='research/v3_0_1')
    p.add_argument('--family');p.add_argument('--scale');p.add_argument('--seed',type=int);p.add_argument('--target',type=int)
    p.add_argument('--push',action='store_true');p.add_argument('--retry-failed',action='store_true');args=p.parse_args()
    spec=json.loads(Path(args.config).read_text());root=Path(args.out);out=Path(args.report)
    import torch;torch.set_num_threads(spec['threads'])
    from prglm.v301_training import initialize,run_one,ResourceLimitError
    from prglm.v301_report import export
    from prglm.learning_curve import atomic_json
    initialize(spec,root);out.mkdir(parents=True,exist_ok=True)
    def report():
        with (root/'report.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);return export(root,out)
    if args.family:
        directory=root/args.scale/args.family/f'seed{args.seed}';directory.mkdir(parents=True,exist_ok=True)
        with (directory/'job.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:run_one(spec,root,args.family,args.scale,args.seed,args.target,callback=report)
            except ResourceLimitError as e:
                atomic_json(directory/'failure.json',{'family':args.family,'scale':args.scale,'seed':args.seed,'target_bytes':args.target,'classification':'resource-limited','reason':str(e)})
        return
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze(out);verify(out);report()
        if args.push:publish(out,'Implement v3.0.1 frozen-checkpoint duration verification')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        def job(family,scale,seed,target):
            directory=root/scale/family/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True);failure=directory/'failure.json'
            if failure.exists():
                if not args.retry_failed:return
                history=directory/'failure_history.json';items=json.loads(history.read_text()) if history.exists() else []
                items.append(json.loads(failure.read_text()));atomic_json(history,items);failure.unlink()
            result=directory/'results.json'
            if result.exists() and any(r['training_tokens']==target for r in json.loads(result.read_text())['results']):return
            reason=None
            if shutil.disk_usage(root).free<spec['resource_limits']['minimum_free_disk_bytes']:reason='preregistered free-disk bound'
            if reason:
                atomic_json(failure,{'family':family,'scale':scale,'seed':seed,'target_bytes':target,'classification':'resource-limited','reason':reason});return
            cmd=[sys.executable,str(Path(__file__).resolve()),'--config',args.config,'--out',args.out,'--report',args.report,'--family',family,'--scale',scale,'--seed',str(seed),'--target',str(target)]
            code=None;reason=None
            started=time.monotonic()
            def record_wall():
                path=directory/'stage_timings.json'
                timings=json.loads(path.read_text()) if path.exists() else []
                timings.append({'target_bytes':target,'wall_seconds':time.monotonic()-started,'exit_code':code,
                    'definition':'Worker stage including load, checkpoint serialization, training, evaluation and report; retries included.'})
                atomic_json(path,timings)
            for attempt in range(2):
                with (directory/'worker.log').open('a') as log:
                    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    try:code=proc.wait(timeout=spec['resource_limits']['job_stage_wall_seconds'])
                    except subprocess.TimeoutExpired:
                        proc.terminate();proc.wait();code=-15;reason='preregistered six-hour job-stage wall limit'
                if code==0:
                    record_wall();return
                if code<0:reason=reason or 'process killed; external cause unknown';break
            atomic_json(failure,{'family':family,'scale':scale,'seed':seed,'target_bytes':target,'classification':'resource-limited' if code<0 else 'failed-resumable','reason':reason or 'worker failed twice; last normal checkpoint retained','exit_code':code})
            record_wall()
        for target in spec['milestones']:
            with ThreadPoolExecutor(max_workers=spec['workers']) as executor:
                futures=[executor.submit(job,f,s,k,target) for f,scales in spec['combinations'].items() for s in scales for k in spec['seeds']]
                for future in futures:future.result() # Completion waits only, no training-log polling.
            d=report();verify(out)
            if args.push:publish(out,f'Update v3.0.1 through {target/1000000:g}M training bytes')
            print(json.dumps({'milestone_group_finished':target,'completed_milestones':d['completed_extension_milestones'],'failures':len(d['failures'])}),flush=True)
        d=export(root,out,finalize=True);audit(d,root,out);final_readme(d)
        if args.push:publish(out,'Finalize v3.0.1 extended-training crossover verification',True)
        print(json.dumps({'runner_finished':True,'complete':d['complete'],'decision':d['decision']}),flush=True)


if __name__=='__main__':main()

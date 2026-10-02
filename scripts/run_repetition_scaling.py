"""Autonomous scaling supervisor: finite job waits, resume, reports and Git pushes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]


def freeze(out):
    p=out/'frozen_manifest.json'
    if p.exists():return
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only','-z','HEAD']).decode().split('\0')
    files=[Path(f) for f in tracked if f and f!='README.md']
    for root in Path('runs').glob('*'):
        if root.is_dir() and not root.name.startswith('v3'):
            files += [f for f in root.rglob('*') if f.is_file() and (f.suffix=='.pt' or f.name=='results.json')]
    from prglm.learning_curve import atomic_json
    atomic_json(p,{'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(files))},
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()})


def verify_frozen(out):
    files=json.loads((out/'frozen_manifest.json').read_text())['files']
    for f,h in files.items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen artifact changed: '+f)
    return len(files)


def publish(out,message,final=False):
    paths=['prglm/v3_model.py','prglm/v3_metrics.py','prglm/v3_training.py','prglm/v3_report.py',
        'configs/repetition_scaling.json','configs/repetition_smoke.json','tests/test_v3.py',
        'scripts/run_repetition_scaling.py',str(out)]
    if final:paths.append('README.md')
    subprocess.run(['git','add','--',*paths],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True,capture_output=True)
    if subprocess.run(['git','diff','--cached','--quiet']).returncode:
        subprocess.run(['git','commit','-m',message],check=True,capture_output=True)
    try:subprocess.run(['git','push','origin','main'],check=True,capture_output=True,timeout=180)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
        from prglm.learning_curve import atomic_json
        atomic_json(out/'publish_failure.json',{'error_type':type(e).__name__,'action':'git push origin main','note':'Training continues; credential values are never printed/stored.'})
        return False
    return True


def final_readme(data):
    p=Path('README.md');text=p.read_text();marker='\n## PRG-v3 Same-Edge Repetition Scaling Study\n'
    if marker in text:return
    final=max(data['spec']['milestones']);summaries=[s for s in data['summaries'] if s['training_tokens']==final and s['validation_bytes']==4096]
    lines=[marker,
        '**Research question:** Can repeated use of the SAME low-bit source node/destination/edge slot substitute for part of numerical weight magnitude, particularly as model scale grows? This means additive reuse of a frozen reference message, not visiting different edges or ordinary W^k composition.',
        'S0–S4 scale32768→131072→524288→2097152→8388608 edge slots. Float32 single-pass, ternary single-pass and learned bounded repetition share topology and neural blocks. Each has seeds42/43/44 and continuous fixed-data budgets12.8k/51.2k/204.8k/819.2k bytes. Large models may be undertrained.',
        'Scaling tests loss, packed storage, active operations and dynamic state, rather than judging a single small model. Region/gateway/mutable topology/fatigue are omitted to isolate core repetition, not to declare those ideas worthless. No dashboard/UI changes; all v0–v2.1 artifacts stay frozen.',
        '', '| Family | Scale | Final loss | Byte PPL | n |','|---|---|---:|---:|---:|']
    for s in summaries:
        fmt=lambda a:f"{a['mean']:.4f} ± {a['std']:.4f}" if a['std'] is not None else f"{a['mean']:.4f} (n=1)"
        lines.append(f"| {s['family']} | {s['scale']} | {fmt(s['loss'])} | {fmt(s['ppl'])} | {s['loss']['n']} |")
    lines+=['',f"Complete={data['complete']}; recorded failed/resource-limited jobs={len(data['failures'])}. Measured same-edge mean-loss sign-change crossovers={data.get('observed_crossover_scales',[])}. No extrapolated crossover is asserted.",
        'Canonical analysis, all17 answers, paired differences, exact memory matches, ablations, costs and static SVG/PNG figures: [research/v3/REPORT.md](research/v3/REPORT.md). Raw machine results: [research/v3/results.json](research/v3/results.json).',
        'Packed exports actually encode2-bit ternary edges; PyTorch execution decodesFP32 and batches controller candidates. Arithmetic k*s*m equivalence is not nonlinear LM equivalence or a hardware-efficiency claim. Fixed16-channel hashed state and biased ST optimization restrict generality; useful repetition must improve measured quality/memory/compute jointly.',
        '', 'Resume: `OMP_NUM_THREADS=1 PYTHONPATH=.deps:. python scripts/run_repetition_scaling.py --push`. Failed checkpoints remain under ignored runs/v3; completed rows skip. Resource-limited/failed jobs are recorded; `--retry-failed` explicitly retries only such jobs, preserving failure history. Install the project research/test extras on a fresh checkout.',
        'The prior AI-assisted development disclosure, novelty/prior-art caveats and ChatGPT-surfaced literature remain intact. The human author set the SAME-edge hypothesis and research direction; OpenAI models assisted discussion, implementation, testing, automation and reporting. No claim of original AI-generated source code or established architecture novelty is made.']
    p.write_text(text+'\n'.join(lines)+'\n')


def audit(data,root,out):
    import torch
    streams={s:set() for s in data['spec']['seeds']};count=0
    for family in data['spec']['families']:
        for scale in data['spec']['scales']:
            for seed in data['spec']['seeds']:
                directory=root/scale/family/f'seed{seed}'
                p=directory/'results.json'
                if not p.exists():continue
                rows=json.loads(p.read_text())['results']
                for row in rows:
                    checkpoint=Path(row['checkpoint_identifier'])
                    assert checkpoint.exists() and hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
                    gradient=row['gradient_diagnostics']['prefix_embedding_gradient_L1']
                    assert math.isfinite(gradient) and gradient>=0
                    for budget in data['spec']['validation_bytes']:
                        metrics=row['validation'][str(budget)];assert metrics['validation_tokens']==budget
                        assert metrics['repetition']['identity_failures']==0
                    streams[seed].add(row['stream_sha256'])
                if any(row['training_tokens']==max(data['spec']['milestones']) for row in rows):
                    saved=torch.load(directory/'latest.pt',weights_only=False,map_location='cpu')
                    assert saved['training_tokens']==max(data['spec']['milestones'])
                    assert saved['optimizers'][0]['state'] and saved['rng']['torch'].numel()
                    assert all(torch.isfinite(v).all() for v in saved['state'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
                    count+=1
    assert all(len(values)==1 for values in streams.values() if values)
    from prglm.learning_curve import atomic_json
    atomic_json(out/'integrity.json',{'study_complete':data['complete'],'verified_final_checkpoints':count,
        'frozen_artifact_count':verify_frozen(out),'completed_milestones':len(data['results']),
        'stream_hashes':{str(k):list(v) for k,v in streams.items()}})


def main():
    os.chdir(REPO)
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/repetition_scaling.json')
    p.add_argument('--out',default='runs/v3');p.add_argument('--report',default='research/v3')
    p.add_argument('--family');p.add_argument('--scale');p.add_argument('--seed',type=int)
    p.add_argument('--push',action='store_true');p.add_argument('--retry-failed',action='store_true')
    args=p.parse_args();spec=json.loads(Path(args.config).read_text());root=Path(args.out);out=Path(args.report)
    import torch
    torch.set_num_threads(spec['threads'])
    from prglm.v3_training import initialize,run_one
    from prglm.v3_report import export
    from prglm.learning_curve import atomic_json
    train,val=initialize(spec,root);out.mkdir(parents=True,exist_ok=True)
    def report():
        with (root/'report.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);return export(root,out)
    if args.family:
        directory=root/args.scale/args.family/f'seed{args.seed}';directory.mkdir(parents=True,exist_ok=True)
        with (directory/'job.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            run_one(spec,root,args.family,args.scale,args.seed,train,val,report_callback=report)
        return
    with (root/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(out);verify_frozen(out);report()
        if args.push:publish(out,'Implement additive same-edge PRG-v3 scaling study and sanity tests')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        def job(scale,family,seed):
            directory=root/scale/family/f'seed{seed}';directory.mkdir(parents=True,exist_ok=True)
            fail=directory/'failure.json'
            if fail.exists() and not args.retry_failed:return
            if fail.exists():
                history=directory/'failure_history.json'
                saved=json.loads(history.read_text()) if history.exists() else []
                saved.append(json.loads(fail.read_text()));atomic_json(history,saved);fail.unlink()
            results=directory/'results.json'
            if results.exists() and any(r['training_tokens']==max(spec['milestones']) for r in json.loads(results.read_text())['results']):return
            if shutil.disk_usage(root).free<spec['resource_limits']['minimum_free_disk_bytes']:
                atomic_json(fail,{'family':family,'scale':scale,'seed':seed,'classification':'resource-limited','reason':'preregistered free-disk bound'});return
            cmd=[sys.executable,str(Path(__file__).resolve()),'--config',args.config,'--out',args.out,'--report',args.report,
                 '--family',family,'--scale',scale,'--seed',str(seed)]
            code=None;resource_limited=False
            for attempt in range(2):
                with (directory/'worker.log').open('a') as log:
                    process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                    try:code=process.wait(timeout=spec['resource_limits']['job_wall_seconds'])
                    except subprocess.TimeoutExpired:
                        process.terminate();process.wait();code=-15;resource_limited=True
                if code==0:return
                if code<0:resource_limited=True;break
            atomic_json(fail,{'family':family,'scale':scale,'seed':seed,'classification':'resource-limited' if resource_limited else 'failed-resumable',
                'exit_code':code,'reason':'killed/timeout; exact external cause may be unknown' if resource_limited else 'worker failed twice; preserved checkpoint and worker.log'})
        for scale in spec['scales']:
            with ThreadPoolExecutor(max_workers=spec['workers']) as executor:
                futures=[executor.submit(job,scale,family,seed) for family in spec['families'] for seed in spec['seeds']]
                for future in futures:future.result() # Blocking completion waits, no batch polling.
            data=report()
            if args.push:publish(out,'Update v3 scaling report through '+scale)
            print(json.dumps({'scale_group_finished':scale,'completed_milestones':len(data['results']),'failures':len(data['failures'])}),flush=True)
        data=export(root,out,finalize=True);audit(data,root,out);final_readme(data)
        if args.push:publish(out,'Finalize same-edge repetition scaling study and measured curves',True)
        print(json.dumps({'runner_finished':True,'complete':data['complete'],'rows':len(data['results'])}),flush=True)


if __name__=='__main__':main()

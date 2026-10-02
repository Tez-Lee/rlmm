"""Autonomous study: blocking worker waits, no batch polling, automatic reports/pushes."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def freeze(report):
    path=report/'frozen_manifest.json'
    if path.exists():return
    files=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0')
    files=[f for f in files if f and f!='README.md']
    # Preserve existing ignored checkpoint/results too, not just tracked exports.
    for root in Path('runs').glob('*'):
        if root.is_dir() and not root.name.startswith('v2_1'):
            files += [str(p) for p in root.rglob('*') if p.is_file() and (p.suffix=='.pt' or p.name=='results.json')]
    snapshot={f:{'bytes':Path(f).stat().st_size,'sha256':hashlib.sha256(Path(f).read_bytes()).hexdigest()} for f in sorted(set(files))}
    from prglm.learning_curve import atomic_json
    atomic_json(path,{'base_commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'files':snapshot})


def verify_frozen(report):
    manifest=json.loads((report/'frozen_manifest.json').read_text())
    for f,info in manifest['files'].items():
        path=Path(f)
        if not path.is_file() or path.stat().st_size!=info['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=info['sha256']:
            raise RuntimeError(f'Frozen artifact changed: {f}')
    return len(manifest['files'])


def publish(message,report,final=False):
    paths=['configs/core_correction.json','configs/core_correction_smoke.json',
           'prglm/v21_config.py','prglm/v21_model.py','prglm/v21_memory.py','prglm/v21_training.py',
           'prglm/v21_diagnostics.py','prglm/v21_report.py','tests/test_v21.py',
           'scripts/run_core_correction.py',str(report)]
    if final:paths.append('README.md')
    subprocess.run(['git','add','--',*paths],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True,capture_output=True)
    changed=subprocess.run(['git','diff','--cached','--quiet']).returncode
    if changed:subprocess.run(['git','commit','-m',message],check=True,capture_output=True)
    try:
        subprocess.run(['git','push','origin','main'],check=True,capture_output=True,timeout=180)
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
        # Publish failure does not stop/restart training. Never expose credential values.
        from prglm.learning_curve import atomic_json
        atomic_json(report/'publish_failure.json',{'action':'git push origin main','error_type':type(error).__name__,
            'message':'Push failed; training continues. Retry Git authentication/network through normal credential path.'})
        return False
    if (report/'publish_failure.json').exists():(report/'publish_failure.json').unlink()
    return True


def final_readme(report,result):
    text=Path('README.md').read_text()
    marker='\n## PRG-v2.1 Core Correction Study\n'
    if marker in text:return
    lookup={s['model']:s for s in result['summaries'] if s['training_tokens']==819200 and s['validation_bytes']==4096 and s['mode']=='sampled'}
    lines=[marker,'Completed36 new milestone results (three models × three seeds × four continuous milestones). Frozen v0/v1/v2 code/results/checkpoints were preserved; no baseline was retrained.',
        'The goal was to test temporal credit, valid cycle-cap semantics and gateway opportunity. A scalar-gated prefix-state pool reuses the existing readout projection; cap output uses a processed source. Clean accumulation changes only accumulator summation. The minimum-one-gateway variant is diagnostic, not the main architecture.',
        '', '| Model | Final loss (nats/byte) | Byte PPL |','|---|---:|---:|']
    for m in ('transformer_core','transformer_total','prg_v1','prg_v2_adaptive','prg_v2_uniform',*result['spec']['models']):
        s=lookup[m];lines.append(f"| {m} | {s['loss']['mean']:.4f} ± {s['loss']['std']:.4f} | {s['ppl']['mean']:.4f} ± {s['ppl']['std']:.4f} |")
    lines+=['','Three-seed means ± sample SD,4096 held-out target bytes. Full16384-byte checks, route metrics, gradient characterization, late slopes and all12 research answers are in [the v2.1 report](research/v2_1/REPORT.md). Canonical results are README, that REPORT and its machine-readable [results.json](research/v2_1/results.json).',
        'No dashboard, HTML, FastAPI or ZIP work was done for v2.1; old artifacts remain preserved and were not part of this research workflow.',
        'Limitations: the core correction is a bundle, global context pooling can bypass sparse routing, the gateway probe changes OUTPUT timing/compute, expected routing is a surrogate and expected probe is undefined. This study does not isolate numerical-precision replacement. No architecture novelty or successful hypothesis is assumed.',
        '', 'Resume or reproduce using `OMP_NUM_THREADS=1 PYTHONPATH=.deps:. python scripts/run_core_correction.py --push`. Checkpoint/model exports stay in ignored `runs/v2_1/`; the runner resumes incomplete jobs and skips completed results. Install project research/test extras first on a fresh checkout.',
        '', '### AI-assisted development disclosure','',
        'The human project author led the research hypotheses, architecture concepts, experimental questions, experiment direction and interpretation criteria. OpenAI ChatGPT and OpenAI coding models substantially assisted literature discovery during ideation, architecture discussion, implementation, test generation, experiment automation, result summarization and documentation. The author reviews the machine-produced analysis. We do not claim AI-generated implementation code itself as original source code. AI assistance and architecture novelty are separate questions.',
        '', '### Prior art and novelty','',
        'Similar prior work exists for the individual components. PRG-LM is currently an **experimental research hypothesis**; novelty of the overall design is not established. It must be evaluated against published literature, patents and existing implementations.',
        '', '### Literature surfaced during ChatGPT-assisted ideation','',
        'The following were surfaced in the project discussion by ChatGPT. This does not assert that they were OpenAI model training data, or that this implementation was directly derived from their source code. These are related discussion references, not a completed novelty search.',
        '', '- Alex Graves, *Adaptive Computation Time for Recurrent Neural Networks*, arXiv:1603.08983 (2016).',
        '- Simon Schug, Frederik Benzing, Angelika Steger, *Presynaptic stochasticity improves energy efficiency and helps alleviate the stability-plasticity dilemma*, eLife (2021).',
        '- Sizhong Lan, *On the Relation of Impulse Propagation to Synaptic Strength*, arXiv:1805.09001 (2018).',
        '- Jack C. Gartside et al., *Reconfigurable training and reservoir computing in an artificial spin-vortex ice via spin-wave fingerprinting*, Nature Nanotechnology (2022).',
        '', 'No additional looped/recurrent-LM or equilibrium citation is added: the inspected repository history did not provide a precise citation previously surfaced in discussion.']
    Path('README.md').write_text(text+'\n'.join(lines)+'\n')


def audit_final(spec,root,report):
    import torch
    streams={s:set() for s in spec['seeds']};count=0
    for seed in spec['seeds']:
        for name in spec['models']:
            directory=root/f'seed{seed}'/name
            rows=json.loads((directory/'results.json').read_text())['results']
            assert sorted(r['training_tokens'] for r in rows)==spec['milestones']
            for r in rows:
                streams[seed].add(r['token_stream_sha256'])
                for budget in spec['validation_bytes']:
                    assert r['validation'][str(budget)]['sampled']['validation_tokens']==budget
                assert Path(r['checkpoint_identifier']).exists()
            checkpoint=torch.load(directory/'latest.pt',weights_only=False,map_location='cpu')
            assert checkpoint['training_tokens']==max(spec['milestones'])
            assert checkpoint['optimizer']['state'] and checkpoint['rng']['torch'].numel()
            assert all(torch.isfinite(v).all() for v in checkpoint['state'].values() if isinstance(v,torch.Tensor) and v.is_floating_point())
            count+=1
    for seed,values in streams.items():
        assert len(values)==1
        baseline=Path(f'runs/phase_a/seed{seed}/prg_v1/results.json')
        if baseline.exists():assert next(iter(values))==json.loads(baseline.read_text())['results'][-1]['token_stream_sha256']
    from prglm.learning_curve import atomic_json
    atomic_json(report/'integrity.json',{'complete':True,'final_checkpoint_count':count,
        'frozen_artifacts_verified':verify_frozen(report),'stream_hash_per_seed':{str(s):next(iter(v)) for s,v in streams.items()},
        'new_completed_milestones':len(spec['models'])*len(spec['seeds'])*len(spec['milestones'])})


def main():
    os.chdir(REPO)
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/core_correction.json')
    p.add_argument('--out',default='runs/v2_1');p.add_argument('--report',default='research/v2_1')
    p.add_argument('--seed',type=int);p.add_argument('--milestone',type=int);p.add_argument('--push',action='store_true')
    args=p.parse_args();spec=json.loads(Path(args.config).read_text());root=Path(args.out);report=Path(args.report)
    import torch
    torch.set_num_threads(spec['threads'])
    from prglm.v21_training import initialize,run_one
    from prglm.v21_report import export
    train,val=initialize(spec,root)
    if args.seed is not None:
        for name in spec['models']:
            with (root/f'{name}_seed{args.seed}.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                run_one(spec,root,name,args.seed,train,val,stop_after=args.milestone)
            with (root/'report.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);export(root,report)
        return
    report.mkdir(parents=True,exist_ok=True)
    with (root/'supervisor.lock').open('w') as supervisor_lock:
        fcntl.flock(supervisor_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        freeze(report);verify_frozen(report);export(root,report)
        if args.push:publish('Implement PRG-v2.1 core corrections and pass sanity tests',report)
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        for milestone in spec['milestones']:
            snapshot=export(root,report)
            if all((m,s,milestone) not in map(tuple,snapshot['missing']) for m in spec['models'] for s in spec['seeds']):continue
            processes=[];logs=[]
            for seed in spec['seeds']:
                log=(root/f'worker_seed{seed}.log').open('a');logs.append(log)
                cmd=[sys.executable,str(Path(__file__).resolve()),'--config',args.config,'--out',args.out,
                     '--report',args.report,'--seed',str(seed),'--milestone',str(milestone)]
                process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
                processes.append((seed,cmd,process))
            failures=[]
            # Actual blocking waits. No reading batch logs or polling progress.
            for seed,cmd,process in processes:
                status=process.wait()
                if status:
                    # Resume only failed worker; completed seed/model/milestones skip.
                    with (root/f'worker_seed{seed}.log').open('a') as log:
                        status=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                    if status:failures.append({'seed':seed,'milestone':milestone,'exit_code':status})
            for log in logs:log.close()
            result=export(root,report)
            if failures:
                from prglm.learning_curve import atomic_json
                atomic_json(report/'failure.json',{'failures':failures,'resume':'python scripts/run_core_correction.py --push'})
                if args.push:publish('Record resumable v2.1 worker failure',report)
                raise RuntimeError(f'Failed workers: {failures}; checkpoint preserved')
            if args.push:publish(f'Update v2.1 report at {milestone:,}-byte milestone',report)
            print(json.dumps({'milestone_group_complete':milestone,'new_results':len(result['results'])}),flush=True)
        final=export(root,report,plots=True)
        if not final['complete']:raise RuntimeError('final results incomplete')
        audit_final(spec,root,report);final_readme(report,final)
        if args.push:publish('Complete PRG-v2.1 core correction study and research disclosure',report,final=True)
        print(json.dumps({'complete':True,'rows':len(final['results']),'report':str(report/'REPORT.md')}),flush=True)


if __name__=='__main__':main()

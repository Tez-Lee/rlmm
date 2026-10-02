"""Final-checkpoint CPU/backend microbenchmark, separate from concurrent training."""
import hashlib,json
from pathlib import Path
from .v2_inference import load_checkpoint
from .inference import generate
from .models import backend
from .curve_report import stats


def main():
    out=Path('research/v2/inference_benchmark.json')
    names=['transformer_core','transformer_total','prg_v1','prg_v2_adaptive',
           'prg_v2_uniform','prg_v2_no_recurrence','prg_v2_no_accumulation']
    settings={'prompt':'To be, or not to be, that is the question.','max_tokens':16,
              'temperature':1.,'record_trace':False,'training_tokens':819200,
              'context':32,'backend':str(backend())}
    paths={}
    for seed in (42,43,44):
        for name in names:
            root=Path('runs/mutable_v2' if name.startswith('prg_v2') else 'runs/phase_a')
            paths[seed,name]=root/f'seed{seed}'/name/'model_tokens_819200.pt'
    fingerprints={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
    if out.exists():
        old=json.loads(out.read_text())
        if old['settings']==settings and old['checkpoint_sha256']==fingerprints:
            print('Final inference benchmark already complete; skipped');return
        raise ValueError('Refusing to overwrite a different inference benchmark')
    rows=[]
    for (seed,name),path in paths.items():
        model,tok=load_checkpoint(path)
        generate(model,tok,settings['prompt'],max_tokens=2,seed=seed,record_trace=False)
        sample=generate(model,tok,settings['prompt'],max_tokens=16,seed=seed,record_trace=False)
        rows.append({'model':name,'seed':seed,'tokens_per_second':sample['tokens_per_second'],
            'latency_seconds_per_token':sample['latency_seconds']/16,'completion':sample['completion']})
    summary={name:{'tokens_per_second':stats([r['tokens_per_second'] for r in rows if r['model']==name]),
                  'latency_seconds_per_token':stats([r['latency_seconds_per_token'] for r in rows if r['model']==name])}
             for name in names}
    out.write_text(json.dumps({'settings':settings,'checkpoint_sha256':fingerprints,
        'notes':'One16-byte sample after2-byte warmup per seed. Context recomputed, no KV cache. Prototype timing, not hardware efficiency.',
        'results':rows,'summary':summary},indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':main()

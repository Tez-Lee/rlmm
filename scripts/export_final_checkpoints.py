"""Bundle seed42 final inference exports; preserve all original run checkpoints."""
import hashlib
import json
import shutil
from pathlib import Path


def main():
    from prglm.v2_report import collect_v2
    result=collect_v2(Path('runs/mutable_v2'),Path('runs/phase_a'))
    if not result['complete']:raise RuntimeError('Finish all main controls before exporting final models')
    out=Path('checkpoints/byte256_seed42_819200');out.mkdir(parents=True,exist_ok=True)
    sources={name:Path('runs/phase_a') for name in ('transformer_core','transformer_total','prg_v1')}
    sources.update({name:Path('runs/mutable_v2') for name in result['protocol']['spec']['models']})
    entries=[]
    for name,root in sources.items():
        source=root/'seed42'/name/'model_tokens_819200.pt';target=out/f'{name}.pt'
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Refusing to overwrite a different checkpoint: {target}')
        else:shutil.copy2(source,target)
        entries.append({'name':name,'file':target.name,'bytes':target.stat().st_size,'sha256':digest,
                        'source':str(source)})
    (out/'manifest.json').write_text(json.dumps({'seed':42,'training_tokens':819200,
        'corpus':'TinyShakespeare byte256','models':entries,
        'format':'Inference-only exports. Full optimizer/mutation/RNG checkpoints remain under runs/.'},indent=2)+'\n')
    print('Exported',len(entries),'models;',sum(e['bytes'] for e in entries),'bytes')

if __name__=='__main__':main()

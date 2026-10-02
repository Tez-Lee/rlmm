"""Combine separately run baselines and PRG into seed-matched dashboard runs."""
import argparse,json,shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',required=True);p.add_argument('--prg',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); base=Path(a.baseline);prg=Path(a.prg);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    left=json.loads((base/'results.json').read_text());right=json.loads((prg/'results.json').read_text())
    assert left['config']['seed']==right['config']['seed']
    assert all(left['config'][k]==right['config'][k] for k in ('vocab_size','context','regions','region_size','max_cycles'))
    left['results']=[r for r in left['results'] if r['model']!='prg']+[r for r in right['results'] if r['model']=='prg']
    (out/'results.json').write_text(json.dumps(left,indent=2))
    for file in ('tokenizer.json','train.bin','val.bin','transformer.pt','looped.pt'):
        shutil.copy2(base/file,out/file)
    shutil.copy2(prg/'prg.pt',out/'prg.pt')


if __name__=='__main__':main()

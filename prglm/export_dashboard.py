"""Export a checkpoint's benchmark and sample trajectory to GitHub Pages assets."""
import argparse, json, shutil
from pathlib import Path
from .inference import load,generate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='runs/first'); p.add_argument('--dest',default='docs')
    p.add_argument('--matched',default='runs/matched')
    a=p.parse_args(); src=Path(a.run); dest=Path(a.dest); dest.mkdir(exist_ok=True)
    for file in ('index.html','app.js'):
        shutil.copy2(Path('web/static')/file,dest/file)
    bench=json.loads((src/'results.json').read_text())
    matched=Path(a.matched)/'results.json'
    if matched.exists():
        row=json.loads(matched.read_text())['results'][0]
        row['model']='transformer (size matched)'
        row['d_model']=44
        bench['results'].append(row)
    summary=src.parent/'summary.json'
    if summary.exists(): bench['seed_summary']=json.loads(summary.read_text())
    (dest/'results.json').write_text(json.dumps(bench,indent=2))
    sample={}
    for name in ('transformer','prg'):
        model,tok=load(src,name)
        sample[name]=generate(model,tok,'Once upon a time',max_tokens=8,seed=42)
    (dest/'sample.json').write_text(json.dumps({'runs':[sample]},indent=2))
    (dest/'.nojekyll').touch()


if __name__=='__main__': main()

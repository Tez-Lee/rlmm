"""Export a downloadable local dashboard with benchmark and trajectory replay."""
import argparse, json, shutil, zipfile
from pathlib import Path
from .inference import load,generate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',default='runs/first'); p.add_argument('--dest',default='dashboard')
    p.add_argument('--matched',default='runs/matched')
    p.add_argument('--archive',default='dashboard-bundle.zip')
    p.add_argument('--v1-run',default='runs/v1_final')
    p.add_argument('--refresh-legacy-sample',action='store_true')
    p.add_argument('--refresh-v1',action='store_true')
    a=p.parse_args(); src=Path(a.run); dest=Path(a.dest); dest.mkdir(exist_ok=True)
    for file in ('index.html','app.js','v1_dashboard.js','learning_curve.js','v2_dashboard.js'):
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
    if a.refresh_legacy_sample or not (dest/'sample.json').exists():
        sample={}
        for name in ('transformer','prg'):
            model,tok=load(src,name)
            sample[name]=generate(model,tok,'Once upon a time',max_tokens=8,seed=42)
        (dest/'sample.json').write_text(json.dumps({'runs':[sample]},indent=2))
    v1=Path(a.v1_run)
    if (v1/'results.json').exists() and (a.refresh_v1 or not (dest/'v1_results.json').exists() or not (dest/'v1_sample.json').exists()):
        v1_bench=json.loads((v1/'results.json').read_text())
        v1_bench['ablations']={}
        for mode in ('recurrence','accumulation','fatigue'):
            result=v1.parent/f'v1_ablate_{mode}'/'results.json'
            if result.exists():
                rows=json.loads(result.read_text())['results']
                v1_bench['ablations'][mode]=[{'seed':row['seed'],'sampled_ppl':row['validation']['sampled']['ppl'],
                    'expected_ppl':row['validation']['expected']['ppl'],
                    'argmax_ppl':row['validation']['argmax']['ppl']} for row in rows if row['model']=='prg_v1']
        (dest/'v1_results.json').write_text(json.dumps(v1_bench,indent=2))
        comparison={}
        for name in ('transformer_core','transformer_total','prg','prg_v1'):
            model,tok=load(v1/'seed42',name)
            comparison[name]=generate(model,tok,'To be',max_tokens=8,seed=42)
        (dest/'v1_sample.json').write_text(json.dumps(comparison,indent=2))
    curve=Path('research/phase_a/learning_curve.json')
    if curve.exists():shutil.copy2(curve,dest/'learning_curve.json')
    v2_report=Path('research/v2/results.json')
    if v2_report.exists():shutil.copy2(v2_report,dest/'v2_results.json')
    with zipfile.ZipFile(a.archive,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for file in ('index.html','app.js','v1_dashboard.js','results.json','sample.json','README.md',
                     'v1_results.json','v1_sample.json','learning_curve.js','learning_curve.json','v2_dashboard.js','v2_results.json'):
            if (dest/file).exists():archive.write(dest/file,arcname=f'dashboard/{file}')


if __name__=='__main__': main()

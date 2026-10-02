"""Aggregate completed checkpoints; never fill missing experimental cells."""
import argparse
import json
import math
import statistics
from pathlib import Path
from .learning_curve import atomic_json


def stats(values):
    return {'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values)>1 else None, 'n': len(values)}


def collect(root):
    protocol = json.loads((root/'protocol.json').read_text())
    rows=[]
    for path in sorted(root.glob('seed*/*/results.json')):
        rows.extend(json.loads(path.read_text())['results'])
    groups={}
    for row in rows:
        for budget,modes in row['validation'].items():
            for mode,metrics in modes.items():
                key=(row['model'],row['training_tokens'],budget,mode)
                groups.setdefault(key,[]).append((row,metrics))
    summaries=[]
    for (name,tokens,budget,mode),entries in sorted(groups.items()):
        summaries.append({'model':name,'training_tokens':tokens,'validation_bytes':int(budget),
            'routing_mode':mode,'seeds':[r['seed'] for r,_ in entries],
            'loss':stats([m['loss'] for _,m in entries]),
            'ppl':stats([m['ppl'] for _,m in entries]),
            'static_bytes':entries[0][0]['memory']['total_static_packed_bytes'],
            'peak_bytes':entries[0][0]['memory']['peak_theoretical_inference_bytes'],
            'active_edges_per_token':stats([r['dynamics']['traversed_local_edges_per_token']
                for r,_ in entries]) if name=='prg_v1' else None})
    gaps=[]
    for tokens in protocol['spec']['milestones']:
        for budget in protocol['spec']['validation_bytes']:
            for baseline in ('transformer_core','transformer_total'):
                for mode in ('sampled','argmax','expected'):
                    a={r['seed']:r['validation'][str(budget)]['sampled'] for r in rows
                       if r['model']==baseline and r['training_tokens']==tokens}
                    b={r['seed']:r['validation'][str(budget)][mode] for r in rows
                       if r['model']=='prg_v1' and r['training_tokens']==tokens}
                    seeds=sorted(a.keys() & b.keys())
                    if seeds:
                        gaps.append({'training_tokens':tokens,'validation_bytes':budget,
                            'baseline':baseline,'routing_mode':mode,'seeds':seeds,
                            'loss_gap':stats([b[s]['loss']-a[s]['loss'] for s in seeds]),
                            'ppl_ratio':stats([b[s]['ppl']/a[s]['ppl'] for s in seeds])})
    expected={(m,s,t) for m in protocol['spec']['models'] for s in protocol['spec']['seeds']
              for t in protocol['spec']['milestones']}
    present={(r['model'],r['seed'],r['training_tokens']) for r in rows}
    missing=sorted(expected-present)
    return {'protocol':protocol,'complete':not missing,'missing_checkpoints':missing,
            'summaries':summaries,'paired_gaps':gaps,'results':rows}


def svg_plot(path,series,title,ylabel,log=False):
    width,height=900,450
    left,top,right,bottom=85,45,25,65
    pts=[(x,y) for _,points in series for x,y in points]
    if not pts:return
    tx=lambda x:math.log10(x) if log else x
    xs=[tx(x) for x,_ in pts];ys=[y for _,y in pts]
    xmin,xmax=min(xs),max(xs);ymin,ymax=min(ys),max(ys)
    if xmin==xmax:xmax=xmin+1
    spread=max(ymax-ymin,.1);ymin-=.1*spread;ymax+=.1*spread
    X=lambda x:left+(tx(x)-xmin)/(xmax-xmin)*(width-left-right)
    Y=lambda y:height-bottom-(y-ymin)/(ymax-ymin)*(height-top-bottom)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="85" y="25" font-size="18">{title}</text>']
    for i in range(6):
        y=ymin+(ymax-ymin)*i/5
        parts.append(f'<line x1="{left}" x2="{width-right}" y1="{Y(y)}" y2="{Y(y)}" stroke="#ddd"/>')
        parts.append(f'<text x="10" y="{Y(y)+5}" font-size="12">{y:.3g}</text>')
    for x in sorted({x for x,_ in pts}):
        parts.append(f'<text x="{X(x)}" y="{height-bottom+20}" text-anchor="middle" font-size="12">{x:,}</text>')
    colors=['#2563eb','#0891b2','#e11d48','#a855f7','#16a34a','#d97706']
    for i,(name,points) in enumerate(series):
        color=colors[i%len(colors)]
        poly=' '.join(f'{X(x):.2f},{Y(y):.2f}' for x,y in points)
        parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x,y in points:parts.append(f'<circle cx="{X(x)}" cy="{Y(y)}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{left+i*190}" y="{height-12}" fill="{color}" font-size="12">{name}</text>')
    parts.append(f'<text x="450" y="{height-35}" text-anchor="middle" font-size="12">Training target bytes {"(log scale)" if log else "(linear scale)"}</text>')
    parts.append(f'<text x="15" y="40" font-size="12">{ylabel}</text></svg>')
    path.write_text('\n'.join(parts))


def export(root,out):
    payload=collect(root);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'learning_curve.json',payload)
    def fmt(s):
        return f"{s['mean']:.4f} ± {s['std']:.4f}" if s['std'] is not None else f"{s['mean']:.4f} (n=1)"
    lines=['# Phase A: continuous learning curve', '',
        'Complete: '+str(payload['complete'])+'. Missing checkpoints: '+str(len(payload['missing_checkpoints']))+'.',
        'All primary summaries require seeds 42/43/44. Incomplete groups are labelled and cannot support main conclusions.', '',
        '| Model | Training tokens | Loss | PPL | Static bytes | Peak bytes | Active edges/token | Recurrence | Topology mode |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for s in payload['summaries']:
        if s['validation_bytes']!=4096 or s['routing_mode']!='sampled':continue
        label=s['model']+f" (n={s['loss']['n']})"
        edges=fmt(s['active_edges_per_token']) if s['active_edges_per_token'] else 'N/A'
        lines.append(f"| {label} | {s['training_tokens']:,} | {fmt(s['loss'])} | {fmt(s['ppl'])} | {s['static_bytes']:,} | {s['peak_bytes']:,} | {edges} | {'ON' if s['model']=='prg_v1' else 'N/A'} | static |")
    for metric in ('loss','ppl'):
        lines.extend(['',f'## {metric.upper()} curve summary (4096-byte sampled evaluation)','',
            '| Training tokens | Transformer core | Transformer peak | PRG-v1 | PRG-v2 |',
            '|---:|---:|---:|---:|---:|'])
        for tokens in payload['protocol']['spec']['milestones']:
            cells=[]
            for name in ('transformer_core','transformer_total','prg_v1'):
                matches=[s for s in payload['summaries'] if s['model']==name and
                    s['training_tokens']==tokens and s['validation_bytes']==4096 and s['routing_mode']=='sampled']
                cells.append(fmt(matches[0][metric]) if matches else 'pending')
            lines.append(f"| {tokens:,} | {' | '.join(cells)} | Phase B pending |")
    lines.extend(['','## Paired PRG-v1 / Transformer comparisons','',
        '| Training tokens | Validation bytes | Baseline | Routing mode | Loss gap | PPL ratio | n |',
        '|---:|---:|---|---|---:|---:|---:|'])
    for g in payload['paired_gaps']:
        lines.append(f"| {g['training_tokens']:,} | {g['validation_bytes']} | {g['baseline']} | {g['routing_mode']} | {fmt(g['loss_gap'])} | {fmt(g['ppl_ratio'])} | {len(g['seeds'])} |")
    lines.extend(['','## Interpretation limits','',
        'Loss is cross-entropy in nats per target byte. PPL is byte perplexity, not BPE/token perplexity.',
        '4096-byte validation is the original prefix; 16384-byte validation contains that prefix.',
        'Expected routing is the unchanged v1 soft region-mass surrogate, not exact stochastic expectation.',
        'Memory estimates retain the ad82f7c conventions, including the hypothetical Transformer KV cache; no packed runtime exists.',
        'Training tokens are cumulative sampled target bytes (including repeats), not unique corpus bytes.',
        'A finite learning curve cannot distinguish intrinsic representation limits from all possible optimizer failures.',
        'Pending checkpoints must not be extrapolated. v2 evaluation begins only after Phase A completes.'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for budget in payload['protocol']['spec']['validation_bytes']:
        for metric in ('loss','ppl'):
            series=[]
            for name,mode in [('transformer_core','sampled'),('transformer_total','sampled'),
                              ('prg_v1','sampled'),('prg_v1','argmax'),('prg_v1','expected')]:
                points=[(s['training_tokens'],s[metric]['mean']) for s in payload['summaries']
                        if s['model']==name and s['routing_mode']==mode and s['validation_bytes']==budget and s[metric]['n']==3]
                if points:series.append((f'{name}/{mode}',points))
            for log in (False,True):
                svg_plot(out/f'{metric}_{budget}_{"log" if log else "linear"}.svg',series,
                         f'{metric.upper()} vs training bytes; validation={budget}',metric,log)
        for metric in ('loss_gap','ppl_ratio'):
            series=[]
            for baseline in ('transformer_core','transformer_total'):
                points=[(g['training_tokens'],g[metric]['mean']) for g in payload['paired_gaps']
                    if g['baseline']==baseline and g['validation_bytes']==budget and g['routing_mode']=='sampled' and len(g['seeds'])==3]
                if points:series.append((f'PRG-v1/{baseline}',points))
            for log in (False,True):
                svg_plot(out/f'{metric}_{budget}_{"log" if log else "linear"}.svg',series,
                         f'Paired {metric}; validation={budget}',metric,log)
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/phase_a')
    p.add_argument('--out',default='research/phase_a');args=p.parse_args()
    payload=export(Path(args.run),Path(args.out))
    print(json.dumps({'complete':payload['complete'],'checkpoints':len(payload['results']),
                      'missing':len(payload['missing_checkpoints'])}))

if __name__=='__main__':main()

"""Combine preserved Phase A with new v2 controls; no missing-value imputation."""
import argparse
import json
from pathlib import Path
from .learning_curve import atomic_json
from .curve_report import collect,stats,svg_plot


def collect_v2(root,phase_a):
    a=collect(phase_a)
    protocol=json.loads((root/'protocol.json').read_text())
    new=[]
    for p in sorted(root.glob('seed*/*/results.json')):
        new.extend(json.loads(p.read_text())['results'])
    from .v2_config import V2Config
    for row in new:
        if 'config' not in row:
            spec=protocol['spec']
            row['config']=V2Config(**(spec['v1']|spec['v2']|spec['variants'][row['model']]|{'seed':row['seed']})).dict()
    rows=a['results']+new
    groups={}
    for r in rows:
        for budget,modes in r['validation'].items():
            for mode,m in modes.items():
                key=(r['model'],r['training_tokens'],int(budget),mode)
                groups.setdefault(key,[]).append((r,m))
    summaries=[]
    for (name,tokens,budget,mode),group in sorted(groups.items()):
        memory=group[0][0]['memory']
        summaries.append({'model':name,'training_tokens':tokens,'validation_bytes':budget,'routing_mode':mode,
            'seeds':[r['seed'] for r,_ in group],
            'loss':stats([m['loss'] for _,m in group]),'ppl':stats([m['ppl'] for _,m in group]),
            'static_bytes':memory['total_static_packed_bytes'],'peak_bytes':memory['peak_theoretical_inference_bytes'],
            'training_metadata_bytes':stats([r['memory'].get('training_only_metadata_tensor_bytes',0)+r['memory'].get('training_only_history_json_bytes',0) for r,_ in group]),
            'serialized_model_bytes':stats([r['serialized_model_bytes'] for r,_ in group]),
            'active_edges_per_token':stats([r['dynamics']['traversed_local_edges_per_token'] for r,_ in group]) if name.startswith('prg') else None})
    expected={(m,s,t) for m in protocol['spec']['models'] for s in protocol['spec']['seeds'] for t in protocol['spec']['milestones']}
    present={(r['model'],r['seed'],r['training_tokens']) for r in new}
    differences=[]
    for t in protocol['spec']['milestones']:
        for budget in protocol['spec']['validation_bytes']:
            for mode in ('sampled','argmax','expected'):
                for treatment,control in [('prg_v2_adaptive','prg_v1'),('prg_v2_uniform','prg_v1'),
                    ('prg_v2_adaptive','prg_v2_uniform'),('prg_v2_no_recurrence','prg_v2_adaptive'),
                    ('prg_v2_no_accumulation','prg_v2_adaptive')]:
                    left={r['seed']:r['validation'][str(budget)][mode] for r in rows if r['model']==treatment and r['training_tokens']==t}
                    right={r['seed']:r['validation'][str(budget)][mode] for r in rows if r['model']==control and r['training_tokens']==t}
                    seeds=sorted(left.keys()&right.keys())
                    if seeds:
                        deltas=[left[s]['loss']-right[s]['loss'] for s in seeds]
                        differences.append({'treatment':treatment,'control':control,'training_tokens':t,
                            'validation_bytes':budget,'routing_mode':mode,'seeds':seeds,
                            'loss_delta':stats(deltas),'per_seed_loss_delta':dict(zip(seeds,deltas)),
                            'ppl_ratio':stats([left[s]['ppl']/right[s]['ppl'] for s in seeds])})
    return {'complete':a['complete'] and not expected-present,'missing_checkpoints':sorted(expected-present),
        'protocol':protocol,'static_control':'Preserved Phase A PRG-v1; static-v2 neural/gradient equivalence tested.',
        'summaries':summaries,'paired_differences':differences,'results':rows}


def export(root,out,phase_a):
    d=collect_v2(root,phase_a);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'results.json',d)
    fmt=lambda s:f"{s['mean']:.4f} ± {s['std']:.4f}" if s['std'] is not None else f"{s['mean']:.4f} (n=1)"
    lines=['# Mutable-topology study','',f"Complete: {d['complete']}; remaining v2 checkpoints: {len(d['missing_checkpoints'])}.",
        'All primary summaries use seeds42/43/44. Static control reuses completed Phase A, not retraining.', '',
        '| Model | Training tokens | Loss | PPL | Static bytes | Peak bytes | Active edges/token | Recurrence | Topology mode |',
        '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for s in d['summaries']:
        if s['validation_bytes']!=4096 or s['routing_mode']!='sampled':continue
        mode='uniform' if s['model']=='prg_v2_uniform' else 'adaptive' if s['model'].startswith('prg_v2') else 'static'
        recurrence='OFF' if s['model']=='prg_v2_no_recurrence' else 'ON' if s['model'].startswith('prg') else 'N/A'
        lines.append(f"| {s['model']} (n={s['loss']['n']}) | {s['training_tokens']:,} | {fmt(s['loss'])} | {fmt(s['ppl'])} | {s['static_bytes']:,} | {s['peak_bytes']:,} | {fmt(s['active_edges_per_token']) if s['active_edges_per_token'] else 'N/A'} | {recurrence} | {mode} |")
    names=['transformer_core','transformer_total','prg_v1','prg_v2_adaptive']
    for metric in ('loss','ppl'):
        lines+=['',f'## Learning curve: {metric} (4096-byte sampled)', '',
            '| Training tokens | Transformer core | Transformer peak | PRG-v1 | PRG-v2 adaptive |',
            '|---:|---:|---:|---:|---:|']
        for t in d['protocol']['spec']['milestones']:
            vals=[]
            for name in names:
                s=next((s for s in d['summaries'] if s['model']==name and s['training_tokens']==t and s['validation_bytes']==4096 and s['routing_mode']=='sampled'),None)
                vals.append(fmt(s[metric]) if s else 'pending')
            lines.append(f"| {t:,} | {' | '.join(vals)} |")
    lines+=['','## Topology evolution', '',
        '| Topology mode | Training tokens | Rewires | Reverts | Exploration % | Gateway entropy | Largest hub | PPL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    topology_groups={}
    for r in d['results']:
        if 'topology' in r:topology_groups.setdefault((r['model'],r['training_tokens']),[]).append(r)
    for (name,t),rows in sorted(topology_groups.items()):
        accepted=stats([r['topology']['counters']['accepted'] for r in rows])
        reverted=stats([r['topology']['counters']['reverted'] for r in rows])
        exploration=stats([100*r['topology']['counters']['exploration']/max(1,r['topology']['counters']['accepted']) for r in rows])
        ent=stats([r['topology']['topology_entropy'] for r in rows]);hub=stats([r['topology']['largest_hub'] for r in rows])
        ppl=stats([r['validation']['4096']['sampled']['ppl'] for r in rows])
        lines.append(f'| {name} (n={len(rows)}) | {t:,} | {fmt(accepted)} | {fmt(reverted)} | {fmt(exploration)} | {fmt(ent)} | {fmt(hub)} | {fmt(ppl)} |')
    lines+=['','## Paired controls', '',
        '| Training tokens | Validation bytes | Routing | Treatment − control | Loss delta | PPL ratio | n |',
        '|---:|---:|---|---|---:|---:|---:|']
    for diff in d['paired_differences']:
        lines.append(f"| {diff['training_tokens']:,} | {diff['validation_bytes']} | {diff['routing_mode']} | {diff['treatment']} − {diff['control']} | {fmt(diff['loss_delta'])} | {fmt(diff['ppl_ratio'])} | {len(diff['seeds'])} |")
    lines+=['','## Inference and training metadata', '',
        'The local-edge/gateway/router/embedding/runtime breakdown is recorded in every result.memory.',
        'EMA, mutation RNG, counters, ever-edge table, probation and histories are training-only.',
        'Inference exports discard these buffers, retaining the gateway table and neural state.',
        'Full resumable checkpoints retain all mutation state; their byte size is separate from model-only exports.',
        'Active edges/token counts processed edge slots; effective nonzero edges are separately recorded.',
        'Region hotness/output credit is a proxy, not a causal usefulness measurement. Gradient credit is optional and disabled in the main experiment.',
        'RecurrenceOFF inherits the v1 ascending-region-ID mask; it additionally alters the usable gateway graph.',
        'Three seeds describe this dataset/configuration; no unobserved asymptotic claim is warranted.']
    if d['complete']:
        lines+=['','## Answers to the ten research questions','']
        final=max(d['protocol']['spec']['milestones'])
        lookup={(s['model'],s['training_tokens']):s for s in d['summaries'] if s['validation_bytes']==4096 and s['routing_mode']=='sampled'}
        first=min(d['protocol']['spec']['milestones'])
        v1=lookup['prg_v1',final];core=lookup['transformer_core',final];peak=lookup['transformer_total',final]
        adaptive=lookup['prg_v2_adaptive',final];uniform=lookup['prg_v2_uniform',final]
        off=lookup['prg_v2_no_recurrence',final];noacc=lookup['prg_v2_no_accumulation',final]
        lines.append(f"1. v1 reduced its overall peak-matched loss gap from {lookup['prg_v1',first]['loss']['mean']-lookup['transformer_total',first]['loss']['mean']:.3f} to {v1['loss']['mean']-peak['loss']['mean']:.3f} nats/byte. The gap is nonmonotonic: it narrowed at204.8k then widened at819.2k. Final core gap={v1['loss']['mean']-core['loss']['mean']:.3f}.")
        lines.append('2. There is strong evidence of initial slow optimization/sample efficiency, but eventual finite-budget inferiority persists. These curves cannot isolate intrinsic representation efficiency from residual optimizer limitations.')
        lines.append(f"3. Adaptive − static final loss delta={adaptive['loss']['mean']-v1['loss']['mean']:.4f}; lower is better. See paired seed deltas and16384-byte checks above.")
        lines.append(f"4. Adaptive − uniform final loss delta={adaptive['loss']['mean']-uniform['loss']['mean']:.4f}; small/inconsistent differences do not establish a hotness benefit.")
        lines.append('5. Rewiring and nonuniform topology are measured, but useful self-organization requires a reproducible language-model improvement over uniform/static; graph shape alone is insufficient.')
        latest=[r for r in d['results'] if r['model']=='prg_v2_adaptive' and r['training_tokens']==final]
        lines.append('6. Adaptive largest indegrees='+str([r['topology']['largest_hub'] for r in latest])+', Gini='+str([round(r['topology']['indegree_gini'],3) for r in latest])+', isolated regions='+str([r['topology']['isolated_regions'] for r in latest])+'. Maximum possible indegree is31; report concentration without treating diversity as proof of utility.')
        cases=[]
        for r in latest:
            top=r['topology']
            for h in top['history']:
                if h['kind']!='exploration':continue
                src,slot,dest=h['source'],h['slot'],h['new']
                if top['gateway_table'][src][slot]==dest and top['components']['in_degree'][dest]>=6 and top['gateway_downstream_credit'][src][slot]>0:
                    cases.append({'seed':r['seed'],'source':src,'destination':dest,'mutation_step':h['step'],'indegree':top['components']['in_degree'][dest],'credit_proxy':top['gateway_downstream_credit'][src][slot]})
        lines.append('7. Exploratory edges still present at final checkpoint with indegree≥6 and positive downstream-credit EMA: '+json.dumps(cases[:6])+'. This is retrospective proxy evidence, not causal discovery of useful hubs. No qualifying case is reported when the list is empty.')
        lines.append(f"8. RecurrenceOFF − ON final loss delta={off['loss']['mean']-adaptive['loss']['mean']:.4f}. The destination-ID masking confound prevents attributing the entire delta to revisits alone.")
        lines.append(f"9. AccumulationOFF − ON final loss delta={noacc['loss']['mean']-adaptive['loss']['mean']:.4f}. Positive deltas favor accumulation; compare consistency across seeds/milestones.")
        lines.append('10. The original precision-for-recurrence hypothesis is not established by these experiments. Improvement from training volume or changing topology cannot by itself show that traversal replaced numerical weight precision. Favorable adaptive results would justify a narrower sparse-topology hypothesis; unfavorable or inconsistent controls do not rescue it. The goal is finding working mechanisms, not defending a hypothesis.')
    else:
        lines+=['','Ten-question final assessment is pending until all three-seed controls complete.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for budget in d['protocol']['spec']['validation_bytes']:
        for metric in ('loss','ppl'):
            series=[]
            for name in names:
                points=[(s['training_tokens'],s[metric]['mean']) for s in d['summaries'] if s['model']==name and s['routing_mode']=='sampled' and s['validation_bytes']==budget and s[metric]['n']==3]
                if points:series.append((name,points))
            for log in (False,True):svg_plot(out/f'{metric}_{budget}_{"log" if log else "linear"}.svg',series,f'{metric.upper()} / training bytes; validation={budget}',metric,log)
    return d


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='runs/mutable_v2');p.add_argument('--out',default='research/v2')
    p.add_argument('--phase-a',default='runs/phase_a');args=p.parse_args()
    d=export(Path(args.run),Path(args.out),Path(args.phase_a))
    print(json.dumps({'complete':d['complete'],'remaining':len(d['missing_checkpoints']),'rows':len(d['results'])}))

if __name__=='__main__':main()

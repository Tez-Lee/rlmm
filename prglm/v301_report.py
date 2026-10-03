"""Measured continuation analysis; no point selection or curve extrapolation."""
import json
import math
import os
from pathlib import Path
from .learning_curve import atomic_json
from .curve_report import stats


def collect(root):
    protocol=json.loads((Path(root)/'protocol.json').read_text());spec=protocol['extension'];base=protocol['source_protocol']['spec']
    historical=json.loads((Path(spec['source_report'])/'results.json').read_text())['results']
    selected=lambda r:r['family'] in spec['combinations'] and r['scale'] in spec['combinations'][r['family']] and r['seed'] in spec['seeds']
    rows=[dict(r,historical_baseline=True) for r in historical if selected(r) and r['training_tokens']==spec['base_training_bytes']]
    for p in sorted(Path(root).glob('S*/*/seed*/results.json')):rows+=json.loads(p.read_text())['results']
    failures=[json.loads(p.read_text()) for p in sorted(Path(root).glob('S*/*/seed*/failure.json'))]
    timings=[{'identifier':str(p),'stages':json.loads(p.read_text())} for p in sorted(Path(root).glob('S*/*/seed*/stage_timings.json'))]
    expected={(f,s,k,t) for f,scales in spec['combinations'].items() for s in scales for k in spec['seeds'] for t in spec['milestones']}
    present={(r['family'],r['scale'],r['seed'],r['training_tokens']) for r in rows}
    summaries=[];pairs=[];slopes=[]
    times=[spec['base_training_bytes']]+spec['milestones']
    for budget in base['validation_bytes']:
        for f,scales in spec['combinations'].items():
            for s in scales:
                series=[]
                for t in times:
                    group=[r for r in rows if r['family']==f and r['scale']==s and r['training_tokens']==t]
                    if not group:continue
                    repkeys=('mean_repeats','fraction_repeat_ge2','fraction_repeat_ge4','fraction_at_max','repeat_probability_mean','unique_edges_per_token','total_edge_operations_per_token','compute_fraction_repetition')
                    item={'family':f,'scale':s,'training_tokens':t,'validation_bytes':budget,'loss':stats([r['validation'][str(budget)]['loss'] for r in group]),'ppl':stats([r['validation'][str(budget)]['ppl'] for r in group]),'repetition':{k:stats([r['validation'][str(budget)]['repetition'][k] for r in group]) for k in repkeys}}
                    summaries.append(item)
                    if len(group)==len(spec['seeds']):series.append(item)
                for a,b in zip(series,series[1:]):
                    slopes.append({'family':f,'scale':s,'validation_bytes':budget,'from_bytes':a['training_tokens'],'to_bytes':b['training_tokens'],'loss_improvement':a['loss']['mean']-b['loss']['mean'],'loss_slope_per_log_bytes':(b['loss']['mean']-a['loss']['mean'])/math.log(b['training_tokens']/a['training_tokens'])})
        for kind,control,scales in [('same_edge','float',[('S2','S2'),('S3','S3'),('S4','S4')]),('same_memory','float',[('S0','S2'),('S1','S3'),('S2','S4')]),('repeat_ternary','ternary',[('S2','S2'),('S3','S3'),('S4','S4')])]:
            for cs,rs in scales:
                for t in times:
                    a={r['seed']:r for r in rows if r['family']=='repeat' and r['scale']==rs and r['training_tokens']==t}
                    b={r['seed']:r for r in rows if r['family']==control and r['scale']==cs and r['training_tokens']==t}
                    seeds=sorted(a.keys()&b.keys())
                    if not seeds:continue
                    delta={str(k):a[k]['validation'][str(budget)]['loss']-b[k]['validation'][str(budget)]['loss'] for k in seeds}
                    first=seeds[0];memratio=a[first]['memory']['packed_static_bytes']/b[first]['memory']['packed_static_bytes']
                    opsratio=stats([a[k]['validation'][str(budget)]['repetition']['total_edge_operations_per_token']/b[k]['validation'][str(budget)]['repetition']['total_edge_operations_per_token'] for k in seeds])
                    pairs.append({'kind':kind,'control':control,'control_scale':cs,'repeat_scale':rs,'training_tokens':t,'validation_bytes':budget,'loss_delta':stats(list(delta.values())),'per_seed_delta':delta,'repeat_better_seeds':sum(v<0 for v in delta.values()),'packed_static_byte_ratio':memratio,'edge_operations_ratio':opsratio,'same_memory_matched_within_one_percent':abs(memratio-1)<=.01})
    return {'spec':spec,'base_spec':base,'complete':not(expected-present),'expected_extension_milestones':len(expected),'completed_extension_milestones':len(expected&present),'latest_completed_training_bytes':max((r['training_tokens'] for r in rows),default=0),'missing':sorted(expected-present),'failures':failures,'stage_wall_timings':timings,'results':rows,'summaries':summaries,'paired_comparisons':pairs,'improvement_slopes':slopes}


def format_stat(x):return f"{x['mean']:.5f} ± {x['std']:.5f}" if x['std'] is not None else f"{x['mean']:.5f} (n={x['n']})"


def conclusions(data):
    final=max(data['spec']['milestones']);last=data['spec']['milestones'][-2:];base=data['spec']['base_training_bytes'];n=len(data['spec']['seeds'])
    p=[x for x in data['paired_comparisons'] if x['validation_bytes']==4096 and x['loss_delta']['n']==n]
    groups=lambda kind,t:[x for x in p if x['kind']==kind and x['training_tokens']==t]
    final_edge=groups('same_edge',final);final_mem=groups('same_memory',final)
    compact=lambda xs:{x['control_scale']+'→'+x['repeat_scale']:{'mean':x['loss_delta']['mean'],'SD':x['loss_delta']['std'],'seed_deltas':x['per_seed_delta'],'repeat_better_seeds':x['repeat_better_seeds']} for x in xs}
    improvement={}
    for family in ('float','repeat'):
        improvement[family]={}
        for scale in ('S2','S3','S4'):
            a=next((x for x in data['summaries'] if x['family']==family and x['scale']==scale and x['training_tokens']==base and x['validation_bytes']==4096 and x['loss']['n']==n),None)
            b=next((x for x in data['summaries'] if x['family']==family and x['scale']==scale and x['training_tokens']==final and x['validation_bytes']==4096 and x['loss']['n']==n),None)
            if a and b:improvement[family][scale]=a['loss']['mean']-b['loss']['mean']
    large=[x for x in final_edge if x['repeat_scale'] in ('S3','S4')]
    status='incomplete' if len(large)!=2 else ('survived' if all(x['loss_delta']['mean']<0 for x in large) else 'disappeared' if all(x['loss_delta']['mean']>=0 for x in large) else 'mixed')
    memory_status='incomplete' if len(final_mem)!=3 else ('yes' if any(x['loss_delta']['mean']<=0 for x in final_mem) else 'no')
    stable=[]
    for pair in final_mem:
        prev=next((x for x in groups('same_memory',last[0]) if x['repeat_scale']==pair['repeat_scale']),None)
        if prev and prev['loss_delta']['mean']<=0 and pair['loss_delta']['mean']<=0 and pair['repeat_better_seeds']>=2 and prev['repeat_better_seeds']>=2:stable.append(pair['repeat_scale'])
    if not data['complete']:case='Incomplete: final decision unavailable; no favorable intermediate checkpoint is substituted.'
    elif status=='disappeared' and memory_status=='no':case='Case A: v3 crossover was substantially explained by unequal convergence under fixed-data-budget scaling. Current same-edge repetition-as-weight-substitution hypothesis is not supported strongly enough to justify further scaling in this architecture.'
    elif stable and any(x['loss_delta']['mean']<0 for x in large):case='Measured stable same-memory advantages support further independent replication and larger-scale testing, but do not establish a universal replacement for numerical weight magnitude. Repetition operation ratios below must be considered separately; memory and operations cannot be converted to energy without hardware evidence.'
    else:case='Case B/mixed: repetition may survive as a mechanism signal, but strong stable memory-efficiency evidence is not demonstrated. Further scaling is not justified by a transient crossover alone.'
    return {'same_edge_crossover':status,'same_memory_crossover':memory_status,'stable_majority_seed_memory_pairs':stable,'case':case},[
        '1. Float final-minus-baseline improvements (positive means lower loss): '+str(improvement['float']),
        '2. Repeat improvements: '+str(improvement['repeat']),
        '3. S3/S4 final same-edge crossover: '+status+'. Final gaps: '+str(compact(final_edge)),
        '4. Last two milestone same-edge gaps: '+str({t:compact(groups('same_edge',t)) for t in last}),
        '5. Repeat−ternary final paired gaps: '+str(compact(groups('repeat_ternary',final))),
        '6. Float S0↔Repeat S2: '+str([x for x in final_mem if x['repeat_scale']=='S2']),
        '7. Float S1↔Repeat S3: '+str([x for x in final_mem if x['repeat_scale']=='S3']),
        '8. Float S2↔Repeat S4: '+str([x for x in final_mem if x['repeat_scale']=='S4']),
        '9. Final measured same-memory crossover: '+memory_status+'. Earlier sign changes alone are not success.',
        '10. Seed directions: '+str(compact(final_mem))+'. Stable majority-seed pairs at both last milestones: '+str(stable),
        '11. Same-memory paired gap trajectories: '+str({t:compact(groups('same_memory',t)) for t in [base]+data['spec']['milestones']}),
        '12. Undertraining interpretation: '+case+' Longer training tests convergence confounding; it does not prove causal architectural superiority.',
        '13. Exact continuation coverage and held-out coverage are tabulated below. Visited fraction measures exposure, not sufficient optimization; baseline whole-history coverage was not recorded.',
        '14. Within-family loss/coverage across scales are measured in the curves. Larger stored capacity alone is not usable capacity, especially with the unchanged16-channel state.',
        '15. Full repeat histograms/probabilities and usage trajectories are in JSON and usage plot; bounded counts and sampled inference are unchanged.',
        '16. Same-memory pairs retain approximately16× structural edges at ratio≈1 packed static bytes. Their measured edge-operation ratios are reported per seed/milestone. Extra repetitions increase additions; no wall-time or energy equivalence is claimed.',
        '17. Mechanism versus optimization artifact: '+case+' Three seeds, one corpus, biased ST and hashed compact state limit attribution.',
        '18. Continue researching? '+('Final evidence is incomplete; defer the decision.' if not data['complete'] else 'Limited independent replication is justified for stable measured memory pairs; do not infer universal weight replacement.' if stable else 'No automatic larger-scale extension is justified; current memory/compute evidence is insufficient.'),
        '19. End this scaling line? '+('Defer until the preregistered final milestone is measured or resource limits are documented.' if not data['complete'] else 'Yes, further scaling of this architecture is not justified by the measured evidence.' if status=='disappeared' and memory_status=='no' else 'Do not claim universal refutation. Restrict any next work to independent validation of the measured tradeoff, rather than unmeasured scaling.')]


def plots(data,out):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/rlmm-matplotlib')
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    full=[s for s in data['summaries'] if s['validation_bytes']==4096 and s['loss']['n']==3]
    def save(fig,name):
        fig.tight_layout();path=out/(name+'.svg');fig.savefig(path,metadata={'Date':None});path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n');fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    for name in ('loss_training','repeat_usage','validation_slope'):
        fig,ax=plt.subplots(figsize=(9,5))
        for f,scales in data['spec']['combinations'].items():
            if name=='repeat_usage' and f!='repeat':continue
            for s in scales:
                points=[x for x in full if x['family']==f and x['scale']==s]
                if name=='validation_slope':
                    points=[x for x in data['improvement_slopes'] if x['family']==f and x['scale']==s and x['validation_bytes']==4096]
                    if points:ax.plot([x['to_bytes'] for x in points],[x['loss_slope_per_log_bytes'] for x in points],marker='o',label=f+' '+s)
                elif points:
                    values=[x['loss'] if name=='loss_training' else x['repetition']['mean_repeats'] for x in points]
                    ax.errorbar([x['training_tokens'] for x in points],[x['mean'] for x in values],yerr=[x['std'] for x in values],marker='o',label=f+' '+s)
        ax.set_xscale('log');ax.set_xlabel('Cumulative training bytes');ax.set_ylabel(name);ax.grid(alpha=.2);ax.legend(fontsize=8);save(fig,name)
    for kind,name in [('same_edge','same_edge_gap'),('same_memory','same_memory_gap'),('repeat_ternary','repeat_ternary_gap')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for scale in ('S2','S3','S4'):
            points=[x for x in data['paired_comparisons'] if x['kind']==kind and x['repeat_scale']==scale and x['validation_bytes']==4096 and x['loss_delta']['n']==3]
            if points:ax.errorbar([x['training_tokens'] for x in points],[x['loss_delta']['mean'] for x in points],yerr=[x['loss_delta']['std'] for x in points],marker='o',label=points[0]['control_scale']+' ↔ '+scale)
        ax.axhline(0,color='gray');ax.set_xscale('log');ax.set_xlabel('Cumulative training bytes');ax.set_ylabel('Repeat − control loss');ax.grid(alpha=.2);ax.legend();save(fig,name)


def export(root=Path('runs/v3_0_1'),out=Path('research/v3_0_1'),finalize=False):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True);d=collect(root)
    failed=sum(x['classification']=='failed-resumable' for x in d['failures']);limited=len(d['failures'])-failed
    lines=['# PRG-LM v3.0.1 — Extended-Training Crossover Verification','',f"Complete: {d['complete']}; completed extension milestones: {d['completed_extension_milestones']}/{d['expected_extension_milestones']}; failed jobs: {failed}; resource-limited jobs: {limited}; latest completed training bytes: {d['latest_completed_training_bytes']:,}.",
        '', '## Frozen hypothesis and protocol','',
        'Does v3 same-edge repetition advantage survive substantially longer training, or did larger numerical-weight baselines converge more slowly? Duration extension ONLY: all v3 model/topology/hash/state/representation/controller/max_repeat8/optimizer/lr3e-4/clipping/tokenizer/context32/split/validation/seeds/objective remain unchanged. No tuning, new architecture, old-run retraining or dashboard work.',
        '33 continuations: Float S0–S4; ternary S2–S4; Repeat S2–S4; seeds42/43/44. Source model, both optimizers, RNG and cursor at819200 are restored. New NumPy sampling stream has a verified byte-for-byte819200 prefix match, then continues the same sampler. Evaluation uses unchanged4096/16384 windows, batch8, sampled seed+batch index. Evaluation RNG is restored before training.',
        'Preregistered cumulative milestones:1638400,3276800,6553600,13107200. Final13.1072M is the primary decision; last6.5536M and13.1072M directions are checked. No early stopping or favorable checkpoint selection. Original819200 rows are copied as historical baseline, not re-evaluated.',
        'The original v3 has five4× edge scales and compact16-channel state. Same-edge S2/S3/S4 and SAME measured packed-memory pairs Float S0↔Repeat S2, S1↔S3, S2↔S4 are fixed. No interpolation/extrapolation. n=3 mean±sample SD and every paired seed delta are reported, without significance claims.',
        'CPU, two one-thread workers. Limits fixed before training: six hours per job per milestone stage, free disk at least2GB before each job/checkpoint. Resource-limited jobs retain completed milestones/checkpoints; they do not supply final conclusions. Full checkpoint recreation is unnecessary: all33 sources are present.',
        'Packed/static/dynamic accounting reuses immutable v3 metrics. Exports pack ternary2-bit edges, PyTorch execution decodesFP32. Controller candidates include inactive lanes. Modeled active additions are not actual FLOPs or energy. Process RSS/optimizer tensors/prototype throughput remain separate.',
        '', '## Progress','', '| Family | Scale | Seed42 | Seed43 | Seed44 | Final |','|---|---|---:|---:|---:|---|']
    for f,scales in d['spec']['combinations'].items():
        for s in scales:
            cells=[]
            for seed in d['spec']['seeds']:
                t=max([r['training_tokens'] for r in d['results'] if r['family']==f and r['scale']==s and r['seed']==seed],default=0)
                bad=any(x['family']==f and x['scale']==s and x['seed']==seed for x in d['failures'])
                cells.append(str(t)+(' failed/resource-limited' if bad else ''))
            lines.append(f"| {f} | {s} | {' | '.join(cells)} | {all(x==str(max(d['spec']['milestones'])) for x in cells)} |")
    sanity=out/'sanity.json'
    if sanity.exists():lines+=['','## Sanity checks','',json.loads(sanity.read_text())['summary']]
    lines+=['','## All validation results','', '| Family | Scale | Training bytes | Validation bytes | Loss | Byte PPL | n |','|---|---|---:|---:|---:|---:|---:|']
    for x in d['summaries']:lines.append(f"| {x['family']} | {x['scale']} | {x['training_tokens']} | {x['validation_bytes']} | {format_stat(x['loss'])} | {format_stat(x['ppl'])} | {x['loss']['n']} |")
    lines+=['','## Paired comparisons','', '| Kind | Control | Repeat | Training bytes | Validation bytes | Paired Δloss | Each seed Δloss | Repeat better seeds | Packed ratio | Addition ratio |','|---|---|---|---:|---:|---:|---|---:|---:|---:|']
    for x in d['paired_comparisons']:lines.append(f"| {x['kind']} | {x['control']} {x['control_scale']} | {x['repeat_scale']} | {x['training_tokens']} | {x['validation_bytes']} | {format_stat(x['loss_delta'])} | {x['per_seed_delta']} | {x['repeat_better_seeds']} | {x['packed_static_byte_ratio']:.6f} | {format_stat(x['edge_operations_ratio'])} |")
    lines+=['','## Undertraining diagnostics','', 'Training coverage starts at819200 because v3 did not record whole-history coverage. Coalesced sparse-gradient source rows track exact distinct activated slots, including zero ternary values. Per-source update counts are optimizer-step touches, not nonzero updates. Gradient norms and edge changes sample first64 touched rows every100 steps; they are NOT unbiased whole-graph statistics. Full counters are training-only checkpoint state. Held-out coverage is exact fixed-window topology coverage. Visit coverage alone cannot establish sufficient convergence.',
        '', '| Family | Scale | Seed | Training bytes | Train loss | Continuation covered fraction | Diagnostic unique edges | Diagnostic fraction | Gradient norm mean | Edge update mean |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['results']:
        if r.get('historical_baseline'):continue
        a=r['undertraining_diagnostics'];probes=a['sampled_gradient_updates'];gn=sum(x['pre_clip_gradient_norm'] for x in probes)/max(len(probes),1);upd=sum(x['sampled_edge_latent_mean_abs_update'] for x in probes)/max(len(probes),1)
        lines.append(f"| {r['family']} | {r['scale']} | {r['seed']} | {r['training_tokens']} | {r['train_loss']:.5f} | {a['training_coverage']['visited_edge_fraction_since_resume']:.6f} | {a['held_out_coverage']['unique_structural_edges']} | {a['held_out_coverage']['visited_edge_fraction']:.6f} | {gn:.5f} | {upd:.8f} |")
    lines+=['','## Improvement slopes','', '| Family | Scale | Validation bytes | From | To | Loss improvement | Δloss/Δlog bytes |','|---|---|---:|---:|---:|---:|---:|']
    for s in d['improvement_slopes']:lines.append(f"| {s['family']} | {s['scale']} | {s['validation_bytes']} | {s['from_bytes']} | {s['to_bytes']} | {s['loss_improvement']:.6f} | {s['loss_slope_per_log_bytes']:.6f} |")
    lines+=['','## Repetition and runtime','', 'Full histograms, medians, ≥2/≥4/max fractions, controller probabilities, additions/unique edges/token, repetition fractions, memory, gradient diagnostics, source/checkpoint/stream hashes, train time/throughput, inference throughput/latency and RSS are preserved in results.json at every milestone. CUDA fields are null on this CPU backend. MPS memory is unavailable here. No custom accelerator or GPU performance claim.',
        '', '## Failures and limits','',str(d['failures']),
        'One corpus and three seeds restrict generality. Longer fixed-lr training may still not establish convergence. Hashed addresses and16-channel state constrain usable capacity; ST gradients are biased. Repeat contribution proxies are partly definitional. Measured memory savings do not imply lower compute/energy. Frozen original files/checkpoints are hashed and reverified.']
    if finalize:
        decision,answers=conclusions(d);d['decision']=decision
        lines+=['','## Final19 questions','']+answers
        plots(d,out)
        for name in ('loss_training','same_edge_gap','same_memory_gap','repeat_ternary_gap','repeat_usage','validation_slope'):lines+=['',f'![{name}]({name}.svg)']
    else:lines+=['','Final19-question decision pending until all planned jobs finish or recorded resource bounds are reached.']
    atomic_json(out/'results.json',d);(out/'REPORT.md').write_text('\n'.join(lines)+'\n');return d

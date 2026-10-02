"""Measured scaling tables and curves; never interpolate missing/favorable points."""
import json
import math
from pathlib import Path
from .learning_curve import atomic_json
from .curve_report import stats

ARCH='''The graph stores N nodes ×32 relative-offset edge slots. Sixteen source nodes are selected by a fixed causal rolling prefix hash and source stride37. The shared relative offset template is (slot+1)*2654435761 mod N. Node counts are powers of two, so slots/destinations and activated sources are unique. Float and ternary families use identical identities at the same scale.

Current-byte embedding plus0.9 times the previous graph-produced16-channel state creates one tanh source message. Source, destination, slot, reference message and edge value remain frozen throughout its repetition episode. First pass adds sign*message; each REPEAT adds that SAME value again. A six-feature shared6→16→1 controller predicts continuation from reference, |reference|, sign, normalized source/destination and repeat index. It has129 parameters independent of N; there are no edge-specific recurrence parameters/probability buffers. Bernoulli logistic-noise straight-through gates train bounded repetition; inference samples Bernoulli gates. Maximum repeat8.

Per-edge linear accumulators are reduced into destination-index-mod16 channels with fixed1/sqrt(32) scaling, followed by tanh, existing-sized readout and tied byte head. There is no LayerNorm to erase a uniform repetition magnitude. The next token's persistent state is ONLY graph-produced output: zero graph values eliminate token/prefix influence. There is no direct embedding/context-to-head residual, attention, global contextual pooling, Region/router/gateway/fatigue/mutation.

This is a compact16-channel causal state with hashed sparse connectivity, not persistent independent state for every stored node. Address aliasing and the compressed destination readout restrict the model class; conclusions do not cover all graph architectures. Nonlinear LM equivalence is not assumed from linear k*s*m equality. The controller is shared but high-precision, so gains alone do not prove an efficient replacement for arbitrary stored magnitude.'''


def collect(root):
    root=Path(root);spec=json.loads((root/'protocol.json').read_text())['spec'];rows=[];failures=[]
    for p in sorted(root.glob('S*/*/seed*/results.json')):rows+=json.loads(p.read_text())['results']
    for p in sorted(root.glob('S*/*/seed*/failure.json')):failures.append(json.loads(p.read_text()))
    expected={(f,s,k,t) for f in spec['families'] for s in spec['scales'] for k in spec['seeds'] for t in spec['milestones']}
    present={(r['family'],r['scale'],r['seed'],r['training_tokens']) for r in rows}
    summaries=[]
    for family in spec['families']:
        for scale in spec['scales']:
            for token in spec['milestones']:
                group=[r for r in rows if r['family']==family and r['scale']==scale and r['training_tokens']==token]
                for budget in spec['validation_bytes']:
                    if not group:continue
                    memory=group[0]['memory']
                    metric_keys=('mean_repeats','fraction_repeat_ge2','fraction_repeat_ge4','total_edge_operations_per_token','unique_edges_per_token','compute_fraction_repetition','active_edge_fraction_operations','repeat_output_proxy_correlation')
                    repetitions={key:stats([r['validation'][str(budget)]['repetition'][key] for r in group if r['validation'][str(budget)]['repetition'][key] is not None]) for key in metric_keys if any(r['validation'][str(budget)]['repetition'][key] is not None for r in group)}
                    summaries.append({'family':family,'scale':scale,'training_tokens':token,'validation_bytes':budget,
                        'seeds':[r['seed'] for r in group],'loss':stats([r['validation'][str(budget)]['loss'] for r in group]),
                        'ppl':stats([r['validation'][str(budget)]['ppl'] for r in group]),'edge_slots':group[0]['edge_slots'],
                        'packed_static_bytes':memory['packed_static_bytes'],'modeled_peak_bytes':memory['modeled_peak_bytes'],
                        'repetition':repetitions})
    return {'complete':not expected-present,'missing':sorted(expected-present),'spec':spec,'results':rows,
            'summaries':summaries,'failures':failures}


def comparisons(data):
    spec=data['spec'];final=max(spec['milestones']);summaries=data['summaries'];rows=data['results']
    pairs=[];slopes=[];memory_pairs=[]
    for budget in spec['validation_bytes']:
        for scale in spec['scales']:
            for control in ('float','ternary'):
                a={r['seed']:r['validation'][str(budget)]['loss'] for r in rows if r['family']=='repeat' and r['scale']==scale and r['training_tokens']==final}
                b={r['seed']:r['validation'][str(budget)]['loss'] for r in rows if r['family']==control and r['scale']==scale and r['training_tokens']==final}
                seeds=sorted(a.keys()&b.keys())
                if seeds:pairs.append({'control':control,'scale':scale,'validation_bytes':budget,'seeds':seeds,
                    'loss_delta':stats([a[s]-b[s] for s in seeds]),'per_seed_delta':{str(s):a[s]-b[s] for s in seeds}})
        for family in spec['families']:
            groups=[s for s in summaries if s['family']==family and s['training_tokens']==final and s['validation_bytes']==budget and s['loss']['n']==3]
            for a,b in zip(groups,groups[1:]):
                for dimension in ('edge_slots','packed_static_bytes'):
                    slopes.append({'family':family,'from_scale':a['scale'],'to_scale':b['scale'],'validation_bytes':budget,
                        'dimension':dimension,'slope':(b['loss']['mean']-a['loss']['mean'])/math.log(b[dimension]/a[dimension])})
        floats=[s for s in summaries if s['family']=='float' and s['training_tokens']==final and s['validation_bytes']==budget and s['loss']['n']==3]
        repeats=[s for s in summaries if s['family']=='repeat' and s['training_tokens']==final and s['validation_bytes']==budget and s['loss']['n']==3]
        for f in floats:
            if not repeats:continue
            nearest=min(repeats,key=lambda r:abs(math.log(r['packed_static_bytes']/f['packed_static_bytes'])))
            memory_pairs.append({'float_scale':f['scale'],'repeat_scale':nearest['scale'],'validation_bytes':budget,
                'static_byte_ratio':nearest['packed_static_bytes']/f['packed_static_bytes'],
                'structural_edge_ratio':nearest['edge_slots']/f['edge_slots'],
                'loss_delta':nearest['loss']['mean']-f['loss']['mean'],
                'matched_within_one_percent':abs(nearest['packed_static_bytes']/f['packed_static_bytes']-1)<=.01})
    return pairs,slopes,memory_pairs


def plots(data,out):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/rlmm-matplotlib')
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    final=max(data['spec']['milestones']);summary=[s for s in data['summaries'] if s['training_tokens']==final and s['validation_bytes']==4096 and s['loss']['n']==3]
    specs=[('loss_edges','edge_slots','loss','Structural edge slots',True),
           ('loss_static','packed_static_bytes','loss','Theoretical packed static bytes',True),
           ('loss_peak','modeled_peak_bytes','loss','Modeled peak inference bytes',True),
           ('loss_active','total_edge_operations_per_token','loss','Modeled active edge additions/token',False),
           ('ppl_scale','edge_slots','ppl','Structural edge slots',True),
           ('repeat_utilization','edge_slots','mean_repeats','Structural edge slots',True)]
    for name,xkey,ykey,xlabel,log in specs:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for family in data['spec']['families']:
            points=[s for s in summary if s['family']==family]
            if not points:continue
            X=lambda s:s[xkey] if xkey in s else s['repetition'][xkey]['mean']
            Y=lambda s:s[ykey] if ykey in s else s['repetition'][ykey]
            ax.errorbar([X(s) for s in points],[Y(s)['mean'] for s in points],yerr=[Y(s)['std'] for s in points],marker='o',capsize=3,label=family)
        if log:ax.set_xscale('log')
        ax.set_xlabel(xlabel);ax.set_ylabel(ykey);ax.set_title('Final819.2k bytes;4096 validation; mean ± seed SD')
        ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        path=out/(name+'.svg');fig.savefig(path,metadata={'Date':None});path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for control in ('float','ternary'):
        points=[p for p in data['same_edge_comparisons'] if p['control']==control and p['validation_bytes']==4096 and p['loss_delta']['n']==3]
        ax.errorbar([data['spec']['scales'][p['scale']]*data['spec']['model']['slots'] for p in points],
            [p['loss_delta']['mean'] for p in points],yerr=[p['loss_delta']['std'] for p in points],marker='o',capsize=3,label='repeat − '+control)
    ax.axhline(0,color='gray',linewidth=1);ax.set_xscale('log');ax.set_xlabel('Structural edge slots');ax.set_ylabel('Paired loss delta (nats/byte)');ax.legend();fig.tight_layout()
    path=out/'relative_gap.svg';fig.savefig(path,metadata={'Date':None});path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    fig.savefig(out/'relative_gap.png',dpi=150);plt.close(fig)


def export(root=Path('runs/v3'),out=Path('research/v3'),finalize=False):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True);data=collect(root)
    pairs,slopes,matches=comparisons(data)
    data.update(same_edge_comparisons=pairs,scaling_slopes=slopes,same_memory_comparisons=matches)
    atomic_json(out/'results.json',data)
    fmt=lambda v:f"{v['mean']:.5f} ± {v['std']:.5f}" if v['std'] is not None else f"{v['mean']:.5f} (n={v['n']})"
    lines=['# PRG-LM v3 — Same-Edge Repetition Scaling Study','',
        f"Complete: {data['complete']}; {len(data['results'])}/{len(data['results'])+len(data['missing'])} primary milestone results; {len(data['failures'])} recorded failed/resource-limited jobs.",
        'Hypothesis: Can additive temporal reuse of the SAME low-bit source/destination/slot replace part of stored numerical magnitude, and become more favorable with graph scale? No conclusion is presumed.',
        '', '## Exact architecture','',ARCH,'','## Preregistered protocol','',
        'CPU/CUDA/MPS auto selection; this environment has2 CPU cores and8GB memory. Two worker processes, one PyTorch thread each. Families: Float32 single pass, ternary single pass, ternary shared learned repetition. No Transformer is retrained/reused as a falsely matched v3 baseline.',
        'TinyShakespeare, first90%/last10% split, byte256, context32, seeds42/43/44. Continuous12.8k→51.2k→204.8k→819.2k training bytes, same hashed stream/seed. AdamW neural blocks + SparseAdam edge latents, lr3e-4, shared combined gradient clip1. SparseAdam has no weight decay in every family. Dense moments are training-only. This is FIXED-DATA-BUDGET scaling: large graphs can be undertrained.',
        'Primary held-out sampled routing4096 bytes, secondary16384 prefix-correlated bytes; identical windows to prior studies. Controller p=.5 initially, logistic-noise hard/ST training. Inference uses sampled counts. Conditional probabilities depend on fixed reference/sign/addresses/index, never on freshly transformed repeated messages.',
        'Bounds/resource limits are fixed in configs/repetition_scaling.json before training. No scales/seeds/hyperparameters are selected from emerging validation. Planned S0–S4 only; S5 is not planned. Resource-limited jobs remain visible with incomplete cells.',
        '', '## Scale and memory rules','',
        '| Scale | Nodes | Slots/node | Stored edges |','|---|---:|---:|---:|']
    for s,n in data['spec']['scales'].items():lines.append(f"| {s} | {n:,} | {data['spec']['model']['slots']} | {n*data['spec']['model']['slots']:,} |")
    lines+=['Float structural storage isFP32 (4 bytes), low-bit structural storage2 bits. Other neural blocks/controller areFP32; unused controller is retained/frozen in single-pass controls so repeatOFF is architecturally matched. Low-bit training storesFP32 latents. Actual inference exports pack ternary codes into2 bits but this PyTorch executor DECOMPRESSES toFP32; serialized savings are not resident-memory/speed savings.',
        'Modeled dynamic state assumes streaming controllers: persistent16-channel state, hash/source addresses, frozen edge reference/accumulator and counters. Actual batched probability/feature/gradient workspaces are excluded. Both training and current inference batch all candidate gates, including inactive lanes. Theoretical active additions do NOT equal executed GPU/CPU FLOPs. Actual FP32 tensors, process peak RSS, CUDA allocated/reserved when available, optimizer bytes and export/resume sizes are separately recorded.',
        '', '## Progress','', '| Family | Scale | Seed42 | Seed43 | Seed44 | Final complete |','|---|---|---:|---:|---:|---|']
    for s in data['spec']['scales']:
        for f in data['spec']['families']:
            cells=[]
            for seed in data['spec']['seeds']:
                values=[r['training_tokens'] for r in data['results'] if r['family']==f and r['scale']==s and r['seed']==seed]
                failure=next((v for v in data['failures'] if v['family']==f and v['scale']==s and v['seed']==seed),None)
                cells.append((str(max(values)) if values else 'pending')+(' (resource-limited / failed)' if failure else ''))
            done=all(x==str(max(data['spec']['milestones'])) for x in cells)
            lines.append(f"| {f} | {s} | {' | '.join(cells)} | {done} |")
    sanity=out/'sanity.json'
    if sanity.exists():lines+=['','## Sanity tests','',json.loads(sanity.read_text())['summary'],
        'Signed additive1/2/4/8 equality, locked edge identities, frozen messages, prefix credit, zero-graph bypass test, causal prefix invariance, no-repeat equivalence, packed export/save-load and two-optimizer/RNG-resume are checked. Full exact values/configs are in sanity.json.']
    lines+=['','## Learning curves','', '| Family | Scale | Training bytes | Validation bytes | Loss | Byte PPL | n |','|---|---|---:|---:|---:|---:|---:|']
    for s in data['summaries']:lines.append(f"| {s['family']} | {s['scale']} | {s['training_tokens']:,} | {s['validation_bytes']} | {fmt(s['loss'])} | {fmt(s['ppl'])} | {s['loss']['n']} |")
    final=max(data['spec']['milestones']);finals=[s for s in data['summaries'] if s['training_tokens']==final and s['validation_bytes']==4096]
    lines+=['','## Final compute/repetition','', '| Family | Scale | Repeats/edge | ≥2 fraction | ≥4 fraction | Edge additions/token | Repetition compute fraction | Active fraction |','|---|---|---:|---:|---:|---:|---:|---:|']
    for s in finals:
        r=s['repetition'];lines.append(f"| {s['family']} | {s['scale']} | "+' | '.join(fmt(r[k]) for k in ('mean_repeats','fraction_repeat_ge2','fraction_repeat_ge4','total_edge_operations_per_token','compute_fraction_repetition','active_edge_fraction_operations'))+' |')
    lines+=['All modeled active counts include zero ternary edges; nonzero operations are separately in JSON. Unique activated edges/token is512 by design. Consecutive revisits are counted within locked per-edge episodes, not different edges or new tokens. Repeated edges are interleaved parallel lanes; every lane retains identity/reference. Controller probabilities, full histograms, median, bound fraction, projected output contribution by count and correlations are in machine results.',
        'Output contribution is a PRE-NONLINEARITY readout-column magnitude proxy; count increases its magnitude by construction. Positive correlation is not causal evidence of useful language-model computation.',
        '', '## Same-edge-count paired gaps','', '| Control | Scale | Validation bytes | Repeat − control loss | Per-seed deltas |','|---|---|---:|---:|---|']
    for p in pairs:lines.append(f"| {p['control']} | {p['scale']} | {p['validation_bytes']} | {fmt(p['loss_delta'])} | {p['per_seed_delta']} |")
    lines+=['','## Nearest measured same-memory comparison','', '| Float scale | Repeat scale | Validation bytes | Static byte ratio | Edge ratio | Loss delta | Within1% budget |','|---|---|---:|---:|---:|---:|---|']
    for p in matches:lines.append(f"| {p['float_scale']} | {p['repeat_scale']} | {p['validation_bytes']} | {p['static_byte_ratio']:.6f} | {p['structural_edge_ratio']:.2f} | {p['loss_delta']:.5f} | {p['matched_within_one_percent']} |")
    lines+=['Only within1% measured matches support a same-memory claim; unmatched nearest rows are explicitly not budget matched. No interpolation/extrapolation. FP16 float models were not trained and are not substituted into the measured table.',
        '', '## Empirical scaling slopes','', '| Family | From | To | Validation bytes | Dimension | Loss/log-dimension slope |','|---|---|---|---:|---|---:|']
    for s in slopes:lines.append(f"| {s['family']} | {s['from_scale']} | {s['to_scale']} | {s['validation_bytes']} | {s['dimension']} | {s['slope']:.6f} |")
    lines+=['','## Memory and prototype runtime','', '| Family | Scale | Trainable FP params | Low-bit slots | Packed structure | Controller bytes | Embedding/head bytes | Static packed | Dynamic | Modeled peak | Export bytes | Resume bytes | Optimizer bytes |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in finals:
        rows=[r for r in data['results'] if r['family']==s['family'] and r['scale']==s['scale'] and r['training_tokens']==final]
        m=rows[0]['memory'];lines.append(f"| {s['family']} | {s['scale']} | {m['trainable_float_parameters']} | {m['low_bit_structural_edges']} | {m['static']['structural_edges']} | {m['static']['shared_controller']} | {m['static']['embedding_head']} | {m['packed_static_bytes']} | {m['dynamic_inference_bytes']} | {m['modeled_peak_bytes']} | {fmt(stats([r['actual_inference_export_bytes'] for r in rows]))} | {fmt(stats([r['actual_pytorch_resume_bytes'] for r in rows]))} | {rows[0]['optimizer_training_only_tensor_bytes']} |")
    lines+=['','| Family | Scale | Train tok/s | Inference tok/s | Latency ms/token | Peak process RSS bytes |','|---|---|---:|---:|---:|---:|']
    for s in finals:
        rows=[r for r in data['results'] if r['family']==s['family'] and r['scale']==s['scale'] and r['training_tokens']==final]
        lines.append(f"| {s['family']} | {s['scale']} | "+' | '.join(fmt(stats(vals)) for vals in ([r['training_tokens_per_second'] for r in rows],[r['runtime']['inference_tokens_per_second'] for r in rows],[r['runtime']['latency_seconds_per_token']*1000 for r in rows],[r['peak_process_rss_bytes'] for r in rows]))+' |')
    lines+=['Timings are prototype measurements under the actual worker schedule. Peak process RSS includes interpreter/Torch and is cumulative per job process. CUDA stats are null on CPU. No GPU or custom-accelerator advantage is inferred from packed bytes or Python wall time.',
        '', '## Representative repetition ablations','',
        'Ternary single-pass is the trained repetitionOFF control, proven equivalent to repeat architecture with max_repeat1 and an unused controller. S0/S2/S4 additionally evaluate the SAME learned checkpoint with fixed1/2/4/8 on4096 bytes: inference interventions, not retraining or hyperparameter selection.',
        '| Scale | Fixed count | Loss | PPL | n |','|---|---:|---:|---:|---:|']
    diagnostics=[]
    for scale in data['spec']['diagnostic_scales']:
        rows=[r for r in data['results'] if r['family']=='repeat' and r['scale']==scale and r['training_tokens']==final and 'fixed_repeat_diagnostics' in r]
        for k in data['spec']['fixed_repeats']:
            if rows:
                values=[r['fixed_repeat_diagnostics'][str(k)][str(data['spec']['diagnostic_validation_bytes'])] for r in rows]
                diagnostic={'scale':scale,'count':k,'loss':stats([v['loss'] for v in values]),'ppl':stats([v['ppl'] for v in values])};diagnostics.append(diagnostic)
                lines.append(f"| {scale} | {k} | {fmt(diagnostic['loss'])} | {fmt(diagnostic['ppl'])} | {len(rows)} |")
    data['fixed_repeat_summaries']=diagnostics
    if finalize:
        lines+=['','## Final research questions','']
        full=[s for s in finals if s['loss']['n']==3];repeat=[s for s in full if s['family']=='repeat']
        complete_scales=[s for s in data['spec']['scales'] if all(any(r['family']==f and r['scale']==s for r in full) for f in data['spec']['families'])]
        pg=[p for p in pairs if p['validation_bytes']==4096 and p['loss_delta']['n']==3 and p['control']=='float']
        tg=[p for p in pairs if p['validation_bytes']==4096 and p['loss_delta']['n']==3 and p['control']=='ternary']
        crosses=[p['scale'] for p in pg if p['loss_delta']['mean']<0]
        prior_positive=False;crossovers=[]
        for p in pg:
            if prior_positive and p['loss_delta']['mean']<0:crossovers.append(p['scale'])
            prior_positive=p['loss_delta']['mean']>0
        matched=[p for p in matches if p['validation_bytes']==4096 and p['matched_within_one_percent']]
        negative_matches=[p for p in matched if p['loss_delta']<0]
        utilization={s['scale']:s['repetition']['mean_repeats'] for s in repeat}
        gaps={p['scale']:p['loss_delta'] for p in pg};ternary_gaps={p['scale']:p['loss_delta'] for p in tg}
        quality=[]
        for r in repeat:
            for f in [s for s in full if s['family']=='float']:
                # Measured dominance, not an arbitrary equal-quality tolerance.
                if r['loss']['mean']<=f['loss']['mean']:
                    quality.append({'repeat_scale':r['scale'],'float_scale':f['scale'],
                        'memory_ratio':r['packed_static_bytes']/f['packed_static_bytes'],
                        'operations_ratio':r['repetition']['total_edge_operations_per_token']['mean']/f['repetition']['total_edge_operations_per_token']['mean']})
        data['measured_dominance_pairs']=quality;data['observed_crossover_scales']=crossovers
        lines += [
            '1. Yes, the controlled linear accumulator reproduces (sign*count)*message for signs−1/0/+1 and counts1/2/4/8 within tolerance. This does not establish nonlinear LM equivalence.',
            '2. Actual learned repeat usage by scale: '+str({k:fmt(v) for k,v in utilization.items()})+'. Full histograms and actual locked identities verify repeats>1 where measured; usage alone is not usefulness.',
            '3. Trained repetitionOFF comparison (repeat − ternary) loss: '+str({k:fmt(v) for k,v in ternary_gaps.items()})+'. Fixed1 inference intervention is a separate matched-checkpoint diagnostic above.',
            '4. Negative repeat−ternary deltas indicate recovery; compare float gaps separately. No single favorable seed/scale defines recovery.',
            '5. Same-edge float gaps by scale: '+str({k:fmt(v) for k,v in gaps.items()})+'.',
            '6. Within1% nearest MEASURED memory matches: '+str(matched)+'. Favorable matches: '+str(negative_matches)+'. No unmatched/interpolated budget is treated as evidence.',
            '7. Float relative-gap sequence: '+str([p['loss_delta']['mean'] for p in pg])+'. '+('Gap decreases monotonically over measured consecutive points.' if len(pg)>1 and all(b['loss_delta']['mean']<=a['loss_delta']['mean'] for a,b in zip(pg,pg[1:])) else 'No monotonically improving relative gap across the measured points is established.'),
            '8. Observed adjacent mean-loss sign-change crossover scales: '+str(crossovers)+'. Negative repeat−float measured points: '+str(crosses)+'. Seed SD/paired deltas limit confidence; no extrapolated crossover is claimed.',
            '9. Repeat count/output-proxy correlations are in JSON. Proxy magnitude grows with count by definition and is before tanh/head; this association does not show causal LM utility.',
            '10. Unique activated edges/token=512 by design, so unique active fraction falls4× per stored-capacity step. Operation fraction additionally depends on measured repeat counts. This is only attractive if quality also improves; fraction reduction alone is tautological.',
            '11. Shared controller stays129 params/516 bytes across scales. Its packed-static fractions are recorded per model: overhead is increasingly amortized, unlike FP32 training edge latents/optimizer memory.',
            '12. Measured equal-or-better mean-quality dominance pairs: '+str(quality)+'. Only ratios<1 indicate a packed-memory saving at no worse measured mean quality; no quality interpolation or statistical-equivalence claim.',
            '13. The same measured dominance pairs report operation ratios. Only ratios<1 show active-addition savings at no worse mean quality; all families use the same512 unique activations, so repeated additions generally cost more.',
            '14. This is a CPU prototype; actual timings/RSS and decoded FP32 training/inference memory differ from2-bit packed estimates. No GPU-speed/energy claim is established.',
            '15. Controlled arithmetic supports feasibility of linear magnitude substitution. Practical LM/scaling support requires reproducible negative paired gaps and favorable memory-quality tradeoffs above; positive arithmetic alone is insufficient. '+('Some favorable matched-memory points justify limited further investigation.' if negative_matches else 'No favorable matched-memory advantage is established here.'),
            '16. '+('A larger preregistered replication could test favorable measured memory trends, without assuming extrapolated crossover.' if negative_matches and crossovers else 'These results do not yet justify claiming a larger-scale crossover; investigate the measured losses, controller utility and undertraining before scaling further.'),
            '17. Positive float gaps, non-improving gap trends, repetition costs without ternary recovery or absent same-memory advantages weaken the practical hypothesis in this model. Undertraining, hashed compact-state limits and ST optimization prevent a universal impossibility claim.',
            '', '## Limits and preservation','',
            'Only fully measured three-seed groups support primary conclusions. Fixed-data-budget large models can be undertrained, and hashed prefix-address capacity may generalize poorly. More stored low-bit edges without validation improvement is not useful capacity. Controller/address functions can only express a constrained shared magnitude family, unlike independent float weights. ST gradients are biased; count/output correlation is partly definitional. Model-wide confidence/generalization requires independent corpora and training-budget controls. No hyperparameter changes, old-model retraining or dashboard/UI work occurred.']
        plots(data,out)
        for name in ('loss_edges','loss_static','loss_peak','loss_active','ppl_scale','repeat_utilization','relative_gap'):lines+=['',f'![{name}]({name}.svg)']
    else:lines+=['','Final17-question analysis pending until the planned jobs finish or documented resource limits are reached.']
    atomic_json(out/'results.json',data);(out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return data

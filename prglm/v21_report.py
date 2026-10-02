"""Automatically generated v2.1 report: frozen baselines, no missing-value guesses."""
import hashlib
import json
import math
from pathlib import Path
from .learning_curve import atomic_json
from .curve_report import stats,svg_plot

BASELINES=('transformer_core','transformer_total','prg_v1','prg_v2_adaptive','prg_v2_uniform',
           'prg_v2_no_recurrence','prg_v2_no_accumulation')
ARCHITECTURE='''PRGLMv21 inherits frozen v1 propagation and shared RouterNet; no topology mutation or attention. Before each token, persistent node state is pooled into K fixed node-index-mod-K channels by sum/sqrt(R*N/K), then tanh. This prefix-only context uses the existing readout projection weight without its bias and a learned scalar sigmoid gate initialized at0 (gate0.5). It is added once to normalized per-walker local readout before existing LayerNorm/vocabulary head. It is a real contextual input, not a gradient-only loss or injected gradient.

Both accumulation settings read tanh of the same current-token accumulator, with the same contextual input, router, state update, dimensions and memory convention. OFF replaces previous accumulator with incoming delta; ON adds it with the configured decay. Router features can consequently differ because accumulator contents differ. This is an intended downstream consequence, not a changed RouterNet. The added tanh is explicit and shared; the bundle comparison with v2 does not isolate the temporal fix alone.

At the hard cycle cap, remaining walkers contribute their LAST PROCESSED source accumulator message, saved before any transition. An unprocessed gateway destination is never gathered for readout. Expected routing terminates surviving mass from processed current sources before final transitions. Valid messages can cancel to zero; the eliminated artifact is reading an uninitialized destination, not a guarantee that every valid signal is nonzero.

The diagnostic probe requires >=1 GATEWAY action per walker before OUTPUT. If needed, LOCAL/RETURN are masked when only required hops plus one destination-processing cycle remain. The cycle bound must exceed minimum hops. Destinations are processed before the bound; no topology changes or auxiliary loss occur. Core has no such constraint. Expected probe routing is deliberately unavailable because its merged soft branches cannot faithfully track per-walker hop requirements. Expected core/OFF remain supplementary region-mass surrogates, not exact expectations.'''


def frozen_baselines():
    rows=[]
    for root in ('runs/phase_a','runs/mutable_v2'):
        for path in sorted(Path(root).glob('seed*/*/results.json')):
            for r in json.loads(path.read_text())['results']:
                if r['model'] not in BASELINES:continue
                rows.append({k:r[k] for k in ('model','seed','training_tokens','validation','memory','serialized_model_bytes')}
                    | {'frozen_source':str(path),'dynamics':{k:v for k,v in r.get('dynamics',{}).items()
                       if k in ('trajectory_example','actual_traversal_count_per_token','routing_entropy','region_utilization')},
                       'gateway_selection_training_ema':r.get('topology',{}).get('gateway_selection_fraction',None)})
    # Reuse canonical committed summaries on a checkout without ignored runs.
    if not rows:
        source=Path('research/v2/results.json')
        for r in json.loads(source.read_text())['results']:
            if r['model'] in BASELINES:
                rows.append({k:r[k] for k in ('model','seed','training_tokens','validation','memory','serialized_model_bytes')}
                    | {'frozen_source':str(source),'dynamics':{k:v for k,v in r.get('dynamics',{}).items()
                       if k in ('trajectory_example','actual_traversal_count_per_token','routing_entropy','region_utilization')},
                       'gateway_selection_training_ema':r.get('topology',{}).get('gateway_selection_fraction',None)})
    return rows


def export(root=Path('runs/v2_1'),out=Path('research/v2_1'),plots=False):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((root/'protocol.json').read_text())['spec']
    new=[]
    for p in sorted(root.glob('seed*/*/results.json')):new+=json.loads(p.read_text())['results']
    old=frozen_baselines();all_rows=old+new
    present={(r['model'],r['seed'],r['training_tokens']) for r in new}
    expected={(m,s,t) for m in spec['models'] for s in spec['seeds'] for t in spec['milestones']}
    complete=not expected-present
    summaries=[]
    for model in BASELINES+tuple(spec['models']):
        for milestone in spec['milestones']:
            rows=[r for r in all_rows if r['model']==model and r['training_tokens']==milestone]
            for budget in spec['validation_bytes']:
                for mode in ('sampled','argmax','expected'):
                    group=[r for r in rows if mode in r['validation'].get(str(budget),{})]
                    if not group:continue
                    summaries.append({'model':model,'training_tokens':milestone,'validation_bytes':budget,
                        'mode':mode,'seeds':[r['seed'] for r in group],
                        'loss':stats([r['validation'][str(budget)][mode]['loss'] for r in group]),
                        'ppl':stats([r['validation'][str(budget)][mode]['ppl'] for r in group])})
    payload={'complete':complete,'missing':sorted(expected-present),'spec':spec,'results':new,
             'frozen_baselines':old,'summaries':summaries}
    sanity=out/'sanity.json'
    if sanity.exists():payload['sanity']=json.loads(sanity.read_text())
    atomic_json(out/'results.json',payload)
    fmt=lambda x:f"{x['mean']:.5f} ± {x['std']:.5f}" if x['std'] is not None else f"{x['mean']:.5f} (n={x['n']})"
    lines=['# PRG-LM v2.1 — Core Correction Study','',
        '## Executive summary','',f'Complete: {complete}. New milestones complete: {len(present)}/{len(expected)}. No frozen baseline was retrained.',
        'Objective: separate temporal-credit, cap/readout and gateway-opportunity limitations with minimal new mechanisms.',
        'Main runs: core, clean_no_accum, diagnostic gateway_probe; seeds42/43/44, continuous12.8k→51.2k→204.8k→819.2k.',
        'No final conclusions are drawn until all primary experiments finish. No dashboard/UI artifacts are modified.',
        '', '## Exact architecture changes','',ARCHITECTURE,'','## Frozen protocol and baselines','',
        'Start commit67ac23b. TinyShakespeare bytes256/context32, R32/N1024/E16/G4/K16/W2/C4/D64. AdamW lr3e-4, batch4, gradient clip1, one CPU thread per seed. Same fixed training starts/seed and validation windows as Phase A. Four cumulative training-byte milestones. Every100 optimizer steps plus milestones saves optimizer/RNG/cursor/EMA. Evaluation RNG is restored before continuing training.',
        '4096-byte primary targets are a prefix subset of16384-byte secondary targets; these are correlated checks, not independent datasets. Original v1/v2 classes, trainers, reports/results, dashboard and checkpoints remain frozen. Existing recurrence/accumulationOFF results are shown separately and retain their documented confounds.',
        '', '## Progress','', '| Model | Seed | 12.8k | 51.2k | 204.8k | 819.2k |','|---|---:|---|---|---|---|']
    for m in spec['models']:
        for seed in spec['seeds']:
            lines.append(f"| {m} | {seed} | "+' | '.join('done' if (m,seed,t) in present else 'pending' for t in spec['milestones'])+' |')
    if sanity.exists():
        s=payload['sanity'];c=s['characterization']
        lines+=['','## Sanity tests','',f"Python tests: {s['pytest_summary']}. Full-shape smoke: {s['smoke_summary']}.",
            f"Forced direct-OUTPUT prefix activation gradient L1: v2={c['v2_before']['prefix_embedding_gradient_L1']:.9g}; v2.1={c['v21_after']['prefix_embedding_gradient_L1']:.9g}. Exact controlled config/seed and per-token values are in sanity.json.",
            'Tests cover valid final-cycle source, clean one-event ON/OFF output+gradient equivalence, multi-event accumulator-only difference, per-walker probe hop+destination processing, eval state freeze, seeded save/load, exact optimizer/RNG-resume and completed-job skip. Existing v1/v2 tests remain unchanged.']
    lines+=['','## Main learning curves','', 'Loss is nats/byte. PPL is byte-level. Mean ± sample SD; only n=3 groups support main comparisons. Expected probe is not applicable.']
    for budget in spec['validation_bytes']:
        lines+=['',f'### Sampled validation: {budget} bytes','',
                '| Model | Training bytes | Loss | PPL | n |','|---|---:|---:|---:|---:|']
        for s in summaries:
            if s['validation_bytes']==budget and s['mode']=='sampled':
                lines.append(f"| {s['model']} | {s['training_tokens']:,} | {fmt(s['loss'])} | {fmt(s['ppl'])} | {s['loss']['n']} |")
    lines+=['','## Supplementary argmax / expected','', '| Model | Training bytes | Validation bytes | Mode | Loss | PPL | n |','|---|---:|---:|---|---:|---:|---:|']
    for s in summaries:
        if s['model'] in spec['models'] and s['mode']!='sampled':lines.append(f"| {s['model']} | {s['training_tokens']:,} | {s['validation_bytes']} | {s['mode']} | {fmt(s['loss'])} | {fmt(s['ppl'])} | {s['loss']['n']} |")
    slopes={}
    lines+=['','## Late scaling slope','', 'Loss delta / ln(819200/204800); more negative is faster improvement. Finite interval, not an asymptotic scaling law.',
            '| Model | Validation bytes | Loss slope | n |','|---|---:|---:|---:|']
    for m in BASELINES+tuple(spec['models']):
        for budget in spec['validation_bytes']:
            a={r['seed']:r['validation'][str(budget)]['sampled']['loss'] for r in all_rows if r['model']==m and r['training_tokens']==204800 and str(budget) in r['validation']}
            b={r['seed']:r['validation'][str(budget)]['sampled']['loss'] for r in all_rows if r['model']==m and r['training_tokens']==819200 and str(budget) in r['validation']}
            seeds=sorted(a.keys()&b.keys())
            if seeds:
                slopes[m,budget]=stats([(b[s]-a[s])/math.log(4) for s in seeds])
                lines.append(f"| {m} | {budget} | {fmt(slopes[m,budget])} | {len(seeds)} |")
    lines+=['','## Gateway utilization','',
            'New held-out routing metrics cover ALL4096 primary validation bytes with sampled routing. Training columns are EMA counts normalized into fractions. Direct OUTPUT is the fraction of tokens where ALL initial walkers output at cycle1; first-cycle OUTPUT additionally reports walker fraction in JSON. Trajectory length is actions/walker; cycles/token and unique Regions/token are separate.',
            '| Model | Training bytes | Train EMA gateway % | Held-out gateway % | Gateway tokens % | Hops/token | Direct OUTPUT % | Actions/walker | Unique Regions/token |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in spec['models']:
        for t in spec['milestones']:
            rows=[r for r in new if r['model']==m and r['training_tokens']==t]
            if not rows:continue
            metrics=[r['validation'][str(spec['validation_bytes'][0])]['sampled']['routing'] for r in rows]
            values=[stats([100*r['routing_training_ema']['gateway_action_fraction'] for r in rows])]
            for key,mult in [('gateway_action_fraction',100),('fraction_tokens_using_gateway',100),('mean_gateway_hops_per_token',1),('direct_OUTPUT_fraction_tokens',100),('mean_trajectory_length_per_walker',1),('unique_regions_visited_per_token',1)]:
                values.append(stats([r[key]*mult for r in metrics]))
            lines.append(f"| {m} (n={len(rows)}) | {t:,} | "+' | '.join(fmt(v) for v in values)+' |')
    lines+=['Frozen v2 reported adaptive training-EMA gateway fractions0.04–0.07% at final. Its saved held-out trace covers32 bytes, unlike the new4096-byte accounting; do not equate different sample sizes. Existing route traces/visit statistics and frozen source references remain in results.json.',
        '', '## Clean accumulation and gateway probe','',
        'Paired per-seed sampled loss deltas relative to core; positive is worse. Clean OFF changes only accumulator update rule; probe is diagnostic, not the main architecture.',
        '| Condition − core | Training bytes | Validation bytes | Loss delta | Per-seed deltas |','|---|---:|---:|---:|---|']
    paired={}
    for other in ('prg_v21_clean_no_accum','prg_v21_gateway_probe'):
        for t in spec['milestones']:
            for budget in spec['validation_bytes']:
                a={r['seed']:r['validation'][str(budget)]['sampled']['loss'] for r in new if r['model']=='prg_v21_core' and r['training_tokens']==t}
                b={r['seed']:r['validation'][str(budget)]['sampled']['loss'] for r in new if r['model']==other and r['training_tokens']==t}
                seeds=sorted(a.keys()&b.keys())
                if seeds:
                    deltas=[b[s]-a[s] for s in seeds];paired[other,t,budget]=stats(deltas)
                    lines.append(f"| {other} − core | {t:,} | {budget} | {fmt(stats(deltas))} | {dict(zip(seeds,deltas))} |")
    lines+=['','## Memory','',
            'Same inherited modeled inference-state convention for all new variants. Extra static: scalar gate4 bytes. Extra peak: gate4 + prefix-context64 + last-valid source128 + hop counters8 =204 bytes for the main shape. These numbers exclude allocator/autograd/operator workspaces; actual model export still stores FP32 edge latents. Existing Transformer position-embedding double-count audit is preserved in the v2 report; no baseline sizes are retuned.',
            '| Model | Static packed | Modeled peak | Actual export bytes | Routing stats training bytes | Optimizer tensor bytes |','|---|---:|---:|---:|---:|---:|']
    for m in spec['models']:
        rows=[r for r in new if r['model']==m and r['training_tokens']==max(spec['milestones'])]
        if rows:
            mem=rows[0]['memory'];lines.append(f"| {m} | {mem['total_static_packed_bytes']} | {mem['peak_theoretical_inference_bytes']} | {fmt(stats([r['serialized_model_bytes'] for r in rows]))} | {mem['training_only_routing_statistics_bytes']} | {mem['training_only_optimizer_tensor_bytes']} |")
    lines+=['Full memory component breakdown, checkpoint identifiers, stream hashes, training EMA/cumulative and every routing mode are in [results.json](results.json). Full optimizer/RNG/resume checkpoints remain under ignored runs/v2_1; no large model exports are committed.','',
            '## Interpretation and known limits','',
            'The core correction bundles temporal context, common tanh readout and cap semantics; its validation improvement cannot be attributed solely to one correction. Fixed global context pooling can itself bypass graph traversal, so better LM results alone do not prove useful gateways. Gateway probe simultaneously removes early OUTPUT shortcuts and changes compute/trajectory, preventing a pure causal estimate of gateway usefulness. Even improved recurrence would require separate low-bit/precision-matched controls before supporting precision replacement.',
            'Hard top-k/node selection remains nondifferentiable and training uses a straight-through route surrogate. Fixed context32/window resets, one corpus, three seeds and finite budget restrict generalization. Modeled memory is not measured physical memory. No hardware, packed execution or topology mutation is introduced. No hyperparameters, validation targets or seeds are tuned to emerging results.']
    if complete and 819200 in spec['milestones']:
        lookup={s['model']:s for s in summaries if s['training_tokens']==819200 and s['validation_bytes']==4096 and s['mode']=='sampled'}
        c=lookup['prg_v21_core'];v1=lookup['prg_v1'];v2=lookup['prg_v2_adaptive'];probe=lookup['prg_v21_gateway_probe']
        final_rows=[r for r in new if r['model']=='prg_v21_core' and r['training_tokens']==819200]
        grads=[r['temporal_credit']['prefix_embedding_gradient_L1'] for r in final_rows]
        gateway=stats([100*r['validation']['4096']['sampled']['routing']['gateway_action_fraction'] for r in final_rows])
        train_gateway=stats([100*r['routing_training_ema']['gateway_action_fraction'] for r in final_rows])
        probe_delta=paired['prg_v21_gateway_probe',819200,4096]['mean']
        noacc_delta=paired['prg_v21_clean_no_accum',819200,4096]['mean']
        core_slope=slopes['prg_v21_core',4096]['mean'];v1_slope=slopes['prg_v1',4096]['mean'];v2_slope=slopes['prg_v2_adaptive',4096]['mean']
        lines+=['','## Final research questions','',
            f"1. Forced direct-OUTPUT prefix gradients survive: final core seeds={grads}. Initial before/after values are above; full ST gradient is a different diagnostic.",
            f"2. Core correction bundle final loss={fmt(c['loss'])}, PPL={fmt(c['ppl'])}; loss delta vs v1={c['loss']['mean']-v1['loss']['mean']:.5f}, vs adaptive v2={c['loss']['mean']-v2['loss']['mean']:.5f}. Negative means improvement; bundle cannot isolate the temporal fix alone.",
            f"3. Late loss/log-byte slope: core={core_slope:.5f}, v1={v1_slope:.5f}, v2={v2_slope:.5f}. Core late improvement is {'faster' if core_slope<min(v1_slope,v2_slope) else 'not faster than both baselines'}.",
            f"4. The finite-budget plateau is {'partly alleviated' if core_slope<min(v1_slope,v2_slope) and c['loss']['mean']<v1['loss']['mean'] else 'not clearly alleviated'} by these measurements. No claim about asymptotic capacity follows.",
            '5. Unprocessed-destination cap reads are removed by construction and controlled tests in hard and expected core engines. Valid signal cancellation remains possible; this is not a universal guarantee of nonzero logits.',
            f"6. Natural gateway action %: training EMA={fmt(train_gateway)}, held-out={fmt(gateway)}. Training mean is {'above' if train_gateway['mean']>.07 else 'not above'} the old0.04–0.07% range; compare all seeds, not different sample scopes. No significance test or matched old full-validation trajectory sample is claimed.",
            f"7. Diagnostic probe loss minus core={fmt(paired['prg_v21_gateway_probe',819200,4096])}; PPL={fmt(probe['ppl'])}. Constraint guarantees opportunity; actual utilization and prefix diagnostics are recorded for every seed.",
            f"8. The probe is {'better and offers limited evidence of gateway-opportunity potential' if probe_delta<0 else 'worse/equal and provides no measured gateway-opportunity benefit here'}. Blocked early OUTPUT and extra processing remain confounds; neither contrast proves traversal causality.",
            f"9. Clean no-accum minus core loss={fmt(paired['prg_v21_clean_no_accum',819200,4096])}; measured contrast favors {'accumulation ON' if noacc_delta>0 else 'replacing the accumulator (OFF)'}. Paired-seed magnitudes above limit confidence; this is cleaner than the frozen v2 OFF readout comparison.",
            '10. These controls test contextual/readout and gateway mechanisms, not replacement of numerical precision. The original precision-for-recurrence hypothesis remains unestablished. Sparse-graph usefulness must be supported by consistent probe/natural traversal benefits rather than a context-only improvement.',
            f"11. {'There is limited reason for a separately controlled mutation experiment because natural use rises and the probe improves' if probe_delta<0 and train_gateway['mean']>.07 else 'These controls do not yet justify another main mutation sweep'}. Separate contextual-bypass and OUTPUT-timing controls before claiming topology usefulness.",
            '12. Preserve negative results. Prioritize a factorial correction control and sparse contextual routing if the global context path dominates; reconsider this architecture before expensive scaling when gateway utility remains absent.']
        lines[4:4]=[f"Final core sampled loss/PPL: {fmt(c['loss'])} / {fmt(c['ppl'])}.",
                    f"Core−v1 loss={c['loss']['mean']-v1['loss']['mean']:.5f}; probe−core={paired['prg_v21_gateway_probe',819200,4096]['mean']:.5f}; no-accum−core={paired['prg_v21_clean_no_accum',819200,4096]['mean']:.5f}."]
    else:lines+=['','Final interpretation and numbered questions: pending until ALL36 new milestone results complete.']
    if complete and plots:
        for budget in spec['validation_bytes']:
            for metric in ('loss','ppl'):
                for group,names in [('core',('transformer_core','transformer_total','prg_v1','prg_v21_core')),
                                    ('controls',tuple(spec['models']))]:
                    series=[]
                    for m in names:
                        points=[(s['training_tokens'],s[metric]['mean'],s[metric]['std']) for s in summaries if s['model']==m and s['validation_bytes']==budget and s['mode']=='sampled']
                        series.append((m,points))
                    path=out/f'{group}_{metric}_{budget}_log.svg'
                    svg_plot(path,series,f'v2.1 {group}, validation={budget}',metric,True)
                    lines+=['',f'![{group} {metric} {budget}]({path.name})']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return payload

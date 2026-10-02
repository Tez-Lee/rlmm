# PRG-LM v2.1 — Core Correction Study

## Executive summary

Complete: False. New milestones complete: 0/36. No frozen baseline was retrained.
Objective: separate temporal-credit, cap/readout and gateway-opportunity limitations with minimal new mechanisms.
Main runs: core, clean_no_accum, diagnostic gateway_probe; seeds42/43/44, continuous12.8k→51.2k→204.8k→819.2k.
No final conclusions are drawn until all primary experiments finish. No dashboard/UI artifacts are modified.

## Exact architecture changes

PRGLMv21 inherits frozen v1 propagation and shared RouterNet; no topology mutation or attention. Before each token, persistent node state is pooled into K fixed node-index-mod-K channels by sum/sqrt(R*N/K), then tanh. This prefix-only context uses the existing readout projection weight without its bias and a learned scalar sigmoid gate initialized at0 (gate0.5). It is added once to normalized per-walker local readout before existing LayerNorm/vocabulary head. It is a real contextual input, not a gradient-only loss or injected gradient.

Both accumulation settings read tanh of the same current-token accumulator, with the same contextual input, router, state update, dimensions and memory convention. OFF replaces previous accumulator with incoming delta; ON adds it with the configured decay. Router features can consequently differ because accumulator contents differ. This is an intended downstream consequence, not a changed RouterNet. The added tanh is explicit and shared; the bundle comparison with v2 does not isolate the temporal fix alone.

At the hard cycle cap, remaining walkers contribute their LAST PROCESSED source accumulator message, saved before any transition. An unprocessed gateway destination is never gathered for readout. Expected routing terminates surviving mass from processed current sources before final transitions. Valid messages can cancel to zero; the eliminated artifact is reading an uninitialized destination, not a guarantee that every valid signal is nonzero.

The diagnostic probe requires >=1 GATEWAY action per walker before OUTPUT. If needed, LOCAL/RETURN are masked when only required hops plus one destination-processing cycle remain. The cycle bound must exceed minimum hops. Destinations are processed before the bound; no topology changes or auxiliary loss occur. Core has no such constraint. Expected probe routing is deliberately unavailable because its merged soft branches cannot faithfully track per-walker hop requirements. Expected core/OFF remain supplementary region-mass surrogates, not exact expectations.

## Frozen protocol and baselines

Start commit67ac23b. TinyShakespeare bytes256/context32, R32/N1024/E16/G4/K16/W2/C4/D64. AdamW lr3e-4, batch4, gradient clip1, one CPU thread per seed. Same fixed training starts/seed and validation windows as Phase A. Four cumulative training-byte milestones. Every100 optimizer steps plus milestones saves optimizer/RNG/cursor/EMA. Evaluation RNG is restored before continuing training.
4096-byte primary targets are a prefix subset of16384-byte secondary targets; these are correlated checks, not independent datasets. Original v1/v2 classes, trainers, reports/results, dashboard and checkpoints remain frozen. Existing recurrence/accumulationOFF results are shown separately and retain their documented confounds.

## Progress

| Model | Seed | 12.8k | 51.2k | 204.8k | 819.2k |
|---|---:|---|---|---|---|
| prg_v21_core | 42 | pending | pending | pending | pending |
| prg_v21_core | 43 | pending | pending | pending | pending |
| prg_v21_core | 44 | pending | pending | pending | pending |
| prg_v21_clean_no_accum | 42 | pending | pending | pending | pending |
| prg_v21_clean_no_accum | 43 | pending | pending | pending | pending |
| prg_v21_clean_no_accum | 44 | pending | pending | pending | pending |
| prg_v21_gateway_probe | 42 | pending | pending | pending | pending |
| prg_v21_gateway_probe | 43 | pending | pending | pending | pending |
| prg_v21_gateway_probe | 44 | pending | pending | pending | pending |

## Sanity tests

Python tests: 42 passed; existing33 + new9 tests. Full-shape smoke: All3 variants, seed42, full main shape,128→256 training bytes; six finite milestones. Frozen baselines not trained..
Forced direct-OUTPUT prefix activation gradient L1: v2=0; v2.1=0.608478107. Exact controlled config/seed and per-token values are in sanity.json.
Tests cover valid final-cycle source, clean one-event ON/OFF output+gradient equivalence, multi-event accumulator-only difference, per-walker probe hop+destination processing, eval state freeze, seeded save/load, exact optimizer/RNG-resume and completed-job skip. Existing v1/v2 tests remain unchanged.

## Main learning curves

Loss is nats/byte. PPL is byte-level. Mean ± sample SD; only n=3 groups support main comparisons. Expected probe is not applicable.

### Sampled validation: 4096 bytes

| Model | Training bytes | Loss | PPL | n |
|---|---:|---:|---:|---:|
| transformer_core | 12,800 | 4.79351 ± 0.06729 | 120.90696 ± 8.16229 | 3 |
| transformer_core | 51,200 | 4.02171 ± 0.06192 | 55.86751 ± 3.45336 | 3 |
| transformer_core | 204,800 | 2.82540 ± 0.05786 | 16.88664 ± 0.98020 | 3 |
| transformer_core | 819,200 | 2.39069 ± 0.00742 | 10.92120 ± 0.08093 | 3 |
| transformer_total | 12,800 | 4.04549 ± 0.07030 | 57.23185 ± 3.93969 | 3 |
| transformer_total | 51,200 | 3.19058 ± 0.03082 | 24.31008 ± 0.74253 | 3 |
| transformer_total | 204,800 | 2.54130 ± 0.00884 | 12.69651 ± 0.11196 | 3 |
| transformer_total | 819,200 | 2.17929 ± 0.01017 | 8.84030 ± 0.08968 | 3 |
| prg_v1 | 12,800 | 5.39993 ± 0.23489 | 225.39509 ± 50.91763 | 3 |
| prg_v1 | 51,200 | 3.80771 ± 0.03034 | 45.06106 ± 1.37197 | 3 |
| prg_v1 | 204,800 | 2.79025 ± 0.01602 | 16.28640 ± 0.25976 | 3 |
| prg_v1 | 819,200 | 2.57549 ± 0.00942 | 13.13818 ± 0.12353 | 3 |
| prg_v2_adaptive | 12,800 | 5.40227 ± 0.23841 | 226.04972 ± 51.90015 | 3 |
| prg_v2_adaptive | 51,200 | 3.80775 ± 0.03619 | 45.06862 ± 1.64745 | 3 |
| prg_v2_adaptive | 204,800 | 2.79282 ± 0.01017 | 16.32749 ± 0.16585 | 3 |
| prg_v2_adaptive | 819,200 | 2.59190 ± 0.00969 | 13.35557 ± 0.12965 | 3 |
| prg_v2_uniform | 12,800 | 5.40205 ± 0.23688 | 225.93600 ± 51.36361 | 3 |
| prg_v2_uniform | 51,200 | 3.80768 ± 0.04523 | 45.07660 ± 2.06260 | 3 |
| prg_v2_uniform | 204,800 | 2.79402 ± 0.00849 | 16.34707 ± 0.13889 | 3 |
| prg_v2_uniform | 819,200 | 2.58145 ± 0.00750 | 13.21648 ± 0.09932 | 3 |
| prg_v2_no_recurrence | 12,800 | 5.33881 ± 0.20492 | 211.09590 ± 40.93544 | 3 |
| prg_v2_no_recurrence | 51,200 | 3.70435 ± 0.04011 | 40.64564 ± 1.63194 | 3 |
| prg_v2_no_recurrence | 204,800 | 2.78110 ± 0.01178 | 16.13754 ± 0.18990 | 3 |
| prg_v2_no_recurrence | 819,200 | 2.58533 ± 0.00843 | 13.26804 ± 0.11189 | 3 |
| prg_v2_no_accumulation | 12,800 | 5.45347 ± 0.22415 | 237.42000 ± 51.38552 | 3 |
| prg_v2_no_accumulation | 51,200 | 3.85414 ± 0.04218 | 47.21620 ± 1.99193 | 3 |
| prg_v2_no_accumulation | 204,800 | 2.82134 ± 0.01598 | 16.80082 ± 0.26931 | 3 |
| prg_v2_no_accumulation | 819,200 | 2.60812 ± 0.01233 | 13.57426 ± 0.16775 | 3 |

### Sampled validation: 16384 bytes

| Model | Training bytes | Loss | PPL | n |
|---|---:|---:|---:|---:|
| transformer_core | 12,800 | 4.81260 ± 0.06736 | 123.23876 ± 8.36196 | 3 |
| transformer_core | 51,200 | 4.04653 ± 0.05779 | 57.26233 ± 3.30097 | 3 |
| transformer_core | 204,800 | 2.84318 ± 0.05947 | 17.19056 ± 1.02946 | 3 |
| transformer_core | 819,200 | 2.36426 ± 0.00385 | 10.63623 ± 0.04104 | 3 |
| transformer_total | 12,800 | 4.07344 ± 0.07198 | 58.85919 ± 4.15912 | 3 |
| transformer_total | 51,200 | 3.23590 ± 0.03396 | 25.43905 ± 0.85864 | 3 |
| transformer_total | 204,800 | 2.53538 ± 0.00862 | 12.62153 ± 0.10905 | 3 |
| transformer_total | 819,200 | 2.16244 ± 0.01079 | 8.69265 ± 0.09377 | 3 |
| prg_v1 | 12,800 | 5.40890 ± 0.22537 | 227.10524 ± 49.28073 | 3 |
| prg_v1 | 51,200 | 3.84125 ± 0.03106 | 46.59854 ± 1.44829 | 3 |
| prg_v1 | 204,800 | 2.80987 ± 0.01238 | 16.60859 ± 0.20522 | 3 |
| prg_v1 | 819,200 | 2.57186 ± 0.00607 | 13.09031 ± 0.07950 | 3 |
| prg_v2_adaptive | 12,800 | 5.40959 ± 0.22497 | 227.24898 ± 49.23843 | 3 |
| prg_v2_adaptive | 51,200 | 3.83511 ± 0.03054 | 46.31293 ± 1.41755 | 3 |
| prg_v2_adaptive | 204,800 | 2.81739 ± 0.01078 | 16.73373 ± 0.18004 | 3 |
| prg_v2_adaptive | 819,200 | 2.57987 ± 0.00817 | 13.19573 ± 0.10764 | 3 |
| prg_v2_uniform | 12,800 | 5.41019 ± 0.22560 | 227.40778 ± 49.42139 | 3 |
| prg_v2_uniform | 51,200 | 3.83827 ± 0.03865 | 46.46827 ± 1.81317 | 3 |
| prg_v2_uniform | 204,800 | 2.81640 ± 0.00864 | 16.71697 ± 0.14439 | 3 |
| prg_v2_uniform | 819,200 | 2.57074 ± 0.00802 | 13.07578 ± 0.10488 | 3 |
| prg_v2_no_recurrence | 12,800 | 5.33610 ± 0.20060 | 210.41749 ± 40.21695 | 3 |
| prg_v2_no_recurrence | 51,200 | 3.73645 ± 0.02981 | 41.96110 ± 1.25234 | 3 |
| prg_v2_no_recurrence | 204,800 | 2.80767 ± 0.01699 | 16.57293 ± 0.28289 | 3 |
| prg_v2_no_recurrence | 819,200 | 2.57538 ± 0.01272 | 13.13707 ± 0.16775 | 3 |
| prg_v2_no_accumulation | 12,800 | 5.46064 ± 0.22565 | 239.19451 ± 52.35482 | 3 |
| prg_v2_no_accumulation | 51,200 | 3.88098 ± 0.03373 | 48.48996 ± 1.63911 | 3 |
| prg_v2_no_accumulation | 204,800 | 2.84727 ± 0.02078 | 17.24311 ± 0.35975 | 3 |
| prg_v2_no_accumulation | 819,200 | 2.59548 ± 0.00967 | 13.40350 ± 0.12932 | 3 |

## Supplementary argmax / expected

| Model | Training bytes | Validation bytes | Mode | Loss | PPL | n |
|---|---:|---:|---|---:|---:|---:|

## Late scaling slope

Loss delta / ln(819200/204800); more negative is faster improvement. Finite interval, not an asymptotic scaling law.
| Model | Validation bytes | Loss slope | n |
|---|---:|---:|---:|
| transformer_core | 4096 | -0.31358 ± 0.03804 | 3 |
| transformer_core | 16384 | -0.34546 ± 0.04335 | 3 |
| transformer_total | 4096 | -0.26114 ± 0.00153 | 3 |
| transformer_total | 16384 | -0.26902 ± 0.00516 | 3 |
| prg_v1 | 4096 | -0.15491 ± 0.00527 | 3 |
| prg_v1 | 16384 | -0.17169 ± 0.00803 | 3 |
| prg_v2_adaptive | 4096 | -0.14493 ± 0.00803 | 3 |
| prg_v2_adaptive | 16384 | -0.17133 ± 0.00194 | 3 |
| prg_v2_uniform | 4096 | -0.15334 ± 0.01139 | 3 |
| prg_v2_uniform | 16384 | -0.17721 ± 0.01201 | 3 |
| prg_v2_no_recurrence | 4096 | -0.14122 ± 0.01457 | 3 |
| prg_v2_no_recurrence | 16384 | -0.16756 ± 0.01924 | 3 |
| prg_v2_no_accumulation | 4096 | -0.15380 ± 0.01971 | 3 |
| prg_v2_no_accumulation | 16384 | -0.18162 ± 0.02196 | 3 |

## Gateway utilization

New held-out routing metrics cover ALL4096 primary validation bytes with sampled routing. Training columns are EMA counts normalized into fractions. Direct OUTPUT is the fraction of tokens where ALL initial walkers output at cycle1; first-cycle OUTPUT additionally reports walker fraction in JSON. Trajectory length is actions/walker; cycles/token and unique Regions/token are separate.
| Model | Training bytes | Train EMA gateway % | Held-out gateway % | Gateway tokens % | Hops/token | Direct OUTPUT % | Actions/walker | Unique Regions/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
Frozen v2 reported adaptive training-EMA gateway fractions0.04–0.07% at final. Its saved held-out trace covers32 bytes, unlike the new4096-byte accounting; do not equate different sample sizes. Existing route traces/visit statistics and frozen source references remain in results.json.

## Clean accumulation and gateway probe

Paired per-seed sampled loss deltas relative to core; positive is worse. Clean OFF changes only accumulator update rule; probe is diagnostic, not the main architecture.
| Condition − core | Training bytes | Validation bytes | Loss delta | Per-seed deltas |
|---|---:|---:|---:|---|

## Memory

Same inherited modeled inference-state convention for all new variants. Extra static: scalar gate4 bytes. Extra peak: gate4 + prefix-context64 + last-valid source128 + hop counters8 =204 bytes for the main shape. These numbers exclude allocator/autograd/operator workspaces; actual model export still stores FP32 edge latents. Existing Transformer position-embedding double-count audit is preserved in the v2 report; no baseline sizes are retuned.
| Model | Static packed | Modeled peak | Actual export bytes | Routing stats training bytes | Optimizer tensor bytes |
|---|---:|---:|---:|---:|---:|
Full memory component breakdown, checkpoint identifiers, stream hashes, training EMA/cumulative and every routing mode are in [results.json](results.json). Full optimizer/RNG/resume checkpoints remain under ignored runs/v2_1; no large model exports are committed.

## Interpretation and known limits

The core correction bundles temporal context, common tanh readout and cap semantics; its validation improvement cannot be attributed solely to one correction. Fixed global context pooling can itself bypass graph traversal, so better LM results alone do not prove useful gateways. Gateway probe simultaneously removes early OUTPUT shortcuts and changes compute/trajectory, preventing a pure causal estimate of gateway usefulness. Even improved recurrence would require separate low-bit/precision-matched controls before supporting precision replacement.
Hard top-k/node selection remains nondifferentiable and training uses a straight-through route surrogate. Fixed context32/window resets, one corpus, three seeds and finite budget restrict generalization. Modeled memory is not measured physical memory. No hardware, packed execution or topology mutation is introduced. No hyperparameters, validation targets or seeds are tuned to emerging results.

Final interpretation and numbered questions: pending until ALL36 new milestone results complete.

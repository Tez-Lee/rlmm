# PRG-LM v2.1 — Core Correction Study

## Executive summary

Complete: True. New milestones complete: 6/6. No frozen baseline was retrained.
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
| prg_v21_core | 42 | done | done |
| prg_v21_clean_no_accum | 42 | done | done |
| prg_v21_gateway_probe | 42 | done | done |

## Main learning curves

Loss is nats/byte. PPL is byte-level. Mean ± sample SD; only n=3 groups support main comparisons. Expected probe is not applicable.

### Sampled validation: 128 bytes

| Model | Training bytes | Loss | PPL | n |
|---|---:|---:|---:|---:|
| prg_v21_core | 128 | 6.14350 (n=1) | 465.68120 (n=1) | 1 |
| prg_v21_core | 256 | 6.15392 (n=1) | 470.55732 (n=1) | 1 |
| prg_v21_clean_no_accum | 128 | 6.09421 (n=1) | 443.28462 (n=1) | 1 |
| prg_v21_clean_no_accum | 256 | 6.14912 (n=1) | 468.30414 (n=1) | 1 |
| prg_v21_gateway_probe | 128 | 6.23379 (n=1) | 509.68592 (n=1) | 1 |
| prg_v21_gateway_probe | 256 | 6.20443 (n=1) | 494.93894 (n=1) | 1 |

### Sampled validation: 256 bytes

| Model | Training bytes | Loss | PPL | n |
|---|---:|---:|---:|---:|
| prg_v21_core | 128 | 6.12588 (n=1) | 457.54729 (n=1) | 1 |
| prg_v21_core | 256 | 6.08988 (n=1) | 441.36718 (n=1) | 1 |
| prg_v21_clean_no_accum | 128 | 6.10673 (n=1) | 448.87065 (n=1) | 1 |
| prg_v21_clean_no_accum | 256 | 6.09335 (n=1) | 442.90305 (n=1) | 1 |
| prg_v21_gateway_probe | 128 | 6.14114 (n=1) | 464.58089 (n=1) | 1 |
| prg_v21_gateway_probe | 256 | 6.11254 (n=1) | 451.48371 (n=1) | 1 |

## Supplementary argmax / expected

| Model | Training bytes | Validation bytes | Mode | Loss | PPL | n |
|---|---:|---:|---|---:|---:|---:|
| prg_v21_core | 128 | 128 | argmax | 6.12116 (n=1) | 455.39179 (n=1) | 1 |
| prg_v21_core | 128 | 128 | expected | 6.18562 (n=1) | 485.71161 (n=1) | 1 |
| prg_v21_core | 128 | 256 | argmax | 6.06967 (n=1) | 432.53615 (n=1) | 1 |
| prg_v21_core | 128 | 256 | expected | 6.11749 (n=1) | 453.72520 (n=1) | 1 |
| prg_v21_core | 256 | 128 | argmax | 6.11539 (n=1) | 452.77316 (n=1) | 1 |
| prg_v21_core | 256 | 128 | expected | 6.18058 (n=1) | 483.27247 (n=1) | 1 |
| prg_v21_core | 256 | 256 | argmax | 6.06324 (n=1) | 429.76380 (n=1) | 1 |
| prg_v21_core | 256 | 256 | expected | 6.11400 (n=1) | 452.14361 (n=1) | 1 |
| prg_v21_clean_no_accum | 128 | 128 | argmax | 6.15956 (n=1) | 473.22126 (n=1) | 1 |
| prg_v21_clean_no_accum | 128 | 128 | expected | 6.16920 (n=1) | 477.80368 (n=1) | 1 |
| prg_v21_clean_no_accum | 128 | 256 | argmax | 6.08379 (n=1) | 438.68651 (n=1) | 1 |
| prg_v21_clean_no_accum | 128 | 256 | expected | 6.11014 (n=1) | 450.40148 (n=1) | 1 |
| prg_v21_clean_no_accum | 256 | 128 | argmax | 6.15169 (n=1) | 469.50952 (n=1) | 1 |
| prg_v21_clean_no_accum | 256 | 128 | expected | 6.16356 (n=1) | 475.11712 (n=1) | 1 |
| prg_v21_clean_no_accum | 256 | 256 | argmax | 6.07655 (n=1) | 435.52591 (n=1) | 1 |
| prg_v21_clean_no_accum | 256 | 256 | expected | 6.10490 (n=1) | 448.04735 (n=1) | 1 |
| prg_v21_gateway_probe | 128 | 128 | argmax | 6.05757 (n=1) | 427.33554 (n=1) | 1 |
| prg_v21_gateway_probe | 128 | 256 | argmax | 6.01259 (n=1) | 408.53868 (n=1) | 1 |
| prg_v21_gateway_probe | 256 | 128 | argmax | 6.04458 (n=1) | 421.81854 (n=1) | 1 |
| prg_v21_gateway_probe | 256 | 256 | argmax | 6.01309 (n=1) | 408.74250 (n=1) | 1 |

## Late scaling slope

Loss delta / ln(819200/204800); more negative is faster improvement. Finite interval, not an asymptotic scaling law.
| Model | Validation bytes | Loss slope | n |
|---|---:|---:|---:|

## Gateway utilization

New held-out routing metrics cover ALL4096 primary validation bytes with sampled routing. Training columns are EMA counts normalized into fractions. Direct OUTPUT is the fraction of tokens where ALL initial walkers output at cycle1; first-cycle OUTPUT additionally reports walker fraction in JSON. Trajectory length is actions/walker; cycles/token and unique Regions/token are separate.
| Model | Training bytes | Train EMA gateway % | Held-out gateway % | Gateway tokens % | Hops/token | Direct OUTPUT % | Actions/walker | Unique Regions/token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prg_v21_core (n=1) | 128 | 58.89029 (n=1) | 57.88820 (n=1) | 93.75000 (n=1) | 3.64062 (n=1) | 3.90625 (n=1) | 3.14453 (n=1) | 4.67188 (n=1) |
| prg_v21_core (n=1) | 256 | 58.53697 (n=1) | 58.02469 (n=1) | 93.75000 (n=1) | 3.67188 (n=1) | 3.90625 (n=1) | 3.16406 (n=1) | 4.67969 (n=1) |
| prg_v21_clean_no_accum (n=1) | 128 | 58.89029 (n=1) | 57.90785 (n=1) | 93.75000 (n=1) | 3.63281 (n=1) | 3.90625 (n=1) | 3.13672 (n=1) | 4.67188 (n=1) |
| prg_v21_clean_no_accum (n=1) | 256 | 58.47325 (n=1) | 57.99257 (n=1) | 93.75000 (n=1) | 3.65625 (n=1) | 3.90625 (n=1) | 3.15234 (n=1) | 4.68750 (n=1) |
| prg_v21_gateway_probe (n=1) | 128 | 63.02703 (n=1) | 63.39869 (n=1) | 100.00000 (n=1) | 4.54688 (n=1) | 0.00000 (n=1) | 3.58594 (n=1) | 5.32812 (n=1) |
| prg_v21_gateway_probe (n=1) | 256 | 62.83860 (n=1) | 63.28976 (n=1) | 100.00000 (n=1) | 4.53906 (n=1) | 0.00000 (n=1) | 3.58594 (n=1) | 5.32812 (n=1) |
Frozen v2 reported adaptive training-EMA gateway fractions0.04–0.07% at final. Its saved held-out trace covers32 bytes, unlike the new4096-byte accounting; do not equate different sample sizes. Existing route traces/visit statistics and frozen source references remain in results.json.

## Clean accumulation and gateway probe

Paired per-seed sampled loss deltas relative to core; positive is worse. Clean OFF changes only accumulator update rule; probe is diagnostic, not the main architecture.
| Condition − core | Training bytes | Validation bytes | Loss delta | Per-seed deltas |
|---|---:|---:|---:|---|
| prg_v21_clean_no_accum − core | 128 | 128 | -0.04929 (n=1) | {42: -0.04928922653198242} |
| prg_v21_clean_no_accum − core | 128 | 256 | -0.01915 (n=1) | {42: -0.019145488739013672} |
| prg_v21_clean_no_accum − core | 256 | 128 | -0.00480 (n=1) | {42: -0.004799842834472656} |
| prg_v21_clean_no_accum − core | 256 | 256 | 0.00347 (n=1) | {42: 0.0034737586975097656} |
| prg_v21_gateway_probe − core | 128 | 128 | 0.09029 (n=1) | {42: 0.09029340744018555} |
| prg_v21_gateway_probe − core | 128 | 256 | 0.01526 (n=1) | {42: 0.015255451202392578} |
| prg_v21_gateway_probe − core | 256 | 128 | 0.05052 (n=1) | {42: 0.050516605377197266} |
| prg_v21_gateway_probe − core | 256 | 256 | 0.02266 (n=1) | {42: 0.02266216278076172} |

## Memory

Same inherited modeled inference-state convention for all new variants. Extra static: scalar gate4 bytes. Extra peak: gate4 + prefix-context64 + last-valid source128 + hop counters8 =204 bytes for the main shape. These numbers exclude allocator/autograd/operator workspaces; actual model export still stores FP32 edge latents. Existing Transformer position-embedding double-count audit is preserved in the v2 report; no baseline sizes are retuned.
| Model | Static packed | Modeled peak | Actual export bytes | Routing stats training bytes | Optimizer tensor bytes |
|---|---:|---:|---:|---:|---:|
| prg_v21_core | 220864 | 483602 | 2195259.00000 (n=1) | 240 | 4373764 |
| prg_v21_clean_no_accum | 220864 | 483602 | 2195259.00000 (n=1) | 240 | 4373764 |
| prg_v21_gateway_probe | 220864 | 483602 | 2195259.00000 (n=1) | 240 | 4373764 |
Full memory component breakdown, checkpoint identifiers, stream hashes, training EMA/cumulative and every routing mode are in [results.json](results.json). Full optimizer/RNG/resume checkpoints remain under ignored runs/v2_1; no large model exports are committed.

## Interpretation and known limits

The core correction bundles temporal context, common tanh readout and cap semantics; its validation improvement cannot be attributed solely to one correction. Fixed global context pooling can itself bypass graph traversal, so better LM results alone do not prove useful gateways. Gateway probe simultaneously removes early OUTPUT shortcuts and changes compute/trajectory, preventing a pure causal estimate of gateway usefulness. Even improved recurrence would require separate low-bit/precision-matched controls before supporting precision replacement.
Hard top-k/node selection remains nondifferentiable and training uses a straight-through route surrogate. Fixed context32/window resets, one corpus, three seeds and finite budget restrict generalization. Modeled memory is not measured physical memory. No hardware, packed execution or topology mutation is introduced. No hyperparameters, validation targets or seeds are tuned to emerging results.

Final interpretation and numbered questions: pending until ALL36 new milestone results complete.

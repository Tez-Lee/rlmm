# PRG-LM v3.0.1 — Extended-Training Crossover Verification

Complete: False; completed extension milestones: 0/132; failed jobs: 0; resource-limited jobs: 0; latest completed training bytes: 819,200.

## Frozen hypothesis and protocol

Does v3 same-edge repetition advantage survive substantially longer training, or did larger numerical-weight baselines converge more slowly? Duration extension ONLY: all v3 model/topology/hash/state/representation/controller/max_repeat8/optimizer/lr3e-4/clipping/tokenizer/context32/split/validation/seeds/objective remain unchanged. No tuning, new architecture, old-run retraining or dashboard work.
33 continuations: Float S0–S4; ternary S2–S4; Repeat S2–S4; seeds42/43/44. Source model, both optimizers, RNG and cursor at819200 are restored. New NumPy sampling stream has a verified byte-for-byte819200 prefix match, then continues the same sampler. Evaluation uses unchanged4096/16384 windows, batch8, sampled seed+batch index. Evaluation RNG is restored before training.
Preregistered cumulative milestones:1638400,3276800,6553600,13107200. Final13.1072M is the primary decision; last6.5536M and13.1072M directions are checked. No early stopping or favorable checkpoint selection. Original819200 rows are copied as historical baseline, not re-evaluated.
The original v3 has five4× edge scales and compact16-channel state. Same-edge S2/S3/S4 and SAME measured packed-memory pairs Float S0↔Repeat S2, S1↔S3, S2↔S4 are fixed. No interpolation/extrapolation. n=3 mean±sample SD and every paired seed delta are reported, without significance claims.
CPU, two one-thread workers. Limits fixed before training: six hours per job per milestone stage, free disk at least2GB before each job/checkpoint. Resource-limited jobs retain completed milestones/checkpoints; they do not supply final conclusions. Full checkpoint recreation is unnecessary: all33 sources are present.
Packed/static/dynamic accounting reuses immutable v3 metrics. Exports pack ternary2-bit edges, PyTorch execution decodesFP32. Controller candidates include inactive lanes. Modeled active additions are not actual FLOPs or energy. Process RSS/optimizer tensors/prototype throughput remain separate.

## Progress

| Family | Scale | Seed42 | Seed43 | Seed44 | Final |
|---|---|---:|---:|---:|---|
| float | S0 | 819200 | 819200 | 819200 | False |
| float | S1 | 819200 | 819200 | 819200 | False |
| float | S2 | 819200 | 819200 | 819200 | False |
| float | S3 | 819200 | 819200 | 819200 | False |
| float | S4 | 819200 | 819200 | 819200 | False |
| ternary | S2 | 819200 | 819200 | 819200 | False |
| ternary | S3 | 819200 | 819200 | 819200 | False |
| ternary | S4 | 819200 | 819200 | 819200 | False |
| repeat | S2 | 819200 | 819200 | 819200 | False |
| repeat | S3 | 819200 | 819200 | 819200 | False |
| repeat | S4 | 819200 | 819200 | 819200 | False |

## Sanity checks

56 tests passed, including all previous tests. Tiny continuation matches an uninterrupted immutable v3 trainer bit-for-bit for weights, both optimizer states, RNG and history. Interrupted extension and milestone-stage barriers are also bit-exact. All33 original819200 checkpoints are present. Actual Float S0/seed42 checkpoint continuation smoke reached832000 bytes (12800 additional), preserving original files. Smoke is separate from primary results and did not select/tune settings.

## All validation results

| Family | Scale | Training bytes | Validation bytes | Loss | Byte PPL | n |
|---|---|---:|---:|---:|---:|---:|
| float | S0 | 819200 | 4096 | 3.08353 ± 0.01534 | 21.83711 ± 0.33618 | 3 |
| float | S1 | 819200 | 4096 | 3.08262 ± 0.01217 | 21.81651 ± 0.26452 | 3 |
| float | S2 | 819200 | 4096 | 3.17757 ± 0.06327 | 24.02069 ± 1.54089 | 3 |
| float | S3 | 819200 | 4096 | 3.29907 ± 0.00160 | 27.08733 ± 0.04345 | 3 |
| float | S4 | 819200 | 4096 | 3.29974 ± 0.00306 | 27.10579 ± 0.08279 | 3 |
| ternary | S2 | 819200 | 4096 | 3.22646 ± 0.06150 | 25.22253 ± 1.57713 | 3 |
| ternary | S3 | 819200 | 4096 | 3.30145 ± 0.00025 | 27.15205 ± 0.00677 | 3 |
| ternary | S4 | 819200 | 4096 | 3.30185 ± 0.00300 | 27.16291 ± 0.08138 | 3 |
| repeat | S2 | 819200 | 4096 | 3.18661 ± 0.02769 | 24.21254 ± 0.67547 | 3 |
| repeat | S3 | 819200 | 4096 | 3.18766 ± 0.02304 | 24.23595 ± 0.56213 | 3 |
| repeat | S4 | 819200 | 4096 | 3.19244 ± 0.02199 | 24.35171 ± 0.53825 | 3 |
| float | S0 | 819200 | 16384 | 3.10628 ± 0.01940 | 22.34068 ± 0.43583 | 3 |
| float | S1 | 819200 | 16384 | 3.09970 ± 0.00876 | 22.19181 ± 0.19399 | 3 |
| float | S2 | 819200 | 16384 | 3.22349 ± 0.06961 | 25.15666 ± 1.78015 | 3 |
| float | S3 | 819200 | 16384 | 3.35217 ± 0.00093 | 28.56468 ± 0.02671 | 3 |
| float | S4 | 819200 | 16384 | 3.35255 ± 0.00143 | 28.57548 ± 0.04091 | 3 |
| ternary | S2 | 819200 | 16384 | 3.27491 ± 0.06613 | 26.47987 ± 1.78282 | 3 |
| ternary | S3 | 819200 | 16384 | 3.35423 ± 0.00114 | 28.62351 ± 0.03276 | 3 |
| ternary | S4 | 819200 | 16384 | 3.35438 ± 0.00143 | 28.62774 ± 0.04081 | 3 |
| repeat | S2 | 819200 | 16384 | 3.22082 ± 0.02726 | 25.05500 ± 0.68678 | 3 |
| repeat | S3 | 819200 | 16384 | 3.22297 ± 0.02363 | 25.10731 ± 0.59593 | 3 |
| repeat | S4 | 819200 | 16384 | 3.22980 ± 0.02578 | 25.28013 ± 0.65462 | 3 |

## Paired comparisons

| Kind | Control | Repeat | Training bytes | Validation bytes | Paired Δloss | Each seed Δloss | Repeat better seeds | Packed ratio | Addition ratio |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| same_edge | float S2 | S2 | 819200 | 4096 | 0.00905 ± 0.03606 | {'42': -0.03043125569820404, '43': 0.0402391254901886, '44': 0.017327740788459778} | 1 | 0.094715 | 1.71044 ± 0.14133 |
| same_edge | float S3 | S3 | 819200 | 4096 | -0.11141 ± 0.02216 | {'42': -0.08593869209289551, '43': -0.1262848824262619, '44': -0.12199458479881287} | 3 | 0.070767 | 1.71730 ± 0.15311 |
| same_edge | float S4 | S4 | 819200 | 4096 | -0.10730 ± 0.02495 | {'42': -0.07898999750614166, '43': -0.12609504163265228, '44': -0.11682750284671783} | 3 | 0.064581 | 1.72301 ± 0.12876 |
| same_memory | float S0 | S2 | 819200 | 4096 | 0.10308 ± 0.03498 | {'42': 0.14088594913482666, '43': 0.0964922308921814, '44': 0.07186663150787354} | 0 | 1.000000 | 1.71044 ± 0.14133 |
| same_memory | float S1 | S3 | 819200 | 4096 | 0.10504 ± 0.03515 | {'42': 0.1456267088651657, '43': 0.08493557572364807, '44': 0.08456158638000488} | 0 | 1.000000 | 1.71730 ± 0.15311 |
| same_memory | float S2 | S4 | 819200 | 4096 | 0.01487 ± 0.04127 | {'42': -0.03168964385986328, '43': 0.04696173965930939, '44': 0.029342010617256165} | 1 | 1.000015 | 1.72301 ± 0.12876 |
| repeat_ternary | ternary S2 | S2 | 819200 | 4096 | -0.03985 ± 0.07617 | {'42': 0.02316346764564514, '43': -0.018220603466033936, '44': -0.1244889497756958} | 2 | 1.000000 | 1.71044 ± 0.14133 |
| repeat_ternary | ternary S3 | S3 | 819200 | 4096 | -0.11379 ± 0.02280 | {'42': -0.08748193085193634, '43': -0.1277533620595932, '44': -0.1261444240808487} | 3 | 1.000000 | 1.71730 ± 0.15311 |
| repeat_ternary | ternary S4 | S4 | 819200 | 4096 | -0.10941 ± 0.02457 | {'42': -0.08137206733226776, '43': -0.1271793097257614, '44': -0.11967706680297852} | 3 | 1.000000 | 1.72301 ± 0.12876 |
| same_edge | float S2 | S2 | 819200 | 16384 | -0.00266 ± 0.04249 | {'42': -0.05134975537657738, '43': 0.026943180710077286, '44': 0.016412563621997833} | 1 | 0.094715 | 1.70233 ± 0.14093 |
| same_edge | float S3 | S3 | 819200 | 16384 | -0.12920 ± 0.02275 | {'42': -0.10404785349965096, '43': -0.14833737909793854, '44': -0.13520938530564308} | 3 | 0.070767 | 1.70994 ± 0.15453 |
| same_edge | float S4 | S4 | 819200 | 16384 | -0.12275 ± 0.02640 | {'42': -0.09333153069019318, '43': -0.14436401426792145, '44': -0.13055985048413277} | 3 | 0.064581 | 1.71609 ± 0.13066 |
| same_memory | float S0 | S2 | 819200 | 16384 | 0.11454 ± 0.03752 | {'42': 0.15634262934327126, '43': 0.10350795090198517, '44': 0.08377399295568466} | 0 | 1.000000 | 1.70233 ± 0.14093 |
| same_memory | float S1 | S3 | 819200 | 16384 | 0.12327 ± 0.03234 | {'42': 0.15950176119804382, '43': 0.09731252864003181, '44': 0.11301052570343018} | 0 | 1.000000 | 1.70994 ± 0.15453 |
| same_memory | float S2 | S4 | 819200 | 16384 | 0.00631 ± 0.04422 | {'42': -0.04452309012413025, '43': 0.03588606417179108, '44': 0.02755865454673767} | 1 | 1.000015 | 1.71609 ± 0.13066 |
| repeat_ternary | ternary S2 | S2 | 819200 | 16384 | -0.05408 ± 0.07678 | {'42': 0.011130429804325104, '43': -0.034673675894737244, '44': -0.1387110948562622} | 2 | 1.000000 | 1.70233 ± 0.14093 |
| repeat_ternary | ternary S3 | S3 | 819200 | 16384 | -0.13126 ± 0.02328 | {'42': -0.10509925708174706, '43': -0.1497029960155487, '44': -0.13896410167217255} | 3 | 1.000000 | 1.70994 ± 0.15453 |
| repeat_ternary | ternary S4 | S4 | 819200 | 16384 | -0.12458 ± 0.02664 | {'42': -0.09493490681052208, '43': -0.1465255171060562, '44': -0.1322760470211506} | 3 | 1.000000 | 1.71609 ± 0.13066 |

## Undertraining diagnostics

Training coverage starts at819200 because v3 did not record whole-history coverage. Coalesced sparse-gradient source rows track exact distinct activated slots, including zero ternary values. Per-source update counts are optimizer-step touches, not nonzero updates. Gradient norms and edge changes sample first64 touched rows every100 steps; they are NOT unbiased whole-graph statistics. Full counters are training-only checkpoint state. Held-out coverage is exact fixed-window topology coverage. Visit coverage alone cannot establish sufficient convergence.

| Family | Scale | Seed | Training bytes | Train loss | Continuation covered fraction | Diagnostic unique edges | Diagnostic fraction | Gradient norm mean | Edge update mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Improvement slopes

| Family | Scale | Validation bytes | From | To | Loss improvement | Δloss/Δlog bytes |
|---|---|---:|---:|---:|---:|---:|

## Repetition and runtime

Full histograms, medians, ≥2/≥4/max fractions, controller probabilities, additions/unique edges/token, repetition fractions, memory, gradient diagnostics, source/checkpoint/stream hashes, train time/throughput, inference throughput/latency and RSS are preserved in results.json at every milestone. CUDA fields are null on this CPU backend. MPS memory is unavailable here. No custom accelerator or GPU performance claim.

## Failures and limits

[]
One corpus and three seeds restrict generality. Longer fixed-lr training may still not establish convergence. Hashed addresses and16-channel state constrain usable capacity; ST gradients are biased. Repeat contribution proxies are partly definitional. Measured memory savings do not imply lower compute/energy. Frozen original files/checkpoints are hashed and reverified.

Final19-question decision pending until all planned jobs finish or recorded resource bounds are reached.

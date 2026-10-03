# PRG-LM v3.0.1 — Extended-Training Crossover Verification

Complete: True; completed extension milestones: 1/1; failed jobs: 0; resource-limited jobs: 0; latest completed training bytes: 832,000.

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
| float | S0 | 832000 | True |

## All validation results

| Family | Scale | Training bytes | Validation bytes | Loss | Byte PPL | n |
|---|---|---:|---:|---:|---:|---:|
| float | S0 | 819200 | 4096 | 3.07760 (n=1) | 21.70627 (n=1) | 1 |
| float | S0 | 832000 | 4096 | 3.07637 (n=1) | 21.67962 (n=1) | 1 |
| float | S0 | 819200 | 16384 | 3.09494 (n=1) | 22.08599 (n=1) | 1 |
| float | S0 | 832000 | 16384 | 3.09137 (n=1) | 22.00727 (n=1) | 1 |

## Paired comparisons

| Kind | Control | Repeat | Training bytes | Validation bytes | Paired Δloss | Each seed Δloss | Repeat better seeds | Packed ratio | Addition ratio |
|---|---|---|---:|---:|---:|---|---:|---:|---:|

## Undertraining diagnostics

Training coverage starts at819200 because v3 did not record whole-history coverage. Coalesced sparse-gradient source rows track exact distinct activated slots, including zero ternary values. Per-source update counts are optimizer-step touches, not nonzero updates. Gradient norms and edge changes sample first64 touched rows every100 steps; they are NOT unbiased whole-graph statistics. Full counters are training-only checkpoint state. Held-out coverage is exact fixed-window topology coverage. Visit coverage alone cannot establish sufficient convergence.

| Family | Scale | Seed | Training bytes | Train loss | Continuation covered fraction | Diagnostic unique edges | Diagnostic fraction | Gradient norm mean | Edge update mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float | S0 | 42 | 832000 | 3.15895 | 1.000000 | 32768 | 1.000000 | 0.55577 | 0.00005795 |

## Improvement slopes

| Family | Scale | Validation bytes | From | To | Loss improvement | Δloss/Δlog bytes |
|---|---|---:|---:|---:|---:|---:|
| float | S0 | 4096 | 819200 | 832000 | 0.001229 | -0.079240 |
| float | S0 | 16384 | 819200 | 832000 | 0.003571 | -0.230302 |

## Repetition and runtime

Full histograms, medians, ≥2/≥4/max fractions, controller probabilities, additions/unique edges/token, repetition fractions, memory, gradient diagnostics, source/checkpoint/stream hashes, train time/throughput, inference throughput/latency and RSS are preserved in results.json at every milestone. CUDA fields are null on this CPU backend. MPS memory is unavailable here. No custom accelerator or GPU performance claim.

## Failures and limits

[]
One corpus and three seeds restrict generality. Longer fixed-lr training may still not establish convergence. Hashed addresses and16-channel state constrain usable capacity; ST gradients are biased. Repeat contribution proxies are partly definitional. Measured memory savings do not imply lower compute/energy. Frozen original files/checkpoints are hashed and reverified.

Final19-question decision pending until all planned jobs finish or recorded resource bounds are reached.

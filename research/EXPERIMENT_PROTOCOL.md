# Continuous learning curves and mutable topology: preregistered protocol

Base: `ad82f7ce83d21fe25fe69a0e1fb4476672982892`. Existing v0/v1 model,
router, configuration and result files are preserved. Results go into new paths.
No phase changes are selected using favorable validation results.

## Phase A (must complete before v2 evaluation)

`configs/learning_curve.json` fixes all architecture and optimizer conditions.
Milestones: 12,800; 51,200; 204,800; 819,200 cumulative training target bytes.
Each seed/model follows one continuous AdamW run, lr=3e-4, defaults unchanged,
clip-norm=1, batch=4, context=32. No scheduler or optimizer reset.
CPU execution uses one Torch thread per worker (three concurrent seed workers);
the original v1 run used two. This changes floating-point reduction order, not
architecture/optimizer/token stream. Discrete routing can amplify tiny arithmetic
differences; first-milestone v1 loss differs from the archived run by at most0.0086
nats/byte. All new curves/controls use the same execution policy.
All models receive exactly the same seeded random training-window stream.
Each checkpoint saves optimizer, Python/NumPy/Torch/backend RNG state and stream
hash. Evaluation saves/restores training RNG. The original validation windows
and seed+batch-index sampled routing policy are retained.

Use the unchanged byte-256 TinyShakespeare split (first90%/last10%). Evaluate
4096 and 16384 contiguous non-overlapping window target bytes from the held-out
prefix. These sets overlap. Evaluate sampled/argmax/expected for PRG-v1.
The expected mode remains a soft surrogate, not exact stochastic expectation.
Report mean and sample SD across seeds 42,43,44; incomplete groups cannot be
used for the main conclusion. Report paired seed loss differences/PPL ratios.

Transformer core: width32,3layers; peak: width84,1layer. Use inherited theoretical
memory estimates for direct comparison with ad82f7c. They are not measurements
of resident memory; Transformer runtime currently recomputes context while the
estimate assumes KV storage. Packed edges are theoretical; training uses FP32
latent edge parameters. Record model-only and resumable checkpoint bytes
separately. Timing from concurrent CPU workers is elapsed throughput under
contention and cannot establish hardware efficiency.

## Proposed Phase B parameters (fixed before v2 results)

Implement a separate PRGLMv2; preserve local edge propagation, token injection,
accumulator, fatigue, RouterNet and hidden widths from v1. Destination IDs are
buffers, never gradient parameters. Mutation is explicit after optimizer step;
forward/validation/inference never mutate topology.

Main configurations: static, uniform mutation, adaptive mutation,
adaptive+recurrenceOFF. Three seeds each. Additional adaptive+accumulationOFF is
needed to directly answer whether accumulation remains beneficial. No-exploration
may be an explicitly exploratory seed42 diagnostic. Milestones and dataset,
optimizer, batches, validation modes remain identical to Phase A.

Defaults: mutation_interval=50 optimizer updates; mutation_probability=0.25 per
eligible source; maximum_rewires_per_interval=8; exploration_epsilon=0.10;
hotness_temperature=0.5; hotness_ema_decay=0.95 per optimizer update.
At most one gateway changes per source in an interval. Source order is shuffled
with a separate checkpointed mutation RNG. Exploration can be uniform or
low-visit-biased (default uniform). Prohibit self, duplicate and invalid targets.

Maintain separate region visit, selection, output-credited path activity,
message/accumulator activity and incoming-degree components, plus per-slot
selection, probability, downstream credit and age. Output contribution and
activity are magnitude/success proxies, not evidence of causal usefulness.
Optional activation*gradient credit must be labeled separately. Destination score
uses standardized output credit per visit + 0.25 standardized visit activity
- 1.0 standardized incoming degree. This penalizes concentration. Replace the
lowest credited/used eligible slot; never silently change these coefficients
based on validation. New slots have a100-step grace/probation window; optional
probation reversion when no downstream contribution is observed. Router slot
parameters and Adam moments remain unchanged (topology is the intended variable).

Training-only EMA, counters, mutation RNG, histories and probation state are
reported as training metadata. Only the gateway table is required at inference;
its packed address bytes remain counted. Preserve all evolution state in training
checkpoints. Report inference memory and training metadata separately.

RecurrenceOFF inherits v1's acyclic destination-ID mask and walker visit mask.
A pre-OFF-training sanity test found that the finite-30 sentinel could allow
forbidden actions when OUTPUT logits fell below-30. V2 OFF therefore uses
strict negative-infinity masks; v1 remains untouched and v2 ON is unchanged.
It therefore changes the usable gateway graph, not only LOCAL/RETURN. Any
recurrence interpretation must disclose this confound; it cannot independently
refute the precision-for-recurrence hypothesis.

## Completion gates and interpretation

Phase A completion means 36 seed/model/milestone checkpoints, each with both
validation budgets (including3routing modes for v1). Then implement/test v2,
smoke actual mutation, train main cases, evaluate controls, and export dashboard.
Do not present pending values as measured or extrapolate future checkpoints.

A persistent finite-budget gap weakens the current implementation; it cannot
prove all representations or all optimizers inefficient. Adaptive versus uniform
is needed to attribute benefits to destination scoring. Destination entropy,
indegree Gini/max, isolated nodes and visit entropy are necessary but do not
alone demonstrate useful self-organization. Track exploration-edge survival and
later output credit to investigate useful discoveries, without posthoc anecdotes
substituting for three-seed comparisons. Hypotheses are assessed, not defended.

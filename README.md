# PRG-LM feasibility experiment

This is a **causal next-token language model** feasibility study. The original shared-stencil model is preserved as **PRG-v0** (`PRGLM`). A separate **PRG-v1** (`PRGLMv1`) tests whether a much larger set of region-owned ternary connections and repeated message traversal can replace some numerical precision. Neither model is assumed to succeed.

## Architecture

- **Transformer:** ordinary causal encoder blocks with tied token head.
- **Looped Transformer:** the same block is reused `layers` times.
- **PRG-v0:** token embedding proposes multiple initial regions with sigmoid activation strengths. Each region holds a vector of `region_size` local nodes. A shared 3-offset stencil updates nodes; its train-time floating parameters are ternarized with a straight-through estimator. The shared RouterNet emits `LOCAL`, ring `NEIGHBOR`, recorded-origin `RETURN`, and `OUTPUT` probabilities. A small origin tensor tracks where each route arrived from. Training propagates the expected route mass. Inference samples categorical actions with a fixed seed, or chooses the highest-probability action deterministically. Fatigue is a separate recurrence penalty and recovers when unused. The maximum cycle count guarantees termination.
- **PRG-v1:** each Region owns its own ternary edge values for relative local offsets. A gateway table contains the only Region addresses. Multiple initial walkers inject token messages at cycle 0, then process only active nodes/edges. A separate accumulator adds incoming messages on every visit. Each walker remembers one previous Region for `RETURN`. A shared RouterNet plus small Region transition logits chooses `LOCAL`, one of four gateways, `RETURN`, or `OUTPUT`. Training uses hard Gumbel-Softmax routing with a straight-through gradient; sampled inference uses categorical actions.

Ternary edges include zero, allowing removal of a local connection. Relative offsets require no arbitrary sparse global addresses. The theoretical packed size counts **only the local edge stencil at 2 bits each**; embeddings, routers and head remain float32. PyTorch serialization stores all trainable tensors as float32. Thus the first implementation does not yet show a meaningful compressed model memory advantage. Training uses a hard top-k forward pass with a sigmoid straight-through gradient for initial region selection. This surrogate and soft route expectation differ from sampled inference and need ablation.

The corpus is public [tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), a fallback because TinyStories is not locally available. The preserved v0 study uses a 2048-symbol BPE vocabulary. The v1 comparison uses exact UTF-8 bytes (256 symbols) for all models to reduce embedding overhead. The first 90% of the file is training data and the last 10% validation data; sequential literary text can create a distribution shift. Generated text from short training is not expected to be coherent.

## Install and run

```bash
python -m pip install -e '.[test]'
mkdir -p data/raw
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/raw/tinyshakespeare.txt
python -m pytest -q
python -m prglm.experiment --steps 8 --batch 4 --context 16 --regions 4 --region-size 32 --cycles 3
PRGLM_RUN=runs/smoke uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The dashboard exposes paired generation, sampled token-level trajectories, region activation/fatigue/probabilities, benchmarks, repeats, and temporary intervention controls. The repository also includes [dashboard-bundle.zip](dashboard-bundle.zip), containing a standalone benchmark and saved trajectory replay. Unzip it and run `python -m http.server 8001` from the directory containing `dashboard/`, then open `http://127.0.0.1:8001/dashboard/`. **Generating new text requires the local FastAPI server** and trained checkpoints. Refresh the attached bundle after new experiments with `python -m prglm.export_dashboard --run runs/final_seed42 --dest dashboard --archive dashboard-bundle.zip`. No GitHub Pages deployment is used.

## Comparability and limits

`--steps`, `--batch`, `--context`, data files, vocabulary, and seed are shared. Model sizes are **not** automatically matched in a smoke run. For a fair follow-up, sweep `--width` and `--layers` for Transformer and `--regions`, `--region-size`, `--cycles` for PRG; select pairs once by parameter count and once by actual/theoretical serialized memory. Report validation loss and compute at the same token budget across at least 3 seeds. `results.json` stores the actual run configuration and metrics. The sampled path is bounded and the sparse stencil is shared, so it may simply underfit; a bad PRG result should not be hidden by choosing a favorable size or seed.

Useful ablations are available through `Config`: `stochastic_recurrence`, `fixed_recurrence_probability`, `fatigue`, `local_router`, `initial_regions`, `edge_type`, `max_cycles`, `region_size`, and `regions`. `edge_type='float'` switches to full-precision local edges. Set `initial_regions=1` for a single seed region. Inference controls can override recurrence probability, fatigue strength/recovery, region selection, and cycle bound without editing checkpoints.

No hardware, micromagnetic, MTJ, or reservoir simulation is included.

## First benchmark (100 updates, batch 8, context 32)

Three seeds (42, 43, 44) on tiny Shakespeare, same tokenizer and 25,600 training tokens per model and seed:

| Model | Parameters | Validation PPL, mean ± SD | Training tokens/s, mean |
| --- | ---: | ---: | ---: |
| Transformer | 233,216 | 968 ± 18 | 26,838 |
| Looped Transformer | 183,232 | 991 ± 25 | 25,457 |
| PRG-LM | 134,828 | 2,866 ± 53 | 1,228 |

A 139,216-parameter Transformer, within 3.3% of PRG-v0's parameter count and 4.2% of its serialized size, obtained PPL 931 in a seed-42 size-matched run. PRG-v0 did not outperform the Transformer. However, PRG-v0 contained only **192 shared ternary local-edge values** and therefore did **not meaningfully test** the intended exchange of numerical precision for substantially larger low-bit connectivity. Its result is a baseline for a small shared-stencil recurrent router, not a definitive test of large-capacity PRG. CPU timing includes Python region loops and is not a measurement of optimized event-driven hardware.

At 500 updates with the same batch and context, seed 42 reached validation PPL 478 (Transformer), 527 (looped), and 1,218 (PRG-v0). The gap narrows with more training but remains large. Seed-42, 100-update PRG-v0 ablations gave PPL 2,947 with fatigue off, 2,698 with recurrence off, and 2,813 with float edges, versus 2,853 for the default. These ablations are exploratory single-seed results for the **v0 shared-stencil model**.

The largest v0 architectural risk is the train/inference difference: training uses expected route mass and a straight-through initial top-k, while inference samples discrete routes. Route effects may also be too weak relative to the token embedding and head.

## PRG-v1 architecture and reproduction

The v1 model is in `prglm/v1_model.py`, separate from `PRGLM`. For each Region, its trainable latent edge tensor has shape `[region_size, edges_per_node]`; a straight-through estimator uses `{-1,0,+1}` in the forward pass. Only 2 packed bits per edge are assumed for the **theoretical** inference model. All 524,288 edge slots have Region-specific values in the final configuration, of which about 320,000 are nonzero after training. Node destinations use one shared, signed relative-offset pattern. Only the small `[regions, gateways]` gateway table stores global Region IDs.

At each token, the Global Router selects multiple initial Regions and `seed_proj` injects the token message once. Each live walker then visits only `active_nodes` source nodes **for local edge propagation**, receives a new message, and lets the Local Router choose `LOCAL`, one of four neighbor gateways, `RETURN` to its previous Region ID, or `OUTPUT`. The accumulator adds incoming messages each cycle; output reads that accumulator. Node state persists across tokens for autoregressive context. The current PyTorch prototype still applies decay to the dense node-state and accumulator tensors each cycle, so its active-edge ratio is **not** a total operation-count ratio. A lazy sparse decay implementation is needed before making event-driven efficiency claims. Recurrence, accumulation, and fatigue are independent switches. Training uses hard Gumbel-Softmax actions and a soft gradient; sampled inference uses categorical actions. The expected validation mode is a **soft Region-mass surrogate**, with one node pattern per Region, rather than an exact expectation over every path.

The v1 comparison uses the same 256-byte UTF-8 vocabulary, corpus split, randomly selected training windows, 32-token context, and 12,800 training tokens per model and seed. Validation uses the same 4,096 held-out target bytes in non-overlapping windows for every model and routing mode. Run it with:

```bash
python -m prglm.v1_experiment --steps 100 --batch 4 --context 32 --validation-tokens 4096 --val-batch 8 --regions 32 --region-size 1024 --edges-per-node 16 --gateways 4 --active-nodes 16 --cycles 4 --seeds 42 43 44 --models transformer_core transformer_total prg prg_v1 --out runs/v1_final
python -m prglm.export_dashboard --run runs/final_seed42 --v1-run runs/v1_final --dest dashboard --archive dashboard-bundle.zip
PRGLM_V1_RUN=runs/v1_final/seed42 uvicorn web.app:app --host 127.0.0.1 --port 8000
```

The export command uses the preserved BPE v0 run in `runs/final_seed42` for the legacy dashboard section; the repository already contains its saved export. If reproducing only the byte study, the final command runs the live **Byte-256 v1 study** comparison without requiring that BPE checkpoint.

`transformer_core` is selected by closest **static packed core bytes excluding token embedding/head**. `transformer_total` is selected by closest **peak theoretical inference bytes including dynamic state**. The Transformer estimate assumes a float32 K/V cache; the current PyTorch implementation recomputes context. No real 2-bit packed runtime exists yet: the checkpoint stores latent edge floats. `results.json` records the full static/dynamic component breakdown and actual serialized bytes.

### PRG-v1 first feasibility result

Final configuration: 32 Regions × 1,024 nodes × 16 edge slots/node = **524,288 Region-owned ternary slots**; 320,660/320,258/319,812 are nonzero after training at seeds 42/43/44. Models share the byte tokenizer and training/validation token budget above. Values are the three-seed mean ± sample standard deviation.

| Model | Matched budget | Sampled validation loss | Sampled PPL | Static packed bytes | Peak inference bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Transformer core | Core static (156,800 vs v1 155,324 bytes) | 4.794 | 120.9 ± 8.2 | 193,664 | 218,240 |
| Transformer total | Peak inference (472,752 vs v1 483,398 bytes) | 4.045 | 57.2 ± 3.9 | 451,248 | 472,752 |
| PRG-v0 retrained on bytes | Reference, not matched | 6.451 | 643.8 ± 141.3 | 79,840 | 82,176 |
| PRG-v1 | Reference | 5.402 | 225.7 ± 50.3 | 220,860 | 483,398 |

PRG-v1 expected soft-route PPL was 208.4 ± 46.6, deterministic argmax 200.3 ± 35.0, and true sampled 225.7 ± 50.3. Its actual serialized PyTorch checkpoint was 2,193,354 bytes, versus 220,860 theoretical static packed bytes. These sizes describe different representations and must not be conflated.

PRG-v1's seed-42 packed static breakdown in bytes: token embedding/tied head 65,536; Global Router 8,320; shared RouterNet 3,868; Region embeddings 2,048; ternary local edges 131,072; gateway addresses 80; transition logits 896; other numerical tensors 9,024; relative-offset metadata 16. Dynamic state per sequence: node state 131,072; accumulator 131,072; fatigue 128; event state 138; other runtime state 128. The local-edge count is large, but only about 1,464–1,624 **edge slots/token** were processed in the 32-token diagnostic samples (about 889–994 nonzero), an active-edge ratio of about 0.07–0.08% per cycle. Mean unique active nodes/token was 62.8. Mean sampled visits/token was 6.06; an independent eight-draw Monte Carlo estimate was 5.67 visits/token. Mean base `LOCAL` probability was 0.148, accumulator magnitude 0.272, and output contribution magnitude 0.352. Visit depth and accumulator magnitude had a positive correlation of about 0.50; base `LOCAL` probability and accumulator magnitude had a weak negative correlation of about −0.12. These short-trace statistics are not corpus-wide estimates or causal proof that recurrence helps.

The v1 result improves over the byte-trained v0 reference, but does not beat either memory-matched Transformer. It is **insufficient to confirm the precision-for-traversal hypothesis**. The soft-route, argmax, and sampled PPL differences show remaining routing-mode sensitivity. Python event-loop throughput is logged but is not a prediction of an optimized event-driven implementation.

### Independent three-seed ablations

All ablations retrain PRG-v1 for the same 12,800 token budget and evaluate the same 4,096 held-out target bytes per seed. Sampled routing PPL (mean ± SD):

| PRG-v1 setting | Sampled PPL | Change from default, paired by seed |
| --- | ---: | ---: |
| Default | 225.7 ± 50.3 | — |
| Recurrence OFF | 211.9 ± 45.4 | −18.5 / −14.1 / −8.8 |
| Accumulation OFF | 237.8 ± 52.5 | +14.1 / +12.6 / +9.7 |
| Fatigue OFF | 228.6 ± 51.7 | +5.6 / +0.7 / +2.4 |

Turning accumulation off worsened PPL for all three seeds, consistent with accumulated messages carrying useful information. Turning recurrence off improved PPL for all three seeds, so the current learned revisit policy does not support the central recurrence claim. The difference is small relative to the absolute PPL and this is a short training run. The next priority is to test longer token budgets and traversal-count interventions at the same packed budget, while replacing dense state decay with a lazy sparse update. An independently evaluated Monte Carlo traversal count is recorded alongside actual counts; route-probability calculations conditioned on a sampled path are labeled separately and are not treated as an exact expected count.

## Continuous learning curves (Phase A, based on ad82f7c)

The original v0/v1 model implementations and archived study remain unchanged.
The new protocol is fixed in [configs/learning_curve.json](configs/learning_curve.json)
and [research/EXPERIMENT_PROTOCOL.md](research/EXPERIMENT_PROTOCOL.md).
Phase A uses one continuous run per seed/model, with milestones 12.8k,51.2k,
204.8k,819.2k target bytes. Both4096/16384 held-out target bytes are evaluated.
The latter contains the original prefix. PRG-v1 uses all three routing modes.

```bash
# Install project requirements using your existing environment, then:
OMP_NUM_THREADS=1 python scripts/run_phase_a.py
# Refresh the report and standalone SVG figures during/after training:
OMP_NUM_THREADS=1 python -m prglm.curve_report
```

In this workspace use `PYTHONPATH=.deps:.` for the locally installed dependencies.
Workers log to `runs/phase_a/worker_seed{42,43,44}.log`; progress and checkpoint
files live under `runs/phase_a/seedSEED/MODEL/`. Repeating the same command
resumes the optimizer and all training RNG states. A per-job lock prevents
concurrent writes to the same run. Evaluation cannot alter the next training
random stream. Checkpoint/token-stream/protocol mismatches are rejected.
The resumable `.pt` includes optimizer state; separate model-only `.pt` sizes
are reported. Local checkpoints are excluded from Git as in the original study.

Results and linear/log-scale SVG charts are exported into `research/phase_a/`.
The dashboard adds a separate learning-curve section; existing v0/v1 sections
and JSON results are preserved. Three-seed groups are required for chart points;
missing results stay pending. See `research/phase_a/REPORT.md` for current status.

The preregistered mutable-topology experiment starts after Phase A completes.
No v2 conclusions are inferred from partial v1 curves. The finite token budget
also cannot prove an intrinsic representation limitation independent of all
possible optimizers.

## PRG-v2: slow discrete gateway adaptation

`PRGLMv2` is a separate class. The existing v1 implementation is untouched.
The neural engine is the v1 hard traversal plus detached training observers.
Tests verify identical static outputs and gradients, accumulation and all three
validation modes. The completed Phase A v1 runs therefore serve as static
controls without retraining. Existing old `runs/prg_v2_*` names are v0-era
experiments and are unrelated; they are preserved.

Fast adaptation remains AdamW of the shared RouterNet/local-edge latent weights.
Slow adaptation changes only gateway destination IDs after optimizer updates.
Defaults fixed before v2 results: interval50 updates, per-source probability0.25,
max8 replacements/opportunity, exploration0.10, temperature0.5, EMA decay0.95,
100-update probation. This delays the first replacement until update100.
Replace the eligible slot with lowest downstream credit +0.01×selection EMA;
at most one slot/source/opportunity. Neural slot parameters and Adam moments
are retained. Never select self, existing row destinations, or invalid IDs.

Destination scoring uses standardized output-path credit per visit
+0.25×standardized visit frequency −1.0×standardized indegree. Credit is shared
across visits in trajectories that contribute OUTPUT or cycle-cap readout.
This is a magnitude/success **proxy**, not causal evidence of loss improvement.
Message/accumulator activity, route probability/selection and success are
reported separately. Optional `gradient_credit` records `|activation×gradient|`
but is OFF in the main study. Exploration can be uniform (default) or low-visit.
Uniform mutation is the random-rewiring control. Probation reverts roads with
zero downstream credit when the original destination remains valid; duplicates
are never introduced. Mutation, probation and EMA are frozen in eval/inference.

The main experiment includes adaptive, uniform, adaptive+recurrenceOFF and
adaptive+accumulationOFF, all with three seeds and four819.2k-token milestones.
RecurrenceOFF inherits v1's ascending-destination-ID restriction: it changes the
usable gateway graph as well as revisit behavior, so conclusions must disclose
this confound. V2 OFF uses strict negative-infinity action masks: a sanity
test exposed forbidden-action leakage with v1 finite-30 sentinels under very low
OUTPUT logits. This was fixed before OFF main training; v1 and ON are unchanged.
No-exploration is optional and is not needed for main conclusions.

```bash
# Requires complete Phase A checkpoint/results; resumes only incomplete jobs.
OMP_NUM_THREADS=1 python scripts/run_mutable_v2.py
# Small standalone smoke (does not contribute to main conclusions):
OMP_NUM_THREADS=1 python -m prglm.v2_experiment --smoke \
  --config configs/mutable_smoke.json --out runs/mutable_smoke_observer_v2
# Refresh measured results and graphs:
OMP_NUM_THREADS=1 python -m prglm.v2_report
```

The mutable runs use a new `runs/mutable_v2/seedSEED/MODEL/` hierarchy. A separate,
checkpointed mutation RNG ensures that structural choices never consume the
neural sampling RNG. EMA, gateway ages, probation, counters and full history are
saved with optimizer/RNG checkpoints. `model_tokens_*.pt` exports discard
training-only topology statistics, retaining actual gateway IDs and neural state.
The packed inference budget is unchanged; training metadata and full checkpoint
serialization are reported separately. No physical or packed execution backend
is implemented. CPU timings under concurrent training are not hardware claims.

`research/v2/REPORT.md` contains complete measured tables, paired controls,
learning curves and the ten-question assessment when the study finishes.
The dashboard preserves v0/v1 and adds v2 graph snapshots, topology diffs,
hotness/indegree, probation history, actual validation-token trajectories and a
live same-prompt comparison. Repository ZIP replay needs no external libraries.
For live generation, start the existing FastAPI app from this repository:

```bash
OMP_NUM_THREADS=1 uvicorn web.app:app --host 127.0.0.1 --port 8000
# Visit http://127.0.0.1:8000 and use the separate PRG-v2 section.
```

Select a completed checkpoint seed/milestone. Generation never changes topology.
Repeats vary the generation seed and show output/trajectory variations. Static
repository replay uses saved held-out trajectories; generating new text requires
the local backend and the local checkpoints (not included in the ZIP).

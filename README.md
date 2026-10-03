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
learning curves and the ten-question assessment. All 84 milestone results are
complete: 36 Phase A and 48 mutable-v2 results. No completed run was retrained.
The dashboard preserves v0/v1 and adds v2 graph snapshots, topology diffs,
hotness/indegree, probation history, actual validation-token trajectories and a
live same-prompt comparison. Repository ZIP replay needs no external libraries.
For live generation, start the existing FastAPI app from this repository:

```bash
OMP_NUM_THREADS=1 uvicorn web.app:app --host 127.0.0.1 --port 8000
# Visit http://127.0.0.1:8000 and use the separate PRG-v2 section.
```

Seed42 final (819,200-byte) inference exports for all seven models are bundled
under `checkpoints/byte256_seed42_819200/` (11.6 MB total, SHA256 manifest).
A fresh checkout can use the live v2 section without training. Select seed42 and
819200; other seeds/milestones require the local `runs/` checkpoints.
Generation never changes topology.
Repeats vary the generation seed and show output/trajectory variations. Static
repository replay uses saved held-out trajectories; generating new text requires
the local backend and model checkpoints (bundled in the repository, not the ZIP).

Memory audit: the inherited Transformer reporting helper double-counts its
learned position embedding. Matching configurations remain unchanged to honor
Phase A. Corrected core-model static/peak estimates are189568/214144 bytes;
peak-matched model440496/462000 bytes, versus PRG220860/483398. Both original
and corrected reporting are retained in the final report. These are modeled
state estimates, not true allocator/workspace bounds or measured RAM peaks.

Ablation interpretation: v1-compatible accumulationOFF also switches readout
from the per-token accumulator to `tanh(persistent node state)`. Thus the toggle
changes the readout source/nonlinearity as well as summation. Report its effect,
but do not interpret the entire difference as isolated evidence for/against
message accumulation. The default accumulator resets each token; persistent
state influences node selection and feedback traversal.

A cycle-cap characterization test exposes another inherited limitation: a final
gateway to an unvisited destination can make accumulationON read a zero
destination accumulator (bias-only); OFF retains the source message. The frozen
v1 behavior is preserved in this study. This horizon/readout confound should be
isolated before making stronger recurrence/accumulation claims.

Under a forced one-cycle direct-OUTPUT path, a characterization test finds zero
prefix-embedding gradient with the default per-token accumulator readout, and
nonzero prefix gradient with accumulationOFF's persistent-state readout. Hard
node selection still depends on prior state, so this is a temporal-credit
limitation rather than proof that inference is stateless. It is particularly
relevant when late learned trajectories terminate immediately.

For a fresh checkout, use your preferred PyTorch backend and install the
research/test extras with `python -m pip install -e '.[research,test]'`.
Matplotlib generates standalone SVG/PNG figures with sample-SD error bars.
The dashboard itself requires no Matplotlib or npm dependencies. Optional
DOM/API checking uses `npm install --prefix .deps/jsdom jsdom`, then
`node tests/dashboard_smoke.cjs` against the running local API.

### Completed Phase A/v2 findings

Final results below are mean ± sample SD across seeds42/43/44, with4096 fixed
held-out target bytes. Loss is nats/byte; PPL is byte-level, not word-level.
The report also includes16384-byte validation and all three PRG routing modes.

| Model | Loss | PPL |
|---|---:|---:|
| Transformer core | 2.3907 ± 0.0074 | 10.9212 ± 0.0809 |
| Transformer peak | 2.1793 ± 0.0102 | 8.8403 ± 0.0897 |
| PRG-v1 static | 2.5755 ± 0.0094 | 13.1382 ± 0.1235 |
| PRG-v2 adaptive | 2.5919 ± 0.0097 | 13.3556 ± 0.1296 |
| PRG-v2 uniform | 2.5814 ± 0.0075 | 13.2165 ± 0.0993 |
| PRG-v2 adaptive recurrenceOFF | 2.5853 ± 0.0084 | 13.2680 ± 0.1119 |
| PRG-v2 adaptive accumulationOFF | 2.6081 ± 0.0123 | 13.5743 ± 0.1678 |

v1 initially catches up, but its peak-matched loss gap decreases from1.354 to
0.249 at204.8k, then widens to0.396 at819.2k. Slow initial optimization is evident;
the finite-budget final gap does not establish an intrinsic capacity limit.
Adaptive mutation improves neither static nor uniform at the final milestone.
No severe hub collapse is observed, but useful self-organization is not established.
RecurrenceOFF is slightly better; accumulationON is slightly better, subject to
the mask/readout confounds documented above. Adaptive training gateway-selection
EMA is only0.04–0.07%; held-out trajectories typically take two initial walkers
straight to OUTPUT. Changing roads has little opportunity to help when bypassed.
The original precision-for-recurrence hypothesis remains unsupported here.

The next diagnostics should isolate temporal credit, cap/readout behavior and
actual gateway usage before another topology sweep. Keep these as new controlled
experiments; the completed models/protocol are preserved without favorable tuning.

`research/v2/inference_benchmark.json` separately records final-checkpoint CPU
generation timing (three seeds,16 generated bytes after warmup, context recomputed).
Core/peak Transformers measured1487/2281 bytes/s; adaptive PRG measured33 bytes/s.
These short prototype measurements do not demonstrate packed/event-driven speed.
All33 Python tests and the dashboard DOM/live-API check pass. Canvas drawing was
mocked in the DOM check; visual browser rendering has not been verified.
Run `python scripts/verify_completed_study.py` to audit results, stream hashes,
resume state, preserved legacy artifacts and bundled export hashes without training.

## PRG-v2.1 Core Correction Study

Completed36 new milestone results (three models × three seeds × four continuous milestones). Frozen v0/v1/v2 code/results/checkpoints were preserved; no baseline was retrained.
The goal was to test temporal credit, valid cycle-cap semantics and gateway opportunity. A scalar-gated prefix-state pool reuses the existing readout projection; cap output uses a processed source. Clean accumulation changes only accumulator summation. The minimum-one-gateway variant is diagnostic, not the main architecture.

| Model | Final loss (nats/byte) | Byte PPL |
|---|---:|---:|
| transformer_core | 2.3907 ± 0.0074 | 10.9212 ± 0.0809 |
| transformer_total | 2.1793 ± 0.0102 | 8.8403 ± 0.0897 |
| prg_v1 | 2.5755 ± 0.0094 | 13.1382 ± 0.1235 |
| prg_v2_adaptive | 2.5919 ± 0.0097 | 13.3556 ± 0.1296 |
| prg_v2_uniform | 2.5814 ± 0.0075 | 13.2165 ± 0.0993 |
| prg_v21_core | 2.5276 ± 0.0128 | 12.5236 ± 0.1605 |
| prg_v21_clean_no_accum | 2.5337 ± 0.0127 | 12.6010 ± 0.1611 |
| prg_v21_gateway_probe | 2.6013 ± 0.0112 | 13.4825 ± 0.1508 |

Three-seed means ± sample SD,4096 held-out target bytes. Full16384-byte checks, route metrics, gradient characterization, late slopes and all12 research answers are in [the v2.1 report](research/v2_1/REPORT.md). Canonical results are README, that REPORT and its machine-readable [results.json](research/v2_1/results.json).
No dashboard, HTML, FastAPI or ZIP work was done for v2.1; old artifacts remain preserved and were not part of this research workflow.
Limitations: the core correction is a bundle, global context pooling can bypass sparse routing, the gateway probe changes OUTPUT timing/compute, expected routing is a surrogate and expected probe is undefined. This study does not isolate numerical-precision replacement. No architecture novelty or successful hypothesis is assumed.

Resume or reproduce using `OMP_NUM_THREADS=1 PYTHONPATH=.deps:. python scripts/run_core_correction.py --push`. Checkpoint/model exports stay in ignored `runs/v2_1/`; the runner resumes incomplete jobs and skips completed results. Install project research/test extras first on a fresh checkout.

### AI-assisted development disclosure

The human project author led the research hypotheses, architecture concepts, experimental questions, experiment direction and interpretation criteria. OpenAI ChatGPT and OpenAI coding models substantially assisted literature discovery during ideation, architecture discussion, implementation, test generation, experiment automation, result summarization and documentation. The author reviews the machine-produced analysis. We do not claim AI-generated implementation code itself as original source code. AI assistance and architecture novelty are separate questions.

### Prior art and novelty

Similar prior work exists for the individual components. PRG-LM is currently an **experimental research hypothesis**; novelty of the overall design is not established. It must be evaluated against published literature, patents and existing implementations.

### Literature surfaced during ChatGPT-assisted ideation

The following were surfaced in the project discussion by ChatGPT. This does not assert that they were OpenAI model training data, or that this implementation was directly derived from their source code. These are related discussion references, not a completed novelty search.

- Alex Graves, *Adaptive Computation Time for Recurrent Neural Networks*, arXiv:1603.08983 (2016).
- Simon Schug, Frederik Benzing, Angelika Steger, *Presynaptic stochasticity improves energy efficiency and helps alleviate the stability-plasticity dilemma*, eLife (2021).
- Sizhong Lan, *On the Relation of Impulse Propagation to Synaptic Strength*, arXiv:1805.09001 (2018).
- Jack C. Gartside et al., *Reconfigurable training and reservoir computing in an artificial spin-vortex ice via spin-wave fingerprinting*, Nature Nanotechnology (2022).

No additional looped/recurrent-LM or equilibrium citation is added: the inspected repository history did not provide a precise citation previously surfaced in discussion.

## PRG-v3 Same-Edge Repetition Scaling Study

**Research question:** Can repeated use of the SAME low-bit source node/destination/edge slot substitute for part of numerical weight magnitude, particularly as model scale grows? This means additive reuse of a frozen reference message, not visiting different edges or ordinary W^k composition.
S0–S4 scale32768→131072→524288→2097152→8388608 edge slots. Float32 single-pass, ternary single-pass and learned bounded repetition share topology and neural blocks. Each has seeds42/43/44 and continuous fixed-data budgets12.8k/51.2k/204.8k/819.2k bytes. Large models may be undertrained.
Scaling tests loss, packed storage, active operations and dynamic state, rather than judging a single small model. Region/gateway/mutable topology/fatigue are omitted to isolate core repetition, not to declare those ideas worthless. No dashboard/UI changes; all v0–v2.1 artifacts stay frozen.

| Family | Scale | Final loss | Byte PPL | n |
|---|---|---:|---:|---:|
| float | S0 | 3.0835 ± 0.0153 | 21.8371 ± 0.3362 | 3 |
| float | S1 | 3.0826 ± 0.0122 | 21.8165 ± 0.2645 | 3 |
| float | S2 | 3.1776 ± 0.0633 | 24.0207 ± 1.5409 | 3 |
| float | S3 | 3.2991 ± 0.0016 | 27.0873 ± 0.0435 | 3 |
| float | S4 | 3.2997 ± 0.0031 | 27.1058 ± 0.0828 | 3 |
| ternary | S0 | 3.0774 ± 0.0264 | 21.7079 ± 0.5756 | 3 |
| ternary | S1 | 3.1074 ± 0.0272 | 22.3692 ± 0.6135 | 3 |
| ternary | S2 | 3.2265 ± 0.0615 | 25.2225 ± 1.5771 | 3 |
| ternary | S3 | 3.3015 ± 0.0002 | 27.1520 ± 0.0068 | 3 |
| ternary | S4 | 3.3018 ± 0.0030 | 27.1629 ± 0.0814 | 3 |
| repeat | S0 | 3.1774 ± 0.0148 | 23.9857 ± 0.3557 | 3 |
| repeat | S1 | 3.1799 ± 0.0279 | 24.0508 ± 0.6713 | 3 |
| repeat | S2 | 3.1866 ± 0.0277 | 24.2125 ± 0.6755 | 3 |
| repeat | S3 | 3.1877 ± 0.0230 | 24.2359 ± 0.5621 | 3 |
| repeat | S4 | 3.1924 ± 0.0220 | 24.3517 ± 0.5382 | 3 |

Complete=True; recorded failed/resource-limited jobs=0. Measured same-edge mean-loss sign-change crossovers=['S3']. No extrapolated crossover is asserted.
Canonical analysis, all17 answers, paired differences, exact memory matches, ablations, costs and static SVG/PNG figures: [research/v3/REPORT.md](research/v3/REPORT.md). Raw machine results: [research/v3/results.json](research/v3/results.json).
Packed exports actually encode2-bit ternary edges; PyTorch execution decodesFP32 and batches controller candidates. Arithmetic k*s*m equivalence is not nonlinear LM equivalence or a hardware-efficiency claim. Fixed16-channel hashed state and biased ST optimization restrict generality; useful repetition must improve measured quality/memory/compute jointly.

Resume: `OMP_NUM_THREADS=1 PYTHONPATH=.deps:. python scripts/run_repetition_scaling.py --push`. Failed checkpoints remain under ignored runs/v3; completed rows skip. Resource-limited/failed jobs are recorded; `--retry-failed` explicitly retries only such jobs, preserving failure history. Install the project research/test extras on a fresh checkout.
The prior AI-assisted development disclosure, novelty/prior-art caveats and ChatGPT-surfaced literature remain intact. The human author set the SAME-edge hypothesis and research direction; OpenAI models assisted discussion, implementation, testing, automation and reporting. No claim of original AI-generated source code or established architecture novelty is made.

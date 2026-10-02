# PRG-LM v3 — Same-Edge Repetition Scaling Study

Complete: False; 0/180 primary milestone results; 0 recorded failed/resource-limited jobs.
Hypothesis: Can additive temporal reuse of the SAME low-bit source/destination/slot replace part of stored numerical magnitude, and become more favorable with graph scale? No conclusion is presumed.

## Exact architecture

The graph stores N nodes ×32 relative-offset edge slots. Sixteen source nodes are selected by a fixed causal rolling prefix hash and source stride37. The shared relative offset template is (slot+1)*2654435761 mod N. Node counts are powers of two, so slots/destinations and activated sources are unique. Float and ternary families use identical identities at the same scale.

Current-byte embedding plus0.9 times the previous graph-produced16-channel state creates one tanh source message. Source, destination, slot, reference message and edge value remain frozen throughout its repetition episode. First pass adds sign*message; each REPEAT adds that SAME value again. A six-feature shared6→16→1 controller predicts continuation from reference, |reference|, sign, normalized source/destination and repeat index. It has129 parameters independent of N; there are no edge-specific recurrence parameters/probability buffers. Bernoulli logistic-noise straight-through gates train bounded repetition; inference samples Bernoulli gates. Maximum repeat8.

Per-edge linear accumulators are reduced into destination-index-mod16 channels with fixed1/sqrt(32) scaling, followed by tanh, existing-sized readout and tied byte head. There is no LayerNorm to erase a uniform repetition magnitude. The next token's persistent state is ONLY graph-produced output: zero graph values eliminate token/prefix influence. There is no direct embedding/context-to-head residual, attention, global contextual pooling, Region/router/gateway/fatigue/mutation.

This is a compact16-channel causal state with hashed sparse connectivity, not persistent independent state for every stored node. Address aliasing and the compressed destination readout restrict the model class; conclusions do not cover all graph architectures. Nonlinear LM equivalence is not assumed from linear k*s*m equality. The controller is shared but high-precision, so gains alone do not prove an efficient replacement for arbitrary stored magnitude.

## Preregistered protocol

CPU/CUDA/MPS auto selection; this environment has2 CPU cores and8GB memory. Two worker processes, one PyTorch thread each. Families: Float32 single pass, ternary single pass, ternary shared learned repetition. No Transformer is retrained/reused as a falsely matched v3 baseline.
TinyShakespeare, first90%/last10% split, byte256, context32, seeds42/43/44. Continuous12.8k→51.2k→204.8k→819.2k training bytes, same hashed stream/seed. AdamW neural blocks + SparseAdam edge latents, lr3e-4, shared combined gradient clip1. SparseAdam has no weight decay in every family. Dense moments are training-only. This is FIXED-DATA-BUDGET scaling: large graphs can be undertrained.
Primary held-out sampled routing4096 bytes, secondary16384 prefix-correlated bytes; identical windows to prior studies. Controller p=.5 initially, logistic-noise hard/ST training. Inference uses sampled counts. Conditional probabilities depend on fixed reference/sign/addresses/index, never on freshly transformed repeated messages.
Bounds/resource limits are fixed in configs/repetition_scaling.json before training. No scales/seeds/hyperparameters are selected from emerging validation. Planned S0–S4 only; S5 is not planned. Resource-limited jobs remain visible with incomplete cells.

## Scale and memory rules

| Scale | Nodes | Slots/node | Stored edges |
|---|---:|---:|---:|
| S0 | 1,024 | 32 | 32,768 |
| S1 | 4,096 | 32 | 131,072 |
| S2 | 16,384 | 32 | 524,288 |
| S3 | 65,536 | 32 | 2,097,152 |
| S4 | 262,144 | 32 | 8,388,608 |
Float structural storage isFP32 (4 bytes), low-bit structural storage2 bits. Other neural blocks/controller areFP32; unused controller is retained/frozen in single-pass controls so repeatOFF is architecturally matched. Low-bit training storesFP32 latents. Actual inference exports pack ternary codes into2 bits but this PyTorch executor DECOMPRESSES toFP32; serialized savings are not resident-memory/speed savings.
Modeled dynamic state assumes streaming controllers: persistent16-channel state, hash/source addresses, frozen edge reference/accumulator and counters. Actual batched probability/feature/gradient workspaces are excluded. Both training and current inference batch all candidate gates, including inactive lanes. Theoretical active additions do NOT equal executed GPU/CPU FLOPs. Actual FP32 tensors, process peak RSS, CUDA allocated/reserved when available, optimizer bytes and export/resume sizes are separately recorded.

## Progress

| Family | Scale | Seed42 | Seed43 | Seed44 | Final complete |
|---|---|---:|---:|---:|---|
| float | S0 | pending | pending | pending | False |
| ternary | S0 | pending | pending | pending | False |
| repeat | S0 | pending | pending | pending | False |
| float | S1 | pending | pending | pending | False |
| ternary | S1 | pending | pending | pending | False |
| repeat | S1 | pending | pending | pending | False |
| float | S2 | pending | pending | pending | False |
| ternary | S2 | pending | pending | pending | False |
| repeat | S2 | pending | pending | pending | False |
| float | S3 | pending | pending | pending | False |
| ternary | S3 | pending | pending | pending | False |
| repeat | S3 | pending | pending | pending | False |
| float | S4 | pending | pending | pending | False |
| ternary | S4 | pending | pending | pending | False |
| repeat | S4 | pending | pending | pending | False |

## Sanity tests

52 tests passed (including all previous tests). Three full-shape smoke families completed; additive maximum absolute error9.54e-7, nonzero prefix activation gradients, exact two-optimizer/RNG resume and causal-prefix invariance verified.
Signed additive1/2/4/8 equality, locked edge identities, frozen messages, prefix credit, zero-graph bypass test, causal prefix invariance, no-repeat equivalence, packed export/save-load and two-optimizer/RNG-resume are checked. Full exact values/configs are in sanity.json.

## Learning curves

| Family | Scale | Training bytes | Validation bytes | Loss | Byte PPL | n |
|---|---|---:|---:|---:|---:|---:|

## Final compute/repetition

| Family | Scale | Repeats/edge | ≥2 fraction | ≥4 fraction | Edge additions/token | Repetition compute fraction | Active fraction |
|---|---|---:|---:|---:|---:|---:|---:|
All modeled active counts include zero ternary edges; nonzero operations are separately in JSON. Unique activated edges/token is512 by design. Consecutive revisits are counted within locked per-edge episodes, not different edges or new tokens. Repeated edges are interleaved parallel lanes; every lane retains identity/reference. Controller probabilities, full histograms, median, bound fraction, projected output contribution by count and correlations are in machine results.
Output contribution is a PRE-NONLINEARITY readout-column magnitude proxy; count increases its magnitude by construction. Positive correlation is not causal evidence of useful language-model computation.

## Same-edge-count paired gaps

| Control | Scale | Validation bytes | Repeat − control loss | Per-seed deltas |
|---|---|---:|---:|---|

## Nearest measured same-memory comparison

| Float scale | Repeat scale | Validation bytes | Static byte ratio | Edge ratio | Loss delta | Within1% budget |
|---|---|---:|---:|---:|---:|---|
Only within1% measured matches support a same-memory claim; unmatched nearest rows are explicitly not budget matched. No interpolation/extrapolation. FP16 float models were not trained and are not substituted into the measured table.

## Empirical scaling slopes

| Family | From | To | Validation bytes | Dimension | Loss/log-dimension slope |
|---|---|---|---:|---|---:|

## Memory and prototype runtime

| Family | Scale | Trainable FP params | Low-bit slots | Packed structure | Controller bytes | Embedding/head bytes | Static packed | Dynamic | Modeled peak | Export bytes | Resume bytes | Optimizer bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

| Family | Scale | Train tok/s | Inference tok/s | Latency ms/token | Peak process RSS bytes |
|---|---|---:|---:|---:|---:|
Timings are prototype measurements under the actual worker schedule. Peak process RSS includes interpreter/Torch and is cumulative per job process. CUDA stats are null on CPU. No GPU or custom-accelerator advantage is inferred from packed bytes or Python wall time.

## Representative repetition ablations

Ternary single-pass is the trained repetitionOFF control, proven equivalent to repeat architecture with max_repeat1 and an unused controller. S0/S2/S4 additionally evaluate the SAME learned checkpoint with fixed1/2/4/8 on4096 bytes: inference interventions, not retraining or hyperparameter selection.
| Scale | Fixed count | Loss | PPL | n |
|---|---:|---:|---:|---:|

Final17-question analysis pending until the planned jobs finish or documented resource limits are reached.

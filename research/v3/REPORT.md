# PRG-LM v3 — Same-Edge Repetition Scaling Study

Complete: False; 108/180 primary milestone results; 0 recorded failed/resource-limited jobs.
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
| float | S0 | 819200 | 819200 | 819200 | True |
| ternary | S0 | 819200 | 819200 | 819200 | True |
| repeat | S0 | 819200 | 819200 | 819200 | True |
| float | S1 | 819200 | 819200 | 819200 | True |
| ternary | S1 | 819200 | 819200 | 819200 | True |
| repeat | S1 | 819200 | 819200 | 819200 | True |
| float | S2 | 819200 | 819200 | 819200 | True |
| ternary | S2 | 819200 | 819200 | 819200 | True |
| repeat | S2 | 819200 | 819200 | 819200 | True |
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
| float | S0 | 12,800 | 4096 | 5.47274 ± 0.01814 | 238.13702 ± 4.30568 | 3 |
| float | S0 | 12,800 | 16384 | 5.47555 ± 0.02026 | 238.81522 ± 4.81976 | 3 |
| float | S0 | 51,200 | 4096 | 5.16368 ± 0.03683 | 174.88440 ± 6.37769 | 3 |
| float | S0 | 51,200 | 16384 | 5.17035 ± 0.03843 | 176.06316 ± 6.69575 | 3 |
| float | S0 | 204,800 | 4096 | 3.30587 ± 0.00623 | 27.27252 ± 0.17030 | 3 |
| float | S0 | 204,800 | 16384 | 3.35969 ± 0.00602 | 28.78049 ± 0.17340 | 3 |
| float | S0 | 819,200 | 4096 | 3.08353 ± 0.01534 | 21.83711 ± 0.33618 | 3 |
| float | S0 | 819,200 | 16384 | 3.10628 ± 0.01940 | 22.34068 ± 0.43583 | 3 |
| float | S1 | 12,800 | 4096 | 5.47249 ± 0.01807 | 238.07720 ± 4.29264 | 3 |
| float | S1 | 12,800 | 16384 | 5.47499 ± 0.01930 | 238.67717 ± 4.59847 | 3 |
| float | S1 | 51,200 | 4096 | 5.16432 ± 0.03606 | 174.99339 ± 6.25297 | 3 |
| float | S1 | 51,200 | 16384 | 5.17032 ± 0.03724 | 176.05277 ± 6.49326 | 3 |
| float | S1 | 204,800 | 4096 | 3.41607 ± 0.08305 | 30.52062 ± 2.59231 | 3 |
| float | S1 | 204,800 | 16384 | 3.46567 ± 0.08206 | 32.07082 ± 2.69020 | 3 |
| float | S1 | 819,200 | 4096 | 3.08262 ± 0.01217 | 21.81651 ± 0.26452 | 3 |
| float | S1 | 819,200 | 16384 | 3.09970 ± 0.00876 | 22.19181 ± 0.19399 | 3 |
| float | S2 | 12,800 | 4096 | 5.47305 ± 0.02030 | 238.21753 ± 4.81811 | 3 |
| float | S2 | 12,800 | 16384 | 5.47580 ± 0.01948 | 238.87123 ± 4.64100 | 3 |
| float | S2 | 51,200 | 4096 | 5.16549 ± 0.03950 | 175.21326 ± 6.85038 | 3 |
| float | S2 | 51,200 | 16384 | 5.17160 ± 0.03858 | 176.28251 ± 6.73309 | 3 |
| float | S2 | 204,800 | 4096 | 3.50565 ± 0.02945 | 33.31274 ± 0.97379 | 3 |
| float | S2 | 204,800 | 16384 | 3.55093 ± 0.02896 | 34.85539 ± 1.00176 | 3 |
| float | S2 | 819,200 | 4096 | 3.17757 ± 0.06327 | 24.02069 ± 1.54089 | 3 |
| float | S2 | 819,200 | 16384 | 3.22349 ± 0.06961 | 25.15666 ± 1.78015 | 3 |
| ternary | S0 | 12,800 | 4096 | 5.47929 ± 0.01728 | 239.69913 ± 4.12456 | 3 |
| ternary | S0 | 12,800 | 16384 | 5.48194 ± 0.01930 | 240.34264 ± 4.61983 | 3 |
| ternary | S0 | 51,200 | 4096 | 5.17522 ± 0.03569 | 176.90963 ± 6.25091 | 3 |
| ternary | S0 | 51,200 | 16384 | 5.18158 ± 0.03701 | 178.04528 ± 6.52272 | 3 |
| ternary | S0 | 204,800 | 4096 | 3.36362 ± 0.02898 | 28.90165 ± 0.83111 | 3 |
| ternary | S0 | 204,800 | 16384 | 3.41444 ± 0.02688 | 30.40706 ± 0.81081 | 3 |
| ternary | S0 | 819,200 | 4096 | 3.07744 ± 0.02638 | 21.70792 ± 0.57565 | 3 |
| ternary | S0 | 819,200 | 16384 | 3.09430 ± 0.03415 | 22.08045 ± 0.75919 | 3 |
| ternary | S1 | 12,800 | 4096 | 5.47777 ± 0.01816 | 239.33898 ± 4.33542 | 3 |
| ternary | S1 | 12,800 | 16384 | 5.48134 ± 0.01982 | 240.20019 ± 4.74647 | 3 |
| ternary | S1 | 51,200 | 4096 | 5.17497 ± 0.03600 | 176.86774 ± 6.30747 | 3 |
| ternary | S1 | 51,200 | 16384 | 5.18107 ± 0.03710 | 177.95419 ± 6.53710 | 3 |
| ternary | S1 | 204,800 | 4096 | 3.43531 ± 0.07421 | 31.09879 ± 2.35162 | 3 |
| ternary | S1 | 204,800 | 16384 | 3.48298 ± 0.07346 | 32.61608 ± 2.44031 | 3 |
| ternary | S1 | 819,200 | 4096 | 3.10743 ± 0.02721 | 22.36916 ± 0.61346 | 3 |
| ternary | S1 | 819,200 | 16384 | 3.12994 ± 0.04392 | 22.88755 ± 1.01733 | 3 |
| ternary | S2 | 12,800 | 4096 | 5.47856 ± 0.02116 | 239.53692 ± 5.04924 | 3 |
| ternary | S2 | 12,800 | 16384 | 5.48237 ± 0.02014 | 240.44790 ± 4.82895 | 3 |
| ternary | S2 | 51,200 | 4096 | 5.17622 ± 0.03968 | 177.10481 ± 6.95754 | 3 |
| ternary | S2 | 51,200 | 16384 | 5.18329 ± 0.03849 | 178.35507 ± 6.79892 | 3 |
| ternary | S2 | 204,800 | 4096 | 3.51603 ± 0.03077 | 33.66124 ± 1.02888 | 3 |
| ternary | S2 | 204,800 | 16384 | 3.56062 ± 0.02951 | 35.19516 ± 1.03145 | 3 |
| ternary | S2 | 819,200 | 4096 | 3.22646 ± 0.06150 | 25.22253 ± 1.57713 | 3 |
| ternary | S2 | 819,200 | 16384 | 3.27491 ± 0.06613 | 26.47987 ± 1.78282 | 3 |
| repeat | S0 | 12,800 | 4096 | 5.52413 ± 0.01691 | 250.69308 ± 4.22878 | 3 |
| repeat | S0 | 12,800 | 16384 | 5.52864 ± 0.02084 | 251.83833 ± 5.22105 | 3 |
| repeat | S0 | 51,200 | 4096 | 5.19844 ± 0.06085 | 181.21241 ± 11.02236 | 3 |
| repeat | S0 | 51,200 | 16384 | 5.20382 ± 0.05938 | 182.18006 ± 10.85428 | 3 |
| repeat | S0 | 204,800 | 4096 | 3.29717 ± 0.00448 | 27.03633 ± 0.12125 | 3 |
| repeat | S0 | 204,800 | 16384 | 3.35439 ± 0.00474 | 28.62825 ± 0.13558 | 3 |
| repeat | S0 | 819,200 | 4096 | 3.17739 ± 0.01477 | 23.98572 ± 0.35572 | 3 |
| repeat | S0 | 819,200 | 16384 | 3.21050 ± 0.01616 | 24.79366 ± 0.40238 | 3 |
| repeat | S1 | 12,800 | 4096 | 5.52316 ± 0.01677 | 250.44884 ± 4.19630 | 3 |
| repeat | S1 | 12,800 | 16384 | 5.52665 ± 0.01888 | 251.33007 ± 4.73286 | 3 |
| repeat | S1 | 51,200 | 4096 | 5.20389 ± 0.04605 | 182.10847 ± 8.45145 | 3 |
| repeat | S1 | 51,200 | 16384 | 5.21042 ± 0.04405 | 183.29096 ± 8.14538 | 3 |
| repeat | S1 | 204,800 | 4096 | 3.29069 ± 0.00795 | 26.86204 ± 0.21313 | 3 |
| repeat | S1 | 204,800 | 16384 | 3.34854 ± 0.00644 | 28.46146 ± 0.18322 | 3 |
| repeat | S1 | 819,200 | 4096 | 3.17991 ± 0.02787 | 24.05080 ± 0.67127 | 3 |
| repeat | S1 | 819,200 | 16384 | 3.21563 ± 0.03037 | 24.92667 ± 0.75698 | 3 |
| repeat | S2 | 12,800 | 4096 | 5.52554 ± 0.02232 | 251.06461 ± 5.58645 | 3 |
| repeat | S2 | 12,800 | 16384 | 5.52929 ± 0.02195 | 252.00478 ± 5.51082 | 3 |
| repeat | S2 | 51,200 | 4096 | 5.18464 ± 0.08629 | 178.95050 ± 15.33266 | 3 |
| repeat | S2 | 51,200 | 16384 | 5.19152 ± 0.08402 | 180.16382 ± 15.04944 | 3 |
| repeat | S2 | 204,800 | 4096 | 3.30284 ± 0.00492 | 27.18995 ± 0.13351 | 3 |
| repeat | S2 | 204,800 | 16384 | 3.35962 ± 0.00690 | 28.77880 ± 0.19898 | 3 |
| repeat | S2 | 819,200 | 4096 | 3.18661 ± 0.02769 | 24.21254 ± 0.67547 | 3 |
| repeat | S2 | 819,200 | 16384 | 3.22082 ± 0.02726 | 25.05500 ± 0.68678 | 3 |

## Final compute/repetition

| Family | Scale | Repeats/edge | ≥2 fraction | ≥4 fraction | Edge additions/token | Repetition compute fraction | Active fraction |
|---|---|---:|---:|---:|---:|---:|---:|
| float | S0 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 512.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.01562 ± 0.00000 |
| float | S1 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 512.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00391 ± 0.00000 |
| float | S2 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 512.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00098 ± 0.00000 |
| ternary | S0 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 512.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.01562 ± 0.00000 |
| ternary | S1 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 512.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00391 ± 0.00000 |
| ternary | S2 | 1.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 | 512.00000 ± 0.00000 | 0.00000 ± 0.00000 | 0.00098 ± 0.00000 |
| repeat | S0 | 1.59473 ± 0.09133 | 0.24618 ± 0.05751 | 0.08265 ± 0.01337 | 816.50106 ± 46.76219 | 0.37160 ± 0.03506 | 0.02492 ± 0.00143 |
| repeat | S1 | 1.71520 ± 0.11790 | 0.24526 ± 0.07362 | 0.10306 ± 0.01425 | 878.18441 ± 60.36343 | 0.41517 ± 0.03950 | 0.00670 ± 0.00046 |
| repeat | S2 | 1.71044 ± 0.14133 | 0.26403 ± 0.06252 | 0.10136 ± 0.02002 | 875.74764 ± 72.35882 | 0.41281 ± 0.04632 | 0.00167 ± 0.00014 |
All modeled active counts include zero ternary edges; nonzero operations are separately in JSON. Unique activated edges/token is512 by design. Consecutive revisits are counted within locked per-edge episodes, not different edges or new tokens. Repeated edges are interleaved parallel lanes; every lane retains identity/reference. Controller probabilities, full histograms, median, bound fraction, projected output contribution by count and correlations are in machine results.
Output contribution is a PRE-NONLINEARITY readout-column magnitude proxy; count increases its magnitude by construction. Positive correlation is not causal evidence of useful language-model computation.

## Same-edge-count paired gaps

| Control | Scale | Validation bytes | Repeat − control loss | Per-seed deltas |
|---|---|---:|---:|---|
| float | S0 | 4096 | 0.09385 ± 0.02574 | {'42': 0.11668145656585693, '43': 0.09892170131206512, '44': 0.06595687568187714} |
| ternary | S0 | 4096 | 0.09994 ± 0.03927 | {'42': 0.13865698873996735, '43': 0.10102854669094086, '44': 0.060138776898384094} |
| float | S1 | 4096 | 0.09729 ± 0.03878 | {'42': 0.14011552929878235, '43': 0.0645456463098526, '44': 0.08721236884593964} |
| ternary | S1 | 4096 | 0.07247 ± 0.01229 | {'42': 0.06988079845905304, '43': 0.06169264018535614, '44': 0.08584915101528168} |
| float | S2 | 4096 | 0.00905 ± 0.03606 | {'42': -0.03043125569820404, '43': 0.0402391254901886, '44': 0.017327740788459778} |
| ternary | S2 | 4096 | -0.03985 ± 0.07617 | {'42': 0.02316346764564514, '43': -0.018220603466033936, '44': -0.1244889497756958} |
| float | S0 | 16384 | 0.10422 ± 0.03034 | {'42': 0.13417574018239975, '43': 0.1049601174890995, '44': 0.07351652160286903} |
| ternary | S0 | 16384 | 0.11620 ± 0.04650 | {'42': 0.16284942254424095, '43': 0.11588728427886963, '44': 0.06985913217067719} |
| float | S1 | 16384 | 0.11593 ± 0.03867 | {'42': 0.15633563697338104, '43': 0.07926047593355179, '44': 0.1122034341096878} |
| ternary | S1 | 16384 | 0.08569 ± 0.02561 | {'42': 0.06554384157061577, '43': 0.07700004428625107, '44': 0.11451354995369911} |
| float | S2 | 16384 | -0.00266 ± 0.04249 | {'42': -0.05134975537657738, '43': 0.026943180710077286, '44': 0.016412563621997833} |
| ternary | S2 | 16384 | -0.05408 ± 0.07678 | {'42': 0.011130429804325104, '43': -0.034673675894737244, '44': -0.1387110948562622} |

## Nearest measured same-memory comparison

| Float scale | Repeat scale | Validation bytes | Static byte ratio | Edge ratio | Loss delta | Within1% budget |
|---|---|---:|---:|---:|---:|---|
| S0 | S2 | 4096 | 1.000000 | 16.00 | 0.10308 | True |
| S1 | S2 | 4096 | 0.343454 | 4.00 | 0.10400 | False |
| S2 | S2 | 4096 | 0.094715 | 1.00 | 0.00905 | False |
| S0 | S2 | 16384 | 1.000000 | 16.00 | 0.11454 | True |
| S1 | S2 | 16384 | 0.343454 | 4.00 | 0.12113 | False |
| S2 | S2 | 16384 | 0.094715 | 1.00 | -0.00266 | False |
Only within1% measured matches support a same-memory claim; unmatched nearest rows are explicitly not budget matched. No interpolation/extrapolation. FP16 float models were not trained and are not substituted into the measured table.

## Empirical scaling slopes

| Family | From | To | Validation bytes | Dimension | Loss/log-dimension slope |
|---|---|---|---:|---|---:|
| float | S0 | S1 | 4096 | edge_slots | -0.000660 |
| float | S0 | S1 | 4096 | packed_static_bytes | -0.000856 |
| float | S1 | S2 | 4096 | edge_slots | 0.068493 |
| float | S1 | S2 | 4096 | packed_static_bytes | 0.073709 |
| ternary | S0 | S1 | 4096 | edge_slots | 0.021634 |
| ternary | S0 | S1 | 4096 | packed_static_bytes | 0.115414 |
| ternary | S1 | S2 | 4096 | edge_slots | 0.085860 |
| ternary | S1 | S2 | 4096 | packed_static_bytes | 0.183149 |
| repeat | S0 | S1 | 4096 | edge_slots | 0.001820 |
| repeat | S0 | S1 | 4096 | packed_static_bytes | 0.009711 |
| repeat | S1 | S2 | 4096 | edge_slots | 0.004837 |
| repeat | S1 | S2 | 4096 | packed_static_bytes | 0.010317 |
| float | S0 | S1 | 16384 | edge_slots | -0.004750 |
| float | S0 | S1 | 16384 | packed_static_bytes | -0.006162 |
| float | S1 | S2 | 16384 | edge_slots | 0.089297 |
| float | S1 | S2 | 16384 | packed_static_bytes | 0.096098 |
| ternary | S0 | S1 | 16384 | edge_slots | 0.025711 |
| ternary | S0 | S1 | 16384 | packed_static_bytes | 0.137166 |
| ternary | S1 | S2 | 16384 | edge_slots | 0.104570 |
| ternary | S1 | S2 | 16384 | packed_static_bytes | 0.223058 |
| repeat | S0 | S1 | 16384 | edge_slots | 0.003701 |
| repeat | S0 | S1 | 16384 | packed_static_bytes | 0.019742 |
| repeat | S1 | S2 | 16384 | edge_slots | 0.003747 |
| repeat | S1 | S2 | 16384 | packed_static_bytes | 0.007992 |

## Memory and prototype runtime

| Family | Scale | Trainable FP params | Low-bit slots | Packed structure | Controller bytes | Embedding/head bytes | Static packed | Dynamic | Modeled peak | Export bytes | Resume bytes | Optimizer bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float | S0 | 51280 | 0 | 131072 | 516 | 65536 | 205700 | 5218 | 210918 | 210416.00000 ± 0.00000 | 812739.33333 ± 36.95042 | 410260 |
| float | S1 | 149584 | 0 | 524288 | 516 | 65536 | 598916 | 5218 | 604134 | 603632.00000 ± 0.00000 | 1992430.00000 ± 0.00000 | 1196692 |
| float | S2 | 542800 | 0 | 2097152 | 516 | 65536 | 2171780 | 5218 | 2176998 | 2176496.00000 ± 0.00000 | 6711022.00000 ± 0.00000 | 4342420 |
| ternary | S0 | 51280 | 32768 | 8192 | 516 | 65536 | 82820 | 5218 | 88038 | 87536.00000 ± 0.00000 | 812760.66667 ± 36.95042 | 410260 |
| ternary | S1 | 149584 | 131072 | 32768 | 516 | 65536 | 107396 | 5218 | 112614 | 112112.00000 ± 0.00000 | 1992430.00000 ± 0.00000 | 1196692 |
| ternary | S2 | 542800 | 524288 | 131072 | 516 | 65536 | 205700 | 5218 | 210918 | 210416.00000 ± 0.00000 | 6711022.00000 ± 0.00000 | 4342420 |
| repeat | S0 | 51409 | 32768 | 8192 | 516 | 65536 | 82820 | 5218 | 88038 | 87536.00000 ± 0.00000 | 817107.33333 ± 36.95042 | 411308 |
| repeat | S1 | 149713 | 131072 | 32768 | 516 | 65536 | 107396 | 5218 | 112614 | 112112.00000 ± 0.00000 | 1996755.33333 ± 36.95042 | 1197740 |
| repeat | S2 | 542929 | 524288 | 131072 | 516 | 65536 | 205700 | 5218 | 210918 | 210416.00000 ± 0.00000 | 6715347.33333 ± 36.95042 | 4343468 |

| Family | Scale | Train tok/s | Inference tok/s | Latency ms/token | Peak process RSS bytes |
|---|---|---:|---:|---:|---:|
| float | S0 | 9315.94308 ± 313.54531 | 722.32423 ± 96.16907 | 1.40217 ± 0.20014 | 294100992.00000 ± 90576.25861 |
| float | S1 | 9482.28540 ± 266.74030 | 784.31057 ± 6.25428 | 1.27506 ± 0.01017 | 297736874.66667 ± 171070.32796 |
| float | S2 | 9203.22578 ± 148.23242 | 703.66307 ± 56.83946 | 1.42717 ± 0.11216 | 308142080.00000 ± 258762.18629 |
| ternary | S0 | 8892.18892 ± 179.88945 | 603.20354 ± 58.68037 | 1.66894 ± 0.17152 | 294638933.33333 ± 59403.76895 |
| ternary | S1 | 8752.57277 ± 164.69970 | 545.49443 ± 145.78233 | 1.94251 ± 0.61356 | 298362197.33333 ± 408581.56053 |
| ternary | S2 | 8478.75013 ± 100.32038 | 619.92826 ± 11.61364 | 1.61347 ± 0.03023 | 309190656.00000 ± 720814.54085 |
| repeat | S0 | 1170.90495 ± 26.27557 | 135.90620 ± 3.66701 | 7.36165 ± 0.20177 | 398875306.66667 ± 1486217.85729 |
| repeat | S1 | 1166.55600 ± 7.29188 | 133.17699 ± 1.65809 | 7.50958 ± 0.09305 | 400594261.33333 ± 3938585.32814 |
| repeat | S2 | 1187.84085 ± 7.51767 | 136.37769 ± 3.96054 | 7.33675 ± 0.21536 | 403162453.33333 ± 2392408.81533 |
Timings are prototype measurements under the actual worker schedule. Peak process RSS includes interpreter/Torch and is cumulative per job process. CUDA stats are null on CPU. No GPU or custom-accelerator advantage is inferred from packed bytes or Python wall time.

## Representative repetition ablations

Ternary single-pass is the trained repetitionOFF control, proven equivalent to repeat architecture with max_repeat1 and an unused controller. S0/S2/S4 additionally evaluate the SAME learned checkpoint with fixed1/2/4/8 on4096 bytes: inference interventions, not retraining or hyperparameter selection.
| Scale | Fixed count | Loss | PPL | n |
|---|---:|---:|---:|---:|
| S0 | 1 | 4.20108 ± 0.23711 | 67.98118 ± 15.34929 | 3 |
| S0 | 2 | 4.30517 ± 0.27935 | 75.93172 ± 19.53363 | 3 |
| S0 | 4 | 4.43755 ± 0.30872 | 87.12972 ± 24.41241 | 3 |
| S0 | 8 | 4.53524 ± 0.31800 | 96.24189 ± 27.71511 | 3 |
| S2 | 1 | 4.21098 ± 0.30899 | 69.67652 ± 22.53876 | 3 |
| S2 | 2 | 4.34855 ± 0.34281 | 80.56076 ± 28.88492 | 3 |
| S2 | 4 | 4.48378 ± 0.37265 | 92.92026 ± 36.35231 | 3 |
| S2 | 8 | 4.57466 ± 0.38478 | 102.10217 ± 41.42335 | 3 |

Final17-question analysis pending until the planned jobs finish or documented resource limits are reached.

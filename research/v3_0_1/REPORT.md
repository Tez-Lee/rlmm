# PRG-LM v3.0.1 — Extended-Training Crossover Verification

Complete: False; completed extension milestones: 33/132; failed jobs: 0; resource-limited jobs: 0; latest completed training bytes: 1,638,400.

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
| float | S0 | 1638400 | 1638400 | 1638400 | False |
| float | S1 | 1638400 | 1638400 | 1638400 | False |
| float | S2 | 1638400 | 1638400 | 1638400 | False |
| float | S3 | 1638400 | 1638400 | 1638400 | False |
| float | S4 | 1638400 | 1638400 | 1638400 | False |
| ternary | S2 | 1638400 | 1638400 | 1638400 | False |
| ternary | S3 | 1638400 | 1638400 | 1638400 | False |
| ternary | S4 | 1638400 | 1638400 | 1638400 | False |
| repeat | S2 | 1638400 | 1638400 | 1638400 | False |
| repeat | S3 | 1638400 | 1638400 | 1638400 | False |
| repeat | S4 | 1638400 | 1638400 | 1638400 | False |

## Sanity checks

56 tests passed, including all previous tests. Tiny continuation matches an uninterrupted immutable v3 trainer bit-for-bit for weights, both optimizer states, RNG and history. Interrupted extension and milestone-stage barriers are also bit-exact. All33 original819200 checkpoints are present. Actual Float S0/seed42 checkpoint continuation smoke reached832000 bytes (12800 additional), preserving original files. Smoke is separate from primary results and did not select/tune settings.

## All validation results

| Family | Scale | Training bytes | Validation bytes | Loss | Byte PPL | n |
|---|---|---:|---:|---:|---:|---:|
| float | S0 | 819200 | 4096 | 3.08353 ± 0.01534 | 21.83711 ± 0.33618 | 3 |
| float | S0 | 1638400 | 4096 | 3.03635 ± 0.00366 | 20.82918 ± 0.07615 | 3 |
| float | S1 | 819200 | 4096 | 3.08262 ± 0.01217 | 21.81651 ± 0.26452 | 3 |
| float | S1 | 1638400 | 4096 | 3.04347 ± 0.00733 | 20.97825 ± 0.15410 | 3 |
| float | S2 | 819200 | 4096 | 3.17757 ± 0.06327 | 24.02069 ± 1.54089 | 3 |
| float | S2 | 1638400 | 4096 | 3.06673 ± 0.00936 | 21.47214 ± 0.20079 | 3 |
| float | S3 | 819200 | 4096 | 3.29907 ± 0.00160 | 27.08733 ± 0.04345 | 3 |
| float | S3 | 1638400 | 4096 | 3.21743 ± 0.06399 | 24.99820 ± 1.62184 | 3 |
| float | S4 | 819200 | 4096 | 3.29974 ± 0.00306 | 27.10579 ± 0.08279 | 3 |
| float | S4 | 1638400 | 4096 | 3.28770 ± 0.00269 | 26.78114 ± 0.07203 | 3 |
| ternary | S2 | 819200 | 4096 | 3.22646 ± 0.06150 | 25.22253 ± 1.57713 | 3 |
| ternary | S2 | 1638400 | 4096 | 3.08977 ± 0.03052 | 21.97884 ± 0.67629 | 3 |
| ternary | S3 | 819200 | 4096 | 3.30145 ± 0.00025 | 27.15205 ± 0.00677 | 3 |
| ternary | S3 | 1638400 | 4096 | 3.28184 ± 0.00921 | 26.62544 ± 0.24583 | 3 |
| ternary | S4 | 819200 | 4096 | 3.30185 ± 0.00300 | 27.16291 ± 0.08138 | 3 |
| ternary | S4 | 1638400 | 4096 | 3.29258 ± 0.00277 | 26.91216 ± 0.07458 | 3 |
| repeat | S2 | 819200 | 4096 | 3.18661 ± 0.02769 | 24.21254 ± 0.67547 | 3 |
| repeat | S2 | 1638400 | 4096 | 3.09249 ± 0.01617 | 22.03371 ± 0.35653 | 3 |
| repeat | S3 | 819200 | 4096 | 3.18766 ± 0.02304 | 24.23595 ± 0.56213 | 3 |
| repeat | S3 | 1638400 | 4096 | 3.09274 ± 0.01092 | 22.03816 ± 0.24138 | 3 |
| repeat | S4 | 819200 | 4096 | 3.19244 ± 0.02199 | 24.35171 ± 0.53825 | 3 |
| repeat | S4 | 1638400 | 4096 | 3.09874 ± 0.01071 | 22.17085 ± 0.23784 | 3 |
| float | S0 | 819200 | 16384 | 3.10628 ± 0.01940 | 22.34068 ± 0.43583 | 3 |
| float | S0 | 1638400 | 16384 | 3.04345 ± 0.00118 | 20.97747 ± 0.02483 | 3 |
| float | S1 | 819200 | 16384 | 3.09970 ± 0.00876 | 22.19181 ± 0.19399 | 3 |
| float | S1 | 1638400 | 16384 | 3.05187 ± 0.00840 | 21.15545 ± 0.17764 | 3 |
| float | S2 | 819200 | 16384 | 3.22349 ± 0.06961 | 25.15666 ± 1.78015 | 3 |
| float | S2 | 1638400 | 16384 | 3.08794 ± 0.00657 | 21.93222 ± 0.14435 | 3 |
| float | S3 | 819200 | 16384 | 3.35217 ± 0.00093 | 28.56468 ± 0.02671 | 3 |
| float | S3 | 1638400 | 16384 | 3.26690 ± 0.06715 | 26.26980 ± 1.78668 | 3 |
| float | S4 | 819200 | 16384 | 3.35255 ± 0.00143 | 28.57548 ± 0.04091 | 3 |
| float | S4 | 1638400 | 16384 | 3.33949 ± 0.00202 | 28.20485 ± 0.05692 | 3 |
| ternary | S2 | 819200 | 16384 | 3.27491 ± 0.06613 | 26.47987 ± 1.78282 | 3 |
| ternary | S2 | 1638400 | 16384 | 3.11485 ± 0.03628 | 22.54003 ± 0.82318 | 3 |
| ternary | S3 | 819200 | 16384 | 3.35423 ± 0.00114 | 28.62351 ± 0.03276 | 3 |
| ternary | S3 | 1638400 | 16384 | 3.33382 ± 0.00954 | 28.04612 ± 0.26790 | 3 |
| ternary | S4 | 819200 | 16384 | 3.35438 ± 0.00143 | 28.62774 ± 0.04081 | 3 |
| ternary | S4 | 1638400 | 16384 | 3.34424 ± 0.00126 | 28.33910 ± 0.03567 | 3 |
| repeat | S2 | 819200 | 16384 | 3.22082 ± 0.02726 | 25.05500 ± 0.68678 | 3 |
| repeat | S2 | 1638400 | 16384 | 3.10446 ± 0.01248 | 22.29831 ± 0.27750 | 3 |
| repeat | S3 | 819200 | 16384 | 3.22297 ± 0.02363 | 25.10731 ± 0.59593 | 3 |
| repeat | S3 | 1638400 | 16384 | 3.10512 ± 0.00918 | 22.31258 ± 0.20431 | 3 |
| repeat | S4 | 819200 | 16384 | 3.22980 ± 0.02578 | 25.28013 ± 0.65462 | 3 |
| repeat | S4 | 1638400 | 16384 | 3.10952 ± 0.00799 | 22.41084 ± 0.17894 | 3 |

## Paired comparisons

| Kind | Control | Repeat | Training bytes | Validation bytes | Paired Δloss | Each seed Δloss | Repeat better seeds | Packed ratio | Addition ratio |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| same_edge | float S2 | S2 | 819200 | 4096 | 0.00905 ± 0.03606 | {'42': -0.03043125569820404, '43': 0.0402391254901886, '44': 0.017327740788459778} | 1 | 0.094715 | 1.71044 ± 0.14133 |
| same_edge | float S2 | S2 | 1638400 | 4096 | 0.02576 ± 0.01346 | {'42': 0.0411495715379715, '43': 0.019963949918746948, '44': 0.016164451837539673} | 0 | 0.094715 | 1.63416 ± 0.04588 |
| same_edge | float S3 | S3 | 819200 | 4096 | -0.11141 ± 0.02216 | {'42': -0.08593869209289551, '43': -0.1262848824262619, '44': -0.12199458479881287} | 3 | 0.070767 | 1.71730 ± 0.15311 |
| same_edge | float S3 | S3 | 1638400 | 4096 | -0.12469 ± 0.05384 | {'42': -0.18425703048706055, '43': -0.11032307147979736, '44': -0.07949119806289673} | 3 | 0.070767 | 1.65640 ± 0.08075 |
| same_edge | float S4 | S4 | 819200 | 4096 | -0.10730 ± 0.02495 | {'42': -0.07898999750614166, '43': -0.12609504163265228, '44': -0.11682750284671783} | 3 | 0.064581 | 1.72301 ± 0.12876 |
| same_edge | float S4 | S4 | 1638400 | 4096 | -0.18896 ± 0.01339 | {'42': -0.17418108880519867, '43': -0.20029814541339874, '44': -0.19238723814487457} | 3 | 0.064581 | 1.68190 ± 0.09758 |
| same_memory | float S0 | S2 | 819200 | 4096 | 0.10308 ± 0.03498 | {'42': 0.14088594913482666, '43': 0.0964922308921814, '44': 0.07186663150787354} | 0 | 1.000000 | 1.71044 ± 0.14133 |
| same_memory | float S0 | S2 | 1638400 | 4096 | 0.05614 ± 0.01272 | {'42': 0.06971126794815063, '43': 0.04449988901615143, '44': 0.054197654128074646} | 0 | 1.000000 | 1.63416 ± 0.04588 |
| same_memory | float S1 | S3 | 819200 | 4096 | 0.10504 ± 0.03515 | {'42': 0.1456267088651657, '43': 0.08493557572364807, '44': 0.08456158638000488} | 0 | 1.000000 | 1.71730 ± 0.15311 |
| same_memory | float S1 | S3 | 1638400 | 4096 | 0.04927 ± 0.01588 | {'42': 0.06502459943294525, '43': 0.033269062638282776, '44': 0.049508750438690186} | 0 | 1.000000 | 1.65640 ± 0.08075 |
| same_memory | float S2 | S4 | 819200 | 4096 | 0.01487 ± 0.04127 | {'42': -0.03168964385986328, '43': 0.04696173965930939, '44': 0.029342010617256165} | 1 | 1.000015 | 1.72301 ± 0.12876 |
| same_memory | float S2 | S4 | 1638400 | 4096 | 0.03201 ± 0.01091 | {'42': 0.04255238175392151, '43': 0.0327104777097702, '44': 0.020775720477104187} | 0 | 1.000015 | 1.68190 ± 0.09758 |
| repeat_ternary | ternary S2 | S2 | 819200 | 4096 | -0.03985 ± 0.07617 | {'42': 0.02316346764564514, '43': -0.018220603466033936, '44': -0.1244889497756958} | 2 | 1.000000 | 1.71044 ± 0.14133 |
| repeat_ternary | ternary S2 | S2 | 1638400 | 4096 | 0.00272 ± 0.03679 | {'42': 0.04033386707305908, '43': 0.0010131150484085083, '44': -0.03319080173969269} | 1 | 1.000000 | 1.63416 ± 0.04588 |
| repeat_ternary | ternary S3 | S3 | 819200 | 4096 | -0.11379 ± 0.02280 | {'42': -0.08748193085193634, '43': -0.1277533620595932, '44': -0.1261444240808487} | 3 | 1.000000 | 1.71730 ± 0.15311 |
| repeat_ternary | ternary S3 | S3 | 1638400 | 4096 | -0.18910 ± 0.00371 | {'42': -0.18700125813484192, '43': -0.19339051842689514, '44': -0.18691778182983398} | 3 | 1.000000 | 1.65640 ± 0.08075 |
| repeat_ternary | ternary S4 | S4 | 819200 | 4096 | -0.10941 ± 0.02457 | {'42': -0.08137206733226776, '43': -0.1271793097257614, '44': -0.11967706680297852} | 3 | 1.000000 | 1.72301 ± 0.12876 |
| repeat_ternary | ternary S4 | S4 | 1638400 | 4096 | -0.19384 ± 0.01293 | {'42': -0.1791389286518097, '43': -0.20343317091464996, '44': -0.19893521070480347} | 3 | 1.000000 | 1.68190 ± 0.09758 |
| same_edge | float S2 | S2 | 819200 | 16384 | -0.00266 ± 0.04249 | {'42': -0.05134975537657738, '43': 0.026943180710077286, '44': 0.016412563621997833} | 1 | 0.094715 | 1.70233 ± 0.14093 |
| same_edge | float S2 | S2 | 1638400 | 16384 | 0.01652 ± 0.01154 | {'42': 0.02926023304462433, '43': 0.006768755614757538, '44': 0.013521376997232437} | 0 | 0.094715 | 1.62781 ± 0.04355 |
| same_edge | float S3 | S3 | 819200 | 16384 | -0.12920 ± 0.02275 | {'42': -0.10404785349965096, '43': -0.14833737909793854, '44': -0.13520938530564308} | 3 | 0.070767 | 1.70994 ± 0.15453 |
| same_edge | float S3 | S3 | 1638400 | 16384 | -0.16178 ± 0.06373 | {'42': -0.22963923960924149, '43': -0.15252140536904335, '44': -0.103182353079319} | 3 | 0.070767 | 1.64846 ± 0.07863 |
| same_edge | float S4 | S4 | 819200 | 16384 | -0.12275 ± 0.02640 | {'42': -0.09333153069019318, '43': -0.14436401426792145, '44': -0.13055985048413277} | 3 | 0.064581 | 1.71609 ± 0.13066 |
| same_edge | float S4 | S4 | 1638400 | 16384 | -0.22997 ± 0.00921 | {'42': -0.22327328100800514, '43': -0.24046996980905533, '44': -0.22616447508335114} | 3 | 0.064581 | 1.67398 ± 0.09588 |
| same_memory | float S0 | S2 | 819200 | 16384 | 0.11454 ± 0.03752 | {'42': 0.15634262934327126, '43': 0.10350795090198517, '44': 0.08377399295568466} | 0 | 1.000000 | 1.70233 ± 0.14093 |
| same_memory | float S0 | S2 | 1638400 | 16384 | 0.06101 ± 0.01176 | {'42': 0.06920654699206352, '43': 0.04753373935818672, '44': 0.06629155948758125} | 0 | 1.000000 | 1.62781 ± 0.04355 |
| same_memory | float S1 | S3 | 819200 | 16384 | 0.12327 ± 0.03234 | {'42': 0.15950176119804382, '43': 0.09731252864003181, '44': 0.11301052570343018} | 0 | 1.000000 | 1.70994 ± 0.15453 |
| same_memory | float S1 | S3 | 1638400 | 16384 | 0.05325 ± 0.01657 | {'42': 0.06049942970275879, '43': 0.03429148346185684, '44': 0.06495499238371849} | 0 | 1.000000 | 1.64846 ± 0.07863 |
| same_memory | float S2 | S4 | 819200 | 16384 | 0.00631 ± 0.04422 | {'42': -0.04452309012413025, '43': 0.03588606417179108, '44': 0.02755865454673767} | 1 | 1.000015 | 1.71609 ± 0.13066 |
| same_memory | float S2 | S4 | 1638400 | 16384 | 0.02158 ± 0.00890 | {'42': 0.031815458089113235, '43': 0.017270177602767944, '44': 0.01565823331475258} | 0 | 1.000015 | 1.67398 ± 0.09588 |
| repeat_ternary | ternary S2 | S2 | 819200 | 16384 | -0.05408 ± 0.07678 | {'42': 0.011130429804325104, '43': -0.034673675894737244, '44': -0.1387110948562622} | 2 | 1.000000 | 1.70233 ± 0.14093 |
| repeat_ternary | ternary S2 | S2 | 1638400 | 16384 | -0.01039 ± 0.03803 | {'42': 0.029694415628910065, '43': -0.014923378825187683, '44': -0.04595063254237175} | 2 | 1.000000 | 1.62781 ± 0.04355 |
| repeat_ternary | ternary S3 | S3 | 819200 | 16384 | -0.13126 ± 0.02328 | {'42': -0.10509925708174706, '43': -0.1497029960155487, '44': -0.13896410167217255} | 3 | 1.000000 | 1.70994 ± 0.15453 |
| repeat_ternary | ternary S3 | S3 | 1638400 | 16384 | -0.22870 ± 0.01074 | {'42': -0.23206106573343277, '43': -0.23734864592552185, '44': -0.2166820615530014} | 3 | 1.000000 | 1.64846 ± 0.07863 |
| repeat_ternary | ternary S4 | S4 | 819200 | 16384 | -0.12458 ± 0.02664 | {'42': -0.09493490681052208, '43': -0.1465255171060562, '44': -0.1322760470211506} | 3 | 1.000000 | 1.71609 ± 0.13066 |
| repeat_ternary | ternary S4 | S4 | 1638400 | 16384 | -0.23472 ± 0.00910 | {'42': -0.2271854504942894, '43': -0.24483058229088783, '44': -0.2321396879851818} | 3 | 1.000000 | 1.67398 ± 0.09588 |

## Undertraining diagnostics

Training coverage starts at819200 because v3 did not record whole-history coverage. Coalesced sparse-gradient source rows track exact distinct activated slots, including zero ternary values. Per-source update counts are optimizer-step touches, not nonzero updates. Gradient norms and edge changes sample first64 touched rows every100 steps; they are NOT unbiased whole-graph statistics. Full counters are training-only checkpoint state. Held-out coverage is exact fixed-window topology coverage. Visit coverage alone cannot establish sufficient convergence.

| Family | Scale | Seed | Training bytes | Train loss | Continuation covered fraction | Diagnostic unique edges | Diagnostic fraction | Gradient norm mean | Edge update mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| float | S0 | 42 | 1638400 | 3.08635 | 1.000000 | 32768 | 1.000000 | 0.47408 | 0.00005595 |
| float | S0 | 43 | 1638400 | 2.78558 | 1.000000 | 32768 | 1.000000 | 0.49546 | 0.00005615 |
| float | S0 | 44 | 1638400 | 3.08019 | 1.000000 | 32768 | 1.000000 | 0.46234 | 0.00005654 |
| float | S1 | 42 | 1638400 | 3.10944 | 1.000000 | 131072 | 1.000000 | 0.46953 | 0.00005888 |
| float | S1 | 43 | 1638400 | 2.80085 | 1.000000 | 131072 | 1.000000 | 0.34253 | 0.00005919 |
| float | S1 | 44 | 1638400 | 3.09442 | 1.000000 | 131072 | 1.000000 | 0.40053 | 0.00005871 |
| float | S2 | 42 | 1638400 | 3.16093 | 1.000000 | 512736 | 0.977966 | 0.42417 | 0.00009147 |
| float | S2 | 43 | 1638400 | 2.78914 | 1.000000 | 512736 | 0.977966 | 0.40117 | 0.00007660 |
| float | S2 | 44 | 1638400 | 3.00208 | 1.000000 | 512736 | 0.977966 | 0.38825 | 0.00007527 |
| repeat | S2 | 42 | 1638400 | 3.11997 | 1.000000 | 512736 | 0.977966 | 0.47007 | 0.00005160 |
| repeat | S2 | 43 | 1638400 | 2.87198 | 1.000000 | 512736 | 0.977966 | 0.48908 | 0.00005455 |
| repeat | S2 | 44 | 1638400 | 3.03891 | 1.000000 | 512736 | 0.977966 | 0.43651 | 0.00005784 |
| ternary | S2 | 42 | 1638400 | 3.15662 | 1.000000 | 512736 | 0.977966 | 0.46264 | 0.00008334 |
| ternary | S2 | 43 | 1638400 | 2.83688 | 1.000000 | 512736 | 0.977966 | 0.35241 | 0.00008092 |
| ternary | S2 | 44 | 1638400 | 3.00495 | 1.000000 | 512736 | 0.977966 | 0.31198 | 0.00009675 |
| float | S3 | 42 | 1638400 | 3.44333 | 1.000000 | 1290912 | 0.615555 | 0.23619 | 0.00008421 |
| float | S3 | 43 | 1638400 | 2.99713 | 1.000000 | 1290912 | 0.615555 | 0.24871 | 0.00011031 |
| float | S3 | 44 | 1638400 | 3.05288 | 1.000000 | 1290912 | 0.615555 | 0.28198 | 0.00011855 |
| repeat | S3 | 42 | 1638400 | 3.15074 | 1.000000 | 1290912 | 0.615555 | 0.51753 | 0.00005591 |
| repeat | S3 | 43 | 1638400 | 2.87014 | 1.000000 | 1290912 | 0.615555 | 0.48833 | 0.00005976 |
| repeat | S3 | 44 | 1638400 | 3.02597 | 1.000000 | 1290912 | 0.615555 | 0.43373 | 0.00006223 |
| ternary | S3 | 42 | 1638400 | 3.44858 | 1.000000 | 1290912 | 0.615555 | 0.23822 | 0.00007995 |
| ternary | S3 | 43 | 1638400 | 3.09382 | 1.000000 | 1290912 | 0.615555 | 0.23935 | 0.00008783 |
| ternary | S3 | 44 | 1638400 | 3.15752 | 1.000000 | 1290912 | 0.615555 | 0.24677 | 0.00009297 |
| float | S4 | 42 | 1638400 | 3.46089 | 1.000000 | 1791872 | 0.213608 | 0.23954 | 0.00010357 |
| float | S4 | 43 | 1638400 | 3.11069 | 1.000000 | 1791872 | 0.213608 | 0.23561 | 0.00010500 |
| float | S4 | 44 | 1638400 | 3.15461 | 1.000000 | 1791872 | 0.213608 | 0.24620 | 0.00010909 |
| repeat | S4 | 42 | 1638400 | 3.10440 | 1.000000 | 1791872 | 0.213608 | 0.52791 | 0.00006380 |
| repeat | S4 | 43 | 1638400 | 2.84369 | 1.000000 | 1791872 | 0.213608 | 0.50293 | 0.00006869 |
| repeat | S4 | 44 | 1638400 | 3.07973 | 1.000000 | 1791872 | 0.213608 | 0.42736 | 0.00007059 |
| ternary | S4 | 42 | 1638400 | 3.46233 | 1.000000 | 1791872 | 0.213608 | 0.24613 | 0.00009389 |
| ternary | S4 | 43 | 1638400 | 3.11813 | 1.000000 | 1791872 | 0.213608 | 0.23781 | 0.00009685 |
| ternary | S4 | 44 | 1638400 | 3.16355 | 1.000000 | 1791872 | 0.213608 | 0.24407 | 0.00009830 |

## Improvement slopes

| Family | Scale | Validation bytes | From | To | Loss improvement | Δloss/Δlog bytes |
|---|---|---:|---:|---:|---:|---:|
| float | S0 | 4096 | 819200 | 1638400 | 0.047182 | -0.068069 |
| float | S1 | 4096 | 819200 | 1638400 | 0.039150 | -0.056481 |
| float | S2 | 4096 | 819200 | 1638400 | 0.110842 | -0.159911 |
| float | S3 | 4096 | 819200 | 1638400 | 0.081639 | -0.117780 |
| float | S4 | 4096 | 819200 | 1638400 | 0.012049 | -0.017383 |
| ternary | S2 | 4096 | 819200 | 1638400 | 0.136695 | -0.197209 |
| ternary | S3 | 4096 | 819200 | 1638400 | 0.019613 | -0.028296 |
| ternary | S4 | 4096 | 819200 | 1638400 | 0.009274 | -0.013379 |
| repeat | S2 | 4096 | 819200 | 1638400 | 0.094127 | -0.135797 |
| repeat | S3 | 4096 | 819200 | 1638400 | 0.094923 | -0.136945 |
| repeat | S4 | 4096 | 819200 | 1638400 | 0.093700 | -0.135181 |
| float | S0 | 16384 | 819200 | 1638400 | 0.062835 | -0.090651 |
| float | S1 | 16384 | 819200 | 1638400 | 0.047823 | -0.068995 |
| float | S2 | 16384 | 819200 | 1638400 | 0.135547 | -0.195553 |
| float | S3 | 16384 | 819200 | 1638400 | 0.085267 | -0.123014 |
| float | S4 | 16384 | 819200 | 1638400 | 0.013056 | -0.018835 |
| ternary | S2 | 16384 | 819200 | 1638400 | 0.160057 | -0.230914 |
| ternary | S3 | 16384 | 819200 | 1638400 | 0.020408 | -0.029442 |
| ternary | S4 | 16384 | 819200 | 1638400 | 0.010133 | -0.014619 |
| repeat | S2 | 16384 | 819200 | 1638400 | 0.116366 | -0.167880 |
| repeat | S3 | 16384 | 819200 | 1638400 | 0.117850 | -0.170021 |
| repeat | S4 | 16384 | 819200 | 1638400 | 0.120273 | -0.173517 |

## Repetition and runtime

Full histograms, medians, ≥2/≥4/max fractions, controller probabilities, additions/unique edges/token, repetition fractions, memory, gradient diagnostics, source/checkpoint/stream hashes, train time/throughput, inference throughput/latency and RSS are preserved in results.json at every milestone. CUDA fields are null on this CPU backend. MPS memory is unavailable here. No custom accelerator or GPU performance claim.

## Failures and limits

[]
One corpus and three seeds restrict generality. Longer fixed-lr training may still not establish convergence. Hashed addresses and16-channel state constrain usable capacity; ST gradients are biased. Repeat contribution proxies are partly definitional. Measured memory savings do not imply lower compute/energy. Frozen original files/checkpoints are hashed and reverified.

Final19-question decision pending until all planned jobs finish or recorded resource bounds are reached.

# Mutable-topology study

Complete: False; remaining v2 checkpoints: 39.
All primary summaries use seeds42/43/44. Static control reuses completed Phase A, not retraining.

| Model | Training tokens | Loss | PPL | Static bytes | Peak bytes | Active edges/token | Recurrence | Topology mode |
|---|---:|---:|---:|---:|---:|---:|---|---|
| prg_v1 (n=3) | 12,800 | 5.3999 ± 0.2349 | 225.3951 ± 50.9176 | 220,860 | 483,398 | 1544.0000 ± 71.1056 | ON | static |
| prg_v1 (n=3) | 51,200 | 3.8077 ± 0.0303 | 45.0611 ± 1.3720 | 220,860 | 483,398 | 613.3333 ± 53.2666 | ON | static |
| prg_v1 (n=3) | 204,800 | 2.7902 ± 0.0160 | 16.2864 ± 0.2598 | 220,860 | 483,398 | 525.3333 ± 4.6188 | ON | static |
| prg_v1 (n=3) | 819,200 | 2.5755 ± 0.0094 | 13.1382 ± 0.1235 | 220,860 | 483,398 | 520.0000 ± 13.8564 | ON | static |
| prg_v2_adaptive (n=3) | 12,800 | 5.4023 ± 0.2384 | 226.0497 ± 51.9002 | 220,860 | 483,398 | 1541.3333 ± 68.0392 | ON | adaptive |
| prg_v2_adaptive (n=3) | 51,200 | 3.8077 ± 0.0362 | 45.0686 ± 1.6475 | 220,860 | 483,398 | 584.0000 ± 41.5692 | ON | adaptive |
| prg_v2_adaptive (n=3) | 204,800 | 2.7928 ± 0.0102 | 16.3275 ± 0.1659 | 220,860 | 483,398 | 517.3333 ± 9.2376 | ON | adaptive |
| transformer_core (n=3) | 12,800 | 4.7935 ± 0.0673 | 120.9070 ± 8.1623 | 193,664 | 218,240 | N/A | N/A | static |
| transformer_core (n=3) | 51,200 | 4.0217 ± 0.0619 | 55.8675 ± 3.4534 | 193,664 | 218,240 | N/A | N/A | static |
| transformer_core (n=3) | 204,800 | 2.8254 ± 0.0579 | 16.8866 ± 0.9802 | 193,664 | 218,240 | N/A | N/A | static |
| transformer_core (n=3) | 819,200 | 2.3907 ± 0.0074 | 10.9212 ± 0.0809 | 193,664 | 218,240 | N/A | N/A | static |
| transformer_total (n=3) | 12,800 | 4.0455 ± 0.0703 | 57.2319 ± 3.9397 | 451,248 | 472,752 | N/A | N/A | static |
| transformer_total (n=3) | 51,200 | 3.1906 ± 0.0308 | 24.3101 ± 0.7425 | 451,248 | 472,752 | N/A | N/A | static |
| transformer_total (n=3) | 204,800 | 2.5413 ± 0.0088 | 12.6965 ± 0.1120 | 451,248 | 472,752 | N/A | N/A | static |
| transformer_total (n=3) | 819,200 | 2.1793 ± 0.0102 | 8.8403 ± 0.0897 | 451,248 | 472,752 | N/A | N/A | static |

## Learning curve: loss (4096-byte sampled)

| Training tokens | Transformer core | Transformer peak | PRG-v1 | PRG-v2 adaptive |
|---:|---:|---:|---:|---:|
| 12,800 | 4.7935 ± 0.0673 | 4.0455 ± 0.0703 | 5.3999 ± 0.2349 | 5.4023 ± 0.2384 |
| 51,200 | 4.0217 ± 0.0619 | 3.1906 ± 0.0308 | 3.8077 ± 0.0303 | 3.8077 ± 0.0362 |
| 204,800 | 2.8254 ± 0.0579 | 2.5413 ± 0.0088 | 2.7902 ± 0.0160 | 2.7928 ± 0.0102 |
| 819,200 | 2.3907 ± 0.0074 | 2.1793 ± 0.0102 | 2.5755 ± 0.0094 | pending |

## Learning curve: ppl (4096-byte sampled)

| Training tokens | Transformer core | Transformer peak | PRG-v1 | PRG-v2 adaptive |
|---:|---:|---:|---:|---:|
| 12,800 | 120.9070 ± 8.1623 | 57.2319 ± 3.9397 | 225.3951 ± 50.9176 | 226.0497 ± 51.9002 |
| 51,200 | 55.8675 ± 3.4534 | 24.3101 ± 0.7425 | 45.0611 ± 1.3720 | 45.0686 ± 1.6475 |
| 204,800 | 16.8866 ± 0.9802 | 12.6965 ± 0.1120 | 16.2864 ± 0.2598 | 16.3275 ± 0.1659 |
| 819,200 | 10.9212 ± 0.0809 | 8.8403 ± 0.0897 | 13.1382 ± 0.1235 | pending |

## Topology evolution

| Topology mode | Training tokens | Rewires | Reverts | Exploration % | Gateway entropy | Largest hub | PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| prg_v2_adaptive (n=3) | 12,800 | 6.0000 ± 2.0000 | 0.0000 ± 0.0000 | 12.5000 ± 12.5000 | 3.4586 ± 0.0010 | 5.3333 ± 0.5774 | 226.0497 ± 51.9002 |
| prg_v2_adaptive (n=3) | 51,200 | 50.0000 ± 2.0000 | 0.0000 ± 0.0000 | 7.9551 ± 1.6831 | 3.4255 ± 0.0011 | 6.3333 ± 0.5774 | 45.0686 ± 1.6475 |
| prg_v2_adaptive (n=3) | 204,800 | 221.6667 ± 3.5119 | 59.0000 ± 3.6056 | 8.8704 ± 0.9051 | 3.4178 ± 0.0133 | 7.0000 ± 1.0000 | 16.3275 ± 0.1659 |

## Paired controls

| Training tokens | Validation bytes | Routing | Treatment − control | Loss delta | PPL ratio | n |
|---:|---:|---|---|---:|---:|---:|
| 12,800 | 4096 | sampled | prg_v2_adaptive − prg_v1 | 0.0023 ± 0.0043 | 1.0024 ± 0.0043 | 3 |
| 12,800 | 4096 | argmax | prg_v2_adaptive − prg_v1 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 3 |
| 12,800 | 4096 | expected | prg_v2_adaptive − prg_v1 | 0.0006 ± 0.0020 | 1.0006 ± 0.0020 | 3 |
| 12,800 | 16384 | sampled | prg_v2_adaptive − prg_v1 | 0.0007 ± 0.0005 | 1.0007 ± 0.0005 | 3 |
| 12,800 | 16384 | argmax | prg_v2_adaptive − prg_v1 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 3 |
| 12,800 | 16384 | expected | prg_v2_adaptive − prg_v1 | 0.0010 ± 0.0026 | 1.0010 ± 0.0026 | 3 |
| 51,200 | 4096 | sampled | prg_v2_adaptive − prg_v1 | 0.0000 ± 0.0122 | 1.0001 ± 0.0121 | 3 |
| 51,200 | 4096 | argmax | prg_v2_adaptive − prg_v1 | -0.0087 ± 0.0088 | 0.9914 ± 0.0087 | 3 |
| 51,200 | 4096 | expected | prg_v2_adaptive − prg_v1 | -0.0108 ± 0.0126 | 0.9894 ± 0.0124 | 3 |
| 51,200 | 16384 | sampled | prg_v2_adaptive − prg_v1 | -0.0061 ± 0.0018 | 0.9939 ± 0.0018 | 3 |
| 51,200 | 16384 | argmax | prg_v2_adaptive − prg_v1 | -0.0077 ± 0.0038 | 0.9923 ± 0.0037 | 3 |
| 51,200 | 16384 | expected | prg_v2_adaptive − prg_v1 | -0.0090 ± 0.0078 | 0.9911 ± 0.0078 | 3 |
| 204,800 | 4096 | sampled | prg_v2_adaptive − prg_v1 | 0.0026 ± 0.0073 | 1.0026 ± 0.0073 | 3 |
| 204,800 | 4096 | argmax | prg_v2_adaptive − prg_v1 | 0.0013 ± 0.0062 | 1.0013 ± 0.0062 | 3 |
| 204,800 | 4096 | expected | prg_v2_adaptive − prg_v1 | 0.0083 ± 0.0167 | 1.0085 ± 0.0168 | 3 |
| 204,800 | 16384 | sampled | prg_v2_adaptive − prg_v1 | 0.0075 ± 0.0017 | 1.0075 ± 0.0017 | 3 |
| 204,800 | 16384 | argmax | prg_v2_adaptive − prg_v1 | 0.0066 ± 0.0012 | 1.0066 ± 0.0012 | 3 |
| 204,800 | 16384 | expected | prg_v2_adaptive − prg_v1 | 0.0138 ± 0.0098 | 1.0139 ± 0.0099 | 3 |

## Inference and training metadata

The local-edge/gateway/router/embedding/runtime breakdown is recorded in every result.memory.
EMA, mutation RNG, counters, ever-edge table, probation and histories are training-only.
Inference exports discard these buffers, retaining the gateway table and neural state.
Full resumable checkpoints retain all mutation state; their byte size is separate from model-only exports.
Active edges/token counts processed edge slots; effective nonzero edges are separately recorded.
Region hotness/output credit is a proxy, not a causal usefulness measurement. Gradient credit is optional and disabled in the main experiment.
RecurrenceOFF inherits the v1 ascending-region-ID mask; it additionally alters the usable gateway graph.
Three seeds describe this dataset/configuration; no unobserved asymptotic claim is warranted.

Ten-question final assessment is pending until all three-seed controls complete.

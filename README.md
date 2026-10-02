# PRG-LM feasibility experiment

This is a small **causal next-token language model** experiment. It tests whether a shared, ternary local stencil plus probabilistic region revisits can replace some numerical connection strength. It does not establish that claim in advance.

## Architecture

- **Transformer:** ordinary causal encoder blocks with tied token head.
- **Looped Transformer:** the same block is reused `layers` times.
- **PRG-LM:** token embedding proposes multiple initial regions with sigmoid activation strengths. Each region holds a vector of `region_size` local nodes. A shared 3-offset stencil updates nodes; its train-time floating parameters are ternarized with a straight-through estimator. The shared RouterNet emits `LOCAL`, ring `NEIGHBOR`, recorded-origin `RETURN`, and `OUTPUT` probabilities. A small origin tensor tracks where each route arrived from. Training propagates the expected route mass. Inference samples categorical actions with a fixed seed, or chooses the highest-probability action deterministically. Fatigue is a separate recurrence penalty and recovers when unused. The maximum cycle count guarantees termination.

Ternary edges include zero, allowing removal of a local connection. Relative offsets require no arbitrary sparse global addresses. The theoretical packed size counts **only the local edge stencil at 2 bits each**; embeddings, routers and head remain float32. PyTorch serialization stores all trainable tensors as float32. Thus the first implementation does not yet show a meaningful compressed model memory advantage. Training uses a hard top-k forward pass with a sigmoid straight-through gradient for initial region selection. This surrogate and soft route expectation differ from sampled inference and need ablation.

The first corpus is public [tiny Shakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare), a fallback because TinyStories is not locally available. A byte-level BPE vocabulary of 2048 is trained on the training split alone. Byte-level BPE handles arbitrary text and is reused by all three models. The first 90% of the file is training data and the last 10% validation data; sequential literary text can create a distribution shift. Generated text from smoke training is not expected to be coherent.

## Install and run

```bash
python -m pip install -e '.[test]'
mkdir -p data/raw
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/raw/tinyshakespeare.txt
python -m pytest -q
python -m prglm.experiment --steps 8 --batch 4 --context 16 --regions 4 --region-size 32 --cycles 3
PRGLM_RUN=runs/smoke uvicorn web.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. The dashboard exposes paired generation, sampled token-level trajectories, region activation/fatigue/probabilities, benchmarks, repeats, and temporary intervention controls. `docs/` contains the GitHub Pages frontend, a benchmark export, and a saved generation replay. Export it with `python -m prglm.export_dashboard --run runs/final_seed42 --dest docs`. The workflow in `.github/workflows/pages.yml` deploys `docs/` from `main` after GitHub Pages is enabled for GitHub Actions. **Live PyTorch generation requires the local API** because GitHub Pages cannot execute Python or PyTorch server code. Set `localStorage.prglmApi` in the browser to an accessible FastAPI origin when using a remotely hosted static page; the default is the page origin.

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

A 139,216-parameter Transformer, within 3.3% of PRG-LM's parameter count and 4.2% of its serialized size, obtained PPL 931 in a seed-42 size-matched run. The PRG-LM result is substantially worse under this small budget. Only 192 local edge values can be packed to 2 bits in this configuration; the memory saving is negligible relative to float32 embeddings and routers. These data do not support the proposed memory advantage. CPU timing includes Python region loops and is not a measurement of optimized event-driven hardware.

At 500 updates with the same batch and context, seed 42 reached validation PPL 478 (Transformer), 527 (looped), and 1,218 (PRG-LM). The gap narrows with more training but remains large. Seed-42, 100-update PRG ablations gave PPL 2,947 with fatigue off, 2,698 with recurrence off, and 2,813 with float edges, versus 2,853 for the default. These ablations are exploratory single-seed results; recurrence-off improving perplexity is an early warning for the core hypothesis.

The largest architectural risk is the train/inference difference: training uses expected route mass and a straight-through initial top-k, while inference samples discrete routes. Route effects may also be too weak relative to the token embedding and head. Next experiments should repeat the recurrence-off result across seeds, compare expected-route and sampled validation loss, and sweep cycle limits at matched model size and token budget.

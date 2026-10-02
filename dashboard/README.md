# Local dashboard bundle

This folder contains the preserved PRG-v0 benchmark and a separate PRG-v1 byte-vocabulary feasibility comparison, plus saved Transformer/PRG generation with token-level Region trajectories. It works without GitHub Pages or a Python ML installation.

From the directory containing this folder, run:

```bash
python -m http.server 8001
```

Open `http://127.0.0.1:8001/dashboard/`. The saved samples, timeline, graph, and benchmark charts are available offline. The PRG-v1 section shows the memory breakdown, 256-byte comparison, actual traversal, accumulator magnitude, and sampled/argmax/soft routing validation results. To generate new text, train the models and start the FastAPI server as described in the repository's main README. The static bundle alone cannot run PyTorch inference.

# Local dashboard bundle

This folder contains the preserved PRG-v0 benchmark and a separate PRG-v1 byte-vocabulary feasibility comparison, plus saved Transformer/PRG generation with token-level Region trajectories. It works without GitHub Pages or a Python ML installation.

From the directory containing this folder, run:

```bash
python -m http.server 8001
```

Open `http://127.0.0.1:8001/dashboard/`. The saved samples, timeline, graph, and benchmark charts are available offline. The PRG-v1 section shows the memory breakdown, 256-byte comparison, actual traversal, accumulator magnitude, and sampled/argmax/soft routing validation results. To generate new text, start the FastAPI server as described in the repository's main README. The static bundle alone cannot run PyTorch inference.

Phase A/v2 additions show completed continuous learning curves and mutable
Region topology snapshots. Choose checkpoint seed/milestone, token and cycle to
inspect actual paths. Green edges were added and dashed red edges removed since
the preceding milestone. The full static JSON preserves rewiring history; the
UI shows its last150 records. The live v2 Generate button needs the local FastAPI
backend. The repository includes seed42 final models in
`checkpoints/byte256_seed42_819200/`; these work without training. Other
seeds/milestones require completed checkpoints under `runs/phase_a` /
`runs/mutable_v2`. All84 research milestone results are attached as static data;
the ZIP includes replay data, not model weights or the Python backend.

# Local dashboard bundle

This folder contains the benchmark export and a saved Transformer/PRG-LM generation with token-level region trajectories. It works without GitHub Pages or a Python ML installation.

From the directory containing this folder, run:

```bash
python -m http.server 8001
```

Open `http://127.0.0.1:8001/dashboard/`. The saved sample, timeline, graph, and benchmark charts are available offline. To generate new text, train the models and start the FastAPI server as described in the repository's main README. The static bundle alone cannot run PyTorch inference.

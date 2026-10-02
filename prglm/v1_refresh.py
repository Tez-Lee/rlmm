"""Recompute trajectory diagnostics from saved v1 checkpoints without retraining."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .models import backend
from .v1_config import V1Config
from .v1_model import PRGLMv1
from .v1_experiment import v1_diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    root=Path(a.run);file=root/'results.json';data=json.loads(file.read_text())
    val=np.fromfile(root/'val.bin',dtype=np.uint8)
    for row in data['results']:
        if row['model']!='prg_v1':continue
        checkpoint=torch.load(root/f"seed{row['seed']}/prg_v1.pt",map_location='cpu',weights_only=False)
        model=PRGLMv1(V1Config(**checkpoint['config']))
        model.load_state_dict(checkpoint['state'])
        model.to(backend()).eval()
        row['dynamics']=v1_diagnostics(model,val,backend(),row['config']['context'],row['seed'])
    file.write_text(json.dumps(data,indent=2))


if __name__=='__main__':main()

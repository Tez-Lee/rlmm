"""Load slim inference exports without training mutation metadata."""
from pathlib import Path
import torch
from .byte_data import ByteTokenizer
from .models import backend


def load_checkpoint(path):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=False)
    name=checkpoint['name']
    if name.startswith('prg_v2_'):
        from .v2_config import V2Config
        from .v2_model import PRGLMv2
        model=PRGLMv2(V2Config(**checkpoint['config']))
        incompatible=model.load_state_dict(checkpoint['state'],strict=False)
        if incompatible.unexpected_keys or any(not k.startswith('topology.') for k in incompatible.missing_keys):
            raise ValueError('invalid inference state keys')
        model.topology=None  # Inference retains no EMA, probation or mutation RNG buffers.
    elif name=='prg_v1':
        from .v1_model import PRGLMv1
        from .v1_config import V1Config
        model=PRGLMv1(V1Config(**checkpoint['config']))
        model.load_state_dict(checkpoint['state'])
    else:
        from .config import Config
        from .models import TransformerLM
        model=TransformerLM(Config(**checkpoint['config']))
        model.load_state_dict(checkpoint['state'])
    return model.to(backend()).eval(),ByteTokenizer()

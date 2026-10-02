import time
import torch
from .config import Config
from .models import make_model,backend
from .data import load_tokenizer


def load(run_dir,name):
    from pathlib import Path
    import json
    root=Path(run_dir); checkpoint=torch.load(root/f'{name}.pt',map_location='cpu',weights_only=False)
    if name=='prg_v1':
        from .v1_config import V1Config
        c=V1Config(**checkpoint['config'])
    else:
        c=Config(**checkpoint['config'])
    model=make_model(name,c)
    model.load_state_dict(checkpoint['state']); model.to(backend()).eval()
    if json.loads((root/'tokenizer.json').read_text()).get('type')=='utf8-byte':
        from .byte_data import ByteTokenizer
        tokenizer=ByteTokenizer()
    else:
        tokenizer=load_tokenizer(root/'tokenizer.json')
    return model,tokenizer


@torch.no_grad()
def generate(model,tok,prompt,max_tokens=20,temperature=1.,seed=42,record_trace=True,**settings):
    ids=tok.encode(prompt).ids or [0]
    device=next(model.parameters()).device
    gen=torch.Generator(device=device); gen.manual_seed(seed)
    timeline=[]; began=time.perf_counter()
    for i in range(max_tokens):
        x=torch.tensor([ids[-model.c.context:]],device=device)
        kwargs={k:v for k,v in settings.items() if v is not None}
        if model.__class__.__name__ in ('PRGLM','PRGLMv1'): kwargs['seed']=seed+i
        logits,traces=model(x,trace=record_trace,**kwargs)
        probs=(logits[0,-1]/max(temperature,1e-4)).softmax(-1)
        next_id=int(torch.multinomial(probs,1,generator=gen))
        ids.append(next_id)
        timeline.append({'token':tok.decode([next_id]),'id':next_id,'steps':traces[-1] if traces else []})
    sec=time.perf_counter()-began
    return {'text':tok.decode(ids),'completion':tok.decode(ids[-max_tokens:]),'timeline':timeline,
            'latency_seconds':sec,'tokens_per_second':max_tokens/sec if sec else None}

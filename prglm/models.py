import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config
from .routing import route_step


def backend():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


class TransformerLM(nn.Module):
    def __init__(self, c: Config, looped=False, low_bit=False):
        super().__init__()
        self.c, self.looped, self.low_bit = c, looped, low_bit
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.context, c.d_model)
        def block():
            return nn.TransformerEncoderLayer(c.d_model, c.heads, 4*c.d_model, dropout=0, batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.ModuleList([block() for _ in range(1 if looped else c.layers)])
        self.norm = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, x, trace=False, **_):
        b,t=x.shape
        assert t <= self.c.context
        h=self.emb(x)+self.pos(torch.arange(t,device=x.device))
        mask=torch.triu(torch.ones(t,t,device=x.device,dtype=torch.bool),1)
        for i in range(self.c.layers):
            h=self.blocks[0 if self.looped else i](h,src_mask=mask)
        return self.head(self.norm(h))/math.sqrt(self.c.d_model), None


class Fatigue(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
    def forward(self,old,usage,enabled=True,recovery=None):
        if not enabled: return torch.zeros_like(old)
        return (self.c.recovery if recovery is None else recovery)*old+usage


def ste_ternary(x):
    q=(x > .33).to(x.dtype)-(x < -.33).to(x.dtype)
    return x+(q-x).detach()


class PRGLM(nn.Module):
    """Shared local stencil; soft expectation in training, sampled paths in inference."""
    def __init__(self,c: Config):
        super().__init__(); self.c=c.validate()
        r,n=c.regions,c.region_size
        self.emb=nn.Embedding(c.vocab_size,n)
        self.region_emb=nn.Parameter(torch.randn(r,n)*.02)
        self.global_router=nn.Linear(n,r)
        self.local_edge=nn.Parameter(torch.randn(3,n)*.2)
        self.local_bias=nn.Parameter(torch.zeros(n))
        self.router=nn.Sequential(nn.Linear(n+4,32),nn.Tanh(),nn.Linear(32,4))
        self.fatigue_module=Fatigue(c)
        self.norm=nn.LayerNorm(n)
        self.head=nn.Linear(n,c.vocab_size,bias=False)
        self.head.weight=self.emb.weight

    def _local(self,s,token):
        w=ste_ternary(self.local_edge) if self.c.edge_type=="ternary" else self.local_edge
        mix=sum(torch.roll(s,k,dims=-1)*w[i] for i,k in enumerate((-1,0,1)))/3
        return torch.tanh(mix+token[:,None,:]+self.region_emb[None,:,:]+self.local_bias)

    def forward(self,x,trace=False,stochastic=None,recurrence=None,fatigue=None,max_cycles=None,
                forced_region=None,recurrence_override=None,fatigue_strength=None,recovery=None,seed=None):
        c=self.c; b,t=x.shape; r,n=c.regions,c.region_size
        cycles=max_cycles or c.max_cycles
        stochastic=(not self.training) if stochastic is None else stochastic
        recurrence=c.stochastic_recurrence if recurrence is None else recurrence
        fatigue=c.fatigue if fatigue is None else fatigue
        fs=c.fatigue_strength if fatigue_strength is None else fatigue_strength
        state=x.new_zeros((b,r,n),dtype=torch.float32)
        tired=x.new_zeros((b,r),dtype=torch.float32)
        logits_out=[]; traces=[]
        generator=None
        if seed is not None:
            generator=torch.Generator(device=x.device); generator.manual_seed(seed)
        for ti in range(t):
            token=self.emb(x[:,ti])
            gl=self.global_router(token)
            if forced_region is not None: gl[:,int(forced_region)]+=10
            if c.initial_regions==1:
                initial=F.one_hot(gl.argmax(-1),r).float()
            else:
                idx=gl.topk(c.initial_regions,dim=-1).indices
                initial=torch.zeros_like(gl).scatter(-1,idx,1.)
                if self.training:
                    soft=torch.sigmoid(gl)
                    initial=initial+soft-soft.detach()
            initial=initial*torch.sigmoid(gl)
            active=initial
            origin=torch.eye(r,device=x.device)[None].expand(b,-1,-1)
            has_previous=torch.zeros_like(active)
            output=torch.zeros(b,n,device=x.device)
            token_trace=[]
            for ci in range(cycles):
                update=self._local(state,token)
                state=state+active[:,:,None]*(update-state)
                prev_id=(origin*torch.arange(r,device=x.device)[None,None,:]).sum(-1)/r
                inp=torch.cat((state+self.region_emb[None], tired[:,:,None],prev_id[:,:,None],
                              active[:,:,None],torch.full((b,r,1),ci/max(cycles,1),device=x.device)),dim=-1)
                route_logits=self.router(inp)
                if not c.local_router: route_logits=route_logits*0
                route_logits=route_logits.clone()
                route_logits[...,0]-=fs*tired
                route_logits[...,2]-=fs*tired
                route_logits[...,2]-=20*(1-has_previous)
                if recurrence_override is not None:
                    p=max(1e-4,min(1-1e-4,float(recurrence_override)))
                    route_logits[...,0]=math.log(p/(1-p))
                elif c.fixed_recurrence_probability is not None:
                    p=max(1e-4,min(1-1e-4,c.fixed_recurrence_probability))
                    route_logits[...,0]=math.log(p/(1-p))
                if not recurrence: route_logits[...,0]=-20
                probs=route_logits.softmax(-1)
                if stochastic:
                    chosen=torch.multinomial(probs.reshape(-1,4),1,generator=generator).reshape(b,r)
                    route=F.one_hot(chosen,4).float()
                elif self.training:
                    route=probs
                    chosen=probs.argmax(-1)
                else:
                    chosen=probs.argmax(-1)
                    route=F.one_hot(chosen,4).float()
                usage=active
                output=output+(active*route[...,3])[:,:,None].mul(state).sum(1)
                if trace:
                    for bi in range(b):
                        destination=[j if int(chosen[bi,j])==0 else (j+1)%r if int(chosen[bi,j])==1 else
                                     int(origin[bi,j].argmax()) if int(chosen[bi,j])==2 else -1 for j in range(r)]
                        token_trace.append({"cycle":ci,"active":[int(j) for j in torch.where(active[bi]>.05)[0]],
                         "activation":active[bi].detach().cpu().tolist(),"fatigue":tired[bi].detach().cpu().tolist(),
                         "recurrence_probability":probs[bi,:,0].detach().cpu().tolist(),
                         "route_probability":probs[bi].detach().cpu().tolist(),
                         "chosen_route":chosen[bi].detach().cpu().tolist(),"destination":destination})
                tired=self.fatigue_module(tired,usage,fatigue,recovery)
                active,origin,has_previous=route_step(active,route,origin,has_previous)
            output=output+(active[:,:,None]*state).sum(1)
            output=output/(1+initial.sum(-1,keepdim=True))
            logits_out.append(self.head(self.norm(output))/math.sqrt(n))
            if trace: traces.append(token_trace)
        return torch.stack(logits_out,dim=1),traces if trace else None


def make_model(name,c):
    if name=="transformer": return TransformerLM(c)
    if name=="looped": return TransformerLM(c,looped=True)
    if name=="prg": return PRGLM(c)
    raise ValueError(name)

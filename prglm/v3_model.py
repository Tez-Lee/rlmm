"""Additive reuse of frozen SAME edge identities/messages; no graph routers."""
from dataclasses import asdict,dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .v1_model import ternary_ste


@dataclass
class V3Config:
    nodes:int=1024
    slots:int=32
    active_sources:int=16
    hidden:int=64
    context:int=32
    max_repeat:int=8
    family:str='repeat'
    state_decay:float=.9

    def validate(self):
        if self.family not in ('float','ternary','repeat'):raise ValueError('family')
        if not self.nodes>self.slots>=1 or not 1<=self.active_sources<=self.nodes:raise ValueError('shape')
        if not 1<=self.max_repeat<=16:raise ValueError('bound')
        if self.nodes & (self.nodes-1):raise ValueError('use power-of-two nodes for unique deterministic templates')
        return self

    def dict(self):return asdict(self)


def additive_repeat(sign,message,count):
    out=torch.zeros_like(message)
    for _ in range(count):out=out+sign*message
    return out


class PRGLMv3(nn.Module):
    def __init__(self,c:V3Config):
        super().__init__();self.c=c.validate();k=c.active_sources
        # Shared block initialization is independent of graph scale/family.
        self.emb=nn.Embedding(256,c.hidden)
        self.seed=nn.Linear(c.hidden,k)
        self.readout=nn.Linear(k,c.hidden)
        self.norm=nn.Identity()  # No normalization that cancels repetition magnitude.
        self.controller=nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,1))
        nn.init.zeros_(self.controller[-1].weight);nn.init.zeros_(self.controller[-1].bias)
        for p in self.controller.parameters():p.requires_grad_(c.family=='repeat')
        self.edges=nn.Embedding(c.nodes,c.slots,sparse=True)
        with torch.no_grad():self.edges.weight.normal_(0,.65)
        self.register_buffer('offsets',((torch.arange(c.slots)+1)*2654435761)%c.nodes)
        self.last_stats=[]

    def _graph(self,token,previous,address,mode,generator,fixed_repeat,collect,trace):
        c=self.c;k=c.active_sources;b=token.shape[0]
        src=(address[:,None]+torch.arange(k,device=token.device)*37)%c.nodes
        dst=(src[...,None]+self.offsets)%c.nodes
        raw=self.edges(src)
        weight=raw if c.family=='float' else ternary_ste(raw)
        # Capture once; no W^k composition or updated message during repeats.
        message=torch.tanh(self.seed(self.emb(token))+c.state_decay*previous)
        reference=message[...,None].expand(-1,-1,c.slots)
        contribution=weight*reference
        accum=contribution # Mandatory first activation of each edge.
        counts=torch.ones_like(contribution,dtype=torch.long)
        alive=torch.ones_like(contribution)
        maximum=fixed_repeat if fixed_repeat is not None else (c.max_repeat if c.family=='repeat' else 1)
        if maximum<1 or maximum>c.max_repeat:raise ValueError('repeat bound')
        if c.family!='repeat' and fixed_repeat not in (None,1):raise ValueError('single-pass family')
        probabilities=[];decision_masks=[]
        if maximum>1 and fixed_repeat is None:
            features=torch.stack((reference,reference.abs(),weight.detach(),
                src[...,None].expand_as(reference)/c.nodes,dst/c.nodes,
                torch.zeros_like(reference)),dim=-1)
            # Controller has no evolving state: probabilities can be batched.
            levels=torch.arange(1,maximum,device=token.device)/c.max_repeat
            features=features[...,None,:].expand(-1,-1,-1,maximum-1,-1).clone()
            features[...,-1]=levels
            logits=self.controller(features).squeeze(-1)
            probability=logits.sigmoid()
            if mode=='train':
                u=torch.rand_like(logits).clamp(1e-6,1-1e-6)
                soft=(logits+u.log()-(1-u).log()).sigmoid()
                gates=soft+((soft>=.5).float()-soft).detach()
            elif mode=='expected':gates=probability
            elif mode=='sampled':gates=(torch.rand(logits.shape,device=token.device,generator=generator)<probability).float()
            else:gates=(probability>=.5).float()
        for repeat in range(1,maximum):
            if fixed_repeat is not None:gate=torch.ones_like(alive);p=gate
            else:gate=gates[...,repeat-1];p=probability[...,repeat-1]
            decision_masks.append(alive.detach());probabilities.append(p.detach())
            alive=alive*gate
            # Exact same source, destination, slot, sign and reference every time.
            accum=accum+contribution*alive
            if mode!='expected':counts=counts+(alive.detach()>0).long()
        channels=(dst%k).flatten(1)
        updated=torch.zeros(b,k,device=token.device).scatter_add(1,channels,accum.flatten(1))/math.sqrt(c.slots)
        state=torch.tanh(updated)
        logits=F.linear(self.norm(self.readout(state)),self.emb.weight)/math.sqrt(c.hidden)
        record=None
        if collect and mode!='expected':
            n=counts.detach().flatten().float();a=accum.detach().abs().flatten()
            # Pre-nonlinearity projected contribution proxy, not causal logit attribution.
            column=self.readout.weight.detach().norm(dim=0)
            out=(accum.detach().abs()*column[dst%k]/math.sqrt(c.slots)).flatten()
            hist=torch.bincount(counts.flatten(),minlength=c.max_repeat+1)[1:c.max_repeat+1]
            grouped=torch.zeros(c.max_repeat,device=token.device).scatter_add(0,counts.flatten()-1,out)
            prob_sum=0.;decisions=0.
            for p,mask in zip(probabilities,decision_masks):
                prob_sum+=float((p*mask).sum());decisions+=float(mask.sum())
            record={'tokens':b,'hist':hist.cpu().tolist(),'contribution_sum_by_count':grouped.cpu().tolist(),
                'moments':[float(n.numel()),float(n.sum()),float(n.square().sum()),float(out.sum()),float(out.square().sum()),float((n*out).sum()),float(a.sum()),float(a.square().sum()),float((n*a).sum())],
                'repeat_probability_sum':prob_sum,'controller_decisions':decisions,
                'nonzero_edge_operations':float((counts*(weight.detach()!=0)).sum()),
                'controller_candidate_evaluations':b*k*c.slots*(maximum-1) if fixed_repeat is None else 0,
                'same_identity_verified':True,'source_nodes':src.detach().cpu().tolist()}
        events=[]
        if trace:
            for j in range(min(k,2)):
                for slot in range(min(c.slots,2)):
                    events.append({'source':int(src[0,j]),'destination':int(dst[0,j,slot]),'slot':slot,
                        'edge_id':int(src[0,j])*c.slots+slot,'reference_message':float(reference[0,j,slot]),
                        'edge_value':float(weight[0,j,slot]),'repeats':int(counts[0,j,slot]),
                        'identity_sequence':[[int(src[0,j]),slot,int(dst[0,j,slot])]]*int(counts[0,j,slot])})
        return logits,state,record,events

    def forward(self,x,routing_mode=None,seed=None,fixed_repeat=None,collect=False,trace=False):
        if x.shape[1]>self.c.context:raise ValueError('context')
        mode=routing_mode or ('train' if self.training else 'sampled')
        generator=None
        if seed is not None:
            # CPU generator avoids MPS generator limitations; CUDA uses CUDA RNG.
            generator=torch.Generator(device=x.device);generator.manual_seed(seed)
        h=torch.zeros(x.shape[0],self.c.active_sources,device=x.device)
        address=torch.zeros(x.shape[0],dtype=torch.long,device=x.device)
        predictions=[];records=[];trajectory=[]
        for ti in range(x.shape[1]):
            address=(address*131+x[:,ti]+1)%self.c.nodes
            y,h,stats,events=self._graph(x[:,ti],h,address,mode,generator,fixed_repeat,collect,trace)
            predictions.append(y)
            if stats:records.append(stats)
            if trace:trajectory.append(events)
        self.last_stats=records
        return torch.stack(predictions,1),trajectory

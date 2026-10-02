"""Slow discrete topology updates; all statistics are training-only buffers.

Output path credit is a magnitude/success proxy, not a causal loss improvement.
No gateway ID has a gradient. Mutation RNG never consumes the neural RNG stream.
"""
import math
import torch
from torch import nn


def standardize(value):
    return (value-value.mean())/value.std(unbiased=False).clamp_min(1e-8)


def entropy(counts):
    p=counts.double()/counts.sum().clamp_min(1)
    return float(-(p*p.clamp_min(1e-30).log()).sum())


class MutableTopology(nn.Module):
    def __init__(self,c,table):
        super().__init__();self.c=c
        r,g=c.regions,c.gateways
        for name in ('visit','selection','output_credit','success','activity','message','accumulator','gradient'):
            self.register_buffer('ema_'+name,torch.zeros(r))
            self.register_buffer('pending_'+name,torch.zeros(r))
        for name in ('selection','probability','credit'):
            self.register_buffer('gateway_'+name,torch.zeros(r,g))
            self.register_buffer('pending_gateway_'+name,torch.zeros(r,g))
        self.register_buffer('age',torch.zeros(r,g,dtype=torch.long))
        self.register_buffer('old_destination',torch.full((r,g),-1,dtype=torch.long))
        self.register_buffer('probation_credit',torch.zeros(r,g))
        self.register_buffer('ever_edges',torch.zeros(r,r,dtype=torch.bool))
        self.ever_edges.scatter_(1,table.cpu(),True)
        self.register_buffer('step',torch.zeros((),dtype=torch.long))
        # Dedicated CPU RNG state is persistent, including across device moves.
        generator=torch.Generator().manual_seed(c.seed+1000003)
        self.register_buffer('rng_state',generator.get_state())
        self.history=[]
        self.counters={'attempts':0,'accepted':0,'reverted':0,'exploration':0,'exploitation':0}
        self.total_lifetime=0;self.finished_lifetimes=0

    def get_extra_state(self):
        return {'history':self.history,'counters':self.counters,
                'total_lifetime':self.total_lifetime,'finished_lifetimes':self.finished_lifetimes}

    def set_extra_state(self,state):
        self.history=state['history'];self.counters=state['counters']
        self.total_lifetime=state['total_lifetime'];self.finished_lifetimes=state['finished_lifetimes']

    def begin_token(self,batch,walkers,device):
        self.path_regions=torch.zeros(batch,walkers,self.c.regions,device=device)
        self.path_gateways=torch.zeros(batch,walkers,self.c.regions,self.c.gateways,device=device)

    @torch.no_grad()
    def observe(self,bi,wi,region,action,prob,message,acc,contribution,output_gate):
        ones=torch.ones_like(region,dtype=self.ema_visit.dtype)
        self.pending_visit.index_add_(0,region,ones)
        activity=(message.detach().abs().mean(-1)+acc.detach().abs().mean(-1))/2
        self.pending_activity.index_add_(0,region,activity)
        self.pending_message.index_add_(0,region,message.detach().abs().mean(-1))
        self.pending_accumulator.index_add_(0,region,acc.detach().abs().mean(-1))
        self.path_regions[bi,wi,region]+=1
        self.pending_gateway_probability.index_add_(0,region,prob.detach()[:,1:self.c.gateways+1])
        gateway=(action>=1)&(action<=self.c.gateways)
        if gateway.any():
            sources=region[gateway];slots=action[gateway]-1
            self.pending_selection.index_add_(0,sources,ones[gateway])
            self.pending_gateway_selection.index_put_((sources,slots),ones[gateway],accumulate=True)
            self.path_gateways[bi[gateway],wi[gateway],sources,slots]+=1
        magnitude=contribution.detach().norm(dim=-1)*output_gate.detach()
        self.credit_paths(bi,wi,magnitude)

    @torch.no_grad()
    def credit_paths(self,bi,wi,magnitude):
        # Divide path credit among visits; a frequently revisited region doesn't
        # receive unbounded credit just because the trajectory is long.
        region_path=self.path_regions[bi,wi]
        credit=region_path/region_path.sum(-1,keepdim=True).clamp_min(1)
        self.pending_output_credit.add_((credit*magnitude[:,None]).sum(0))
        self.pending_success.add_((credit*(magnitude>0)[:,None]).sum(0))
        gateway_path=self.path_gateways[bi,wi]
        gateway_credit=gateway_path/gateway_path.sum((1,2),keepdim=True).clamp_min(1)
        credited=(gateway_credit*magnitude[:,None,None]).sum(0)
        self.pending_gateway_credit.add_(credited)
        self.probation_credit.add_(credited)

    def gradient_hook(self,activation,regions):
        if not self.c.gradient_credit or not activation.requires_grad:return
        detached=activation.detach();regions=regions.detach()
        def hook(grad):
            with torch.no_grad():
                self.pending_gradient.index_add_(0,regions,(detached*grad.detach()).abs().mean(-1))
            return grad
        activation.register_hook(hook)

    def score_components(self,table):
        indegree=torch.bincount(table.flatten(),minlength=self.c.regions).float()
        usefulness=self.ema_output_credit/self.ema_visit.clamp_min(1e-8)
        visits=self.ema_visit/self.ema_visit.sum().clamp_min(1e-8)
        return {'usefulness':usefulness,'visit_activity':visits,'in_degree':indegree,
                'success_per_visit':self.ema_success/self.ema_visit.clamp_min(1e-8),
                'combined_message_accumulator_activity':self.ema_activity/self.ema_visit.clamp_min(1e-8),
                'message_activity':self.ema_message/self.ema_visit.clamp_min(1e-8),
                'accumulator_activity':self.ema_accumulator/self.ema_visit.clamp_min(1e-8),
                'route_selection_frequency':self.ema_selection/self.ema_visit.clamp_min(1e-8),
                'gradient_credit':self.ema_gradient/self.ema_visit.clamp_min(1e-8)}

    def hot_score(self,table):
        v=self.score_components(table)
        return (self.c.usefulness_weight*standardize(v['usefulness'])+
                self.c.visit_weight*standardize(v['visit_activity'])-
                self.c.congestion_weight*standardize(v['in_degree']))

    def candidate_distribution(self,source,table,exploration=False):
        valid=torch.ones(self.c.regions,dtype=torch.bool,device=table.device)
        valid[source]=False;valid[table[source]]=False
        if exploration or self.c.topology_mode=='uniform':
            weights=torch.ones(self.c.regions,device=table.device)
            if exploration and self.c.exploration_policy=='low_visit':
                weights=1/(self.ema_visit+1/self.c.regions)
            weights=weights*valid
        else:
            logits=(self.hot_score(table)/self.c.hotness_temperature).masked_fill(~valid,-torch.inf)
            if not valid.any():return torch.zeros_like(logits)
            weights=logits.softmax(0)
        return weights/weights.sum().clamp_min(1e-20)

    @torch.no_grad()
    def after_optimizer_step(self,table,training):
        if not training:return
        self.step.add_(1);self.age.add_(1)
        decay=self.c.hotness_ema_decay
        for name in ('visit','selection','output_credit','success','activity','message','accumulator','gradient'):
            pending=getattr(self,'pending_'+name)
            getattr(self,'ema_'+name).mul_(decay).add_(pending,alpha=1-decay)
            pending.zero_()
        for name in ('selection','probability','credit'):
            pending=getattr(self,'pending_gateway_'+name)
            getattr(self,'gateway_'+name).mul_(decay).add_(pending,alpha=1-decay)
            pending.zero_()
        if self.c.topology_mode=='static':return
        step=int(self.step)
        # Decisions only on the specified slow timescale, including probation.
        if step % self.c.mutation_interval:return
        if self.c.probation:
            due=(self.old_destination>=0)&(self.age>=self.c.probation_steps)
            for source,slot in due.nonzero().tolist():
                old=int(self.old_destination[source,slot]);new=int(table[source,slot])
                if float(self.probation_credit[source,slot])<=0 and not (table[source]==old).any():
                    self.total_lifetime+=int(self.age[source,slot]);self.finished_lifetimes+=1
                    table[source,slot]=old;self.counters['reverted']+=1
                    self.history.append({'step':step,'source':source,'slot':slot,'old':new,'new':old,
                                         'reason':'no downstream output credit during probation','kind':'revert','age':int(self.age[source,slot])})
                    self.age[source,slot]=0
                else:
                    self.history.append({'step':step,'source':source,'slot':slot,'old':old,'new':new,
                        'reason':'positive downstream credit' if float(self.probation_credit[source,slot])>0 else 'old destination occupied by another gateway',
                        'kind':'survive','age':int(self.age[source,slot]),'downstream_credit':float(self.probation_credit[source,slot])})
                self.old_destination[source,slot]=-1
        generator=torch.Generator();generator.set_state(self.rng_state.cpu())
        changed=0
        for source in torch.randperm(self.c.regions,generator=generator).tolist():
            if changed>=self.c.maximum_rewires_per_interval:break
            eligible=(self.old_destination[source]<0)&(self.age[source]>=self.c.probation_steps)
            if not eligible.any():continue
            if float(torch.rand((),generator=generator))>=self.c.mutation_probability:continue
            self.counters['attempts']+=1
            # Score both probability/usage and downstream proxy credit.
            utility=self.gateway_credit[source]+.01*self.gateway_selection[source]
            slot=int(utility.masked_fill(~eligible,torch.inf).argmin())
            exploration=self.c.topology_mode=='uniform' or float(torch.rand((),generator=generator))<self.c.exploration_epsilon
            weights=self.candidate_distribution(source,table,exploration)
            if not weights.any():continue
            destination=int(torch.multinomial(weights.cpu(),1,generator=generator))
            old=int(table[source,slot]);age=int(self.age[source,slot])
            self.total_lifetime+=age;self.finished_lifetimes+=1
            kind='exploration' if exploration else 'exploitation'
            self.history.append({'step':step,'source':source,'slot':slot,'old':old,'new':destination,
                'reason':'least credited/used eligible slot','kind':kind,'age':age,
                'candidate_probability':float(weights[destination]),
                'hot_score':float(self.hot_score(table)[destination]),
                'replaced_selection_ema':float(self.gateway_selection[source,slot]),
                'replaced_probability_ema':float(self.gateway_probability[source,slot]),
                'replaced_downstream_credit_ema':float(self.gateway_credit[source,slot])})
            table[source,slot]=destination
            self.old_destination[source,slot]=old if self.c.probation else -1
            self.age[source,slot]=0;self.probation_credit[source,slot]=0
            for name in ('selection','probability','credit'):getattr(self,'gateway_'+name)[source,slot]=0
            self.ever_edges[source,destination]=True
            self.counters['accepted']+=1;self.counters[kind]+=1;changed+=1
        self.rng_state.copy_(generator.get_state().to(self.rng_state.device))

    def snapshot(self,table):
        comp=self.score_components(table)
        degrees=comp['in_degree'].cpu().double()
        sorted_degree=degrees.sort().values;n=len(degrees)
        gini=float((2*torch.arange(1,n+1)*sorted_degree).sum()/(n*sorted_degree.sum())-(n+1)/n)
        return {'step':int(self.step),'gateway_table':table.detach().cpu().tolist(),
            'hotness':self.hot_score(table).detach().cpu().tolist(),
            'components':{k:v.detach().cpu().tolist() for k,v in comp.items()},
            'counters':dict(self.counters),'gateway_age':self.age.cpu().tolist(),
            'gateway_selection':self.gateway_selection.cpu().tolist(),
            'gateway_probability':self.gateway_probability.cpu().tolist(),
            'gateway_downstream_credit':self.gateway_credit.cpu().tolist(),
            'topology_entropy':entropy(degrees),'visit_entropy':entropy(self.ema_visit),
            'indegree_gini':gini,'largest_hub':int(degrees.max()),
            'isolated_regions':int((degrees==0).sum()),'near_isolated_regions':int((degrees<=1).sum()),
            'unique_destination_regions':int((degrees>0).sum()),
            'unique_edges_ever':int(self.ever_edges.sum()),
            'current_unique_edges':int(table.numel()),
            'mean_completed_gateway_lifetime':self.total_lifetime/max(1,self.finished_lifetimes),
            'mean_current_gateway_age':float(self.age.float().mean()),
            'training_metadata_tensor_bytes':sum(b.numel()*b.element_size() for b in self.buffers()),
            'history':list(self.history)}

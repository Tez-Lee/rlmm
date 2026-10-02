"""Separate v2 subclass. The hard-token engine is v1 plus detached observers.

Neural initialization and random draws are identical to v1 with mutation off.
Mutation is explicit after the optimizer update; eval never changes topology.
"""
import math
import torch
from torch.nn import functional as F
from .v1_model import PRGLMv1
from .v2_config import V2Config
from .topology import MutableTopology


class PRGLMv2(PRGLMv1):
    def __init__(self,c:V2Config):
        super().__init__(c)
        self.topology=MutableTopology(c,self.gateway_table)

    def after_optimizer_step(self):
        self.topology.after_optimizer_step(self.gateway_table,self.training)

    def _router_logits(self,regions,previous,message,fatigue,acc_message,cycle,fatigue_strength=None):
        base,prob,logits=super()._router_logits(regions,previous,message,fatigue,acc_message,cycle,fatigue_strength)
        if not self.c.recurrence:
            # v1 uses finite -30 sentinels. v2 OFF means strictly forbidden,
            # even if the network assigns an OUTPUT logit below -30.
            mask=torch.zeros_like(logits,dtype=torch.bool)
            mask[:,0]=True;mask[:,self.c.gateways+1]=True
            mask[:,1:self.c.gateways+1]=self.gateway_table[regions]<=regions[:,None]
            logits=logits.masked_fill(mask,-torch.inf)
            prob=logits.softmax(-1)
            base=prob  # Fatigue penalties affect only the forbidden LOCAL/RETURN actions.
        return base,prob,logits

    def _hard_token(self,token_ids,state,fatigue,mode,generator,trace,forced_region=None,
                    recurrence=None,accumulation=None,fatigue_on=None,max_cycles=None,
                    recurrence_override=None,fatigue_strength=None,recovery=None):
        c=self.c;b=token_ids.shape[0];w=c.initial_regions;r=c.regions;n=c.region_size
        if self.training:self.topology.begin_token(b,w,token_ids.device)
        token=self.emb(token_ids)
        global_logits=self.global_router(token)
        if forced_region is not None:
            global_logits=global_logits.clone();global_logits[:,int(forced_region)]+=20
        current=global_logits.topk(w,dim=-1).indices
        initial_strength=torch.sigmoid(global_logits.gather(1,current))
        indices=self._initial_indices(token_ids[:,None],current)
        message=self.seed_proj(token)[:,None,:]*initial_strength[...,None]
        previous=torch.full_like(current,-1)
        alive=torch.ones_like(current,dtype=torch.bool)
        visited=torch.zeros(b,w,r,device=token_ids.device,dtype=torch.bool)
        accum=torch.zeros_like(state)
        readout=torch.zeros(b,c.d_model,device=token_ids.device)
        steps=[];edge_visits=0;nonzero_edge_visits=0;region_visits=torch.zeros(b,r,device=token_ids.device)
        visited_regions=[];base_local=[];expected_visits=[];acc_magnitudes=[];contributions=[]
        cycles=max_cycles or c.max_cycles
        for cycle in range(cycles):
            bi,wi=alive.nonzero(as_tuple=True)
            if bi.numel()==0: break
            region=current[bi,wi]; old_prev=previous[bi,wi]
            visited=visited.index_put((bi,wi,region),torch.ones_like(region,dtype=torch.bool))
            idx=indices[bi,wi];msg=message[bi,wi]
            region_visits=region_visits.flatten().scatter_add(0,bi*r+region,
                torch.ones_like(region,dtype=region_visits.dtype)).view(b,r)
            old_acc=accum[bi[:,None],region[:,None],idx].abs().mean(-1)
            state,accum,next_idx,next_msg,acc_msg,used,nonzero_used=self._propagate(
                state,accum,bi,region,idx,msg,torch.ones_like(region,dtype=msg.dtype),accumulation)
            edge_visits+=used;nonzero_edge_visits+=nonzero_used
            tired=fatigue[bi,region]
            base_prob,prob,route_logits=self._router_logits(region,old_prev,next_msg,tired,acc_msg,cycle,fatigue_strength)
            if recurrence_override is not None:
                p=max(1e-4,min(1-1e-4,float(recurrence_override)))
                route_logits=route_logits.clone()
                route_logits[:,0]=math.log(p/(1-p))+torch.logsumexp(route_logits[:,1:],dim=-1)
                prob=route_logits.softmax(-1)
            if recurrence is False or (recurrence is None and not c.recurrence):
                route_logits=route_logits.clone();route_logits[:,0]=-torch.inf
                route_logits[:,c.gateways+1]=-torch.inf
                for gateway in range(c.gateways):
                    target=self.gateway_table[region,gateway]
                    route_logits[visited[bi,wi,target],gateway+1]=-torch.inf
                prob=route_logits.softmax(-1)
            if fatigue_on is False and c.fatigue:
                route_logits=route_logits.clone()
                strength=c.fatigue_strength if fatigue_strength is None else fatigue_strength
                route_logits[:,0]+=strength*tired
                route_logits[:,c.gateways+1]+=strength*tired
                prob=route_logits.softmax(-1)
            if mode=='train':
                route=F.gumbel_softmax(route_logits,tau=1.,hard=True)
                action=route.argmax(-1)
            elif mode=='sampled':
                action=torch.multinomial(prob,1,generator=generator).squeeze(-1)
                route=F.one_hot(action,c.gateways+3).float()
            else:
                action=prob.argmax(-1)
                route=F.one_hot(action,c.gateways+3).float()
            gateway_action=(action>=1)&(action<=c.gateways)
            returning=action==c.gateways+1
            outputting=action==c.gateways+2
            gateway_index=(action-1).clamp(0,c.gateways-1)
            gateway_dest=self.gateway_table[region,gateway_index]
            destination=torch.where(gateway_action,gateway_dest,torch.where(returning,old_prev,region))
            destination=torch.where(outputting,torch.full_like(destination,-1),destination)
            contribution=self.readout_proj(acc_msg if (accumulation is None and c.accumulation) or accumulation else next_msg)
            output_gate=route[:,c.gateways+2]
            if self.training:
                self.topology.observe(bi,wi,region,action,prob,next_msg,acc_msg,contribution,output_gate)
                self.topology.gradient_hook(acc_msg,region)
            readout=readout.index_add(0,bi,contribution*output_gate[:,None])
            route_gain=route.gather(1,action[:,None]).squeeze(-1)
            indices=indices.index_put((bi,wi),next_idx)
            message=message.index_put((bi,wi),next_msg*route_gain[:,None])
            current=current.index_put((bi,wi),destination.clamp_min(0))
            new_prev=torch.where(action==0,old_prev,region)
            previous=previous.index_put((bi,wi),new_prev)
            alive=alive.index_put((bi,wi),~outputting)
            decay=c.recovery if recovery is None else recovery
            fatigue=fatigue*decay if (fatigue_on is None and c.fatigue) or fatigue_on else fatigue*0
            usage=torch.zeros_like(fatigue).flatten().scatter_add(0,bi*r+region,
                torch.ones_like(region,dtype=fatigue.dtype)).view_as(fatigue)
            fatigue=fatigue+usage if (fatigue_on is None and c.fatigue) or fatigue_on else fatigue
            visited_regions.extend(region.detach().cpu().tolist())
            base_local.extend(base_prob[:,0].detach().cpu().tolist())
            expected_visits.extend((1-prob[:,-1]).detach().cpu().tolist())
            acc_magnitudes.extend(acc_msg.abs().mean(-1).detach().cpu().tolist())
            contributions.extend((contribution.norm(dim=-1)*output_gate).detach().cpu().tolist())
            if trace:
                events=[]
                for j in range(region.numel()):
                    if int(bi[j])!=0: continue
                    events.append({'region':int(region[j]),'previous_region':int(old_prev[j]),
                        'destination':int(destination[j]),'action':int(action[j]),
                        'base_probability':base_prob[j].detach().cpu().tolist(),
                        'route_probability':prob[j].detach().cpu().tolist(),
                        'accumulator_before':float(old_acc[j]),
                        'accumulator_magnitude':float(acc_msg[j].abs().mean()),
                        'contribution_magnitude':float(contribution[j].norm()*output_gate[j]),
                        'walker':int(wi[j]),
                        'fatigue':float(tired[j]),'active_nodes':c.active_nodes,
                        'node_indices':idx[j].detach().cpu().tolist(),
                        'nonzero_edges':int((self.local_edge[region[j],idx[j]].detach().abs()>.33).sum())})
                steps.append({'cycle':cycle,'events':events,'active':[e['region'] for e in events],
                    'fatigue':fatigue[0].detach().cpu().tolist(),
                    'accumulator_magnitude':accum[0].abs().mean(-1).detach().cpu().tolist(),
                    'region_visits':region_visits[0].detach().cpu().tolist()})
        if alive.any():
            bi,wi=alive.nonzero(as_tuple=True)
            region=current[bi,wi];idx=indices[bi,wi]
            last=accum[bi[:,None],region[:,None],idx] if (accumulation is None and c.accumulation) or accumulation else message[bi,wi]
            timeout_contribution=self.readout_proj(last)
            readout=readout.index_add(0,bi,timeout_contribution)
            if self.training:self.topology.credit_paths(bi,wi,timeout_contribution.detach().norm(dim=-1))
        logits=self.head(self.norm(readout/w))/math.sqrt(c.d_model)
        diagnostics={'edge_visits':edge_visits,'nonzero_edge_visits':nonzero_edge_visits,
            'region_visits':region_visits.detach().cpu().tolist(),
            'visited_regions':visited_regions,'base_local_probability':base_local,
            'expected_nonoutput_probability':expected_visits,'accumulator_magnitudes':acc_magnitudes,
            'contribution_magnitudes':contributions}
        return logits,state,fatigue,steps,diagnostics

"""PRG-v2.1: tiny contextual readout, valid cap source, clean accumulation.

No mutable topology or extra neural block. Uses v1 propagation/RouterNet as-is.
Expected routing remains a soft region-mass surrogate, not an exact expectation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .v1_model import PRGLMv1
from .v21_config import V21Config


class PRGLMv21(PRGLMv1):
    def __init__(self,c:V21Config):
        super().__init__(c)
        self.context_gate=nn.Parameter(torch.zeros(()))

    def _context(self,state):
        # Fixed node-index-mod-K channels, variance-preserving sum/sqrt(M).
        c=self.c;cells=c.regions*c.region_size//c.active_nodes
        return torch.tanh(state.reshape(state.shape[0],c.regions,
            c.region_size//c.active_nodes,c.active_nodes).sum((1,2))/math.sqrt(cells))

    def _finish(self,readout,context):
        # Reuse existing projection weight; no second projection bias.
        contextual=F.linear(context,self.readout_proj.weight)
        combined=readout/self.c.initial_regions+torch.sigmoid(self.context_gate)*contextual
        return self.head(self.norm(combined))/math.sqrt(self.c.d_model)

    def _probe_logits(self,logits,hops,cycle,cycles):
        if not self.c.minimum_gateway_hops:return logits
        needed=(self.c.minimum_gateway_hops-hops).clamp_min(0)
        mask=torch.zeros_like(logits,dtype=torch.bool)
        mask[:,-1]=needed>0
        # Reserve a cycle to PROCESS the final required destination.
        must_gateway=(needed>0)&(cycles-cycle<=needed+1)
        mask[:,0]=must_gateway;mask[:,self.c.gateways+1]=must_gateway
        return logits.masked_fill(mask,-torch.inf)

    def _router_logits(self,regions,previous,message,fatigue,acc_message,cycle,fatigue_strength=None):
        base,prob,logits=super()._router_logits(regions,previous,message,fatigue,acc_message,cycle,fatigue_strength)
        # RETURN with no previous Region is invalid, never destination -1.
        mask=torch.zeros_like(logits,dtype=torch.bool)
        mask[:,self.c.gateways+1]=previous<0
        if not self.c.recurrence:
            mask[:,0]=True;mask[:,self.c.gateways+1]=True
            mask[:,1:self.c.gateways+1]=self.gateway_table[regions]<=regions[:,None]
        logits=logits.masked_fill(mask,-torch.inf)
        return base,logits.softmax(-1),logits

    def _hard_token(self,token_ids,state,fatigue,mode,generator,trace,forced_region=None,
                    recurrence=None,accumulation=None,fatigue_on=None,max_cycles=None,
                    recurrence_override=None,fatigue_strength=None,recovery=None):
        c=self.c;b=token_ids.shape[0];w=c.initial_regions;r=c.regions;n=c.region_size
        context=self._context(state)  # Prefix-only state before this token.
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
        hops=torch.zeros_like(current)
        last_valid=torch.zeros_like(message)
        action_counts=torch.zeros(c.gateways+3,device=token_ids.device)
        first_outputs=torch.zeros(b,w,device=token_ids.device)
        token_cycles=torch.zeros(b,device=token_ids.device)
        visited=torch.zeros(b,w,r,device=token_ids.device,dtype=torch.bool)
        accum=torch.zeros_like(state)
        readout=torch.zeros(b,c.d_model,device=token_ids.device)
        steps=[];edge_visits=0;nonzero_edge_visits=0;region_visits=torch.zeros(b,r,device=token_ids.device)
        visited_regions=[];base_local=[];expected_visits=[];acc_magnitudes=[];contributions=[]
        cycles=max_cycles or c.max_cycles
        if c.minimum_gateway_hops>=cycles:raise ValueError("probe needs an extra processing cycle")
        for cycle in range(cycles):
            bi,wi=alive.nonzero(as_tuple=True)
            if bi.numel()==0: break
            token_cycles=token_cycles+alive.any(1).float()
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
            route_logits=self._probe_logits(route_logits,hops[bi,wi],cycle,cycles)
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
            # ON/OFF both use exactly this accumulator readout and nonlinearity.
            contribution=self.readout_proj(torch.tanh(acc_msg))
            last_valid=last_valid.index_put((bi,wi),acc_msg)
            hops=hops.index_put((bi,wi),hops[bi,wi]+gateway_action.long())
            action_counts=action_counts+F.one_hot(action,c.gateways+3).sum(0).detach()
            if cycle==0:first_outputs=first_outputs.index_put((bi,wi),outputting.float())
            output_gate=route[:,c.gateways+2]
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
            # Last processed source is valid even after a final-cycle gateway.
            timeout_contribution=self.readout_proj(torch.tanh(last_valid[bi,wi]))
            readout=readout.index_add(0,bi,timeout_contribution)
        logits=self._finish(readout,context)
        diagnostics={'edge_visits':edge_visits,'nonzero_edge_visits':nonzero_edge_visits,
            'region_visits':region_visits.detach().cpu().tolist(),
            'visited_regions':visited_regions,'base_local_probability':base_local,
            'expected_nonoutput_probability':expected_visits,'accumulator_magnitudes':acc_magnitudes,
            'contribution_magnitudes':contributions}
        diagnostics['routing_counts']={
            'tokens':b,'actions':action_counts.detach().cpu().tolist(),
            'gateway_hops':int(hops.sum()),'tokens_with_gateway':int((hops.sum(1)>0).sum()),
            'first_cycle_outputs':float(first_outputs.sum()),'initial_walkers':b*w,
            'all_walkers_output_first':int((first_outputs.sum(1)==w).sum()),
            'unique_regions':float((region_visits>0).sum()),
            'trajectory_cycles':float(token_cycles.sum())}
        return logits,state,fatigue,steps,diagnostics

    def _expected_token(self,token_ids,state,fatigue,trace=False):
        """Soft region-mass surrogate, without sampled categorical decisions."""
        c=self.c;b=token_ids.shape[0];r=c.regions;k=c.active_nodes
        if c.minimum_gateway_hops:raise ValueError("expected probe is undefined; use sampled/argmax")
        context=self._context(state)
        token=self.emb(token_ids);gl=self.global_router(token)
        selected=gl.topk(c.initial_regions,dim=-1).indices
        mass=torch.zeros_like(gl).scatter(1,selected,torch.sigmoid(gl.gather(1,selected)))
        regions=torch.arange(r,device=token_ids.device)[None].expand(b,-1)
        indices=self._initial_indices(token_ids[:,None],regions)
        message=self.seed_proj(token)[:,None,:].expand(-1,r,-1)
        origin=torch.eye(r,device=token_ids.device)[None].expand(b,-1,-1)
        valid_prev=torch.zeros_like(mass)
        accum=torch.zeros_like(state);readout=torch.zeros(b,c.d_model,device=token_ids.device)
        for cycle in range(c.max_cycles):
            bi,ri=(mass>1e-7).nonzero(as_tuple=True)
            if bi.numel()==0:break
            state,accum,idx,msg,acc_msg,_,_=self._propagate(state,accum,bi,ri,indices[bi,ri],message[bi,ri],mass[bi,ri])
            next_indices=indices.index_put((bi,ri),idx)
            next_message=message.index_put((bi,ri),msg)
            prev_id=(origin[bi,ri]*torch.arange(r,device=token_ids.device)).sum(-1)
            prev_id=torch.where(valid_prev[bi,ri]>1e-6,prev_id,torch.full_like(prev_id,-1)).long()
            base,prob,_=self._router_logits(ri,prev_id,msg,fatigue[bi,ri],acc_msg,cycle)
            probs=torch.zeros(b,r,c.gateways+3,device=token_ids.device).index_put((bi,ri),prob)
            readout=readout.index_add(0,bi,self.readout_proj(torch.tanh(acc_msg))*mass[bi,ri,None]*prob[:,-1,None])
            if cycle==c.max_cycles-1:
                readout=readout.index_add(0,bi,self.readout_proj(torch.tanh(acc_msg))*mass[bi,ri,None]*(1-prob[:,-1,None]))
                mass=torch.zeros_like(mass)
                break
            next_mass=mass*probs[:,:,0]
            next_msg_mass=next_message*next_mass[:,:,None]
            next_orig_mass=origin*next_mass[:,:,None]
            next_valid_mass=valid_prev*next_mass
            for gateway in range(c.gateways):
                destination=self.gateway_table[:,gateway][None].expand(b,-1)
                flux=mass*probs[:,:,gateway+1]
                next_mass=next_mass.scatter_add(1,destination,flux)
                next_msg_mass=next_msg_mass.scatter_add(1,destination[:,:,None].expand(-1,-1,k),next_message*flux[:,:,None])
                address=(torch.arange(b,device=mass.device)[:,None]*r+destination).reshape(-1)
                source_onehot=F.one_hot(regions,r).float()*flux[:,:,None]
                next_orig_mass=next_orig_mass.reshape(b*r,r).index_add(0,address,source_onehot.reshape(b*r,r)).view(b,r,r)
                next_valid_mass=next_valid_mass.scatter_add(1,destination,flux)
            return_flux=origin*(mass*probs[:,:,c.gateways+1])[:,:,None]
            next_mass=next_mass+return_flux.sum(1)
            next_msg_mass=next_msg_mass+torch.einsum('bsd,bsk->bdk',return_flux,next_message)
            next_orig_mass=next_orig_mass+return_flux.transpose(1,2)
            next_valid_mass=next_valid_mass+return_flux.sum(1)
            message=next_msg_mass/next_mass.clamp_min(1e-8)[:,:,None]
            origin=next_orig_mass/next_mass.clamp_min(1e-8)[:,:,None]
            valid_prev=next_valid_mass/next_mass.clamp_min(1e-8)
            mass=next_mass
            indices=next_indices
            fatigue=c.recovery*fatigue+mass if c.fatigue else torch.zeros_like(fatigue)
        return self._finish(readout,context),state,fatigue
